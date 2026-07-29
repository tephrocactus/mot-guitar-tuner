//! Filesystem-backed model browser and per-model tone settings.
//!
//! All file work belongs on the plugin's loader/UI worker.  None of this
//! module is suitable for an audio callback.

use std::collections::BTreeMap;
use std::env;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::cabinet::{CabinetIrImportOptions, MAX_IR_SAMPLES, PreparedCabinetIr};
use crate::capture::CaptureTarget;
use crate::model::{
    ModelError, ModelMetadata, ModelRef, ModelRuntimeLimits, MotModel, Sha256Digest, sha256,
};
use crate::nam_import::{MAX_NAM_SOURCE_BYTES, convert_nam};
use crate::wav_io::decode_mono_wav;

pub const TONE_SETTINGS_VERSION: u32 = 1;
pub const IR_IMPORT_METADATA_VERSION: u32 = 1;

const APPLICATION_SUPPORT_RELATIVE: &str = "Library/Application Support/Plut&Mot/MOT Guitar Plugin";
const MODELS_DIRECTORY: &str = "Models";
const MODEL_SETTINGS_DIRECTORY: &str = "Model Settings";
const IRS_DIRECTORY: &str = "IRs";
const CAPTURE_RECORDS_DIRECTORY: &str = "Capture Records";
static TEMP_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelLibraryPaths {
    pub plugin_root: PathBuf,
    pub models: PathBuf,
    pub model_settings: PathBuf,
    pub irs: PathBuf,
    pub capture_records: PathBuf,
}

impl ModelLibraryPaths {
    #[must_use]
    pub fn from_home(home: impl AsRef<Path>) -> Self {
        let plugin_root = home.as_ref().join(APPLICATION_SUPPORT_RELATIVE);
        Self::from_plugin_root(plugin_root)
    }

    #[must_use]
    pub fn from_plugin_root(plugin_root: impl Into<PathBuf>) -> Self {
        let plugin_root = plugin_root.into();
        Self {
            models: plugin_root.join(MODELS_DIRECTORY),
            model_settings: plugin_root.join(MODEL_SETTINGS_DIRECTORY),
            irs: plugin_root.join(IRS_DIRECTORY),
            capture_records: plugin_root.join(CAPTURE_RECORDS_DIRECTORY),
            plugin_root,
        }
    }

    pub fn for_current_user() -> Result<Self, LibraryError> {
        let home = env::var_os("HOME").ok_or(LibraryError::HomeDirectoryUnavailable)?;
        Ok(Self::from_home(home))
    }
}

#[derive(Clone, Debug)]
pub struct ModelEntry {
    pub path: PathBuf,
    pub reference: ModelRef,
    pub metadata: ModelMetadata,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScanIssue {
    pub path: PathBuf,
    pub message: String,
}

#[derive(Clone, Debug, Default)]
pub struct ModelScan {
    pub models: Vec<ModelEntry>,
    pub issues: Vec<ScanIssue>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrainerCapturePreset {
    pub model_id: String,
    pub target: CaptureTarget,
    pub amplifier: String,
    pub amplifier_channel: String,
    pub control_positions: String,
    pub interface_output: String,
    pub interface_input: String,
    pub reamp_box: String,
    pub reactive_load: String,
    pub load_impedance_ohms: Option<u16>,
    pub return_gain_note: String,
}

#[derive(Clone, Debug)]
pub struct IrLibraryEntry {
    pub path: PathBuf,
    pub metadata: IrImportMetadata,
}

#[derive(Clone, Debug, Default)]
pub struct IrLibraryScan {
    pub entries: Vec<IrLibraryEntry>,
    pub issues: Vec<ScanIssue>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IrProcessingMode {
    MinimumPhaseAutoTrim,
    Raw,
}

impl IrProcessingMode {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::MinimumPhaseAutoTrim => "minimum_phase_auto_trim",
            Self::Raw => "raw",
        }
    }
}

impl FromStr for IrProcessingMode {
    type Err = LibraryError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "minimum_phase_auto_trim" => Ok(Self::MinimumPhaseAutoTrim),
            "raw" => Ok(Self::Raw),
            _ => Err(LibraryError::InvalidToneSettings(format!(
                "unknown IR processing mode {value:?}"
            ))),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IrReference {
    pub ir_id: String,
    pub sha256: Sha256Digest,
    pub filename_hint: String,
    pub processing: IrProcessingMode,
}

/// Immutable provenance recorded beside an imported cabinet IR.
///
/// The archived WAV is always the exact source byte stream. Processing remains
/// a runtime choice; these fields describe the default MPT + auto-trim result
/// without replacing or rewriting the RAW asset.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IrImportMetadata {
    pub schema_version: u32,
    pub ir_id: String,
    pub sha256: Sha256Digest,
    pub archived_filename: String,
    pub original_filename: String,
    pub sample_rate_hz: u32,
    pub sample_count: u32,
    pub default_processing: IrProcessingMode,
    pub default_trim_leading_samples: u32,
}

impl IrImportMetadata {
    pub fn validate(&self) -> Result<(), LibraryError> {
        if self.schema_version != IR_IMPORT_METADATA_VERSION {
            return Err(LibraryError::InvalidIrMetadata(format!(
                "unsupported schema version {}",
                self.schema_version
            )));
        }
        validate_identifier("IR metadata ir_id", &self.ir_id)
            .map_err(|error| LibraryError::InvalidIrMetadata(error.to_string()))?;
        validate_filename_hint("IR metadata archived_filename", &self.archived_filename)
            .map_err(|error| LibraryError::InvalidIrMetadata(error.to_string()))?;
        validate_filename_hint("IR metadata original_filename", &self.original_filename)
            .map_err(|error| LibraryError::InvalidIrMetadata(error.to_string()))?;
        if self.sample_rate_hz != crate::model::REQUIRED_SAMPLE_RATE_HZ {
            return Err(LibraryError::InvalidIrMetadata(format!(
                "sample rate must be {} Hz",
                crate::model::REQUIRED_SAMPLE_RATE_HZ
            )));
        }
        if self.sample_count == 0 || self.sample_count as usize > MAX_IR_SAMPLES {
            return Err(LibraryError::InvalidIrMetadata(format!(
                "sample_count must be within 1..={MAX_IR_SAMPLES}"
            )));
        }
        if self.default_processing != IrProcessingMode::MinimumPhaseAutoTrim {
            return Err(LibraryError::InvalidIrMetadata(
                "default processing must be minimum_phase_auto_trim".to_owned(),
            ));
        }
        if self.default_trim_leading_samples >= self.sample_count {
            return Err(LibraryError::InvalidIrMetadata(
                "default trim must leave at least one sample".to_owned(),
            ));
        }
        Ok(())
    }

    #[must_use]
    pub fn reference(&self) -> IrReference {
        IrReference {
            ir_id: self.ir_id.clone(),
            sha256: self.sha256,
            filename_hint: self.archived_filename.clone(),
            processing: self.default_processing,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImportedIr {
    pub archived_path: PathBuf,
    pub reference: IrReference,
    pub metadata: IrImportMetadata,
}

#[derive(Clone, Debug)]
pub struct ImportedNam {
    pub entry: ModelEntry,
    pub provenance_path: PathBuf,
    pub notice: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToneSettings {
    pub schema_version: u32,
    pub model_id: String,
    pub model_sha256: Sha256Digest,
    pub input_gain_db: f32,
    pub tight_percent: f32,
    pub bite_percent: f32,
    pub ir: Option<IrReference>,
}

impl ToneSettings {
    #[must_use]
    pub fn defaults_for(model: &ModelRef) -> Self {
        Self {
            schema_version: TONE_SETTINGS_VERSION,
            model_id: model.model_id.clone(),
            model_sha256: model.sha256,
            input_gain_db: 0.0,
            tight_percent: 0.0,
            bite_percent: 0.0,
            ir: None,
        }
    }

    pub fn validate(&self) -> Result<(), LibraryError> {
        if self.schema_version != TONE_SETTINGS_VERSION {
            return Err(LibraryError::UnsupportedToneSettingsVersion(
                self.schema_version,
            ));
        }
        validate_identifier("model_id", &self.model_id)?;
        validate_range("input_gain_db", self.input_gain_db, -24.0, 24.0)?;
        validate_range("tight_percent", self.tight_percent, 0.0, 100.0)?;
        validate_range("bite_percent", self.bite_percent, 0.0, 100.0)?;
        if let Some(ir) = &self.ir {
            validate_identifier("ir_id", &ir.ir_id)?;
            validate_filename_hint("IR filename_hint", &ir.filename_hint)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct ModelLibrary {
    paths: ModelLibraryPaths,
}

impl ModelLibrary {
    #[must_use]
    pub const fn new(paths: ModelLibraryPaths) -> Self {
        Self { paths }
    }

    pub fn for_current_user() -> Result<Self, LibraryError> {
        Ok(Self::new(ModelLibraryPaths::for_current_user()?))
    }

    #[must_use]
    pub const fn paths(&self) -> &ModelLibraryPaths {
        &self.paths
    }

    pub fn ensure_directories(&self) -> Result<(), LibraryError> {
        fs::create_dir_all(&self.paths.models)?;
        fs::create_dir_all(&self.paths.model_settings)?;
        fs::create_dir_all(&self.paths.irs)?;
        fs::create_dir_all(&self.paths.capture_records)?;
        Ok(())
    }

    /// Scans every `.motmodel`, verifies its digest and applies the runtime
    /// compatibility policy.  Invalid files are reported without hiding the
    /// remaining usable models.
    pub fn scan(&self, limits: &ModelRuntimeLimits) -> Result<ModelScan, LibraryError> {
        let paths = collect_model_paths(&self.paths.models)?;
        let mut scan = ModelScan::default();
        for path in paths {
            match read_validated_model(&path, limits) {
                Ok(model) => {
                    let filename_hint = file_name_string(&path)?;
                    scan.models.push(ModelEntry {
                        path,
                        reference: model.model_ref(filename_hint),
                        metadata: model.metadata().clone(),
                    });
                }
                Err(error) => scan.issues.push(ScanIssue {
                    path,
                    message: error.to_string(),
                }),
            }
        }
        sort_model_scan(&mut scan);
        Ok(scan)
    }

    /// Scans valid model containers without applying the current Player
    /// architecture policy. Trainer uses this metadata-only catalog to offer
    /// older models as retraining presets even when their runtime architecture
    /// is no longer supported by the current Player.
    pub fn scan_catalog(&self) -> Result<ModelScan, LibraryError> {
        let paths = collect_model_paths(&self.paths.models)?;
        let mut scan = ModelScan::default();
        for path in paths {
            match MotModel::read(&path) {
                Ok(model) => {
                    let filename_hint = file_name_string(&path)?;
                    scan.models.push(ModelEntry {
                        path,
                        reference: model.model_ref(filename_hint),
                        metadata: model.metadata().clone(),
                    });
                }
                Err(error) => scan.issues.push(ScanIssue {
                    path,
                    message: error.to_string(),
                }),
            }
        }
        sort_model_scan(&mut scan);
        Ok(scan)
    }

    /// Loads editable Trainer metadata associated with an immutable model.
    /// Missing capture records are valid for imported models.
    pub fn load_trainer_capture_preset(
        &self,
        model_id: &str,
    ) -> Result<Option<TrainerCapturePreset>, LibraryError> {
        validate_identifier("model_id", model_id)?;
        let path = self
            .paths
            .capture_records
            .join(model_id)
            .join("capture.json");
        let json = match fs::read_to_string(path) {
            Ok(json) => json,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(LibraryError::Io(error)),
        };
        capture_preset_from_json(&json, model_id).map(Some)
    }

    /// Scans the managed IR archive and verifies each RAW WAV against its
    /// provenance sidecar. The runtime loader repeats the same exact-content
    /// check immediately before preparing the live convolver.
    ///
    /// This is a blocking library operation and belongs on a worker thread.
    pub fn scan_irs(&self) -> Result<IrLibraryScan, LibraryError> {
        let paths = collect_ir_paths(&self.paths.irs)?;
        let mut scan = IrLibraryScan::default();
        for path in paths {
            match self.load_ir_metadata(&path) {
                Ok(Some(metadata)) => scan.entries.push(IrLibraryEntry { path, metadata }),
                Ok(None) => scan.issues.push(ScanIssue {
                    path,
                    message: "managed IR provenance sidecar is missing".to_owned(),
                }),
                Err(error) => scan.issues.push(ScanIssue {
                    path,
                    message: error.to_string(),
                }),
            }
        }
        scan.entries.sort_by(|left, right| {
            left.metadata
                .original_filename
                .to_lowercase()
                .cmp(&right.metadata.original_filename.to_lowercase())
                .then_with(|| left.path.cmp(&right.path))
        });
        scan.issues
            .sort_by(|left, right| left.path.cmp(&right.path));
        Ok(scan)
    }

    /// Resolves only an exact model ID + content digest pair.  The filename is
    /// merely a hint, so renaming/moving a model inside the library is safe.
    /// No similar or first-available model is ever substituted.
    pub fn resolve_exact(
        &self,
        reference: &ModelRef,
        limits: &ModelRuntimeLimits,
    ) -> Result<ModelEntry, LibraryError> {
        validate_identifier("model_id", &reference.model_id)?;
        let mut exact = Vec::new();
        let mut id_seen = false;
        let mut digest_seen = false;
        let mut hinted_issue = None;
        for path in collect_model_paths(&self.paths.models)? {
            match MotModel::read(&path) {
                Ok(model) => {
                    let id_matches = model.metadata().model_id == reference.model_id;
                    let digest_matches = model.content_sha256() == reference.sha256;
                    id_seen |= id_matches;
                    digest_seen |= digest_matches;
                    if id_matches && digest_matches {
                        // Compatibility is checked only after immutable
                        // identity is established, so a renamed overweight or
                        // unsupported model is never misreported as missing.
                        model.validate_for_runtime(limits)?;
                        let filename_hint = file_name_string(&path)?;
                        exact.push(ModelEntry {
                            path,
                            reference: model.model_ref(filename_hint),
                            metadata: model.metadata().clone(),
                        });
                    }
                }
                Err(error) if filename_matches(&path, &reference.filename_hint) => {
                    hinted_issue = Some((path, error.to_string()));
                }
                Err(_) => {}
            }
        }

        exact.sort_by(|left, right| {
            let left_hint = filename_matches(&left.path, &reference.filename_hint);
            let right_hint = filename_matches(&right.path, &reference.filename_hint);
            right_hint
                .cmp(&left_hint)
                .then_with(|| left.path.cmp(&right.path))
        });
        if let Some(entry) = exact.into_iter().next() {
            return Ok(entry);
        }

        if id_seen {
            return Err(LibraryError::ModelHashMismatch {
                model_id: reference.model_id.clone(),
            });
        }
        if digest_seen {
            return Err(LibraryError::ModelIdMismatch {
                model_id: reference.model_id.clone(),
            });
        }
        if let Some((path, message)) = hinted_issue {
            return Err(LibraryError::UnreadableModelCandidate { path, message });
        }
        Err(LibraryError::ModelNotFound(reference.model_id.clone()))
    }

    pub fn load_exact(
        &self,
        reference: &ModelRef,
        limits: &ModelRuntimeLimits,
    ) -> Result<MotModel, LibraryError> {
        let entry = self.resolve_exact(reference, limits)?;
        // Resolve is intentionally followed by a fresh read so that the
        // returned model owns one coherent byte snapshot. Re-check immutable
        // identity after that read: a file replaced between directory
        // resolution and loading must fail closed, never become the selected
        // model merely because it occupies the resolved path.
        let model = read_validated_model(&entry.path, limits)?;
        if model.metadata().model_id != reference.model_id {
            return Err(LibraryError::ModelIdMismatch {
                model_id: reference.model_id.clone(),
            });
        }
        if model.content_sha256() != reference.sha256 {
            return Err(LibraryError::ModelHashMismatch {
                model_id: reference.model_id.clone(),
            });
        }
        Ok(model)
    }

    #[must_use]
    pub fn tone_settings_path(&self, model_id: &str) -> PathBuf {
        self.paths.model_settings.join(format!("{model_id}.json"))
    }

    /// Atomically replaces the one saved tone associated with a model ID.
    pub fn save_tone(&self, settings: &ToneSettings) -> Result<(), LibraryError> {
        settings.validate()?;
        fs::create_dir_all(&self.paths.model_settings)?;
        let path = self.tone_settings_path(&settings.model_id);
        atomic_replace(&path, tone_to_json(settings).as_bytes())
    }

    /// Loads saved library defaults only when both model ID and digest match.
    /// DAW instance state should be handled separately and takes precedence.
    pub fn load_tone(&self, model: &ModelRef) -> Result<Option<ToneSettings>, LibraryError> {
        validate_identifier("model_id", &model.model_id)?;
        let path = self.tone_settings_path(&model.model_id);
        let json = match fs::read_to_string(&path) {
            Ok(json) => json,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(LibraryError::Io(error)),
        };
        let settings = tone_from_json(&json)?;
        settings.validate()?;
        if settings.model_id != model.model_id {
            return Err(LibraryError::ToneModelIdMismatch {
                expected: model.model_id.clone(),
                found: settings.model_id,
            });
        }
        if settings.model_sha256 != model.sha256 {
            return Err(LibraryError::ToneModelHashMismatch {
                model_id: model.model_id.clone(),
            });
        }
        Ok(Some(settings))
    }

    /// Converts a compatible NAM A2/C3 model and publishes one immutable,
    /// content-addressed `.motmodel` in the local model library.
    ///
    /// The source `.nam` is only read. Conversion and publication are blocking
    /// worker operations and must never run in the audio callback or editor
    /// paint path.
    pub fn import_nam(&self, source: &Path) -> Result<ImportedNam, LibraryError> {
        let source_filename = file_name_string(source)?;
        if !fs::metadata(source)?.is_file() {
            return Err(LibraryError::InvalidNamImport(
                "NAM source must be a regular file".to_owned(),
            ));
        }
        let source_file = File::open(source)?;
        let source_metadata = source_file.metadata()?;
        if !source_metadata.is_file() {
            return Err(LibraryError::InvalidNamImport(
                "NAM source changed and is no longer a regular file".to_owned(),
            ));
        }
        let source_size = source_metadata.len();
        if source_size > MAX_NAM_SOURCE_BYTES as u64 {
            return Err(LibraryError::InvalidNamImport(format!(
                "NAM source is {source_size} bytes; maximum is {MAX_NAM_SOURCE_BYTES}"
            )));
        }
        let mut source_bytes =
            Vec::with_capacity(usize::try_from(source_size).unwrap_or(MAX_NAM_SOURCE_BYTES));
        source_file
            .take(MAX_NAM_SOURCE_BYTES as u64 + 1)
            .read_to_end(&mut source_bytes)?;
        if source_bytes.len() > MAX_NAM_SOURCE_BYTES {
            return Err(LibraryError::InvalidNamImport(format!(
                "NAM source exceeds the {MAX_NAM_SOURCE_BYTES}-byte maximum"
            )));
        }
        let converted = convert_nam(&source_bytes, &source_filename)
            .map_err(|error| LibraryError::InvalidNamImport(error.to_string()))?;
        let model_bytes = converted.model.to_bytes()?;
        let provenance_bytes = converted
            .provenance_json()
            .map_err(|error| LibraryError::InvalidNamImport(error.to_string()))?;
        let digest = converted.model.content_sha256();
        let filename_hint = format!("{digest}.motmodel");
        let path = self.paths.models.join(&filename_hint);
        let provenance_path = self.paths.models.join(format!("{digest}.nam-import.json"));
        let mut notice = converted.selection.notice();
        if converted.has_calibration_metadata() {
            let calibration_notice =
                "NAM dBu calibration metadata was retained but is not applied automatically.";
            notice = Some(match notice {
                Some(mut existing) => {
                    existing.push(' ');
                    existing.push_str(calibration_notice);
                    existing
                }
                None => calibration_notice.to_owned(),
            });
        }

        fs::create_dir_all(&self.paths.models)?;
        let _ = publish_immutable(&provenance_path, &provenance_bytes)?;
        let _ = publish_immutable(&path, &model_bytes)?;
        let entry = ModelEntry {
            path,
            reference: converted.model.model_ref(filename_hint),
            metadata: converted.model.metadata().clone(),
        };
        Ok(ImportedNam {
            entry,
            provenance_path,
            notice,
        })
    }

    /// Imports one mono 48 kHz cabinet IR into the immutable local RAW archive.
    ///
    /// Validation and minimum-phase trim analysis happen before any library
    /// file is published. The archived `.wav` is byte-for-byte identical to
    /// `source`; its JSON sidecar records provenance and the default processing
    /// result. This method is worker-only and must never run in the editor or
    /// `process`.
    pub fn import_ir(&self, source: &Path) -> Result<ImportedIr, LibraryError> {
        let original_filename = file_name_string(source)?;
        let source_bytes = fs::read(source)?;
        let digest = sha256(&source_bytes);
        let wav = decode_mono_wav(&source_bytes)
            .map_err(|error| LibraryError::InvalidIrImport(error.to_string()))?;
        if wav.channels != 1 {
            return Err(LibraryError::InvalidIrImport(format!(
                "cabinet IR must be mono; source has {} channels",
                wav.channels
            )));
        }
        if wav.sample_rate != crate::model::REQUIRED_SAMPLE_RATE_HZ {
            return Err(LibraryError::InvalidIrImport(format!(
                "cabinet IR must be {} Hz; source is {} Hz",
                crate::model::REQUIRED_SAMPLE_RATE_HZ,
                wav.sample_rate
            )));
        }
        if wav.samples.len() > MAX_IR_SAMPLES {
            return Err(LibraryError::InvalidIrImport(format!(
                "cabinet IR has {} samples; maximum is {MAX_IR_SAMPLES}",
                wav.samples.len()
            )));
        }
        let prepared = PreparedCabinetIr::prepare(
            &wav.samples,
            wav.sample_rate,
            CabinetIrImportOptions::default(),
        )
        .map_err(|error| LibraryError::InvalidIrImport(error.to_string()))?;

        fs::create_dir_all(&self.paths.irs)?;
        let digest_hex = digest.to_string();
        let ir_id = format!("ir-{digest_hex}");
        let sample_count = u32::try_from(wav.samples.len()).map_err(|_| {
            LibraryError::InvalidIrImport("cabinet IR sample count is too large".to_owned())
        })?;
        let default_trim_leading_samples = u32::try_from(prepared.trimmed_leading_samples())
            .map_err(|_| {
                LibraryError::InvalidIrImport("cabinet IR trim count is too large".to_owned())
            })?;

        // Full-digest filenames make the archive content-addressed and remove
        // all basename/Unicode collision ambiguity.
        let archived_filename = format!("{digest_hex}.wav");
        let archived_path = self.paths.irs.join(&archived_filename);
        let metadata_path = ir_metadata_path(&archived_path);
        let metadata = IrImportMetadata {
            schema_version: IR_IMPORT_METADATA_VERSION,
            ir_id,
            sha256: digest,
            archived_filename,
            original_filename,
            sample_rate_hz: wav.sample_rate,
            sample_count,
            default_processing: IrProcessingMode::MinimumPhaseAutoTrim,
            default_trim_leading_samples,
        };
        metadata.validate()?;

        match publish_immutable(&archived_path, &source_bytes) {
            Ok(PublishImmutable::Created) => {}
            Ok(PublishImmutable::AlreadyIdentical) => {}
            Err(error) => return Err(error),
        }

        let metadata_bytes = ir_metadata_to_json(&metadata);
        if metadata_path.exists() {
            let existing = self.load_ir_metadata(&archived_path)?.ok_or_else(|| {
                LibraryError::InvalidIrMetadata(format!(
                    "{} disappeared during import",
                    metadata_path.display()
                ))
            })?;
            return Ok(ImportedIr {
                archived_path,
                reference: existing.reference(),
                metadata: existing,
            });
        }
        match publish_immutable(&metadata_path, metadata_bytes.as_bytes()) {
            Ok(PublishImmutable::Created) => {}
            Ok(PublishImmutable::AlreadyIdentical) => {
                // The first importer owns provenance such as original filename.
                // Load that authoritative sidecar instead of creating a
                // duplicate content asset for a second source path.
                let existing = self.load_ir_metadata(&archived_path)?.ok_or_else(|| {
                    LibraryError::InvalidIrMetadata(format!(
                        "{} disappeared during import",
                        metadata_path.display()
                    ))
                })?;
                return Ok(ImportedIr {
                    archived_path,
                    reference: existing.reference(),
                    metadata: existing,
                });
            }
            Err(LibraryError::ImmutableAssetCollision(_)) => {
                // A concurrent import of the same RAW bytes may have won with
                // a different original filename. Its complete valid sidecar
                // becomes authoritative.
                let existing = self.load_ir_metadata(&archived_path)?.ok_or_else(|| {
                    LibraryError::InvalidIrMetadata(format!(
                        "{} disappeared during import",
                        metadata_path.display()
                    ))
                })?;
                return Ok(ImportedIr {
                    archived_path,
                    reference: existing.reference(),
                    metadata: existing,
                });
            }
            Err(error) => return Err(error),
        }
        Ok(ImportedIr {
            archived_path,
            reference: metadata.reference(),
            metadata,
        })
    }

    /// Loads and validates the provenance sidecar for one archived RAW IR.
    ///
    /// A present sidecar is authoritative: changing the WAV bytes without also
    /// changing the persisted expected digest is reported as corruption.
    pub fn load_ir_metadata(
        &self,
        archived_path: &Path,
    ) -> Result<Option<IrImportMetadata>, LibraryError> {
        let Some(metadata) = self.read_ir_metadata_sidecar(archived_path)? else {
            return Ok(None);
        };
        let bytes = fs::read(archived_path)?;
        if sha256(&bytes) != metadata.sha256 {
            return Err(LibraryError::InvalidIrMetadata(format!(
                "SHA-256 does not match {}",
                archived_path.display()
            )));
        }
        Ok(Some(metadata))
    }

    /// Parses the small provenance sidecar without reading or hashing the WAV.
    ///
    /// This is suitable for building a worker-owned library snapshot without
    /// hashing the WAV twice. The selected digest is still enforced by
    /// [`crate::runtime::RuntimeLoader`] on its worker lane before any IR
    /// reaches the live runtime.
    pub fn read_ir_metadata_sidecar(
        &self,
        archived_path: &Path,
    ) -> Result<Option<IrImportMetadata>, LibraryError> {
        let metadata_path = ir_metadata_path(archived_path);
        let json = match fs::read_to_string(&metadata_path) {
            Ok(json) => json,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(LibraryError::Io(error)),
        };
        let metadata = ir_metadata_from_json(&json)?;
        metadata.validate()?;
        let actual_filename = file_name_string(archived_path)?;
        if metadata.archived_filename != actual_filename {
            return Err(LibraryError::InvalidIrMetadata(format!(
                "sidecar names {}, but belongs to {actual_filename}",
                metadata.archived_filename
            )));
        }
        Ok(Some(metadata))
    }
}

#[derive(Debug)]
pub enum LibraryError {
    Io(io::Error),
    Model(ModelError),
    HomeDirectoryUnavailable,
    InvalidToneSettings(String),
    UnsupportedToneSettingsVersion(u32),
    InvalidNamImport(String),
    InvalidIrImport(String),
    InvalidIrMetadata(String),
    InvalidCaptureMetadata(String),
    ImmutableAssetCollision(PathBuf),
    ToneModelIdMismatch { expected: String, found: String },
    ToneModelHashMismatch { model_id: String },
    ModelNotFound(String),
    ModelHashMismatch { model_id: String },
    ModelIdMismatch { model_id: String },
    UnreadableModelCandidate { path: PathBuf, message: String },
    InvalidFilename(PathBuf),
}

impl fmt::Display for LibraryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "{error}"),
            Self::Model(error) => write!(formatter, "{error}"),
            Self::HomeDirectoryUnavailable => {
                formatter.write_str("HOME is unavailable; cannot locate the MOT model library")
            }
            Self::InvalidToneSettings(message) => {
                write!(formatter, "invalid model tone settings: {message}")
            }
            Self::UnsupportedToneSettingsVersion(version) => {
                write!(formatter, "unsupported tone-settings version {version}")
            }
            Self::InvalidNamImport(message) => {
                write!(formatter, "invalid NAM import: {message}")
            }
            Self::InvalidIrImport(message) => {
                write!(formatter, "invalid cabinet IR import: {message}")
            }
            Self::InvalidIrMetadata(message) => {
                write!(formatter, "invalid cabinet IR metadata: {message}")
            }
            Self::InvalidCaptureMetadata(message) => {
                write!(formatter, "invalid Trainer capture metadata: {message}")
            }
            Self::ImmutableAssetCollision(path) => {
                write!(formatter, "immutable asset collision at {}", path.display())
            }
            Self::ToneModelIdMismatch { expected, found } => write!(
                formatter,
                "tone settings belong to model {found}, expected {expected}"
            ),
            Self::ToneModelHashMismatch { model_id } => write!(
                formatter,
                "tone settings belong to a different revision of model {model_id}"
            ),
            Self::ModelNotFound(model_id) => write!(formatter, "model {model_id} was not found"),
            Self::ModelHashMismatch { model_id } => {
                write!(
                    formatter,
                    "model {model_id} exists but its SHA-256 does not match"
                )
            }
            Self::ModelIdMismatch { model_id } => write!(
                formatter,
                "requested SHA-256 exists but belongs to a different model ID than {model_id}"
            ),
            Self::UnreadableModelCandidate { path, message } => {
                write!(formatter, "{} cannot be loaded: {message}", path.display())
            }
            Self::InvalidFilename(path) => {
                write!(formatter, "{} has no valid UTF-8 filename", path.display())
            }
        }
    }
}

impl std::error::Error for LibraryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Model(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for LibraryError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<ModelError> for LibraryError {
    fn from(error: ModelError) -> Self {
        Self::Model(error)
    }
}

fn read_validated_model(
    path: &Path,
    limits: &ModelRuntimeLimits,
) -> Result<MotModel, LibraryError> {
    let model = MotModel::read(path)?;
    model.validate_for_runtime(limits)?;
    Ok(model)
}

fn sort_model_scan(scan: &mut ModelScan) {
    scan.models.sort_by(|left, right| {
        left.metadata
            .display_name
            .to_lowercase()
            .cmp(&right.metadata.display_name.to_lowercase())
            .then_with(|| left.reference.model_id.cmp(&right.reference.model_id))
            .then_with(|| left.path.cmp(&right.path))
    });
    scan.issues
        .sort_by(|left, right| left.path.cmp(&right.path));
}

fn collect_model_paths(directory: &Path) -> Result<Vec<PathBuf>, LibraryError> {
    let entries = match fs::read_dir(directory) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(LibraryError::Io(error)),
    };
    let mut paths = Vec::new();
    for entry in entries {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if !file_type.is_file() {
            continue;
        }
        let path = entry.path();
        if path
            .extension()
            .and_then(std::ffi::OsStr::to_str)
            .is_some_and(|extension| extension.eq_ignore_ascii_case("motmodel"))
        {
            paths.push(path);
        }
    }
    paths.sort();
    Ok(paths)
}

fn collect_ir_paths(directory: &Path) -> Result<Vec<PathBuf>, LibraryError> {
    let entries = match fs::read_dir(directory) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(LibraryError::Io(error)),
    };
    let mut paths = Vec::new();
    for entry in entries {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            continue;
        }
        let path = entry.path();
        if path
            .extension()
            .and_then(std::ffi::OsStr::to_str)
            .is_some_and(|extension| extension.eq_ignore_ascii_case("wav"))
        {
            paths.push(path);
        }
    }
    paths.sort();
    Ok(paths)
}

fn file_name_string(path: &Path) -> Result<String, LibraryError> {
    path.file_name()
        .and_then(std::ffi::OsStr::to_str)
        .map(str::to_owned)
        .ok_or_else(|| LibraryError::InvalidFilename(path.to_owned()))
}

fn filename_matches(path: &Path, hint: &str) -> bool {
    !hint.is_empty()
        && path
            .file_name()
            .and_then(std::ffi::OsStr::to_str)
            .is_some_and(|filename| filename == hint)
}

fn validate_identifier(field: &str, value: &str) -> Result<(), LibraryError> {
    if value.is_empty()
        || value.len() > 128
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err(LibraryError::InvalidToneSettings(format!(
            "{field} is not a safe identifier"
        )));
    }
    Ok(())
}

fn validate_filename_hint(field: &str, value: &str) -> Result<(), LibraryError> {
    if value.trim().is_empty()
        || value.len() > 1_024
        || value.chars().any(char::is_control)
        || Path::new(value)
            .file_name()
            .and_then(std::ffi::OsStr::to_str)
            != Some(value)
    {
        return Err(LibraryError::InvalidToneSettings(format!(
            "{field} is invalid"
        )));
    }
    Ok(())
}

fn validate_range(field: &str, value: f32, minimum: f32, maximum: f32) -> Result<(), LibraryError> {
    if !value.is_finite() || !(minimum..=maximum).contains(&value) {
        return Err(LibraryError::InvalidToneSettings(format!(
            "{field} must be finite and within {minimum}..={maximum}"
        )));
    }
    Ok(())
}

fn atomic_replace(path: &Path, bytes: &[u8]) -> Result<(), LibraryError> {
    let parent = path.parent().ok_or_else(|| {
        LibraryError::InvalidToneSettings("settings path has no parent".to_owned())
    })?;
    let temporary = unique_temporary_path(parent, path.file_name());
    let result = (|| -> io::Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        fs::rename(&temporary, path)?;
        if let Ok(directory) = File::open(parent) {
            let _ = directory.sync_all();
        }
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result.map_err(LibraryError::Io)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PublishImmutable {
    Created,
    AlreadyIdentical,
}

fn publish_immutable(path: &Path, bytes: &[u8]) -> Result<PublishImmutable, LibraryError> {
    match fs::read(path) {
        Ok(existing) if existing == bytes => return Ok(PublishImmutable::AlreadyIdentical),
        Ok(_) => return Err(LibraryError::ImmutableAssetCollision(path.to_owned())),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(LibraryError::Io(error)),
    }

    let parent = path
        .parent()
        .ok_or_else(|| LibraryError::InvalidIrImport("archive path has no parent".to_owned()))?;
    let temporary = unique_temporary_path(parent, path.file_name());
    let result = (|| -> Result<PublishImmutable, LibraryError> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);

        match fs::hard_link(&temporary, path) {
            Ok(()) => {
                if let Ok(directory) = File::open(parent) {
                    let _ = directory.sync_all();
                }
                Ok(PublishImmutable::Created)
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                let existing = fs::read(path)?;
                if existing == bytes {
                    Ok(PublishImmutable::AlreadyIdentical)
                } else {
                    Err(LibraryError::ImmutableAssetCollision(path.to_owned()))
                }
            }
            Err(error) => Err(LibraryError::Io(error)),
        }
    })();
    let _ = fs::remove_file(&temporary);
    result
}

#[must_use]
pub fn ir_metadata_path(archived_path: &Path) -> PathBuf {
    archived_path.with_extension("motir.json")
}

fn ir_metadata_to_json(metadata: &IrImportMetadata) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema_version\": {},\n",
            "  \"ir_id\": \"{}\",\n",
            "  \"sha256\": \"{}\",\n",
            "  \"archived_filename\": \"{}\",\n",
            "  \"original_filename\": \"{}\",\n",
            "  \"sample_rate_hz\": {},\n",
            "  \"sample_count\": {},\n",
            "  \"default_processing\": \"{}\",\n",
            "  \"default_trim_leading_samples\": {}\n",
            "}}\n"
        ),
        metadata.schema_version,
        escape_json(&metadata.ir_id),
        metadata.sha256,
        escape_json(&metadata.archived_filename),
        escape_json(&metadata.original_filename),
        metadata.sample_rate_hz,
        metadata.sample_count,
        metadata.default_processing.as_str(),
        metadata.default_trim_leading_samples,
    )
}

fn ir_metadata_from_json(json: &str) -> Result<IrImportMetadata, LibraryError> {
    let value = JsonParser::new(json).parse().map_err(|error| {
        LibraryError::InvalidIrMetadata(format!("cannot parse sidecar: {error}"))
    })?;
    let object = value
        .as_object("IR metadata")
        .map_err(|error| LibraryError::InvalidIrMetadata(error.to_string()))?;
    let metadata = IrImportMetadata {
        schema_version: object_u32(object, "schema_version")
            .map_err(|error| LibraryError::InvalidIrMetadata(error.to_string()))?,
        ir_id: object_string(object, "ir_id")
            .map_err(|error| LibraryError::InvalidIrMetadata(error.to_string()))?
            .to_owned(),
        sha256: object_string(object, "sha256")
            .map_err(|error| LibraryError::InvalidIrMetadata(error.to_string()))?
            .parse::<Sha256Digest>()
            .map_err(|error| LibraryError::InvalidIrMetadata(error.to_string()))?,
        archived_filename: object_string(object, "archived_filename")
            .map_err(|error| LibraryError::InvalidIrMetadata(error.to_string()))?
            .to_owned(),
        original_filename: object_string(object, "original_filename")
            .map_err(|error| LibraryError::InvalidIrMetadata(error.to_string()))?
            .to_owned(),
        sample_rate_hz: object_u32(object, "sample_rate_hz")
            .map_err(|error| LibraryError::InvalidIrMetadata(error.to_string()))?,
        sample_count: object_u32(object, "sample_count")
            .map_err(|error| LibraryError::InvalidIrMetadata(error.to_string()))?,
        default_processing: object_string(object, "default_processing")
            .map_err(|error| LibraryError::InvalidIrMetadata(error.to_string()))?
            .parse()
            .map_err(|error: LibraryError| LibraryError::InvalidIrMetadata(error.to_string()))?,
        default_trim_leading_samples: object_u32(object, "default_trim_leading_samples")
            .map_err(|error| LibraryError::InvalidIrMetadata(error.to_string()))?,
    };
    metadata.validate()?;
    Ok(metadata)
}

fn unique_temporary_path(parent: &Path, filename: Option<&std::ffi::OsStr>) -> PathBuf {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos());
    let sequence = TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let filename = filename
        .and_then(std::ffi::OsStr::to_str)
        .unwrap_or("settings");
    parent.join(format!(
        ".{filename}.{}.{}.tmp",
        std::process::id(),
        timestamp ^ u128::from(sequence)
    ))
}

fn tone_to_json(settings: &ToneSettings) -> String {
    let ir = settings.ir.as_ref().map_or_else(
        || "null".to_owned(),
        |ir| {
            format!(
                concat!(
                    "{{\n",
                    "    \"ir_id\": \"{}\",\n",
                    "    \"sha256\": \"{}\",\n",
                    "    \"filename_hint\": \"{}\",\n",
                    "    \"processing\": \"{}\"\n",
                    "  }}"
                ),
                escape_json(&ir.ir_id),
                ir.sha256,
                escape_json(&ir.filename_hint),
                ir.processing.as_str(),
            )
        },
    );
    format!(
        concat!(
            "{{\n",
            "  \"schema_version\": {},\n",
            "  \"model_id\": \"{}\",\n",
            "  \"model_sha256\": \"{}\",\n",
            "  \"input_gain_db\": {},\n",
            "  \"tight_percent\": {},\n",
            "  \"bite_percent\": {},\n",
            "  \"ir\": {}\n",
            "}}\n"
        ),
        settings.schema_version,
        escape_json(&settings.model_id),
        settings.model_sha256,
        settings.input_gain_db,
        settings.tight_percent,
        settings.bite_percent,
        ir,
    )
}

fn tone_from_json(json: &str) -> Result<ToneSettings, LibraryError> {
    let value = JsonParser::new(json).parse()?;
    let object = value.as_object("root")?;
    let schema_version = object_u32(object, "schema_version")?;
    let model_id = object_string(object, "model_id")?.to_owned();
    let model_sha256 = object_string(object, "model_sha256")?
        .parse::<Sha256Digest>()
        .map_err(LibraryError::Model)?;
    let input_gain_db = object_f32(object, "input_gain_db")?;
    let tight_percent = object_f32(object, "tight_percent")?;
    let bite_percent = object_f32(object, "bite_percent")?;
    let ir = match object.get("ir") {
        Some(JsonValue::Null) | None => None,
        Some(value) => {
            let object = value.as_object("ir")?;
            Some(IrReference {
                ir_id: object_string(object, "ir_id")?.to_owned(),
                sha256: object_string(object, "sha256")?
                    .parse::<Sha256Digest>()
                    .map_err(LibraryError::Model)?,
                filename_hint: object_string(object, "filename_hint")?.to_owned(),
                processing: object_string(object, "processing")?.parse()?,
            })
        }
    };
    Ok(ToneSettings {
        schema_version,
        model_id,
        model_sha256,
        input_gain_db,
        tight_percent,
        bite_percent,
        ir,
    })
}

fn capture_preset_from_json(
    json: &str,
    expected_model_id: &str,
) -> Result<TrainerCapturePreset, LibraryError> {
    let value = JsonParser::new(json)
        .parse()
        .map_err(|error| invalid_capture_metadata(error.to_string()))?;
    let root = value
        .as_object("capture root")
        .map_err(|error| invalid_capture_metadata(error.to_string()))?;
    let schema_version = capture_u32(root, "schema_version")?;
    if !(1..=6).contains(&schema_version) {
        return Err(invalid_capture_metadata(format!(
            "unsupported schema version {schema_version}"
        )));
    }
    let model_id = capture_required_string(root, "model_id")?;
    if model_id != expected_model_id {
        return Err(invalid_capture_metadata(format!(
            "record belongs to model {model_id}, expected {expected_model_id}"
        )));
    }
    let target = match capture_required_string(root, "target")?.as_str() {
        "software_plugin_chain" => CaptureTarget::SoftwarePluginChain,
        "full_amp_unfiltered_load" => CaptureTarget::FullAmpUnfilteredLoad,
        value => {
            return Err(invalid_capture_metadata(format!(
                "unknown capture target {value:?}"
            )));
        }
    };
    let hardware = match root.get("hardware") {
        Some(value) => value
            .as_object("hardware")
            .map_err(|error| invalid_capture_metadata(error.to_string()))?,
        None => root,
    };

    Ok(TrainerCapturePreset {
        model_id,
        target,
        amplifier: capture_optional_string(hardware, "amplifier")?,
        amplifier_channel: capture_optional_string(hardware, "amplifier_channel")?,
        control_positions: capture_optional_string(hardware, "control_positions")?,
        interface_output: capture_optional_string(hardware, "interface_output")?,
        interface_input: capture_optional_string(hardware, "interface_input")?,
        reamp_box: capture_optional_string(hardware, "reamp_box")?,
        reactive_load: capture_optional_string(hardware, "reactive_load")?,
        load_impedance_ohms: capture_optional_u16(hardware, "load_impedance_ohms")?,
        return_gain_note: capture_optional_string(hardware, "return_gain_note")?,
    })
}

fn invalid_capture_metadata(message: impl Into<String>) -> LibraryError {
    LibraryError::InvalidCaptureMetadata(message.into())
}

fn capture_required_string(
    object: &BTreeMap<String, JsonValue>,
    field: &str,
) -> Result<String, LibraryError> {
    match object.get(field) {
        Some(JsonValue::String(value)) => Ok(value.clone()),
        Some(_) => Err(invalid_capture_metadata(format!(
            "{field} must be a string"
        ))),
        None => Err(invalid_capture_metadata(format!("{field} is missing"))),
    }
}

fn capture_optional_string(
    object: &BTreeMap<String, JsonValue>,
    field: &str,
) -> Result<String, LibraryError> {
    match object.get(field) {
        Some(JsonValue::String(value)) => Ok(value.clone()),
        Some(JsonValue::Null) | None => Ok(String::new()),
        Some(_) => Err(invalid_capture_metadata(format!(
            "{field} must be a string or null"
        ))),
    }
}

fn capture_u32(object: &BTreeMap<String, JsonValue>, field: &str) -> Result<u32, LibraryError> {
    match object.get(field) {
        Some(JsonValue::Number(value))
            if value.is_finite()
                && value.fract() == 0.0
                && (0.0..=f64::from(u32::MAX)).contains(value) =>
        {
            Ok(*value as u32)
        }
        Some(_) => Err(invalid_capture_metadata(format!(
            "{field} must be an unsigned integer"
        ))),
        None => Err(invalid_capture_metadata(format!("{field} is missing"))),
    }
}

fn capture_optional_u16(
    object: &BTreeMap<String, JsonValue>,
    field: &str,
) -> Result<Option<u16>, LibraryError> {
    match object.get(field) {
        Some(JsonValue::Null) | None => Ok(None),
        Some(JsonValue::Number(value))
            if value.is_finite()
                && value.fract() == 0.0
                && (1.0..=f64::from(u16::MAX)).contains(value) =>
        {
            Ok(Some(*value as u16))
        }
        Some(_) => Err(invalid_capture_metadata(format!(
            "{field} must be null or a positive 16-bit integer"
        ))),
    }
}

fn escape_json(value: &str) -> String {
    let mut output = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            '\u{08}' => output.push_str("\\b"),
            '\u{0c}' => output.push_str("\\f"),
            character if character.is_control() => {
                use std::fmt::Write as _;
                let _ = write!(output, "\\u{:04x}", u32::from(character));
            }
            character => output.push(character),
        }
    }
    output
}

#[derive(Clone, Debug, PartialEq)]
enum JsonValue {
    Null,
    Bool,
    Number(f64),
    String(String),
    Array,
    Object(BTreeMap<String, JsonValue>),
}

impl JsonValue {
    fn as_object(&self, field: &str) -> Result<&BTreeMap<String, JsonValue>, LibraryError> {
        match self {
            Self::Object(object) => Ok(object),
            _ => Err(LibraryError::InvalidToneSettings(format!(
                "{field} must be an object"
            ))),
        }
    }
}

fn object_string<'a>(
    object: &'a BTreeMap<String, JsonValue>,
    field: &str,
) -> Result<&'a str, LibraryError> {
    match object.get(field) {
        Some(JsonValue::String(value)) => Ok(value),
        Some(_) => Err(LibraryError::InvalidToneSettings(format!(
            "{field} must be a string"
        ))),
        None => Err(LibraryError::InvalidToneSettings(format!(
            "{field} is missing"
        ))),
    }
}

fn object_number(object: &BTreeMap<String, JsonValue>, field: &str) -> Result<f64, LibraryError> {
    match object.get(field) {
        Some(JsonValue::Number(value)) if value.is_finite() => Ok(*value),
        Some(_) => Err(LibraryError::InvalidToneSettings(format!(
            "{field} must be a finite number"
        ))),
        None => Err(LibraryError::InvalidToneSettings(format!(
            "{field} is missing"
        ))),
    }
}

fn object_u32(object: &BTreeMap<String, JsonValue>, field: &str) -> Result<u32, LibraryError> {
    let value = object_number(object, field)?;
    if value.fract() != 0.0 || !(0.0..=f64::from(u32::MAX)).contains(&value) {
        return Err(LibraryError::InvalidToneSettings(format!(
            "{field} must be an unsigned integer"
        )));
    }
    Ok(value as u32)
}

fn object_f32(object: &BTreeMap<String, JsonValue>, field: &str) -> Result<f32, LibraryError> {
    let value = object_number(object, field)?;
    if value < f64::from(f32::MIN) || value > f64::from(f32::MAX) {
        return Err(LibraryError::InvalidToneSettings(format!(
            "{field} is outside the f32 range"
        )));
    }
    Ok(value as f32)
}

struct JsonParser<'a> {
    input: &'a str,
    offset: usize,
}

impl<'a> JsonParser<'a> {
    const fn new(input: &'a str) -> Self {
        Self { input, offset: 0 }
    }

    fn parse(mut self) -> Result<JsonValue, LibraryError> {
        self.skip_whitespace();
        let value = self.parse_value()?;
        self.skip_whitespace();
        if self.offset != self.input.len() {
            return self.error("trailing JSON data");
        }
        Ok(value)
    }

    fn parse_value(&mut self) -> Result<JsonValue, LibraryError> {
        self.skip_whitespace();
        match self.peek_byte() {
            Some(b'{') => self.parse_object(),
            Some(b'[') => self.parse_array(),
            Some(b'"') => self.parse_string().map(JsonValue::String),
            Some(b'-' | b'0'..=b'9') => self.parse_number().map(JsonValue::Number),
            Some(b't') => {
                self.consume_keyword("true")?;
                Ok(JsonValue::Bool)
            }
            Some(b'f') => {
                self.consume_keyword("false")?;
                Ok(JsonValue::Bool)
            }
            Some(b'n') => {
                self.consume_keyword("null")?;
                Ok(JsonValue::Null)
            }
            _ => self.error("expected a JSON value"),
        }
    }

    fn parse_object(&mut self) -> Result<JsonValue, LibraryError> {
        self.expect_byte(b'{')?;
        self.skip_whitespace();
        let mut object = BTreeMap::new();
        if self.consume_byte(b'}') {
            return Ok(JsonValue::Object(object));
        }
        loop {
            self.skip_whitespace();
            let key = self.parse_string()?;
            self.skip_whitespace();
            self.expect_byte(b':')?;
            let value = self.parse_value()?;
            if object.insert(key, value).is_some() {
                return self.error("duplicate object key");
            }
            self.skip_whitespace();
            if self.consume_byte(b'}') {
                break;
            }
            self.expect_byte(b',')?;
        }
        Ok(JsonValue::Object(object))
    }

    fn parse_array(&mut self) -> Result<JsonValue, LibraryError> {
        self.expect_byte(b'[')?;
        self.skip_whitespace();
        if self.consume_byte(b']') {
            return Ok(JsonValue::Array);
        }
        loop {
            let _ = self.parse_value()?;
            self.skip_whitespace();
            if self.consume_byte(b']') {
                break;
            }
            self.expect_byte(b',')?;
        }
        Ok(JsonValue::Array)
    }

    fn parse_string(&mut self) -> Result<String, LibraryError> {
        self.expect_byte(b'"')?;
        let mut output = String::new();
        loop {
            let Some(byte) = self.peek_byte() else {
                return self.error("unterminated string");
            };
            match byte {
                b'"' => {
                    self.offset += 1;
                    return Ok(output);
                }
                b'\\' => {
                    self.offset += 1;
                    let escaped = self
                        .next_byte()
                        .ok_or_else(|| self.make_error("unterminated escape"))?;
                    match escaped {
                        b'"' => output.push('"'),
                        b'\\' => output.push('\\'),
                        b'/' => output.push('/'),
                        b'b' => output.push('\u{08}'),
                        b'f' => output.push('\u{0c}'),
                        b'n' => output.push('\n'),
                        b'r' => output.push('\r'),
                        b't' => output.push('\t'),
                        b'u' => output.push(self.parse_unicode_escape()?),
                        _ => return self.error("invalid string escape"),
                    }
                }
                0x00..=0x1f => return self.error("control character in string"),
                _ => {
                    let character = self.input[self.offset..]
                        .chars()
                        .next()
                        .ok_or_else(|| self.make_error("invalid UTF-8 string"))?;
                    self.offset += character.len_utf8();
                    output.push(character);
                }
            }
        }
    }

    fn parse_unicode_escape(&mut self) -> Result<char, LibraryError> {
        let first = self.parse_hex_u16()?;
        if (0xd800..=0xdbff).contains(&first) {
            self.expect_byte(b'\\')?;
            self.expect_byte(b'u')?;
            let second = self.parse_hex_u16()?;
            if !(0xdc00..=0xdfff).contains(&second) {
                return self.error("invalid Unicode surrogate pair");
            }
            let scalar =
                0x1_0000 + ((u32::from(first) - 0xd800) << 10) + (u32::from(second) - 0xdc00);
            char::from_u32(scalar).ok_or_else(|| self.make_error("invalid Unicode scalar"))
        } else if (0xdc00..=0xdfff).contains(&first) {
            self.error("unpaired Unicode low surrogate")
        } else {
            char::from_u32(u32::from(first))
                .ok_or_else(|| self.make_error("invalid Unicode scalar"))
        }
    }

    fn parse_hex_u16(&mut self) -> Result<u16, LibraryError> {
        let mut value = 0_u16;
        for _ in 0..4 {
            let byte = self
                .next_byte()
                .ok_or_else(|| self.make_error("truncated Unicode escape"))?;
            let digit =
                decode_hex(byte).ok_or_else(|| self.make_error("invalid Unicode escape"))?;
            value = (value << 4) | u16::from(digit);
        }
        Ok(value)
    }

    fn parse_number(&mut self) -> Result<f64, LibraryError> {
        let start = self.offset;
        self.consume_byte(b'-');
        match self.peek_byte() {
            Some(b'0') => {
                self.offset += 1;
                if self.peek_byte().is_some_and(|byte| byte.is_ascii_digit()) {
                    return self.error("leading zero in number");
                }
            }
            Some(b'1'..=b'9') => {
                while self.peek_byte().is_some_and(|byte| byte.is_ascii_digit()) {
                    self.offset += 1;
                }
            }
            _ => return self.error("invalid number"),
        }
        if self.consume_byte(b'.') {
            let fraction_start = self.offset;
            while self.peek_byte().is_some_and(|byte| byte.is_ascii_digit()) {
                self.offset += 1;
            }
            if self.offset == fraction_start {
                return self.error("missing fractional digits");
            }
        }
        if self
            .peek_byte()
            .is_some_and(|byte| matches!(byte, b'e' | b'E'))
        {
            self.offset += 1;
            if self
                .peek_byte()
                .is_some_and(|byte| matches!(byte, b'+' | b'-'))
            {
                self.offset += 1;
            }
            let exponent_start = self.offset;
            while self.peek_byte().is_some_and(|byte| byte.is_ascii_digit()) {
                self.offset += 1;
            }
            if self.offset == exponent_start {
                return self.error("missing exponent digits");
            }
        }
        self.input[start..self.offset]
            .parse::<f64>()
            .map_err(|_| self.make_error("invalid number"))
    }

    fn consume_keyword(&mut self, keyword: &str) -> Result<(), LibraryError> {
        if self.input[self.offset..].starts_with(keyword) {
            self.offset += keyword.len();
            Ok(())
        } else {
            self.error("invalid keyword")
        }
    }

    fn expect_byte(&mut self, expected: u8) -> Result<(), LibraryError> {
        if self.consume_byte(expected) {
            Ok(())
        } else {
            self.error(&format!("expected {:?}", expected as char))
        }
    }

    fn consume_byte(&mut self, expected: u8) -> bool {
        if self.peek_byte() == Some(expected) {
            self.offset += 1;
            true
        } else {
            false
        }
    }

    fn next_byte(&mut self) -> Option<u8> {
        let byte = self.peek_byte()?;
        self.offset += 1;
        Some(byte)
    }

    fn peek_byte(&self) -> Option<u8> {
        self.input.as_bytes().get(self.offset).copied()
    }

    fn skip_whitespace(&mut self) {
        while self
            .peek_byte()
            .is_some_and(|byte| matches!(byte, b' ' | b'\n' | b'\r' | b'\t'))
        {
            self.offset += 1;
        }
    }

    fn error<T>(&self, message: &str) -> Result<T, LibraryError> {
        Err(self.make_error(message))
    }

    fn make_error(&self, message: &str) -> LibraryError {
        LibraryError::InvalidToneSettings(format!("{message} at byte {}", self.offset))
    }
}

const fn decode_hex(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    use crate::a2::{
        A2_CHANNELS, A2_DILATIONS, A2_KERNEL_SIZES, A2_LAYER_COUNT, A2_LEAKY_RELU_SLOPE, A2Model,
    };
    use crate::model::{REQUIRED_SAMPLE_RATE_HZ, SupportedArchitecture, sha256};
    use crate::wav_io::write_mono_f32_wav;

    fn metadata(model_id: &str, display_name: &str) -> ModelMetadata {
        ModelMetadata {
            model_id: model_id.to_owned(),
            display_name: display_name.to_owned(),
            architecture_id: "mot-causal-tcn".to_owned(),
            architecture_version: 1,
            sample_rate_hz: REQUIRED_SAMPLE_RATE_HZ,
            causal: true,
            lookahead_samples: 0,
            runtime_latency_samples: 0,
            estimated_macs_per_sample: 20_000,
        }
    }

    fn limits() -> ModelRuntimeLimits {
        ModelRuntimeLimits::new(
            30_000,
            vec![SupportedArchitecture::exact("mot-causal-tcn", 1)],
        )
    }

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new() -> Self {
            let unique = format!(
                "mot-library-tests-{}-{}-{}",
                std::process::id(),
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos(),
                TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed)
            );
            let path = env::temp_dir().join(unique);
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn test_library(directory: &TestDirectory) -> ModelLibrary {
        ModelLibrary::new(ModelLibraryPaths::from_plugin_root(
            directory.0.join("library"),
        ))
    }

    fn direct_nam(name: &str) -> Vec<u8> {
        let model = A2Model::zeros();
        serde_json::to_vec(&json!({
            "version": "0.7.0",
            "metadata": {"name": name},
            "architecture": "WaveNet",
            "config": {
                "layers": [{
                    "input_size": 1,
                    "condition_size": 1,
                    "head": {
                        "out_channels": 1,
                        "kernel_size": 16,
                        "bias": true
                    },
                    "channels": A2_CHANNELS,
                    "kernel_sizes": A2_KERNEL_SIZES,
                    "dilations": A2_DILATIONS,
                    "activation": (0..A2_LAYER_COUNT)
                        .map(|_| json!({
                            "type": "LeakyReLU",
                            "negative_slope": A2_LEAKY_RELU_SLOPE
                        }))
                        .collect::<Vec<_>>(),
                    "bottleneck": A2_CHANNELS,
                    "head1x1": {"active": false},
                    "layer1x1": {"active": true, "groups": 1},
                    "groups_input": 1,
                    "groups_input_mixin": 1,
                    "gating_mode": vec!["none"; A2_LAYER_COUNT],
                    "secondary_activation": vec![Value::Null; A2_LAYER_COUNT],
                    "slimmable": null
                }],
                "head": null,
                "head_scale": 0.01
            },
            "weights": model.weights.to_official_weight_vec(),
            "sample_rate": 48000
        }))
        .unwrap()
    }

    #[test]
    fn macos_paths_match_the_product_contract() {
        let paths = ModelLibraryPaths::from_home("/Users/example");
        assert_eq!(
            paths.models,
            Path::new(
                "/Users/example/Library/Application Support/Plut&Mot/MOT Guitar Plugin/Models"
            )
        );
        assert_eq!(
            paths.model_settings,
            Path::new(
                "/Users/example/Library/Application Support/Plut&Mot/MOT Guitar Plugin/Model Settings"
            )
        );
        assert_eq!(
            paths.irs,
            Path::new("/Users/example/Library/Application Support/Plut&Mot/MOT Guitar Plugin/IRs")
        );
        assert_eq!(
            paths.capture_records,
            Path::new(
                "/Users/example/Library/Application Support/Plut&Mot/MOT Guitar Plugin/Capture Records"
            )
        );
    }

    #[test]
    fn ir_import_archives_raw_bytes_and_round_trips_trim_metadata() {
        let directory = TestDirectory::new();
        let library = test_library(&directory);
        library.ensure_directories().unwrap();
        let source = directory.0.join("Кабинет V30.wav");
        let samples = [0.0, 0.0, 1.0, -0.25, 0.125];
        write_mono_f32_wav(&source, REQUIRED_SAMPLE_RATE_HZ, &samples).unwrap();
        let source_bytes = fs::read(&source).unwrap();
        let source_digest = sha256(&source_bytes);

        let imported = library.import_ir(&source).unwrap();
        assert_eq!(fs::read(&imported.archived_path).unwrap(), source_bytes);
        assert_eq!(
            imported
                .archived_path
                .file_name()
                .unwrap()
                .to_string_lossy(),
            format!("{source_digest}.wav")
        );
        assert_eq!(imported.reference.ir_id, format!("ir-{source_digest}"));
        assert_eq!(imported.reference.sha256, source_digest);
        assert_eq!(
            imported.reference.processing,
            IrProcessingMode::MinimumPhaseAutoTrim
        );
        assert_eq!(imported.metadata.original_filename, "Кабинет V30.wav");
        assert_eq!(imported.metadata.sample_rate_hz, REQUIRED_SAMPLE_RATE_HZ);
        assert_eq!(imported.metadata.sample_count, samples.len() as u32);
        assert_eq!(imported.metadata.default_trim_leading_samples, 2);
        assert_eq!(
            library.load_ir_metadata(&imported.archived_path).unwrap(),
            Some(imported.metadata.clone())
        );
        let scan = library.scan_irs().unwrap();
        assert_eq!(scan.entries.len(), 1);
        assert!(scan.issues.is_empty());
        assert_eq!(scan.entries[0].metadata, imported.metadata);

        // Content identity wins over the second source filename: no duplicate
        // RAW asset is created and the first immutable provenance is retained.
        let renamed_source = directory.0.join("same bytes, renamed.wav");
        fs::write(&renamed_source, &source_bytes).unwrap();
        let repeated = library.import_ir(&renamed_source).unwrap();
        assert_eq!(repeated.archived_path, imported.archived_path);
        assert_eq!(repeated.metadata, imported.metadata);
        let wav_count = fs::read_dir(&library.paths().irs)
            .unwrap()
            .filter_map(Result::ok)
            .filter(|entry| {
                entry
                    .path()
                    .extension()
                    .and_then(std::ffi::OsStr::to_str)
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("wav"))
            })
            .count();
        assert_eq!(wav_count, 1);

        // Browser snapshots never expose an IR whose RAW bytes no longer match
        // the immutable provenance sidecar.
        fs::write(&imported.archived_path, b"tampered").unwrap();
        let corrupt_scan = library.scan_irs().unwrap();
        assert!(corrupt_scan.entries.is_empty());
        assert_eq!(corrupt_scan.issues.len(), 1);
        assert_eq!(corrupt_scan.issues[0].path, imported.archived_path);
    }

    #[test]
    fn nam_import_is_source_preserving_content_addressed_and_idempotent() {
        let directory = TestDirectory::new();
        let library = test_library(&directory);
        library.ensure_directories().unwrap();
        let source = directory.0.join("external model.nam");
        let source_bytes = direct_nam("External A2");
        fs::write(&source, &source_bytes).unwrap();

        let imported = library.import_nam(&source).unwrap();
        assert_eq!(fs::read(&source).unwrap(), source_bytes);
        assert_eq!(imported.entry.metadata.display_name, "External A2");
        assert_eq!(imported.notice, None);
        assert_eq!(
            imported.entry.path.file_name().unwrap().to_string_lossy(),
            format!("{}.motmodel", imported.entry.reference.sha256)
        );
        assert_eq!(
            imported
                .provenance_path
                .file_name()
                .unwrap()
                .to_string_lossy(),
            format!("{}.nam-import.json", imported.entry.reference.sha256)
        );
        let provenance: Value =
            serde_json::from_slice(&fs::read(&imported.provenance_path).unwrap()).unwrap();
        assert_eq!(
            provenance["source_sha256"],
            sha256(&source_bytes).to_string()
        );
        assert_eq!(
            provenance["selection"]["runtime_variant"],
            json!("A2 C3 Nano")
        );
        let stored = MotModel::read(&imported.entry.path).unwrap();
        assert_eq!(stored.content_sha256(), imported.entry.reference.sha256);
        assert_eq!(stored.metadata(), &imported.entry.metadata);

        let renamed_source = directory.0.join("same bytes renamed.nam");
        fs::write(&renamed_source, &source_bytes).unwrap();
        let repeated = library.import_nam(&renamed_source).unwrap();
        assert_eq!(repeated.entry.path, imported.entry.path);
        assert_eq!(repeated.entry.reference, imported.entry.reference);
        assert_eq!(repeated.provenance_path, imported.provenance_path);
        let model_count = fs::read_dir(&library.paths().models)
            .unwrap()
            .filter_map(Result::ok)
            .filter(|entry| {
                entry
                    .path()
                    .extension()
                    .and_then(std::ffi::OsStr::to_str)
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("motmodel"))
            })
            .count();
        assert_eq!(model_count, 1);
    }

    #[test]
    fn scan_is_sorted_and_quarantines_bad_or_overweight_files() {
        let directory = TestDirectory::new();
        let library = test_library(&directory);
        library.ensure_directories().unwrap();

        MotModel::new(metadata("z-model", "Zulu"), vec![1])
            .unwrap()
            .write_new(library.paths().models.join("z.motmodel"))
            .unwrap();
        MotModel::new(metadata("a-model", "alpha"), vec![2])
            .unwrap()
            .write_new(library.paths().models.join("a.motmodel"))
            .unwrap();
        let mut expensive = metadata("expensive", "Heavy");
        expensive.estimated_macs_per_sample = 30_001;
        MotModel::new(expensive, vec![3])
            .unwrap()
            .write_new(library.paths().models.join("heavy.motmodel"))
            .unwrap();
        let mut legacy = metadata("legacy", "Legacy");
        legacy.architecture_id = "mot.diagonal-rnn-tanh".to_owned();
        MotModel::new(legacy, vec![4])
            .unwrap()
            .write_new(library.paths().models.join("legacy.motmodel"))
            .unwrap();
        fs::write(library.paths().models.join("bad.motmodel"), b"broken").unwrap();
        fs::write(library.paths().models.join("ignore.wav"), b"not a model").unwrap();

        let scan = library.scan(&limits()).unwrap();
        let ids: Vec<_> = scan
            .models
            .iter()
            .map(|entry| entry.reference.model_id.as_str())
            .collect();
        assert_eq!(ids, vec!["a-model", "z-model"]);
        assert_eq!(scan.issues.len(), 3);
        assert!(
            scan.issues
                .iter()
                .any(|issue| issue.path.ends_with("bad.motmodel"))
        );
        assert!(
            scan.issues
                .iter()
                .any(|issue| issue.path.ends_with("heavy.motmodel"))
        );

        let catalog = library.scan_catalog().unwrap();
        let catalog_ids: Vec<_> = catalog
            .models
            .iter()
            .map(|entry| entry.reference.model_id.as_str())
            .collect();
        assert_eq!(
            catalog_ids,
            vec!["a-model", "expensive", "legacy", "z-model"]
        );
        assert_eq!(catalog.issues.len(), 1);
    }

    #[test]
    fn trainer_capture_preset_reads_legacy_flat_metadata() {
        let json = r#"{
            "schema_version": 1,
            "model_id": "amp-legacy",
            "target": "software_plugin_chain",
            "amplifier": "ПИВО 5153",
            "amplifier_channel": "Red",
            "control_positions": "Gain 6",
            "interface_output": "Out 3",
            "interface_input": "In 1",
            "reamp_box": "Reamp",
            "reactive_load": "Captor X",
            "load_impedance_ohms": null,
            "return_gain_note": "+12 dB"
        }"#;
        let preset = capture_preset_from_json(json, "amp-legacy").unwrap();

        assert_eq!(preset.model_id, "amp-legacy");
        assert_eq!(preset.target, CaptureTarget::SoftwarePluginChain);
        assert_eq!(preset.amplifier, "ПИВО 5153");
        assert_eq!(preset.amplifier_channel, "Red");
        assert_eq!(preset.control_positions, "Gain 6");
        assert_eq!(preset.interface_output, "Out 3");
        assert_eq!(preset.interface_input, "In 1");
        assert_eq!(preset.reamp_box, "Reamp");
        assert_eq!(preset.reactive_load, "Captor X");
        assert_eq!(preset.load_impedance_ohms, None);
        assert_eq!(preset.return_gain_note, "+12 dB");
    }

    #[test]
    fn trainer_capture_preset_reads_current_nested_metadata() {
        let json = r#"{
            "schema_version": 6,
            "model_id": "amp-current",
            "target": "full_amp_unfiltered_load",
            "hardware": {
                "amplifier": "EVH 5153",
                "amplifier_channel": "Blue",
                "control_positions": "Gain 5",
                "interface_output": "Line 3",
                "interface_input": "Input 1",
                "reamp_box": "Radial",
                "reactive_load": "Suhr",
                "load_impedance_ohms": 8,
                "return_gain_note": "Pad on"
            }
        }"#;
        let preset = capture_preset_from_json(json, "amp-current").unwrap();

        assert_eq!(preset.target, CaptureTarget::FullAmpUnfilteredLoad);
        assert_eq!(preset.amplifier, "EVH 5153");
        assert_eq!(preset.load_impedance_ohms, Some(8));
        assert_eq!(preset.return_gain_note, "Pad on");
    }

    #[test]
    fn trainer_capture_preset_rejects_the_wrong_model() {
        let json = r#"{
            "schema_version": 5,
            "model_id": "other-model",
            "target": "software_plugin_chain",
            "hardware": {}
        }"#;
        assert!(matches!(
            capture_preset_from_json(json, "expected-model"),
            Err(LibraryError::InvalidCaptureMetadata(_))
        ));
    }

    #[test]
    fn exact_lookup_survives_rename_but_never_falls_back_by_filename_or_id() {
        let directory = TestDirectory::new();
        let library = test_library(&directory);
        library.ensure_directories().unwrap();

        let model = MotModel::new(metadata("amp-1", "Amp"), vec![1, 2, 3]).unwrap();
        let original_path = library.paths().models.join("original.motmodel");
        model.write_new(&original_path).unwrap();
        let reference = model.model_ref("original.motmodel");
        let renamed_path = library.paths().models.join("renamed.motmodel");
        fs::rename(original_path, &renamed_path).unwrap();

        let entry = library.resolve_exact(&reference, &limits()).unwrap();
        assert_eq!(entry.path, renamed_path);

        let wrong_hash = ModelRef {
            model_id: reference.model_id.clone(),
            sha256: sha256(b"different"),
            filename_hint: "renamed.motmodel".to_owned(),
        };
        assert!(matches!(
            library.resolve_exact(&wrong_hash, &limits()),
            Err(LibraryError::ModelHashMismatch { .. })
        ));
    }

    #[test]
    fn exact_lookup_reports_runtime_incompatibility_after_a_rename() {
        let directory = TestDirectory::new();
        let library = test_library(&directory);
        library.ensure_directories().unwrap();

        let mut heavy_metadata = metadata("heavy-amp", "Heavy Amp");
        heavy_metadata.estimated_macs_per_sample = 30_001;
        let model = MotModel::new(heavy_metadata, vec![1, 2, 3]).unwrap();
        model
            .write_new(library.paths().models.join("renamed.motmodel"))
            .unwrap();
        let reference = model.model_ref("old-name.motmodel");

        assert!(matches!(
            library.resolve_exact(&reference, &limits()),
            Err(LibraryError::Model(ModelError::ModelTooExpensive { .. }))
        ));
    }

    #[test]
    fn tone_sidecar_round_trips_unicode_and_ir_settings() {
        let directory = TestDirectory::new();
        let library = test_library(&directory);
        library.ensure_directories().unwrap();
        let model = MotModel::new(metadata("amp-1", "Amp"), vec![1, 2, 3]).unwrap();
        let reference = model.model_ref("Amp.motmodel");
        let settings = ToneSettings {
            schema_version: TONE_SETTINGS_VERSION,
            model_id: reference.model_id.clone(),
            model_sha256: reference.sha256,
            input_gain_db: -3.5,
            tight_percent: 61.0,
            bite_percent: 42.5,
            ir: Some(IrReference {
                ir_id: "ir-001".to_owned(),
                sha256: sha256(b"IR"),
                filename_hint: "Кабинет \"V30\".wav".to_owned(),
                processing: IrProcessingMode::MinimumPhaseAutoTrim,
            }),
        };

        library.save_tone(&settings).unwrap();
        assert_eq!(library.load_tone(&reference).unwrap(), Some(settings));
    }

    #[test]
    fn sidecar_is_not_applied_to_a_different_model_revision() {
        let directory = TestDirectory::new();
        let library = test_library(&directory);
        library.ensure_directories().unwrap();
        let first = MotModel::new(metadata("amp-1", "Amp"), vec![1]).unwrap();
        let second = MotModel::new(metadata("amp-1", "Amp"), vec![2]).unwrap();
        library
            .save_tone(&ToneSettings::defaults_for(
                &first.model_ref("first.motmodel"),
            ))
            .unwrap();

        assert!(matches!(
            library.load_tone(&second.model_ref("second.motmodel")),
            Err(LibraryError::ToneModelHashMismatch { .. })
        ));
    }

    #[test]
    fn tone_save_replaces_atomically_and_leaves_no_temporary_files() {
        let directory = TestDirectory::new();
        let library = test_library(&directory);
        library.ensure_directories().unwrap();
        let reference = MotModel::new(metadata("amp-1", "Amp"), vec![1])
            .unwrap()
            .model_ref("amp.motmodel");
        let mut settings = ToneSettings::defaults_for(&reference);
        library.save_tone(&settings).unwrap();
        settings.input_gain_db = 4.5;
        library.save_tone(&settings).unwrap();

        assert_eq!(
            library
                .load_tone(&reference)
                .unwrap()
                .unwrap()
                .input_gain_db,
            4.5
        );
        let filenames: Vec<_> = fs::read_dir(&library.paths().model_settings)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        assert_eq!(filenames, vec![std::ffi::OsString::from("amp-1.json")]);
    }

    #[test]
    fn malformed_or_out_of_range_tone_is_rejected() {
        let directory = TestDirectory::new();
        let library = test_library(&directory);
        library.ensure_directories().unwrap();
        let reference = MotModel::new(metadata("amp-1", "Amp"), vec![1])
            .unwrap()
            .model_ref("amp.motmodel");
        fs::write(
            library.tone_settings_path("amp-1"),
            r#"{
                "schema_version": 1,
                "model_id": "amp-1",
                "model_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
                "input_gain_db": 99,
                "tight_percent": 0,
                "bite_percent": 0,
                "ir": null
            }"#,
        )
        .unwrap();

        assert!(matches!(
            library.load_tone(&reference),
            Err(LibraryError::InvalidToneSettings(_))
        ));
    }
}
