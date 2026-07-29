//! MOT PLAYER — zero-latency mono neural amp and cabinet player.
//!
//! The wrapper owns only Player parameters, background tasks, and UI state.
//! Format-independent DSP, model validation, persistence, and IR preparation
//! live in `mot-core`.

mod editor;

use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use crossbeam_queue::ArrayQueue;
use mot_core::amp::AmpControls;
use mot_core::model::{ModelRef, Sha256Digest};
use mot_core::model_library::{
    ImportedIr, ImportedNam, IrLibraryScan, IrProcessingMode, IrReference, ModelEntry,
    ModelLibrary, ModelScan, ToneSettings,
};
use mot_core::runtime::{
    PreparedRuntime, RuntimeAsset, RuntimeLoadRequest, RuntimeLoadStatus, RuntimeLoader,
    RuntimeMailbox, RuntimeMuteReason, RuntimeUpdate,
};
use mot_core::signal_chain::{GuitarSignalChain, OutputMute, RuntimeApplyStatus};
use truce::prelude::*;
use truce_egui::EguiEditor;

use editor::{MotPlayerUi, WINDOW_SIZE};

#[derive(Params)]
pub struct MotPlayerParams {
    #[param(name = "Bypass", flags = "automatable | bypass")]
    pub bypass: BoolParam,

    #[param(name = "Mute", flags = "automatable")]
    pub mute: BoolParam,

    #[param(
        name = "Input Gain",
        range = "linear(-24, 24)",
        default = 0,
        flags = "automatable"
    )]
    pub input_gain: FloatParam,

    #[param(
        name = "Tight",
        range = "linear(0, 100)",
        default = 0,
        flags = "automatable"
    )]
    pub tight: FloatParam,

    #[param(
        name = "Bite",
        range = "linear(0, 100)",
        default = 0,
        flags = "automatable"
    )]
    pub bite: FloatParam,

    #[param(name = "IR Processing", range = "discrete(0, 1)", default = 0)]
    pub ir_processing: IntParam,

    #[persist]
    pub selected_model_id: RwLock<String>,
    #[persist]
    pub selected_model_sha256: RwLock<String>,
    #[persist]
    pub selected_model_filename_hint: RwLock<String>,
    #[persist]
    pub selected_ir_path: RwLock<String>,
    #[persist]
    pub selected_ir_id: RwLock<String>,
    #[persist]
    pub selected_ir_sha256: RwLock<String>,
    #[persist]
    pub selected_ir_filename_hint: RwLock<String>,
    #[persist]
    pub runtime_generation: AtomicCell<u64>,

    #[skip]
    pub runtime_mailbox: Arc<RuntimeMailbox>,
    #[skip]
    pub runtime_request_epoch: AtomicCell<u64>,
    #[skip]
    pub runtime_control: Arc<RuntimeStatusControl>,
    #[skip]
    pub ir_import_control: Arc<IrImportControl>,
    #[skip]
    pub library_control: Arc<LibraryControl>,

    #[meter]
    pub runtime_status: MeterSlot,
}

pub(crate) use MotPlayerParamsParamId as P;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) enum RuntimeUiState {
    #[default]
    Transparent,
    Loading,
    Ready {
        model_name: String,
        ir_name: Option<String>,
    },
    SafeMuted {
        asset: RuntimeAsset,
        message: String,
    },
}

#[derive(Debug, Default)]
pub struct RuntimeStatusControl {
    state: RwLock<RuntimeUiState>,
}

impl RuntimeStatusControl {
    pub(crate) fn set(&self, state: RuntimeUiState) {
        if let Ok(mut current) = self.state.write() {
            *current = state;
        }
    }

    pub(crate) fn get(&self) -> RuntimeUiState {
        self.state
            .read()
            .map_or_else(|_| RuntimeUiState::Loading, |state| state.clone())
    }
}

const IR_IMPORT_OUTCOME_CAPACITY: usize = 2;

#[derive(Debug)]
pub(crate) enum IrImportOutcome {
    Imported(Box<ImportedIr>),
    Error(String),
}

#[derive(Debug)]
pub struct IrImportControl {
    busy: AtomicBool,
    outcomes: ArrayQueue<IrImportOutcome>,
}

impl Default for IrImportControl {
    fn default() -> Self {
        Self {
            busy: AtomicBool::new(false),
            outcomes: ArrayQueue::new(IR_IMPORT_OUTCOME_CAPACITY),
        }
    }
}

impl IrImportControl {
    pub(crate) fn try_begin(&self) -> bool {
        self.busy
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    pub(crate) fn cancel_begin(&self) {
        self.busy.store(false, Ordering::Release);
    }

    pub(crate) fn is_busy(&self) -> bool {
        self.busy.load(Ordering::Acquire)
    }

    pub(crate) fn take_outcome(&self) -> Option<IrImportOutcome> {
        self.outcomes.pop()
    }

    fn finish(&self, outcome: IrImportOutcome) {
        let _ = self.outcomes.force_push(outcome);
        self.busy.store(false, Ordering::Release);
    }
}

pub(crate) struct ImportIrTask {
    pub source: PathBuf,
}

impl BackgroundTask for ImportIrTask {
    type Params = MotPlayerParams;
    const SERIALIZED: bool = true;

    fn run(self, params: &Self::Params) {
        let outcome = if !has_extension(&self.source, "wav") {
            IrImportOutcome::Error("Select a .wav cabinet IR file".to_owned())
        } else {
            ModelLibrary::for_current_user()
                .and_then(|library| library.import_ir(&self.source))
                .map_or_else(
                    |error| IrImportOutcome::Error(error.to_string()),
                    |imported| IrImportOutcome::Imported(Box::new(imported)),
                )
        };
        params.ir_import_control.finish(outcome);
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ExternalFilePickerKind {
    NamModel,
    CabinetIr,
}

impl ExternalFilePickerKind {
    const fn prompt(self) -> &'static str {
        match self {
            Self::NamModel => "Import NAM Model (.nam)",
            Self::CabinetIr => "Import Cabinet IR (.wav)",
        }
    }
}

#[cfg(target_os = "macos")]
pub(crate) fn spawn_external_file_picker(request: ExternalFilePickerKind) -> Result<Child, String> {
    const SCRIPT: &str = r#"ObjC.import("AppKit");
function run(argv) {
    try {
        const app = $.NSApplication.sharedApplication;
        app.setActivationPolicy($.NSApplicationActivationPolicyAccessory);
        $.NSRunningApplication.currentApplication.activateWithOptions(0);
        app.activateIgnoringOtherApps(true);
        const panel = $.NSOpenPanel.openPanel;
        panel.setCanChooseFiles(true);
        panel.setCanChooseDirectories(false);
        panel.setAllowsMultipleSelection(false);
        panel.setResolvesAliases(true);
        panel.setTitle("MOT PLAYER");
        panel.setMessage(argv[0] || "Import File");
        panel.setPrompt("Import");
        const response = Number(panel.runModal);
        if (response !== Number($.NSModalResponseOK)) {
            return JSON.stringify({status: "cancel"});
        }
        const url = panel.URL;
        if (url == null) {
            return JSON.stringify({status: "error", message: "No file URL returned"});
        }
        return JSON.stringify({status: "ok", path: ObjC.unwrap(url.path)});
    } catch (error) {
        return JSON.stringify({status: "error", message: String(error)});
    }
}"#;
    Command::new("/usr/bin/osascript")
        .args(["-l", "JavaScript", "-e", SCRIPT, "--", request.prompt()])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("cannot launch isolated macOS picker: {error}"))
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn spawn_external_file_picker(
    _request: ExternalFilePickerKind,
) -> Result<Child, String> {
    Err("isolated file selection is currently available only on macOS".to_owned())
}

pub(crate) fn parse_external_file_picker_output(
    succeeded: bool,
    stdout: &[u8],
    stderr: &[u8],
) -> Result<Option<PathBuf>, String> {
    if !succeeded {
        let detail = String::from_utf8_lossy(stderr);
        let detail = detail.trim();
        return Err(if detail.is_empty() {
            "isolated file picker exited unsuccessfully".to_owned()
        } else {
            format!("isolated file picker failed: {detail}")
        });
    }

    let response: serde_json::Value = serde_json::from_slice(stdout)
        .map_err(|error| format!("isolated file picker returned invalid JSON: {error}"))?;
    match response.get("status").and_then(serde_json::Value::as_str) {
        Some("cancel") => Ok(None),
        Some("ok") => {
            let path = response
                .get("path")
                .and_then(serde_json::Value::as_str)
                .filter(|path| !path.is_empty())
                .ok_or_else(|| "isolated file picker returned an empty path".to_owned())?;
            Ok(Some(PathBuf::from(path)))
        }
        Some("error") => Err(response
            .get("message")
            .and_then(serde_json::Value::as_str)
            .filter(|message| !message.is_empty())
            .unwrap_or("isolated file picker reported an unknown error")
            .to_owned()),
        _ => Err("isolated file picker returned an unexpected response".to_owned()),
    }
}

fn has_extension(path: &std::path::Path, expected: &str) -> bool {
    path.extension()
        .and_then(std::ffi::OsStr::to_str)
        .is_some_and(|extension| extension.eq_ignore_ascii_case(expected))
}

const LIBRARY_OUTCOME_CAPACITY: usize = 4;

#[derive(Debug)]
pub(crate) struct SelectedToneSnapshot {
    pub reference: ModelRef,
    pub tone: Result<Option<ToneSettings>, String>,
}

#[derive(Debug)]
pub(crate) struct LibrarySnapshot {
    pub models: Result<ModelScan, String>,
    pub irs: Result<IrLibraryScan, String>,
    pub selected_tone: Option<SelectedToneSnapshot>,
}

#[derive(Debug)]
pub(crate) enum LibraryOutcome {
    Scanned {
        request_id: u64,
        result: Result<Box<LibrarySnapshot>, String>,
    },
    ToneLoaded {
        request_id: u64,
        entry: Box<ModelEntry>,
        guard_model_id: String,
        guard_model_sha256: String,
        result: Result<Option<ToneSettings>, String>,
    },
    ToneSaved {
        request_id: u64,
        settings: ToneSettings,
        result: Result<(), String>,
    },
    NamImported {
        request_id: u64,
        result: Result<Box<ImportedNam>, String>,
    },
    FolderOpened {
        request_id: u64,
        result: Result<(), String>,
    },
}

impl LibraryOutcome {
    pub(crate) const fn request_id(&self) -> u64 {
        match self {
            Self::Scanned { request_id, .. }
            | Self::ToneLoaded { request_id, .. }
            | Self::ToneSaved { request_id, .. }
            | Self::NamImported { request_id, .. }
            | Self::FolderOpened { request_id, .. } => *request_id,
        }
    }
}

#[derive(Debug)]
pub struct LibraryControl {
    busy: AtomicBool,
    next_request_id: AtomicU64,
    latest_request_id: AtomicU64,
    outcomes: ArrayQueue<LibraryOutcome>,
}

impl Default for LibraryControl {
    fn default() -> Self {
        Self {
            busy: AtomicBool::new(false),
            next_request_id: AtomicU64::new(0),
            latest_request_id: AtomicU64::new(0),
            outcomes: ArrayQueue::new(LIBRARY_OUTCOME_CAPACITY),
        }
    }
}

impl LibraryControl {
    fn next_request_id(&self) -> u64 {
        self.next_request_id
            .fetch_add(1, Ordering::AcqRel)
            .wrapping_add(1)
            .max(1)
    }

    pub(crate) fn try_begin(&self) -> Option<u64> {
        self.busy
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .ok()?;
        let request_id = self.next_request_id();
        self.latest_request_id.store(request_id, Ordering::Release);
        Some(request_id)
    }

    pub(crate) fn invalidate_pending(&self) {
        let invalidation_id = self.next_request_id();
        self.latest_request_id
            .store(invalidation_id, Ordering::Release);
    }

    pub(crate) fn cancel_begin(&self, request_id: u64) {
        if self.latest_request_id.load(Ordering::Acquire) == request_id {
            self.busy.store(false, Ordering::Release);
        }
    }

    pub(crate) fn is_busy(&self) -> bool {
        self.busy.load(Ordering::Acquire)
    }

    pub(crate) fn is_current(&self, request_id: u64) -> bool {
        self.latest_request_id.load(Ordering::Acquire) == request_id
    }

    pub(crate) fn take_outcome(&self) -> Option<LibraryOutcome> {
        self.outcomes.pop()
    }

    fn finish(&self, outcome: LibraryOutcome) {
        let _ = self.outcomes.force_push(outcome);
        self.busy.store(false, Ordering::Release);
    }
}

#[derive(Debug)]
pub(crate) enum LibraryTaskOperation {
    Scan {
        selected_model: Option<ModelRef>,
    },
    LoadTone {
        entry: Box<ModelEntry>,
        guard_model_id: String,
        guard_model_sha256: String,
    },
    SaveTone {
        model_reference: ModelRef,
        settings: ToneSettings,
    },
    ImportNam {
        source: PathBuf,
    },
    OpenFolder,
}

#[derive(Debug)]
pub(crate) struct LibraryTask {
    pub request_id: u64,
    pub operation: LibraryTaskOperation,
}

impl BackgroundTask for LibraryTask {
    type Params = MotPlayerParams;
    const SERIALIZED: bool = true;

    fn run(self, params: &Self::Params) {
        let outcome = run_library_task(self);
        params.library_control.finish(outcome);
    }
}

fn run_library_task(task: LibraryTask) -> LibraryOutcome {
    let request_id = task.request_id;
    let library = match ModelLibrary::for_current_user() {
        Ok(library) => library,
        Err(error) => {
            return library_unavailable_outcome(request_id, task.operation, error.to_string());
        }
    };

    match task.operation {
        LibraryTaskOperation::Scan { selected_model } => {
            let models = library
                .scan(&mot_core::runtime::tracking_runtime_limits())
                .map_err(|error| error.to_string());
            let irs = library.scan_irs().map_err(|error| error.to_string());
            let selected_tone = selected_model.map(|reference| {
                let tone = library
                    .load_exact(&reference, &mot_core::runtime::tracking_runtime_limits())
                    .map_err(|error| error.to_string())
                    .and_then(|_| {
                        library
                            .load_tone(&reference)
                            .map_err(|error| error.to_string())
                    });
                SelectedToneSnapshot { reference, tone }
            });
            LibraryOutcome::Scanned {
                request_id,
                result: Ok(Box::new(LibrarySnapshot {
                    models,
                    irs,
                    selected_tone,
                })),
            }
        }
        LibraryTaskOperation::LoadTone {
            entry,
            guard_model_id,
            guard_model_sha256,
        } => {
            let result = library
                .load_exact(
                    &entry.reference,
                    &mot_core::runtime::tracking_runtime_limits(),
                )
                .map_err(|error| error.to_string())
                .and_then(|_| {
                    library
                        .load_tone(&entry.reference)
                        .map_err(|error| error.to_string())
                });
            LibraryOutcome::ToneLoaded {
                request_id,
                entry,
                guard_model_id,
                guard_model_sha256,
                result,
            }
        }
        LibraryTaskOperation::SaveTone {
            model_reference,
            settings,
        } => {
            let result = library
                .load_exact(
                    &model_reference,
                    &mot_core::runtime::tracking_runtime_limits(),
                )
                .map_err(|error| error.to_string())
                .and_then(|_| {
                    library
                        .save_tone(&settings)
                        .map_err(|error| error.to_string())
                });
            LibraryOutcome::ToneSaved {
                request_id,
                settings,
                result,
            }
        }
        LibraryTaskOperation::ImportNam { source } => LibraryOutcome::NamImported {
            request_id,
            result: if has_extension(&source, "nam") {
                library
                    .import_nam(&source)
                    .map(Box::new)
                    .map_err(|error| error.to_string())
            } else {
                Err("Select a .nam model file".to_owned())
            },
        },
        LibraryTaskOperation::OpenFolder => LibraryOutcome::FolderOpened {
            request_id,
            result: open_library_folder(&library),
        },
    }
}

fn library_unavailable_outcome(
    request_id: u64,
    operation: LibraryTaskOperation,
    message: String,
) -> LibraryOutcome {
    match operation {
        LibraryTaskOperation::Scan { .. } => LibraryOutcome::Scanned {
            request_id,
            result: Err(message),
        },
        LibraryTaskOperation::LoadTone {
            entry,
            guard_model_id,
            guard_model_sha256,
        } => LibraryOutcome::ToneLoaded {
            request_id,
            entry,
            guard_model_id,
            guard_model_sha256,
            result: Err(message),
        },
        LibraryTaskOperation::SaveTone { settings, .. } => LibraryOutcome::ToneSaved {
            request_id,
            settings,
            result: Err(message),
        },
        LibraryTaskOperation::ImportNam { .. } => LibraryOutcome::NamImported {
            request_id,
            result: Err(message),
        },
        LibraryTaskOperation::OpenFolder => LibraryOutcome::FolderOpened {
            request_id,
            result: Err(message),
        },
    }
}

fn open_library_folder(library: &ModelLibrary) -> Result<(), String> {
    library
        .ensure_directories()
        .map_err(|error| error.to_string())?;
    #[cfg(target_os = "macos")]
    {
        Command::new("open")
            .arg(&library.paths().plugin_root)
            .spawn()
            .map_err(|error| format!("Cannot open model library: {error}"))?;
        Ok(())
    }
    #[cfg(not(target_os = "macos"))]
    {
        Err(format!(
            "Open this folder manually: {}",
            library.paths().plugin_root.display()
        ))
    }
}

#[derive(Clone, Copy, Debug)]
struct LoadRuntimeTask {
    generation: u64,
    request_epoch: u64,
    host_sample_rate_hz: u32,
    host_max_block_size: usize,
}

impl BackgroundTask for LoadRuntimeTask {
    type Params = MotPlayerParams;
    const SERIALIZED: bool = true;

    fn run(self, params: &Self::Params) {
        params.runtime_mailbox.drain_retired();
        params.runtime_control.set(RuntimeUiState::Loading);

        let model_id = read_shared_string(&params.selected_model_id);
        if model_id.is_empty() {
            let mut runtime = PreparedRuntime::transparent();
            runtime.reset(&AudioConfig::new(
                f64::from(self.host_sample_rate_hz),
                self.host_max_block_size.max(1),
            ));
            if runtime_request_is_current(params, self.generation, self.request_epoch) {
                params.runtime_mailbox.publish_latest(RuntimeUpdate::Ready {
                    generation: self.generation,
                    runtime: Box::new(runtime),
                });
                params.runtime_control.set(RuntimeUiState::Transparent);
            }
            return;
        }

        let prepared = prepare_runtime_request(params, self, &model_id);
        match prepared {
            Ok(request) => {
                let loader = match ModelLibrary::for_current_user() {
                    Ok(library) => RuntimeLoader::new(library),
                    Err(error) => {
                        publish_runtime_error(
                            params,
                            self,
                            RuntimeAsset::RuntimeConfiguration,
                            RuntimeMuteReason::CorruptConfiguration,
                            error.to_string(),
                        );
                        return;
                    }
                };
                let outcome = loader.load(request);
                if runtime_request_is_current(params, self.generation, self.request_epoch) {
                    let ui_state = runtime_ui_state(&outcome.status);
                    params.runtime_mailbox.publish_latest(outcome.update);
                    params.runtime_control.set(ui_state);
                }
            }
            Err((asset, reason, message)) => {
                publish_runtime_error(params, self, asset, reason, message);
            }
        }
    }
}

fn prepare_runtime_request(
    params: &MotPlayerParams,
    task: LoadRuntimeTask,
    model_id: &str,
) -> Result<RuntimeLoadRequest, (RuntimeAsset, RuntimeMuteReason, String)> {
    let digest_text = read_shared_string(&params.selected_model_sha256);
    let digest = Sha256Digest::from_str(&digest_text).map_err(|error| {
        (
            RuntimeAsset::RuntimeConfiguration,
            RuntimeMuteReason::CorruptConfiguration,
            format!("selected model SHA-256 is invalid: {error}"),
        )
    })?;
    let filename_hint = {
        let hint = read_shared_string(&params.selected_model_filename_hint);
        if hint.is_empty() {
            format!("{model_id}.motmodel")
        } else {
            hint
        }
    };
    let model_reference = ModelRef {
        model_id: model_id.to_owned(),
        sha256: digest,
        filename_hint,
    };
    let mut tone = ToneSettings::defaults_for(&model_reference);
    tone.input_gain_db = params.input_gain.value();
    tone.tight_percent = params.tight.value();
    tone.bite_percent = params.bite.value();

    let ir_path_text = read_shared_string(&params.selected_ir_path);
    let ir_path = if ir_path_text.is_empty() {
        None
    } else {
        let ir_digest_text = read_shared_string(&params.selected_ir_sha256);
        let ir_digest = Sha256Digest::from_str(&ir_digest_text).map_err(|error| {
            (
                RuntimeAsset::RuntimeConfiguration,
                RuntimeMuteReason::CorruptConfiguration,
                format!("selected IR SHA-256 is invalid: {error}"),
            )
        })?;
        let ir_id = read_shared_string(&params.selected_ir_id);
        let filename_hint = read_shared_string(&params.selected_ir_filename_hint);
        if ir_id.is_empty() || filename_hint.is_empty() {
            return Err((
                RuntimeAsset::RuntimeConfiguration,
                RuntimeMuteReason::CorruptConfiguration,
                "selected IR identity is incomplete".to_owned(),
            ));
        }
        tone.ir = Some(IrReference {
            ir_id,
            sha256: ir_digest,
            filename_hint,
            processing: if params.ir_processing.value_i32() == 1 {
                IrProcessingMode::Raw
            } else {
                IrProcessingMode::MinimumPhaseAutoTrim
            },
        });
        Some(PathBuf::from(ir_path_text))
    };

    let mut request = RuntimeLoadRequest::new(task.generation, model_reference);
    request.tone = Some(tone);
    request.ir_path = ir_path;
    request.host_sample_rate_hz = task.host_sample_rate_hz;
    request.host_max_block_size = task.host_max_block_size.max(1);
    Ok(request)
}

fn publish_runtime_error(
    params: &MotPlayerParams,
    task: LoadRuntimeTask,
    asset: RuntimeAsset,
    reason: RuntimeMuteReason,
    message: String,
) {
    if runtime_request_is_current(params, task.generation, task.request_epoch) {
        params.runtime_mailbox.publish_latest(RuntimeUpdate::Mute {
            generation: task.generation,
            reason,
        });
        params
            .runtime_control
            .set(RuntimeUiState::SafeMuted { asset, message });
    }
}

fn runtime_ui_state(status: &RuntimeLoadStatus) -> RuntimeUiState {
    match status {
        RuntimeLoadStatus::Ready {
            model_reference,
            ir_reference,
            ..
        } => RuntimeUiState::Ready {
            model_name: model_reference.model_id.clone(),
            ir_name: ir_reference
                .as_ref()
                .map(|reference| reference.filename_hint.clone()),
        },
        RuntimeLoadStatus::Missing { asset, message, .. }
        | RuntimeLoadStatus::Corrupt { asset, message, .. } => RuntimeUiState::SafeMuted {
            asset: *asset,
            message: message.clone(),
        },
    }
}

fn runtime_request_is_current(
    params: &MotPlayerParams,
    generation: u64,
    request_epoch: u64,
) -> bool {
    params.runtime_generation.load() == generation
        && params.runtime_request_epoch.load() == request_epoch
}

fn advance_runtime_request_epoch(params: &MotPlayerParams) -> u64 {
    let next = params.runtime_request_epoch.load().wrapping_add(1).max(1);
    params.runtime_request_epoch.store(next);
    next
}

pub(crate) fn read_shared_string(value: &RwLock<String>) -> String {
    value
        .read()
        .map_or_else(|_| String::new(), |value| value.clone())
}

pub(crate) fn write_shared_string(value: &RwLock<String>, edited: &str) {
    if let Ok(mut current) = value.write() {
        *current = edited.to_owned();
    }
}

pub struct MotPlayer {
    signal_chain: GuitarSignalChain,
    output_mute: OutputMute,
    sample_rate: f32,
    max_block_size: usize,
    sample_rate_compatible: bool,
    requested_runtime_generation: u64,
    runtime_status: f32,
}

impl Default for MotPlayer {
    fn default() -> Self {
        Self {
            signal_chain: GuitarSignalChain::default(),
            output_mute: OutputMute::default(),
            sample_rate: 48_000.0,
            max_block_size: 0,
            sample_rate_compatible: true,
            requested_runtime_generation: u64::MAX,
            runtime_status: 0.5,
        }
    }
}

impl PluginLogic for MotPlayer {
    type Params = MotPlayerParams;
    type DspState = Self;

    fn init(_params: &Self::Params, _context: &InitContext) -> Self::DspState {
        Self::default()
    }

    fn supports_in_place() -> bool {
        false
    }

    fn bus_layouts() -> Vec<BusLayout> {
        vec![BusLayout::mono()]
    }

    fn reset(state: &mut Self::DspState, params: &Self::Params, config: &AudioConfig) {
        state.sample_rate = config.sample_rate as f32;
        state.max_block_size = config.max_block_size.max(1);
        state.sample_rate_compatible =
            config.sample_rate.round() as u32 == mot_core::model::REQUIRED_SAMPLE_RATE_HZ;
        state.signal_chain.reset(config);
        state
            .output_mute
            .reset(state.sample_rate, params.mute.value());
        advance_runtime_request_epoch(params);
        state.requested_runtime_generation = u64::MAX;
        state.runtime_status = if state.sample_rate_compatible {
            0.5
        } else {
            0.0
        };
    }

    fn state_changed(state: &mut Self::DspState, params: &Self::Params) {
        // A restored DAW state may contain different model/IR references while
        // carrying the same persisted generation as the runtime currently in
        // use. Invalidate both the in-flight request and the DSP-side
        // generation cache here; the next process block will schedule the
        // actual I/O and runtime construction on the background worker.
        advance_runtime_request_epoch(params);
        state.requested_runtime_generation = u64::MAX;
        state.runtime_status = if state.sample_rate_compatible {
            0.5
        } else {
            0.0
        };
    }

    fn process(
        state: &mut Self::DspState,
        params: &Self::Params,
        buffer: &mut AudioBuffer,
        _events: &EventList,
        context: &mut ProcessContext,
    ) -> ProcessStatus {
        let generation = params.runtime_generation.load();
        if generation != state.requested_runtime_generation {
            if let Some(spawner) = context.tasks::<LoadRuntimeTask>() {
                let request_epoch = advance_runtime_request_epoch(params);
                spawner.spawn_coalescing(LoadRuntimeTask {
                    generation,
                    request_epoch,
                    host_sample_rate_hz: state.sample_rate.round() as u32,
                    host_max_block_size: state.max_block_size,
                });
                state.runtime_status = 0.5;
            }
            state.requested_runtime_generation = generation;
        }

        if let Some(status) = state.signal_chain.poll_runtime(&params.runtime_mailbox) {
            state.runtime_status = match status {
                RuntimeApplyStatus::Ready { .. } => 1.0,
                RuntimeApplyStatus::SafeMuted { .. } => 0.0,
            };
        }
        state.signal_chain.set_controls(AmpControls {
            input_gain_db: params.input_gain.value(),
            tight: params.tight.value() / 100.0,
            bite: params.bite.value() / 100.0,
        });

        let bypassed = params.bypass.value();
        let muted = params.mute.value();
        if buffer.num_input_channels() > 0 && buffer.num_output_channels() > 0 {
            let (input, output) = buffer.io_pair(0, 0);
            if state.sample_rate_compatible {
                state.signal_chain.process_block(input, output);
            } else {
                output.fill(0.0);
                state.runtime_status = 0.0;
            }
            for index in 0..output.len() {
                let mute_gain = state.output_mute.next_gain(muted);
                if bypassed {
                    output[index] = input[index];
                } else {
                    output[index] *= mute_gain;
                }
            }
        }

        context.set_meter(P::RuntimeStatus, state.runtime_status.clamp(0.0, 1.0));
        ProcessStatus::Normal
    }

    fn latency(state: &Self::DspState) -> u32 {
        state.signal_chain.latency_samples()
    }

    fn tail(state: &Self::DspState) -> u32 {
        state.signal_chain.tail_samples()
    }

    fn editor(params: Arc<MotPlayerParams>) -> Box<dyn Editor> {
        EguiEditor::with_ui(params, WINDOW_SIZE, MotPlayerUi).into_editor()
    }
}

truce::plugin! {
    logic: MotPlayer,
    params: MotPlayerParams,
    tasks: [LoadRuntimeTask, ImportIrTask, LibraryTask],
}

truce::enable_rt_paranoid!();

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn player_exposes_only_the_expected_audio_parameters() {
        let params = MotPlayerParams::new();
        assert_eq!(params.input_gain.value(), 0.0);
        assert_eq!(params.tight.value(), 0.0);
        assert_eq!(params.bite.value(), 0.0);
        assert!(!params.mute.value());
    }

    #[test]
    fn player_runtime_reports_zero_samples_of_latency() {
        let mut player = MotPlayer::default();
        player.signal_chain.reset(&AudioConfig::new(48_000.0, 512));
        assert_eq!(player.signal_chain.latency_samples(), 0);
    }

    #[test]
    fn editor_has_a_headless_render_path() {
        let params = Arc::new(MotPlayerParams::new());
        let mut editor = EguiEditor::with_ui(Arc::clone(&params), WINDOW_SIZE, MotPlayerUi);
        let erased: Arc<dyn truce::params::Params> = params;
        assert_eq!(Editor::size(&editor), WINDOW_SIZE);
        if let Some((_, width, height)) = Editor::screenshot(&mut editor, erased) {
            assert_eq!((width, height), (WINDOW_SIZE.0 * 2, WINDOW_SIZE.1 * 2));
        }
    }

    #[test]
    fn isolated_file_picker_protocol_preserves_paths_cancel_and_errors() {
        assert_eq!(
            parse_external_file_picker_output(
                true,
                "{\"status\":\"ok\",\"path\":\"/Users/test/Модель\\nOne.nam\"}\n".as_bytes(),
                b"",
            )
            .unwrap(),
            Some(PathBuf::from("/Users/test/Модель\nOne.nam"))
        );
        assert_eq!(
            parse_external_file_picker_output(true, b"{\"status\":\"cancel\"}\n", b"").unwrap(),
            None
        );
        assert_eq!(
            parse_external_file_picker_output(false, b"", b"helper failed\n").unwrap_err(),
            "isolated file picker failed: helper failed"
        );
        assert!(
            parse_external_file_picker_output(true, b"unexpected\n", b"")
                .unwrap_err()
                .contains("invalid JSON")
        );
        assert!(has_extension(std::path::Path::new("/tmp/MODEL.NAM"), "nam"));
        assert!(!has_extension(
            std::path::Path::new("/tmp/model.wav"),
            "nam"
        ));
    }

    #[test]
    fn restored_state_forces_runtime_reload_when_generation_is_unchanged() {
        let params = MotPlayerParams::new();
        params.runtime_generation.store(7);
        params.runtime_request_epoch.store(11);

        let mut player = MotPlayer {
            requested_runtime_generation: 7,
            runtime_status: 1.0,
            ..MotPlayer::default()
        };

        <MotPlayer as PluginLogic>::state_changed(&mut player, &params);

        assert_eq!(player.requested_runtime_generation, u64::MAX);
        assert_eq!(params.runtime_request_epoch.load(), 12);
        assert_eq!(player.runtime_status, 0.5);
        assert_ne!(
            params.runtime_generation.load(),
            player.requested_runtime_generation
        );
    }
}
