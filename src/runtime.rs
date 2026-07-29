//! Off-thread runtime preparation and lock-free audio-thread handoff.
//!
//! Filesystem access, model decoding, WAV parsing, IR preparation, and object
//! destruction happen on the loader thread.  The audio side only moves an
//! already prepared runtime out of a single-slot queue and moves the retired
//! runtime into a bounded queue.

use std::fmt;
use std::fs;
use std::io;
use std::path::PathBuf;

use crossbeam_queue::ArrayQueue;
use truce::prelude::AudioConfig;

use crate::amp::{AmpControls, AmpProcessor, MAX_MODEL_MACS_PER_SAMPLE};
use crate::cabinet::{CabinetIrImportOptions, CabinetIrMode, CabinetProcessor, PreparedCabinetIr};
use crate::model::{
    DIAGONAL_RNN_ARCHITECTURE_ID, DIAGONAL_RNN_ARCHITECTURE_VERSION, ModelMetadata, ModelRef,
    ModelRuntimeLimits, REQUIRED_SAMPLE_RATE_HZ, SupportedArchitecture, sha256,
};
use crate::model_library::{
    IrProcessingMode, IrReference, LibraryError, ModelLibrary, ToneSettings,
};
use crate::trainer::{MODEL_ARCHITECTURE_ID, decode_model_payload, model_descriptor};
use crate::wav_io::{WavError, decode_mono_wav};

pub const DEFAULT_RETIRED_RUNTIME_CAPACITY: usize = 8;
const DEFAULT_IR_TRIM_THRESHOLD_DB: f32 = -80.0;

/// Fully constructed amp + cabinet state.
///
/// Construction is intentionally private: every instance has passed model,
/// payload, sample-rate, and optional IR validation before it can reach the
/// audio callback.
#[derive(Clone, Debug)]
pub struct PreparedRuntime {
    amp: AmpProcessor,
    cabinet: CabinetProcessor,
    model_reference: Option<ModelRef>,
    ir_reference: Option<IrReference>,
}

impl Default for PreparedRuntime {
    fn default() -> Self {
        Self::transparent()
    }
}

impl PreparedRuntime {
    /// Startup runtime: raw amp, no cabinet IR, bit-exact pass-through.
    #[must_use]
    pub fn transparent() -> Self {
        Self {
            amp: AmpProcessor::default(),
            cabinet: CabinetProcessor::default(),
            model_reference: None,
            ir_reference: None,
        }
    }

    #[must_use]
    pub const fn model_reference(&self) -> Option<&ModelRef> {
        self.model_reference.as_ref()
    }

    #[must_use]
    pub const fn ir_reference(&self) -> Option<&IrReference> {
        self.ir_reference.as_ref()
    }

    #[must_use]
    pub const fn amp(&self) -> &AmpProcessor {
        &self.amp
    }

    #[must_use]
    pub const fn cabinet(&self) -> &CabinetProcessor {
        &self.cabinet
    }

    pub fn set_controls(&mut self, controls: AmpControls) {
        self.amp.set_controls(controls);
    }

    /// Reinitializes causal state without allocating.
    pub fn reset(&mut self, config: &AudioConfig) {
        self.amp.reset(config);
        self.cabinet.reset(config);
    }

    /// Processes directly into caller-owned scratch/output buffers.
    ///
    /// The scratch buffer must contain at least `input.len()` samples and is
    /// normally owned by the signal chain.  No allocation or internal block
    /// accumulation occurs here.
    #[inline]
    pub fn process_block(&mut self, input: &[f32], amp_scratch: &mut [f32], output: &mut [f32]) {
        debug_assert_eq!(input.len(), output.len());
        debug_assert!(amp_scratch.len() >= input.len());
        let amp_output = &mut amp_scratch[..input.len()];
        self.amp.process_block(input, amp_output);
        self.cabinet.process_block(amp_output, output);
    }

    #[must_use]
    pub fn latency_samples(&self) -> u32 {
        self.amp.latency_samples() + self.cabinet.latency_samples()
    }

    #[must_use]
    pub fn tail_samples(&self) -> u32 {
        self.amp.tail_samples() + self.cabinet.tail_samples()
    }

    #[must_use]
    pub fn into_processors(self) -> (AmpProcessor, CabinetProcessor) {
        (self.amp, self.cabinet)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RuntimeAsset {
    Model,
    CabinetIr,
    RuntimeConfiguration,
}

/// Detailed loader/UI status.  It intentionally never enters the audio queue
/// because its strings may allocate and deallocate.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RuntimeLoadStatus {
    Ready {
        generation: u64,
        model_reference: ModelRef,
        ir_reference: Option<IrReference>,
    },
    Missing {
        generation: u64,
        asset: RuntimeAsset,
        message: String,
    },
    Corrupt {
        generation: u64,
        asset: RuntimeAsset,
        message: String,
    },
}

impl RuntimeLoadStatus {
    #[cfg(test)]
    #[must_use]
    pub const fn is_ready(&self) -> bool {
        matches!(self, Self::Ready { .. })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RuntimeMuteReason {
    MissingModel,
    MissingCabinetIr,
    CorruptModel,
    CorruptCabinetIr,
    CorruptConfiguration,
}

/// Heap-free control message consumed at an audio block boundary.
///
/// A `Ready` value may own heap-backed DSP state, but queueing/taking it only
/// moves ownership.  The audio thread must return replaced runtimes through
/// [`RuntimeMailbox::try_retire`] instead of dropping them.
#[derive(Debug)]
pub enum RuntimeUpdate {
    Ready {
        generation: u64,
        runtime: Box<PreparedRuntime>,
    },
    Mute {
        generation: u64,
        reason: RuntimeMuteReason,
    },
}

impl RuntimeUpdate {
    #[must_use]
    pub const fn generation(&self) -> u64 {
        match self {
            Self::Ready { generation, .. } | Self::Mute { generation, .. } => *generation,
        }
    }
}

pub struct RuntimeLoadOutcome {
    pub status: RuntimeLoadStatus,
    pub update: RuntimeUpdate,
}

impl fmt::Debug for RuntimeLoadOutcome {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RuntimeLoadOutcome")
            .field("status", &self.status)
            .field("update_generation", &self.update.generation())
            .finish_non_exhaustive()
    }
}

/// One pending prepared runtime plus a bounded reverse retirement queue.
///
/// `publish_latest()` and `drain_retired()` are loader-thread methods.
/// `take_latest()` and `try_retire()` are audio-thread methods.  The queues
/// allocate only in `new()`.
pub struct RuntimeMailbox {
    pending: ArrayQueue<RuntimeUpdate>,
    retired: ArrayQueue<Box<PreparedRuntime>>,
}

impl Default for RuntimeMailbox {
    fn default() -> Self {
        Self::new(DEFAULT_RETIRED_RUNTIME_CAPACITY)
    }
}

impl RuntimeMailbox {
    #[must_use]
    pub fn new(retired_capacity: usize) -> Self {
        Self {
            pending: ArrayQueue::new(1),
            retired: ArrayQueue::new(retired_capacity.max(1)),
        }
    }

    /// Publishes a loader result, replacing and destroying an older unconsumed
    /// result on this (non-real-time) thread.
    pub fn publish_latest(&self, mut update: RuntimeUpdate) {
        self.drain_retired();
        loop {
            match self.pending.push(update) {
                Ok(()) => return,
                Err(returned) => {
                    update = returned;
                    if let Some(stale) = self.pending.pop() {
                        // Any Vec/String-backed DSP state is destroyed here,
                        // never inside the audio callback.
                        drop(stale);
                    } else {
                        // The consumer won the race between push and pop.
                        std::hint::spin_loop();
                    }
                }
            }
        }
    }

    /// Audio-thread operation: takes the only pending update, if any.
    #[inline]
    pub fn take_latest(&self) -> Option<RuntimeUpdate> {
        self.pending.pop()
    }

    /// Audio-thread operation: transfers ownership of an old runtime to the
    /// loader.  On backpressure, ownership is returned to the caller; callers
    /// must retain it and retry rather than dropping it on the audio thread.
    #[inline]
    pub fn try_retire(&self, runtime: Box<PreparedRuntime>) -> Result<(), Box<PreparedRuntime>> {
        self.retired.push(runtime)
    }

    /// Loader-thread operation: destroys all runtimes returned by audio.
    pub fn drain_retired(&self) -> usize {
        let mut drained = 0;
        while let Some(runtime) = self.retired.pop() {
            drop(runtime);
            drained += 1;
        }
        drained
    }

    #[must_use]
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    #[must_use]
    pub fn retired_len(&self) -> usize {
        self.retired.len()
    }
}

#[derive(Clone, Debug)]
pub struct RuntimeLoadRequest {
    pub generation: u64,
    pub model_reference: ModelRef,
    /// Explicit DAW-state tone.  `None` means load the library tone, falling
    /// back to the model defaults if no sidecar exists.
    pub tone: Option<ToneSettings>,
    /// Required only when the selected tone contains an IR reference.
    pub ir_path: Option<PathBuf>,
    pub host_sample_rate_hz: u32,
    pub host_max_block_size: usize,
}

impl RuntimeLoadRequest {
    #[must_use]
    pub fn new(generation: u64, model_reference: ModelRef) -> Self {
        Self {
            generation,
            model_reference,
            tone: None,
            ir_path: None,
            host_sample_rate_hz: REQUIRED_SAMPLE_RATE_HZ,
            host_max_block_size: 512,
        }
    }
}

/// Blocking loader intended for a dedicated worker or Truce background task.
///
/// The type performs no work in its constructor.  `load()` and
/// `load_and_publish()` must never be called from an audio callback.
#[derive(Clone, Debug)]
pub struct RuntimeLoader {
    library: ModelLibrary,
    limits: ModelRuntimeLimits,
}

impl RuntimeLoader {
    #[must_use]
    pub fn new(library: ModelLibrary) -> Self {
        Self {
            library,
            limits: tracking_runtime_limits(),
        }
    }

    /// Loads and prepares a complete runtime synchronously on the caller's
    /// loader thread.
    #[must_use]
    pub fn load(&self, request: RuntimeLoadRequest) -> RuntimeLoadOutcome {
        let generation = request.generation;
        if request.host_sample_rate_hz != REQUIRED_SAMPLE_RATE_HZ {
            return corrupt_outcome(
                generation,
                RuntimeAsset::RuntimeConfiguration,
                RuntimeMuteReason::CorruptConfiguration,
                format!(
                    "normal runtime requires {REQUIRED_SAMPLE_RATE_HZ} Hz, host requested {} Hz",
                    request.host_sample_rate_hz
                ),
            );
        }
        if request.host_max_block_size == 0 {
            return corrupt_outcome(
                generation,
                RuntimeAsset::RuntimeConfiguration,
                RuntimeMuteReason::CorruptConfiguration,
                "host max block size must be greater than zero".to_owned(),
            );
        }

        let container = match self
            .library
            .load_exact(&request.model_reference, &self.limits)
        {
            Ok(model) => model,
            Err(error) => return model_error_outcome(generation, error),
        };
        if let Err(message) = validate_outer_metadata(container.metadata()) {
            return corrupt_outcome(
                generation,
                RuntimeAsset::Model,
                RuntimeMuteReason::CorruptModel,
                message,
            );
        }
        let model = match decode_model_payload(container.payload()) {
            Ok(model) => model,
            Err(error) => {
                return corrupt_outcome(
                    generation,
                    RuntimeAsset::Model,
                    RuntimeMuteReason::CorruptModel,
                    error.to_string(),
                );
            }
        };
        if let Err(message) = validate_payload_consistency(container.metadata(), &model) {
            return corrupt_outcome(
                generation,
                RuntimeAsset::Model,
                RuntimeMuteReason::CorruptModel,
                message,
            );
        }

        let tone = match request.tone {
            Some(tone) => tone,
            None => match self.library.load_tone(&request.model_reference) {
                Ok(Some(tone)) => tone,
                Ok(None) => ToneSettings::defaults_for(&request.model_reference),
                Err(error) => {
                    return corrupt_outcome(
                        generation,
                        RuntimeAsset::RuntimeConfiguration,
                        RuntimeMuteReason::CorruptConfiguration,
                        error.to_string(),
                    );
                }
            },
        };
        if let Err(message) = validate_tone_binding(&tone, &request.model_reference) {
            return corrupt_outcome(
                generation,
                RuntimeAsset::RuntimeConfiguration,
                RuntimeMuteReason::CorruptConfiguration,
                message,
            );
        }

        let audio_config = AudioConfig::new(
            f64::from(request.host_sample_rate_hz),
            request.host_max_block_size,
        );
        let mut amp = AmpProcessor::default();
        amp.reset(&audio_config);
        if let Err(error) = amp.load_model(model) {
            return corrupt_outcome(
                generation,
                RuntimeAsset::Model,
                RuntimeMuteReason::CorruptModel,
                error.to_string(),
            );
        }
        amp.set_controls_immediate(AmpControls {
            input_gain_db: tone.input_gain_db,
            tight: tone.tight_percent / 100.0,
            bite: tone.bite_percent / 100.0,
        });

        let ir_reference = tone.ir.clone();
        let mut cabinet = match (&ir_reference, &request.ir_path) {
            (None, None) => CabinetProcessor::default(),
            (Some(reference), Some(path)) => match prepare_cabinet(path, reference) {
                Ok(cabinet) => cabinet,
                Err(IrLoadFailure::Missing(message)) => {
                    return missing_outcome(
                        generation,
                        RuntimeAsset::CabinetIr,
                        RuntimeMuteReason::MissingCabinetIr,
                        message,
                    );
                }
                Err(IrLoadFailure::Corrupt(message)) => {
                    return corrupt_outcome(
                        generation,
                        RuntimeAsset::CabinetIr,
                        RuntimeMuteReason::CorruptCabinetIr,
                        message,
                    );
                }
            },
            (Some(reference), None) => {
                return missing_outcome(
                    generation,
                    RuntimeAsset::CabinetIr,
                    RuntimeMuteReason::MissingCabinetIr,
                    format!(
                        "IR {} ({}) has no resolved WAV path",
                        reference.ir_id, reference.filename_hint
                    ),
                );
            }
            (None, Some(path)) => {
                return corrupt_outcome(
                    generation,
                    RuntimeAsset::RuntimeConfiguration,
                    RuntimeMuteReason::CorruptConfiguration,
                    format!(
                        "IR path {} was supplied but the tone has no IR reference",
                        path.display()
                    ),
                );
            }
        };
        cabinet.reset(&audio_config);

        let runtime = PreparedRuntime {
            amp,
            cabinet,
            model_reference: Some(request.model_reference.clone()),
            ir_reference: ir_reference.clone(),
        };
        debug_assert_eq!(runtime.latency_samples(), 0);
        RuntimeLoadOutcome {
            status: RuntimeLoadStatus::Ready {
                generation,
                model_reference: request.model_reference,
                ir_reference,
            },
            update: RuntimeUpdate::Ready {
                generation,
                runtime: Box::new(runtime),
            },
        }
    }

    /// Convenience worker entry point: load, publish the heap-safe audio
    /// command, and return the detailed UI status to the caller.
    #[cfg(test)]
    pub fn load_and_publish(
        &self,
        request: RuntimeLoadRequest,
        mailbox: &RuntimeMailbox,
    ) -> RuntimeLoadStatus {
        let outcome = self.load(request);
        let RuntimeLoadOutcome { status, update } = outcome;
        mailbox.publish_latest(update);
        status
    }
}

#[must_use]
pub fn tracking_runtime_limits() -> ModelRuntimeLimits {
    ModelRuntimeLimits::new(
        u64::from(MAX_MODEL_MACS_PER_SAMPLE),
        vec![SupportedArchitecture::exact(
            DIAGONAL_RNN_ARCHITECTURE_ID,
            DIAGONAL_RNN_ARCHITECTURE_VERSION,
        )],
    )
}

fn validate_outer_metadata(metadata: &ModelMetadata) -> Result<(), String> {
    if DIAGONAL_RNN_ARCHITECTURE_ID != MODEL_ARCHITECTURE_ID {
        return Err("trainer/runtime architecture constants disagree".to_owned());
    }
    if metadata.architecture_id != DIAGONAL_RNN_ARCHITECTURE_ID
        || metadata.architecture_version != DIAGONAL_RNN_ARCHITECTURE_VERSION
    {
        return Err(format!(
            "container architecture {} v{} is not {} v{}",
            metadata.architecture_id,
            metadata.architecture_version,
            DIAGONAL_RNN_ARCHITECTURE_ID,
            DIAGONAL_RNN_ARCHITECTURE_VERSION
        ));
    }
    Ok(())
}

fn validate_payload_consistency(
    metadata: &ModelMetadata,
    model: &crate::amp::CompactCausalModel,
) -> Result<(), String> {
    let descriptor = model_descriptor(model);
    let consistent = descriptor.architecture_id == metadata.architecture_id
        && descriptor.architecture_version == metadata.architecture_version
        && descriptor.causal == metadata.causal
        && descriptor.lookahead_samples == metadata.lookahead_samples
        && descriptor.runtime_latency_samples == metadata.runtime_latency_samples
        && descriptor.sample_rate_hz == metadata.sample_rate_hz
        && u64::from(descriptor.estimated_macs_per_sample) == metadata.estimated_macs_per_sample;
    if consistent {
        Ok(())
    } else {
        Err("container metadata does not match the amplifier payload".to_owned())
    }
}

fn validate_tone_binding(tone: &ToneSettings, model: &ModelRef) -> Result<(), String> {
    tone.validate().map_err(|error| error.to_string())?;
    if tone.model_id != model.model_id {
        return Err(format!(
            "tone belongs to model {}, expected {}",
            tone.model_id, model.model_id
        ));
    }
    if tone.model_sha256 != model.sha256 {
        return Err(format!(
            "tone belongs to a different revision of model {}",
            model.model_id
        ));
    }
    Ok(())
}

enum IrLoadFailure {
    Missing(String),
    Corrupt(String),
}

fn prepare_cabinet(
    path: &PathBuf,
    reference: &IrReference,
) -> Result<CabinetProcessor, IrLoadFailure> {
    let bytes = fs::read(path).map_err(|error| {
        if error.kind() == io::ErrorKind::NotFound {
            IrLoadFailure::Missing(format!("IR WAV {} was not found", path.display()))
        } else {
            IrLoadFailure::Corrupt(format!("cannot read IR WAV {}: {error}", path.display()))
        }
    })?;
    let actual_digest = sha256(&bytes);
    if actual_digest != reference.sha256 {
        return Err(IrLoadFailure::Corrupt(format!(
            "IR {} SHA-256 does not match {}",
            reference.ir_id,
            path.display()
        )));
    }
    let wav = decode_mono_wav(&bytes).map_err(|error| match error {
        WavError::Io(error) if error.kind() == io::ErrorKind::NotFound => {
            IrLoadFailure::Missing(format!("IR WAV {} was not found", path.display()))
        }
        error => IrLoadFailure::Corrupt(error.to_string()),
    })?;
    if wav.channels != 1 {
        return Err(IrLoadFailure::Corrupt(format!(
            "IR {} has {} channels; mono is required",
            reference.ir_id, wav.channels
        )));
    }
    if wav.sample_rate != REQUIRED_SAMPLE_RATE_HZ {
        return Err(IrLoadFailure::Corrupt(format!(
            "IR {} is {} Hz; {} Hz is required",
            reference.ir_id, wav.sample_rate, REQUIRED_SAMPLE_RATE_HZ
        )));
    }
    let options = match reference.processing {
        IrProcessingMode::MinimumPhaseAutoTrim => CabinetIrImportOptions {
            mode: CabinetIrMode::MinimumPhase,
            trim_leading_silence: true,
            trim_threshold_db: DEFAULT_IR_TRIM_THRESHOLD_DB,
        },
        IrProcessingMode::Raw => CabinetIrImportOptions {
            mode: CabinetIrMode::Raw,
            trim_leading_silence: false,
            trim_threshold_db: DEFAULT_IR_TRIM_THRESHOLD_DB,
        },
    };
    let prepared = PreparedCabinetIr::prepare(&wav.samples, wav.sample_rate, options)
        .map_err(|error| IrLoadFailure::Corrupt(error.to_string()))?;
    Ok(CabinetProcessor::from_prepared(prepared))
}

fn model_error_outcome(generation: u64, error: LibraryError) -> RuntimeLoadOutcome {
    let missing = match &error {
        LibraryError::ModelNotFound(_) => true,
        LibraryError::Io(io_error) => io_error.kind() == io::ErrorKind::NotFound,
        _ => false,
    };
    if missing {
        missing_outcome(
            generation,
            RuntimeAsset::Model,
            RuntimeMuteReason::MissingModel,
            error.to_string(),
        )
    } else {
        corrupt_outcome(
            generation,
            RuntimeAsset::Model,
            RuntimeMuteReason::CorruptModel,
            error.to_string(),
        )
    }
}

fn missing_outcome(
    generation: u64,
    asset: RuntimeAsset,
    reason: RuntimeMuteReason,
    message: String,
) -> RuntimeLoadOutcome {
    RuntimeLoadOutcome {
        status: RuntimeLoadStatus::Missing {
            generation,
            asset,
            message,
        },
        update: RuntimeUpdate::Mute { generation, reason },
    }
}

fn corrupt_outcome(
    generation: u64,
    asset: RuntimeAsset,
    reason: RuntimeMuteReason,
    message: String,
) -> RuntimeLoadOutcome {
    RuntimeLoadOutcome {
        status: RuntimeLoadStatus::Corrupt {
            generation,
            asset,
            message,
        },
        update: RuntimeUpdate::Mute { generation, reason },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::path::Path;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::amp::CompactCausalModel;
    use crate::model::{ModelMetadata, MotModel};
    use crate::model_library::{ModelLibraryPaths, TONE_SETTINGS_VERSION};
    use crate::trainer::encode_model_payload;
    use crate::wav_io::write_mono_f32_wav;

    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new() -> Self {
            let unique = format!(
                "mot-runtime-tests-{}-{}-{}",
                std::process::id(),
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .expect("clock")
                    .as_nanos(),
                TEST_COUNTER.fetch_add(1, Ordering::Relaxed)
            );
            let path = env::temp_dir().join(unique);
            fs::create_dir(&path).expect("create test directory");
            Self(path)
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn setup() -> (TestDirectory, ModelLibrary, RuntimeLoader) {
        let directory = TestDirectory::new();
        let library = ModelLibrary::new(ModelLibraryPaths::from_plugin_root(
            directory.0.join("library"),
        ));
        library.ensure_directories().expect("library directories");
        let loader = RuntimeLoader::new(library.clone());
        (directory, library, loader)
    }

    fn metadata_for(model: &CompactCausalModel, model_id: &str) -> ModelMetadata {
        let descriptor = model_descriptor(model);
        ModelMetadata {
            model_id: model_id.to_owned(),
            display_name: model_id.to_owned(),
            architecture_id: descriptor.architecture_id.to_owned(),
            architecture_version: descriptor.architecture_version,
            sample_rate_hz: descriptor.sample_rate_hz,
            causal: descriptor.causal,
            lookahead_samples: descriptor.lookahead_samples,
            runtime_latency_samples: descriptor.runtime_latency_samples,
            estimated_macs_per_sample: u64::from(descriptor.estimated_macs_per_sample),
        }
    }

    fn install_model(
        library: &ModelLibrary,
        model_id: &str,
        model: &CompactCausalModel,
    ) -> ModelRef {
        let container =
            MotModel::new(metadata_for(model, model_id), encode_model_payload(model)).unwrap();
        let filename = format!("{model_id}.motmodel");
        container
            .write_new(library.paths().models.join(&filename))
            .unwrap();
        container.model_ref(filename)
    }

    fn extract_ready(update: RuntimeUpdate) -> Box<PreparedRuntime> {
        match update {
            RuntimeUpdate::Ready { runtime, .. } => runtime,
            RuntimeUpdate::Mute { reason, .. } => panic!("unexpected mute: {reason:?}"),
        }
    }

    fn write_ir(path: &Path, samples: &[f32], sample_rate: u32) -> IrReference {
        write_mono_f32_wav(path, sample_rate, samples).unwrap();
        let bytes = fs::read(path).unwrap();
        IrReference {
            ir_id: "test-ir".to_owned(),
            sha256: sha256(&bytes),
            filename_hint: path.file_name().unwrap().to_string_lossy().into_owned(),
            processing: IrProcessingMode::MinimumPhaseAutoTrim,
        }
    }

    #[test]
    fn exact_model_payload_builds_a_zero_latency_runtime() {
        let (_directory, library, loader) = setup();
        let reference = install_model(&library, "identity", &CompactCausalModel::raw());
        let outcome = loader.load(RuntimeLoadRequest::new(7, reference.clone()));
        assert!(outcome.status.is_ready());
        let mut runtime = extract_ready(outcome.update);
        assert_eq!(runtime.model_reference(), Some(&reference));
        assert_eq!(runtime.latency_samples(), 0);
        assert_eq!(runtime.tail_samples(), 0);

        let input = [0.0, -0.5, 0.25, 1.0, -1.0];
        let mut scratch = [0.0; 5];
        let mut output = [f32::NAN; 5];
        runtime.process_block(&input, &mut scratch, &mut output);
        assert_eq!(output, input);
    }

    #[test]
    fn transparent_runtime_is_bit_exact_with_no_model_or_ir() {
        let mut runtime = PreparedRuntime::transparent();
        assert_eq!(runtime.model_reference(), None);
        assert_eq!(runtime.ir_reference(), None);
        assert_eq!(runtime.latency_samples(), 0);
        assert_eq!(runtime.tail_samples(), 0);

        let input = [0.0, -1.0, 0.125, 0.75, 1.0];
        let mut scratch = [f32::NAN; 5];
        let mut output = [f32::NAN; 5];
        runtime.process_block(&input, &mut scratch, &mut output);
        assert_eq!(output, input);
    }

    #[test]
    fn outer_and_payload_metadata_must_match() {
        let (_directory, library, loader) = setup();
        let model = CompactCausalModel::raw();
        let mut metadata = metadata_for(&model, "mismatch");
        metadata.estimated_macs_per_sample += 1;
        let container = MotModel::new(metadata, encode_model_payload(&model)).unwrap();
        container
            .write_new(library.paths().models.join("mismatch.motmodel"))
            .unwrap();
        let outcome = loader.load(RuntimeLoadRequest::new(
            9,
            container.model_ref("mismatch.motmodel"),
        ));
        assert!(matches!(
            outcome.status,
            RuntimeLoadStatus::Corrupt {
                asset: RuntimeAsset::Model,
                ..
            }
        ));
        assert!(matches!(
            outcome.update,
            RuntimeUpdate::Mute {
                reason: RuntimeMuteReason::CorruptModel,
                ..
            }
        ));
    }

    #[test]
    fn library_tone_loads_by_default_and_explicit_daw_tone_wins() {
        let (_directory, library, loader) = setup();
        let reference = install_model(&library, "identity", &CompactCausalModel::raw());
        let mut saved = ToneSettings::defaults_for(&reference);
        saved.input_gain_db = 6.0;
        saved.tight_percent = 50.0;
        saved.bite_percent = 25.0;
        library.save_tone(&saved).unwrap();

        let saved_runtime = extract_ready(
            loader
                .load(RuntimeLoadRequest::new(1, reference.clone()))
                .update,
        );
        let saved_controls = saved_runtime.amp().target_controls();
        assert!((saved_controls.input_gain_db - 6.0).abs() < 1.0e-4);
        assert!((saved_controls.tight - 0.5).abs() < 1.0e-6);
        assert!((saved_controls.bite - 0.25).abs() < 1.0e-6);

        let mut explicit = ToneSettings::defaults_for(&reference);
        explicit.input_gain_db = -3.5;
        explicit.tight_percent = 10.0;
        explicit.bite_percent = 80.0;
        let mut request = RuntimeLoadRequest::new(2, reference);
        request.tone = Some(explicit);
        let project_runtime = extract_ready(loader.load(request).update);
        let project_controls = project_runtime.amp().target_controls();
        assert!((project_controls.input_gain_db + 3.5).abs() < 1.0e-4);
        assert!((project_controls.tight - 0.1).abs() < 1.0e-6);
        assert!((project_controls.bite - 0.8).abs() < 1.0e-6);
    }

    #[test]
    fn missing_and_hash_mismatched_models_request_safe_mute() {
        let (_directory, library, loader) = setup();
        let existing = install_model(&library, "identity", &CompactCausalModel::raw());
        let missing = ModelRef {
            model_id: "missing".to_owned(),
            sha256: sha256(b"missing"),
            filename_hint: "missing.motmodel".to_owned(),
        };
        let missing_outcome = loader.load(RuntimeLoadRequest::new(1, missing));
        assert!(matches!(
            missing_outcome.status,
            RuntimeLoadStatus::Missing {
                asset: RuntimeAsset::Model,
                ..
            }
        ));

        let wrong_hash = ModelRef {
            model_id: existing.model_id,
            sha256: sha256(b"wrong"),
            filename_hint: existing.filename_hint,
        };
        let corrupt_outcome = loader.load(RuntimeLoadRequest::new(2, wrong_hash));
        assert!(matches!(
            corrupt_outcome.status,
            RuntimeLoadStatus::Corrupt {
                asset: RuntimeAsset::Model,
                ..
            }
        ));
    }

    #[test]
    fn minimum_phase_auto_trim_ir_preserves_sample_zero_onset() {
        let (directory, library, loader) = setup();
        let reference = install_model(&library, "identity", &CompactCausalModel::raw());
        let ir_path = directory.0.join("cab.wav");
        let ir_reference = write_ir(&ir_path, &[0.0, 0.0, 1.0, 0.5], REQUIRED_SAMPLE_RATE_HZ);
        let tone = ToneSettings {
            schema_version: TONE_SETTINGS_VERSION,
            model_id: reference.model_id.clone(),
            model_sha256: reference.sha256,
            input_gain_db: 0.0,
            tight_percent: 0.0,
            bite_percent: 0.0,
            ir: Some(ir_reference.clone()),
        };
        let mut request = RuntimeLoadRequest::new(3, reference);
        request.tone = Some(tone);
        request.ir_path = Some(ir_path);
        let outcome = loader.load(request);
        assert!(outcome.status.is_ready());
        let mut runtime = extract_ready(outcome.update);
        assert_eq!(runtime.ir_reference(), Some(&ir_reference));
        assert_eq!(runtime.latency_samples(), 0);

        let input = [1.0, 0.0, 0.0, 0.0];
        let mut scratch = [0.0; 4];
        let mut output = [0.0; 4];
        runtime.process_block(&input, &mut scratch, &mut output);
        assert!(output[0].abs() > 0.1);
    }

    #[test]
    fn missing_wrong_hash_wrong_rate_and_oversize_irs_are_rejected() {
        let (directory, library, loader) = setup();
        let reference = install_model(&library, "identity", &CompactCausalModel::raw());

        let make_request = |ir_reference: IrReference, path: PathBuf, generation: u64| {
            let tone = ToneSettings {
                schema_version: TONE_SETTINGS_VERSION,
                model_id: reference.model_id.clone(),
                model_sha256: reference.sha256,
                input_gain_db: 0.0,
                tight_percent: 0.0,
                bite_percent: 0.0,
                ir: Some(ir_reference),
            };
            let mut request = RuntimeLoadRequest::new(generation, reference.clone());
            request.tone = Some(tone);
            request.ir_path = Some(path);
            request
        };

        let missing_path = directory.0.join("missing.wav");
        let missing_reference = IrReference {
            ir_id: "missing-ir".to_owned(),
            sha256: sha256(b"missing"),
            filename_hint: "missing.wav".to_owned(),
            processing: IrProcessingMode::Raw,
        };
        assert!(matches!(
            loader
                .load(make_request(missing_reference, missing_path, 1))
                .status,
            RuntimeLoadStatus::Missing {
                asset: RuntimeAsset::CabinetIr,
                ..
            }
        ));

        let wrong_hash_path = directory.0.join("wrong-hash.wav");
        let persisted_reference = write_ir(&wrong_hash_path, &[1.0], REQUIRED_SAMPLE_RATE_HZ);
        // Simulate a library file being replaced after its exact identity was
        // stored in project state. The new bytes must never become the new
        // expected digest merely because the path stayed the same.
        write_mono_f32_wav(&wrong_hash_path, REQUIRED_SAMPLE_RATE_HZ, &[0.5]).unwrap();
        let replaced = loader.load(make_request(persisted_reference, wrong_hash_path, 2));
        assert!(matches!(
            replaced.status,
            RuntimeLoadStatus::Corrupt {
                asset: RuntimeAsset::CabinetIr,
                ..
            }
        ));
        assert!(matches!(
            replaced.update,
            RuntimeUpdate::Mute {
                reason: RuntimeMuteReason::CorruptCabinetIr,
                ..
            }
        ));

        let wrong_rate_path = directory.0.join("wrong-rate.wav");
        let wrong_rate_reference = write_ir(&wrong_rate_path, &[1.0], 44_100);
        assert!(matches!(
            loader
                .load(make_request(wrong_rate_reference, wrong_rate_path, 3))
                .status,
            RuntimeLoadStatus::Corrupt {
                asset: RuntimeAsset::CabinetIr,
                ..
            }
        ));

        let oversize_path = directory.0.join("oversize.wav");
        let oversize_reference =
            write_ir(&oversize_path, &vec![1.0; 8_193], REQUIRED_SAMPLE_RATE_HZ);
        assert!(matches!(
            loader
                .load(make_request(oversize_reference, oversize_path, 4))
                .status,
            RuntimeLoadStatus::Corrupt {
                asset: RuntimeAsset::CabinetIr,
                ..
            }
        ));
    }

    #[test]
    fn mailbox_keeps_only_latest_and_returns_retirement_to_loader() {
        let (_directory, library, loader) = setup();
        let reference = install_model(&library, "identity", &CompactCausalModel::raw());
        let mailbox = RuntimeMailbox::new(2);
        loader.load_and_publish(RuntimeLoadRequest::new(1, reference.clone()), &mailbox);
        loader.load_and_publish(RuntimeLoadRequest::new(2, reference), &mailbox);
        assert_eq!(mailbox.pending_len(), 1);

        let update = mailbox.take_latest().expect("latest update");
        assert_eq!(update.generation(), 2);
        let runtime = extract_ready(update);
        mailbox.try_retire(runtime).expect("retirement capacity");
        assert_eq!(mailbox.retired_len(), 1);
        assert_eq!(mailbox.drain_retired(), 1);
        assert_eq!(mailbox.retired_len(), 0);
    }

    #[test]
    fn retirement_backpressure_returns_ownership_to_audio_caller() {
        let (_directory, library, loader) = setup();
        let reference = install_model(&library, "identity", &CompactCausalModel::raw());
        let first = extract_ready(
            loader
                .load(RuntimeLoadRequest::new(1, reference.clone()))
                .update,
        );
        let second = extract_ready(loader.load(RuntimeLoadRequest::new(2, reference)).update);
        let mailbox = RuntimeMailbox::new(1);
        mailbox.try_retire(first).expect("first retirement");
        let returned = mailbox
            .try_retire(second)
            .expect_err("queue must return ownership");
        assert_eq!(returned.latency_samples(), 0);
        // Destruction happens after leaving the simulated audio operation.
        drop(returned);
        assert_eq!(mailbox.drain_retired(), 1);
    }

    #[cfg(feature = "rt-paranoid")]
    #[test]
    fn audio_side_mailbox_operations_allocate_nothing() {
        let (_directory, library, loader) = setup();
        let reference = install_model(&library, "identity", &CompactCausalModel::raw());
        let mailbox = RuntimeMailbox::new(2);
        loader.load_and_publish(RuntimeLoadRequest::new(1, reference), &mailbox);

        let (update, take_allocations) = truce::rt::audit(|| {
            let _section = truce::rt::RtSection::enter();
            mailbox.take_latest()
        });
        assert_eq!(take_allocations, 0);
        let runtime = extract_ready(update.expect("runtime"));

        let (retirement, retire_allocations) = truce::rt::audit(|| {
            let _section = truce::rt::RtSection::enter();
            mailbox.try_retire(runtime)
        });
        assert_eq!(retire_allocations, 0);
        assert!(retirement.is_ok());
        assert_eq!(mailbox.drain_retired(), 1);
    }
}
