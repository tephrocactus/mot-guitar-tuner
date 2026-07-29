//! Off-thread preparation and training glue for the two-instance capture lab.
//!
//! The types in [`crate::capture`] contain the real-time state machine. This
//! module owns all disk I/O, WAV decoding, alignment, training, and immutable
//! model publication. Audio communicates with it through bounded queues and
//! atomics only.

use std::fs;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::atomic::{AtomicU8, AtomicU32, Ordering};
use std::sync::{Arc, OnceLock, RwLock};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use crossbeam_queue::ArrayQueue;
use truce::prelude::BackgroundTask;

use crate::MotStrobeParams;
use crate::capture::{
    AlignmentConfig, CAPTURE_SAMPLE_RATE_HZ, CaptureBinding, CaptureCoordinator, CaptureEngine,
    CaptureProgram, CaptureRole, CaptureSessionId, CaptureTarget, CompletedReturn,
    HardwareCaptureMetadata, extract_aligned_excitation, measure_alignment,
};
use crate::model::{ModelMetadata, ModelRef, MotModel, sha256};
use crate::model_library::ModelLibrary;
use crate::trainer::{
    CancellationToken, TrainerConfig, TrainingData, TrainingStopReason, encode_model_payload,
    model_descriptor, train_compact_model,
};
#[cfg(test)]
use crate::wav_io::read_mono_wav;
use crate::wav_io::{decode_mono_wav, write_mono_f32_wav};

pub const CAPTURE_ASSET_RELATIVE_PATH: &str = "Capture Assets/input.wav";
pub const CAPTURE_ASSET_SHA256: &str =
    "70f8ec7f25686a1bd77f25973de8e51a6721e957e81eec121822e5e53366bc41";

const READY_CAPACITY: usize = 2;
const RETIRED_CAPACITY: usize = 8;
const RECYCLED_CAPACITY: usize = 2;
const SYNC_HEADER_SAMPLES: usize = 4_096;
const CAPTURE_COORDINATOR_CAPACITY: usize = 16;

static COORDINATOR: OnceLock<Arc<CaptureCoordinator>> = OnceLock::new();
// Only successful asset loads are cached. A missing asset can therefore be
// installed while the DAW remains open and retried without restarting it.
static CAPTURE_PROGRAM: OnceLock<Arc<CaptureProgram>> = OnceLock::new();

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum CaptureWorkerStatus {
    #[default]
    Idle = 0,
    Preparing = 1,
    Ready = 2,
    Capturing = 3,
    Aligning = 4,
    Training = 5,
    ModelSaved = 6,
    Error = 7,
}

impl CaptureWorkerStatus {
    #[must_use]
    pub fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::Preparing,
            2 => Self::Ready,
            3 => Self::Capturing,
            4 => Self::Aligning,
            5 => Self::Training,
            6 => Self::ModelSaved,
            7 => Self::Error,
            _ => Self::Idle,
        }
    }
}

/// A fully allocated capture instance. Moving it across a queue is constant
/// time; construction, WAV loading, and coordinator binding happen off-thread.
#[derive(Debug)]
pub struct PreparedCaptureRuntime {
    pub request_generation: u64,
    pub role: CaptureRole,
    pub session_id: CaptureSessionId,
    pub engine: CaptureEngine,
    pub binding: Option<CaptureBinding>,
    pub program: Option<Arc<CaptureProgram>>,
}

impl PreparedCaptureRuntime {
    #[must_use]
    pub fn normal(request_generation: u64, session_id: CaptureSessionId) -> Self {
        Self {
            request_generation,
            role: CaptureRole::Normal,
            session_id,
            engine: CaptureEngine::new(CaptureRole::Normal, session_id),
            binding: None,
            program: None,
        }
    }
}

/// Shared lock-free handoff owned by one plugin instance.
#[derive(Debug)]
pub struct CaptureControl {
    ready: ArrayQueue<Box<PreparedCaptureRuntime>>,
    retired: ArrayQueue<Box<PreparedCaptureRuntime>>,
    recycled: ArrayQueue<CompletedReturn>,
    status: AtomicU8,
    progress_bits: AtomicU32,
    last_error: RwLock<String>,
    last_saved_model: RwLock<Option<ModelRef>>,
    cancellation: CancellationToken,
}

impl Default for CaptureControl {
    fn default() -> Self {
        Self {
            ready: ArrayQueue::new(READY_CAPACITY),
            retired: ArrayQueue::new(RETIRED_CAPACITY),
            recycled: ArrayQueue::new(RECYCLED_CAPACITY),
            status: AtomicU8::new(CaptureWorkerStatus::Idle as u8),
            progress_bits: AtomicU32::new(0.0_f32.to_bits()),
            last_error: RwLock::new(String::new()),
            last_saved_model: RwLock::new(None),
            cancellation: CancellationToken::default(),
        }
    }
}

impl CaptureControl {
    /// Worker-side latest-wins publication. A displaced runtime is destroyed
    /// on the worker thread, never in the audio callback.
    pub fn publish_runtime(&self, runtime: Box<PreparedCaptureRuntime>) {
        let _ = self.ready.force_push(runtime);
    }

    /// Audio-side non-blocking receive.
    #[must_use]
    pub fn take_runtime(&self) -> Option<Box<PreparedCaptureRuntime>> {
        self.ready.pop()
    }

    /// Audio-side retirement. On the exceptionally unlikely full-queue path,
    /// ownership is returned so the DSP state can keep it until a later block.
    pub fn retire_runtime(
        &self,
        runtime: Box<PreparedCaptureRuntime>,
    ) -> Result<(), Box<PreparedCaptureRuntime>> {
        self.retired.push(runtime)
    }

    /// Worker/UI-side cleanup of audio-retired runtimes.
    pub fn drain_retired(&self) {
        while self.retired.pop().is_some() {}
    }

    /// Worker-side return of the preallocated capture storage.
    fn publish_recycled(&self, mut completed: CompletedReturn) {
        loop {
            match self.recycled.push(completed) {
                Ok(()) => break,
                Err(returned) => {
                    completed = returned;
                    thread::yield_now();
                }
            }
        }
    }

    /// Audio-side receive of a storage buffer after training.
    #[must_use]
    pub fn take_recycled(&self) -> Option<CompletedReturn> {
        self.recycled.pop()
    }

    pub fn set_status(&self, status: CaptureWorkerStatus) {
        self.status.store(status as u8, Ordering::Release);
    }

    #[must_use]
    pub fn status(&self) -> CaptureWorkerStatus {
        CaptureWorkerStatus::from_u8(self.status.load(Ordering::Acquire))
    }

    pub fn set_progress(&self, progress: f32) {
        self.progress_bits
            .store(progress.clamp(0.0, 1.0).to_bits(), Ordering::Release);
    }

    #[must_use]
    pub fn progress(&self) -> f32 {
        f32::from_bits(self.progress_bits.load(Ordering::Acquire)).clamp(0.0, 1.0)
    }

    pub fn set_error(&self, message: impl Into<String>) {
        if let Ok(mut error) = self.last_error.write() {
            *error = message.into();
        }
        self.set_status(CaptureWorkerStatus::Error);
    }

    #[must_use]
    pub fn last_error(&self) -> String {
        self.last_error.read().map_or_else(
            |_| "capture error lock poisoned".to_owned(),
            |error| error.clone(),
        )
    }

    #[must_use]
    pub fn last_saved_model(&self) -> Option<ModelRef> {
        self.last_saved_model
            .read()
            .ok()
            .and_then(|model| model.clone())
    }

    pub fn cancel_training(&self) {
        self.cancellation.cancel();
    }

    /// Clears a previous cancellation before ownership of a new completed
    /// capture is handed to the background task. Call this exactly once when
    /// the Return creates the task, not inside the worker thread: otherwise a
    /// user cancellation issued while the task is queued could be lost.
    pub fn prepare_training(&self) {
        self.cancellation.reset();
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PrepareCaptureTask {
    pub request_generation: u64,
    pub role: u8,
    pub session_id: u64,
    pub instance_id: u64,
}

impl BackgroundTask for PrepareCaptureTask {
    type Params = MotStrobeParams;
    const SERIALIZED: bool = true;

    fn run(self, params: &Self::Params) {
        let control = &params.capture_control;
        control.drain_retired();
        control.set_status(CaptureWorkerStatus::Preparing);

        let role = match self.role {
            1 => CaptureRole::Source,
            2 => CaptureRole::Return,
            _ => CaptureRole::Normal,
        };
        let session_id = CaptureSessionId::new(self.session_id)
            .or_else(|| CaptureSessionId::new(1))
            .expect("one is a valid capture session");
        if role == CaptureRole::Normal {
            control.publish_runtime(Box::new(PreparedCaptureRuntime::normal(
                self.request_generation,
                session_id,
            )));
            control.set_status(CaptureWorkerStatus::Ready);
            return;
        }

        let prepared = (|| -> Result<PreparedCaptureRuntime, String> {
            let program = load_default_capture_program()?;
            let mut engine = CaptureEngine::new(role, session_id);
            engine
                .prepare(Arc::clone(&program), 48_000)
                .map_err(|error| error.to_string())?;
            let binding = coordinator()
                .bind(session_id, role, self.instance_id)
                .map_err(|error| error.to_string())?;
            Ok(PreparedCaptureRuntime {
                request_generation: self.request_generation,
                role,
                session_id,
                engine,
                binding: Some(binding),
                program: Some(program),
            })
        })();
        match prepared {
            Ok(runtime) => {
                control.publish_runtime(Box::new(runtime));
                control.set_status(CaptureWorkerStatus::Ready);
            }
            Err(error) => control.set_error(error),
        }
    }
}

/// Owned, zero-copy Return handoff scheduled by the audio callback.
#[derive(Debug)]
pub struct StartTrainingTask {
    pub completed: CompletedReturn,
    pub program: Arc<CaptureProgram>,
    pub source_send_trim_db: f32,
    pub target: CaptureTarget,
}

impl BackgroundTask for StartTrainingTask {
    type Params = MotStrobeParams;

    fn run(self, params: &Self::Params) {
        let control = Arc::clone(&params.capture_control);
        let max_passes = params.max_passes.value_i32().clamp(1, 400) as u16;
        let display_name = params
            .capture_model_name
            .read()
            .map_or_else(|_| "Captured Amp".to_owned(), |name| name.clone());
        let mut capture_metadata = HardwareCaptureMetadata::uncalibrated_full_amp();
        capture_metadata.target = self.target;
        capture_metadata.amplifier = crate::read_shared_string(&params.capture_amplifier);
        capture_metadata.amplifier_channel =
            crate::read_shared_string(&params.capture_amplifier_channel);
        capture_metadata.control_positions =
            crate::read_shared_string(&params.capture_control_positions);
        capture_metadata.interface_output =
            crate::read_shared_string(&params.capture_interface_output);
        capture_metadata.interface_input =
            crate::read_shared_string(&params.capture_interface_input);
        capture_metadata.reamp_box = crate::read_shared_string(&params.capture_reamp_box);
        capture_metadata.reactive_load = crate::read_shared_string(&params.capture_reactive_load);
        capture_metadata.load_impedance_ohms =
            u16::try_from(params.capture_load_impedance_ohms.load())
                .ok()
                .filter(|impedance| *impedance > 0);
        capture_metadata.return_gain_note =
            crate::read_shared_string(&params.capture_return_gain_note);
        let thread_name = format!("mot-capture-trainer-{}", self.completed.generation);
        let spawn_result = thread::Builder::new().name(thread_name).spawn(move || {
            run_training_job(
                &control,
                self.completed,
                self.program,
                self.source_send_trim_db,
                max_passes,
                &display_name,
                capture_metadata,
            );
        });
        if let Err(error) = spawn_result {
            params
                .capture_control
                .set_error(format!("cannot start trainer thread: {error}"));
        }
    }
}

fn run_training_job(
    control: &CaptureControl,
    completed: CompletedReturn,
    program: Arc<CaptureProgram>,
    source_send_trim_db: f32,
    max_passes: u16,
    display_name: &str,
    mut capture_metadata: HardwareCaptureMetadata,
) {
    control.set_progress(0.0);
    control.set_status(CaptureWorkerStatus::Aligning);

    let result = (|| -> Result<Option<ModelRef>, String> {
        let library = ModelLibrary::for_current_user().map_err(|error| error.to_string())?;
        library
            .ensure_directories()
            .map_err(|error| error.to_string())?;
        let paths = library.paths();
        let (model_id, timestamp) =
            unique_model_id(completed.session_id.get(), completed.generation);
        let capture_dir = paths.plugin_root.join("Capture Records").join(&model_id);
        fs::create_dir_all(&capture_dir).map_err(|error| error.to_string())?;
        // Preserve the exact preallocated Return before correlation,
        // fractional alignment, or any trainer-side preparation.
        write_raw_return(&capture_dir, completed.audio())?;

        let alignment = measure_alignment(&program, completed.audio(), AlignmentConfig::default())
            .map_err(|error| error.to_string())?;
        let target = extract_aligned_excitation(&program, completed.audio(), alignment)
            .map_err(|error| error.to_string())?;
        let emitted = emitted_training_input(&program, source_send_trim_db);
        write_mono_f32_wav(&capture_dir.join("emitted.wav"), 48_000, &emitted)
            .map_err(|error| error.to_string())?;
        write_mono_f32_wav(&capture_dir.join("aligned-return.wav"), 48_000, &target)
            .map_err(|error| error.to_string())?;

        control.set_status(CaptureWorkerStatus::Training);
        let config = TrainerConfig {
            max_passes,
            ..TrainerConfig::default()
        };
        let outcome = train_compact_model(
            TrainingData {
                input: &emitted,
                target: &target,
                sample_rate_hz: 48_000,
            },
            config,
            &control.cancellation,
            |progress| {
                control.set_progress(
                    f32::from(progress.completed_passes)
                        / f32::from(progress.maximum_passes.max(1)),
                );
            },
        )
        .map_err(|error| error.to_string())?;
        if outcome.stop_reason == TrainingStopReason::Cancelled {
            return Ok(None);
        }
        let descriptor = model_descriptor(&outcome.best_model);
        let metadata = ModelMetadata {
            model_id: model_id.clone(),
            display_name: normalized_display_name(display_name, &model_id),
            architecture_id: descriptor.architecture_id.to_owned(),
            architecture_version: descriptor.architecture_version,
            sample_rate_hz: descriptor.sample_rate_hz,
            causal: descriptor.causal,
            lookahead_samples: descriptor.lookahead_samples,
            runtime_latency_samples: descriptor.runtime_latency_samples,
            estimated_macs_per_sample: u64::from(descriptor.estimated_macs_per_sample),
        };
        let model = MotModel::new(metadata, encode_model_payload(&outcome.best_model))
            .map_err(|error| error.to_string())?;
        let filename = format!("{model_id}.motmodel");
        model
            .write_new(paths.models.join(&filename))
            .map_err(|error| error.to_string())?;

        capture_metadata.source_send_trim_db = source_send_trim_db;
        capture_metadata.measured_latency_samples = Some(alignment.fractional_latency_samples);
        capture_metadata.return_peak_dbfs =
            Some(crate::capture::linear_to_dbfs(completed.peak_linear));
        capture_metadata.return_rms_dbfs = Some(completed.rms_dbfs);
        let mut excitation_bytes = Vec::with_capacity(std::mem::size_of_val(program.excitation()));
        for sample in program.excitation() {
            excitation_bytes.extend_from_slice(&sample.to_le_bytes());
        }
        capture_metadata.excitation_hash = sha256(&excitation_bytes).to_string();

        let metadata_json = capture_record_json(
            &model_id,
            timestamp,
            alignment.integer_latency_samples,
            alignment.normalized_correlation,
            outcome.completed_passes,
            outcome.best_pass,
            outcome.best_validation_loss,
            outcome.stop_reason,
            &capture_metadata,
        );
        fs::write(capture_dir.join("capture.json"), metadata_json)
            .map_err(|error| error.to_string())?;
        Ok(Some(model.model_ref(filename)))
    })();

    match result {
        Ok(Some(model_ref)) => {
            if let Ok(mut saved) = control.last_saved_model.write() {
                *saved = Some(model_ref);
            }
            control.set_progress(1.0);
            control.set_status(CaptureWorkerStatus::ModelSaved);
        }
        Ok(None) => {
            control.set_progress(0.0);
            control.set_status(CaptureWorkerStatus::Ready);
        }
        Err(error) => control.set_error(error),
    }
    control.publish_recycled(completed);
}

fn write_raw_return(capture_dir: &Path, return_audio: &[f32]) -> Result<(), String> {
    write_mono_f32_wav(
        &capture_dir.join("raw-return.wav"),
        CAPTURE_SAMPLE_RATE_HZ,
        return_audio,
    )
    .map_err(|error| error.to_string())
}

fn emitted_training_input(program: &CaptureProgram, source_send_trim_db: f32) -> Vec<f32> {
    let input_gain = 10.0_f32.powf(source_send_trim_db.clamp(-40.0, 0.0) / 20.0);
    program
        .excitation()
        .iter()
        .map(|sample| sample * input_gain)
        .collect()
}

fn coordinator() -> &'static Arc<CaptureCoordinator> {
    COORDINATOR.get_or_init(|| CaptureCoordinator::with_capacity(CAPTURE_COORDINATOR_CAPACITY))
}

pub fn capture_asset_path() -> Result<PathBuf, String> {
    let library = ModelLibrary::for_current_user().map_err(|error| error.to_string())?;
    Ok(library
        .paths()
        .plugin_root
        .join(CAPTURE_ASSET_RELATIVE_PATH))
}

pub fn load_default_capture_program() -> Result<Arc<CaptureProgram>, String> {
    if let Some(program) = CAPTURE_PROGRAM.get() {
        return Ok(Arc::clone(program));
    }
    let path = capture_asset_path()?;
    let program = load_capture_program(&path)?;
    let _ = CAPTURE_PROGRAM.set(Arc::clone(&program));
    Ok(CAPTURE_PROGRAM.get().map_or(program, Arc::clone))
}

pub fn load_capture_program(path: &Path) -> Result<Arc<CaptureProgram>, String> {
    let bytes = fs::read(path)
        .map_err(|error| format!("cannot read capture asset {}: {error}", path.display()))?;
    let expected =
        crate::model::Sha256Digest::from_str(CAPTURE_ASSET_SHA256).map_err(|e| e.to_string())?;
    let actual = sha256(&bytes);
    if actual != expected {
        return Err(format!(
            "capture asset SHA-256 mismatch: expected {expected}, found {actual}"
        ));
    }
    // Hash and decode the same immutable byte snapshot. Reading the path a
    // second time would allow a replacement between verification and use.
    let wav = decode_mono_wav(&bytes).map_err(|error| error.to_string())?;
    if wav.sample_rate != 48_000 {
        return Err(format!(
            "capture asset must be 48000 Hz, found {} Hz",
            wav.sample_rate
        ));
    }
    if wav.samples.len() != 9_120_000 {
        return Err(format!(
            "capture asset must contain 9,120,000 samples, found {}",
            wav.samples.len()
        ));
    }
    let sync_header: Arc<[f32]> = generate_sync_header().into();
    let excitation: Arc<[f32]> = wav.samples.into();
    CaptureProgram::new(sync_header, excitation)
        .map(Arc::new)
        .map_err(|error| error.to_string())
}

fn generate_sync_header() -> Vec<f32> {
    let mut state = 0x00c0_ffee_u32;
    let mut header = Vec::with_capacity(SYNC_HEADER_SAMPLES);
    for index in 0..SYNC_HEADER_SAMPLES {
        // xorshift32 is deterministic and has a nearly white bipolar output.
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let sign = if state & 1 == 0 { -1.0 } else { 1.0 };
        let edge = index.min(SYNC_HEADER_SAMPLES - 1 - index).min(127) as f32 / 127.0;
        header.push(sign * 0.18 * edge);
    }
    header
}

fn unique_model_id(session_id: u64, generation: u64) -> (String, u64) {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs());
    (
        format!("capture-{timestamp}-{session_id}-{generation}"),
        timestamp,
    )
}

fn normalized_display_name(requested: &str, fallback: &str) -> String {
    let trimmed = requested.trim();
    if trimmed.is_empty() {
        fallback.to_owned()
    } else {
        trimmed.chars().take(96).collect()
    }
}

#[allow(clippy::too_many_arguments)]
fn capture_record_json(
    model_id: &str,
    timestamp: u64,
    integer_latency_samples: i64,
    correlation: f64,
    completed_passes: u16,
    best_pass: u16,
    best_validation_loss: f64,
    stop_reason: TrainingStopReason,
    metadata: &HardwareCaptureMetadata,
) -> String {
    let target_name = match metadata.target {
        CaptureTarget::SoftwarePluginChain => "software_plugin_chain",
        CaptureTarget::FullAmpUnfilteredLoad => "full_amp_unfiltered_load",
    };
    let stop_reason_name = match stop_reason {
        TrainingStopReason::MaximumPasses => "maximum_passes",
        TrainingStopReason::EarlyStopping => "early_stopping",
        TrainingStopReason::Cancelled => "cancelled",
    };
    let load_impedance = metadata
        .load_impedance_ohms
        .map_or_else(|| "null".to_owned(), |impedance| impedance.to_string());
    format!(
        concat!(
            "{{\n",
            "  \"schema_version\": 1,\n",
            "  \"model_id\": \"{}\",\n",
            "  \"created_unix_seconds\": {},\n",
            "  \"target\": \"{}\",\n",
            "  \"calibration\": {{\n",
            "    \"status\": \"uncalibrated\",\n",
            "    \"input_level_dbu\": null,\n",
            "    \"output_level_dbu\": null\n",
            "  }},\n",
            "  \"sample_rate_hz\": 48000,\n",
            "  \"source_send_trim_db\": {:.3},\n",
            "  \"integer_latency_samples\": {},\n",
            "  \"fractional_latency_samples\": {:.8},\n",
            "  \"sync_correlation\": {:.8},\n",
            "  \"return_peak_dbfs\": {:.3},\n",
            "  \"return_rms_dbfs\": {:.3},\n",
            "  \"excitation_sha256\": \"{}\",\n",
            "  \"completed_passes\": {},\n",
            "  \"best_pass\": {},\n",
            "  \"best_validation_loss\": {:.12},\n",
            "  \"training_stop_reason\": \"{}\",\n",
            "  \"amplifier\": \"{}\",\n",
            "  \"amplifier_channel\": \"{}\",\n",
            "  \"control_positions\": \"{}\",\n",
            "  \"interface_output\": \"{}\",\n",
            "  \"interface_input\": \"{}\",\n",
            "  \"reamp_box\": \"{}\",\n",
            "  \"reactive_load\": \"{}\",\n",
            "  \"load_impedance_ohms\": {},\n",
            "  \"return_gain_note\": \"{}\"\n",
            "}}\n"
        ),
        escape_json(model_id),
        timestamp,
        target_name,
        metadata.source_send_trim_db,
        integer_latency_samples,
        metadata.measured_latency_samples.unwrap_or(0.0),
        correlation,
        metadata.return_peak_dbfs.unwrap_or(f32::NEG_INFINITY),
        metadata.return_rms_dbfs.unwrap_or(f32::NEG_INFINITY),
        metadata.excitation_hash,
        completed_passes,
        best_pass,
        best_validation_loss,
        stop_reason_name,
        escape_json(&metadata.amplifier),
        escape_json(&metadata.amplifier_channel),
        escape_json(&metadata.control_positions),
        escape_json(&metadata.interface_output),
        escape_json(&metadata.interface_input),
        escape_json(&metadata.reamp_box),
        escape_json(&metadata.reactive_load),
        load_impedance,
        escape_json(&metadata.return_gain_note),
    )
}

fn escape_json(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            character if character.is_control() => {
                use std::fmt::Write;
                let _ = write!(escaped, "\\u{:04x}", u32::from(character));
            }
            character => escaped.push(character),
        }
    }
    escaped
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sync_header_is_deterministic_and_faded() {
        let left = generate_sync_header();
        let right = generate_sync_header();
        assert_eq!(left, right);
        assert_eq!(left.len(), SYNC_HEADER_SAMPLES);
        assert_eq!(left[0], 0.0);
        assert_eq!(left[SYNC_HEADER_SAMPLES - 1], 0.0);
        assert!(left.iter().all(|sample| sample.abs() <= 0.18));
    }

    #[test]
    fn worker_status_round_trips() {
        for status in [
            CaptureWorkerStatus::Idle,
            CaptureWorkerStatus::Preparing,
            CaptureWorkerStatus::Ready,
            CaptureWorkerStatus::Capturing,
            CaptureWorkerStatus::Aligning,
            CaptureWorkerStatus::Training,
            CaptureWorkerStatus::ModelSaved,
            CaptureWorkerStatus::Error,
        ] {
            assert_eq!(CaptureWorkerStatus::from_u8(status as u8), status);
        }
    }

    #[test]
    fn capture_control_progress_is_finite_and_clamped() {
        let control = CaptureControl::default();
        control.set_progress(1.5);
        assert_eq!(control.progress(), 1.0);
        control.set_progress(-1.0);
        assert_eq!(control.progress(), 0.0);
    }

    #[test]
    fn raw_return_is_exact_and_saving_it_does_not_change_trainer_input() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let capture_dir = std::env::temp_dir().join(format!("mot-capture-raw-return-{nonce}"));
        fs::create_dir_all(&capture_dir).unwrap();

        let program = CaptureProgram::new(vec![0.1, -0.1], vec![0.25, -0.5, 0.75, -1.0]).unwrap();
        let emitted = emitted_training_input(&program, -6.0);
        let expected_emitted = emitted.clone();
        let exact_return = [0.0, -0.125, 0.333_333_34, 0.875, -0.75];

        write_raw_return(&capture_dir, &exact_return).unwrap();

        // Raw persistence only borrows the completed Return. The trainer input
        // remains the exact Source excitation after its one recorded trim.
        assert_eq!(emitted, expected_emitted);
        let decoded = read_mono_wav(&capture_dir.join("raw-return.wav")).unwrap();
        assert_eq!(decoded.sample_rate, CAPTURE_SAMPLE_RATE_HZ);
        assert_eq!(decoded.samples, exact_return);
        let _ = fs::remove_dir_all(capture_dir);
    }

    #[test]
    fn json_escape_handles_control_characters() {
        assert_eq!(escape_json("a\"b\\c\n"), "a\\\"b\\\\c\\n");
    }

    #[test]
    fn capture_record_preserves_hardware_metadata() {
        let mut metadata = HardwareCaptureMetadata::uncalibrated_full_amp();
        metadata.source_send_trim_db = -18.4;
        metadata.measured_latency_samples = Some(137.25);
        metadata.return_peak_dbfs = Some(-3.2);
        metadata.return_rms_dbfs = Some(-19.7);
        metadata.excitation_hash = "abc123".to_owned();
        metadata.amplifier = "EVH 5153 \"Blue\"".to_owned();
        metadata.amplifier_channel = "Blue".to_owned();
        metadata.control_positions = "Gain 5 / Bass 4".to_owned();
        metadata.interface_output = "Line Out 3".to_owned();
        metadata.interface_input = "Input 1".to_owned();
        metadata.reamp_box = "Reamp".to_owned();
        metadata.reactive_load = "Captor X RAW".to_owned();
        metadata.load_impedance_ohms = Some(8);
        metadata.return_gain_note = "0 dB, pad off".to_owned();

        let json = capture_record_json(
            "capture-test",
            123,
            137,
            0.94,
            400,
            287,
            0.000_123,
            TrainingStopReason::EarlyStopping,
            &metadata,
        );

        assert!(json.contains("\"target\": \"full_amp_unfiltered_load\""));
        assert!(json.contains("\"status\": \"uncalibrated\""));
        assert!(json.contains("\"input_level_dbu\": null"));
        assert!(json.contains("\"source_send_trim_db\": -18.400"));
        assert!(json.contains("\"fractional_latency_samples\": 137.25000000"));
        assert!(json.contains("\"amplifier\": \"EVH 5153 \\\"Blue\\\"\""));
        assert!(json.contains("\"load_impedance_ohms\": 8"));
        assert!(json.contains("\"return_gain_note\": \"0 dB, pad off\""));
        assert!(json.contains("\"training_stop_reason\": \"early_stopping\""));
    }
}
