mod editor;

use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicU8, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use crossbeam_queue::ArrayQueue;
use mot_core::a2::encode_a2_payload;
use mot_core::a2_train::{
    A2CancellationToken, A2PublicationQuality, A2TrainerConfig, A2TrainingData, A2TrainingOutcome,
    A2TrainingProgress, A2TrainingStopReason, train_a2,
};
use mot_core::capture::{
    CAPTURE_SAMPLE_RATE_HZ, CalibrationStatus, CaptureTarget, HardwareCaptureMetadata,
    RETURN_CLIP_THRESHOLD_LINEAR, TransportInfo, extract_aligned_excitation, linear_to_dbfs,
    measure_alignment,
};
use mot_core::capture_asset::{
    CAPTURE_ASSET_SHA256, CAPTURE_PROTOCOL_VERSION, load_default_capture_program,
};
use mot_core::model::{
    A2_ARCHITECTURE_ID, A2_ARCHITECTURE_VERSION, ModelMetadata, ModelRef, MotModel,
};
use mot_core::model_library::ModelLibrary;
use mot_core::split_capture::{CompletedTrainerRecording, SplitCaptureState, TrainerRecorder};
use mot_core::wav_io::write_mono_f32_wav;
use truce::prelude::*;
use truce_egui::EguiEditor;

use editor::{MotTrainerUi, WINDOW_SIZE};

const PREPARED_CAPACITY: usize = 2;
const RETIRED_CAPACITY: usize = 4;
static NEXT_CAPTURE_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum TrainerStatus {
    #[default]
    Loading = 0,
    Ready = 1,
    Armed = 2,
    Waiting = 3,
    Recording = 4,
    Captured = 5,
    Aligning = 6,
    Training = 7,
    ModelSaved = 8,
    Invalid = 9,
    Error = 10,
}

impl TrainerStatus {
    fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::Ready,
            2 => Self::Armed,
            3 => Self::Waiting,
            4 => Self::Recording,
            5 => Self::Captured,
            6 => Self::Aligning,
            7 => Self::Training,
            8 => Self::ModelSaved,
            9 => Self::Invalid,
            10 => Self::Error,
            _ => Self::Loading,
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::Loading => "LOADING CAPTURE ASSET",
            Self::Ready => "READY",
            Self::Armed => "ARMED",
            Self::Waiting => "WAITING FOR TRANSPORT",
            Self::Recording => "RECORDING",
            Self::Captured => "CAPTURED",
            Self::Aligning => "ALIGNING",
            Self::Training => "TRAINING",
            Self::ModelSaved => "MODEL SAVED",
            Self::Invalid => "INVALID CAPTURE",
            Self::Error => "ERROR",
        }
    }
}

#[derive(Clone, Debug)]
pub struct TrainingSnapshot {
    pub progress: f32,
    pub epoch: u32,
    pub maximum_epochs: u32,
    pub best_esr: f64,
    pub epoch_seconds: f64,
    pub elapsed_seconds: f64,
    pub device: String,
}

impl Default for TrainingSnapshot {
    fn default() -> Self {
        Self {
            progress: 0.0,
            epoch: 0,
            maximum_epochs: 400,
            best_esr: 1.0,
            epoch_seconds: 0.0,
            elapsed_seconds: 0.0,
            device: String::new(),
        }
    }
}

pub struct TrainerControl {
    prepared: ArrayQueue<Box<TrainerRecorder>>,
    retired: ArrayQueue<Box<TrainerRecorder>>,
    status: AtomicU8,
    capture_progress_bits: AtomicU32,
    return_peak_bits: AtomicU32,
    training_progress_bits: AtomicU32,
    training: RwLock<TrainingSnapshot>,
    last_error: RwLock<String>,
    last_saved_model: RwLock<Option<ModelRef>>,
    cancellation: A2CancellationToken,
    training_generation: AtomicU64,
}

impl Default for TrainerControl {
    fn default() -> Self {
        Self {
            prepared: ArrayQueue::new(PREPARED_CAPACITY),
            retired: ArrayQueue::new(RETIRED_CAPACITY),
            status: AtomicU8::new(TrainerStatus::Loading as u8),
            capture_progress_bits: AtomicU32::new(0.0_f32.to_bits()),
            return_peak_bits: AtomicU32::new(0.0_f32.to_bits()),
            training_progress_bits: AtomicU32::new(0.0_f32.to_bits()),
            training: RwLock::new(TrainingSnapshot::default()),
            last_error: RwLock::new(String::new()),
            last_saved_model: RwLock::new(None),
            cancellation: A2CancellationToken::default(),
            training_generation: AtomicU64::new(1),
        }
    }
}

impl TrainerControl {
    fn publish_prepared(&self, recorder: Box<TrainerRecorder>) {
        let _ = self.prepared.force_push(recorder);
        self.set_capture_progress(0.0);
        self.set_return_peak(0.0);
    }

    fn take_prepared(&self) -> Option<Box<TrainerRecorder>> {
        self.prepared.pop()
    }

    fn retire_recorder(&self, recorder: Box<TrainerRecorder>) -> Result<(), Box<TrainerRecorder>> {
        self.retired.push(recorder)
    }

    fn drain_retired(&self) {
        while self.retired.pop().is_some() {}
    }

    fn set_status(&self, status: TrainerStatus) {
        self.status.store(status as u8, Ordering::Release);
    }

    pub fn status(&self) -> TrainerStatus {
        TrainerStatus::from_u8(self.status.load(Ordering::Acquire))
    }

    fn set_capture_progress(&self, value: f32) {
        self.capture_progress_bits
            .store(value.clamp(0.0, 1.0).to_bits(), Ordering::Release);
    }

    fn capture_progress(&self) -> f32 {
        f32::from_bits(self.capture_progress_bits.load(Ordering::Acquire)).clamp(0.0, 1.0)
    }

    fn set_return_peak(&self, value: f32) {
        self.return_peak_bits
            .store(value.clamp(0.0, 1.0).to_bits(), Ordering::Release);
    }

    fn return_peak(&self) -> f32 {
        f32::from_bits(self.return_peak_bits.load(Ordering::Acquire)).clamp(0.0, 1.0)
    }

    fn set_status_for_generation(&self, generation: u64, status: TrainerStatus) -> bool {
        if !self.is_training_generation_current(generation) {
            return false;
        }
        self.set_status(status);
        true
    }

    fn update_training(&self, generation: u64, progress: A2TrainingProgress) {
        if !self.is_training_generation_current(generation) {
            return;
        }
        let normalized = progress.completed_epochs as f32 / progress.maximum_epochs.max(1) as f32;
        self.training_progress_bits
            .store(normalized.clamp(0.0, 1.0).to_bits(), Ordering::Release);
        if let Ok(mut snapshot) = self.training.write() {
            *snapshot = TrainingSnapshot {
                progress: normalized,
                epoch: progress.completed_epochs,
                maximum_epochs: progress.maximum_epochs,
                best_esr: progress.best_validation_esr,
                epoch_seconds: progress.epoch_seconds,
                elapsed_seconds: progress.elapsed_seconds,
                device: progress.device_status.label().to_owned(),
            };
        }
    }

    pub fn training_snapshot(&self) -> TrainingSnapshot {
        self.training
            .read()
            .map_or_else(|_| TrainingSnapshot::default(), |value| value.clone())
    }

    fn reset_training(&self, generation: u64, maximum_epochs: u32) -> bool {
        if !self.is_training_generation_current(generation) {
            return false;
        }
        self.cancellation.reset();
        if !self.is_training_generation_current(generation) {
            self.cancellation.cancel();
            return false;
        }
        self.training_progress_bits
            .store(0.0_f32.to_bits(), Ordering::Release);
        if let Ok(mut snapshot) = self.training.write() {
            *snapshot = TrainingSnapshot {
                maximum_epochs,
                ..TrainingSnapshot::default()
            };
        }
        true
    }

    pub fn cancel_training(&self) {
        self.cancellation.cancel();
    }

    fn begin_training_generation(&self) -> u64 {
        let generation = self
            .training_generation
            .fetch_add(1, Ordering::AcqRel)
            .wrapping_add(1);
        self.cancellation.cancel();
        generation
    }

    fn is_training_generation_current(&self, generation: u64) -> bool {
        self.training_generation.load(Ordering::Acquire) == generation
    }

    fn training_is_active(&self, generation: u64) -> bool {
        self.is_training_generation_current(generation) && !self.cancellation.is_cancelled()
    }

    fn training_progress(&self) -> f32 {
        f32::from_bits(self.training_progress_bits.load(Ordering::Acquire)).clamp(0.0, 1.0)
    }

    fn set_error(&self, message: impl Into<String>) {
        if let Ok(mut error) = self.last_error.write() {
            *error = message.into();
        }
        self.set_status(TrainerStatus::Error);
    }

    fn set_error_for_generation(&self, generation: u64, message: impl Into<String>) {
        if self.is_training_generation_current(generation) {
            self.set_error(message);
        }
    }

    pub fn clear_error(&self) {
        if let Ok(mut error) = self.last_error.write() {
            error.clear();
        }
    }

    pub fn last_error(&self) -> String {
        self.last_error
            .read()
            .map_or_else(|_| "trainer error lock poisoned".to_owned(), |e| e.clone())
    }

    pub fn last_saved_model(&self) -> Option<ModelRef> {
        self.last_saved_model
            .read()
            .ok()
            .and_then(|model| model.clone())
    }
}

#[derive(Params)]
pub struct MotTrainerParams {
    #[param(name = "Bypass", flags = "automatable | bypass")]
    pub bypass: BoolParam,
    #[param(name = "Target", range = "discrete(0, 1)", default = 0)]
    pub target: IntParam,
    #[param(name = "Max Passes", range = "discrete(1, 400)", default = 400)]
    pub max_epochs: IntParam,

    #[persist]
    pub model_name: RwLock<String>,
    #[persist]
    pub amplifier: RwLock<String>,
    #[persist]
    pub amplifier_channel: RwLock<String>,
    #[persist]
    pub control_positions: RwLock<String>,
    #[persist]
    pub interface_output: RwLock<String>,
    #[persist]
    pub interface_input: RwLock<String>,
    #[persist]
    pub reamp_box: RwLock<String>,
    #[persist]
    pub reactive_load: RwLock<String>,
    #[persist]
    pub load_impedance_ohms: AtomicCell<u64>,
    #[persist]
    pub return_gain_note: RwLock<String>,

    #[skip]
    pub control: Arc<TrainerControl>,
    #[skip]
    pub arm_generation: AtomicU64,
    #[skip]
    pub prepare_generation: AtomicCell<u64>,

    #[meter]
    pub capture_progress: MeterSlot,
    #[meter]
    pub return_peak: MeterSlot,
    #[meter]
    pub training_progress: MeterSlot,
}

pub(crate) use MotTrainerParamsParamId as P;

#[derive(Clone, Copy, Debug)]
struct PrepareTrainerTask {
    generation: u64,
    sample_rate_hz: u32,
}

impl BackgroundTask for PrepareTrainerTask {
    type Params = MotTrainerParams;
    const SERIALIZED: bool = true;

    fn run(self, params: &Self::Params) {
        params.control.drain_retired();
        if params.prepare_generation.load() != self.generation {
            return;
        }
        let result = load_default_capture_program()
            .and_then(|program| {
                TrainerRecorder::new(program, self.sample_rate_hz).map_err(|e| e.to_string())
            })
            .map(Box::new);
        if params.prepare_generation.load() != self.generation {
            return;
        }
        match result {
            Ok(recorder) => params.control.publish_prepared(recorder),
            Err(error) => params.control.set_error(error),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct RetireRecorderTask;

impl BackgroundTask for RetireRecorderTask {
    type Params = MotTrainerParams;
    const SERIALIZED: bool = true;

    fn run(self, params: &Self::Params) {
        params.control.drain_retired();
    }
}

#[derive(Debug)]
struct TrainRecordingTask {
    recording: CompletedTrainerRecording,
    generation: u64,
}

impl BackgroundTask for TrainRecordingTask {
    type Params = MotTrainerParams;
    const SERIALIZED: bool = true;

    fn run(self, params: &Self::Params) {
        params.control.drain_retired();
        if !params.control.reset_training(
            self.generation,
            params.max_epochs.value_i32().clamp(1, 400) as u32,
        ) {
            return;
        }
        let max_epochs = params.max_epochs.value_i32().clamp(1, 400) as u32;
        let metadata = metadata_from_params(params);
        let display_name = {
            let name = read_lock_string(&params.model_name);
            if name.trim().is_empty() {
                "Captured Amp".to_owned()
            } else {
                name
            }
        };

        let result = train_and_publish(
            &params.control,
            self.generation,
            self.recording,
            max_epochs,
            &display_name,
            metadata,
        );
        if let Err(error) = result {
            params
                .control
                .set_error_for_generation(self.generation, error);
        }

        // Rebuild the fixed recording pool off the audio thread so another
        // capture can be armed without waiting for a DAW/plugin reload.
        if params
            .control
            .is_training_generation_current(self.generation)
            && let Ok(program) = load_default_capture_program()
            && let Ok(recorder) = TrainerRecorder::new(program, CAPTURE_SAMPLE_RATE_HZ)
            && params
                .control
                .is_training_generation_current(self.generation)
        {
            if params.control.status() != TrainerStatus::Error {
                params.control.publish_prepared(Box::new(recorder));
            } else {
                let _ = params.control.prepared.force_push(Box::new(recorder));
            }
        }
        params.control.drain_retired();
    }
}

pub struct MotTrainer {
    recorder: Option<Box<TrainerRecorder>>,
    pending_retired: Option<Box<TrainerRecorder>>,
    pending_training: Option<TrainRecordingTask>,
    recorder_active: bool,
    observed_arm_generation: u64,
    requested_prepare_generation: u64,
    sample_rate_hz: u32,
}

impl Default for MotTrainer {
    fn default() -> Self {
        Self {
            recorder: None,
            pending_retired: None,
            pending_training: None,
            recorder_active: false,
            observed_arm_generation: 0,
            requested_prepare_generation: 0,
            sample_rate_hz: CAPTURE_SAMPLE_RATE_HZ,
        }
    }
}

impl PluginLogic for MotTrainer {
    type Params = MotTrainerParams;
    type DspState = Self;

    fn init(params: &Self::Params, _context: &InitContext) -> Self::DspState {
        if params.prepare_generation.load() == 0 {
            params.prepare_generation.store(1);
        }
        Self::default()
    }

    fn supports_in_place() -> bool {
        false
    }

    fn bus_layouts() -> Vec<BusLayout> {
        vec![BusLayout::mono()]
    }

    fn reset(state: &mut Self::DspState, params: &Self::Params, config: &AudioConfig) {
        state.sample_rate_hz = config.sample_rate.round() as u32;
        state.recorder_active = false;
        state.observed_arm_generation = params.arm_generation.load(Ordering::Acquire);
        params.control.begin_training_generation();
        let generation = params.prepare_generation.load().wrapping_add(1).max(1);
        params.prepare_generation.store(generation);
        state.requested_prepare_generation = 0;
        params.control.set_status(TrainerStatus::Loading);
    }

    fn process(
        state: &mut Self::DspState,
        params: &Self::Params,
        buffer: &mut AudioBuffer,
        _events: &EventList,
        context: &mut ProcessContext,
    ) -> ProcessStatus {
        schedule_retired_recorders(state, params, context);
        let generation = params.prepare_generation.load();
        if generation != state.requested_prepare_generation {
            if let Some(spawner) = context.tasks::<PrepareTrainerTask>() {
                spawner.spawn_coalescing(PrepareTrainerTask {
                    generation,
                    sample_rate_hz: state.sample_rate_hz,
                });
            }
            state.requested_prepare_generation = generation;
        }
        if state.pending_retired.is_none()
            && let Some(prepared) = params.control.take_prepared()
        {
            install_prepared_recorder(state, params, prepared);
            mark_initial_recorder_ready(&params.control);
            schedule_retired_recorders(state, params, context);
        }
        schedule_pending_training(state, context);

        let arm_generation = params.arm_generation.load(Ordering::Acquire);
        if arm_generation != state.observed_arm_generation {
            state.observed_arm_generation = arm_generation;
            if let Some(recorder) = &mut state.recorder {
                match recorder.arm(context.transport.playing) {
                    Ok(()) => {
                        state.recorder_active = true;
                        params.control.set_status(TrainerStatus::Armed);
                    }
                    Err(_error) => params.control.set_status(TrainerStatus::Error),
                }
            }
        }

        if buffer.num_input_channels() > 0 && buffer.num_output_channels() > 0 {
            let (input, output) = buffer.io_pair(0, 0);
            let transport = TransportInfo {
                playing: context.transport.playing,
                recording: context.transport.recording,
                timeline_sample: Some(context.transport.position_samples),
                loop_active: context.transport.loop_active,
                discontinuity: false,
                sample_rate_hz: state.sample_rate_hz,
            };
            if params.bypass.value() {
                output.copy_from_slice(input);
                if state.recorder_active
                    && let Some(recorder) = &mut state.recorder
                    && fail_closed_capture_on_bypass(recorder, transport)
                {
                    set_capture_status_if_authoritative(
                        &params.control,
                        state.recorder_active,
                        recorder.state(),
                    );
                    state.recorder_active = false;
                }
            } else {
                output.fill(0.0);
                if state.recorder_active
                    && let Some(recorder) = &mut state.recorder
                {
                    recorder.process_block(input, transport);
                    let progress = recorder.completed_samples() as f32
                        / recorder.total_samples().max(1) as f32;
                    params.control.set_capture_progress(progress);
                    params.control.set_return_peak(recorder.peak_linear());
                    set_capture_status_if_authoritative(
                        &params.control,
                        state.recorder_active,
                        recorder.state(),
                    );

                    if state.pending_training.is_none()
                        && let Some(recording) = recorder.take_completed()
                    {
                        params.control.set_status(TrainerStatus::Captured);
                        state.recorder_active = false;
                        state.pending_training = Some(TrainRecordingTask {
                            recording,
                            generation: params.control.begin_training_generation(),
                        });
                    }
                }
            }
        }
        schedule_pending_training(state, context);

        context.set_meter(P::CaptureProgress, params.control.capture_progress());
        context.set_meter(P::ReturnPeak, params.control.return_peak());
        context.set_meter(P::TrainingProgress, params.control.training_progress());
        ProcessStatus::Normal
    }

    fn latency(_state: &Self::DspState) -> u32 {
        0
    }

    fn tail(_state: &Self::DspState) -> u32 {
        0
    }

    fn editor(params: Arc<MotTrainerParams>) -> Box<dyn Editor> {
        EguiEditor::with_ui(params, WINDOW_SIZE, MotTrainerUi).into_editor()
    }
}

fn install_prepared_recorder(
    state: &mut MotTrainer,
    params: &MotTrainerParams,
    prepared: Box<TrainerRecorder>,
) {
    if let Some(previous) = state.recorder.replace(prepared)
        && let Err(returned) = params.control.retire_recorder(previous)
    {
        debug_assert!(state.pending_retired.is_none());
        state.pending_retired = Some(returned);
    }
    // A prepared recorder is storage, not an active capture authority. It
    // becomes authoritative only after a successful explicit ARM.
    state.recorder_active = false;
}

fn mark_initial_recorder_ready(control: &TrainerControl) {
    if control.status() == TrainerStatus::Loading {
        control.set_status(TrainerStatus::Ready);
    }
}

fn schedule_retired_recorders(
    state: &mut MotTrainer,
    params: &MotTrainerParams,
    context: &ProcessContext,
) {
    if let Some(retired) = state.pending_retired.take()
        && let Err(returned) = params.control.retire_recorder(retired)
    {
        state.pending_retired = Some(returned);
    }
    if !params.control.retired.is_empty()
        && let Some(spawner) = context.tasks::<RetireRecorderTask>()
    {
        spawner.spawn_coalescing(RetireRecorderTask);
    }
}

fn fail_closed_capture_on_bypass(recorder: &mut TrainerRecorder, transport: TransportInfo) -> bool {
    if !matches!(
        recorder.state(),
        SplitCaptureState::Armed
            | SplitCaptureState::WaitingForTransport
            | SplitCaptureState::PreRoll { .. }
            | SplitCaptureState::Program { .. }
            | SplitCaptureState::Tail { .. }
            | SplitCaptureState::AlignmentMargin { .. }
    ) {
        return false;
    }
    recorder.process_block(
        &[],
        TransportInfo {
            discontinuity: true,
            ..transport
        },
    );
    true
}

fn set_capture_status_if_authoritative(
    control: &TrainerControl,
    authoritative: bool,
    capture_state: SplitCaptureState,
) {
    if authoritative {
        control.set_status(status_from_capture(capture_state));
    }
}

fn schedule_pending_training(state: &mut MotTrainer, context: &mut ProcessContext) {
    let Some(task) = state.pending_training.take() else {
        return;
    };
    if let Some(spawner) = context.tasks::<TrainRecordingTask>() {
        if let Err(returned) = spawner.try_spawn(task) {
            state.pending_training = Some(returned);
        }
    } else {
        state.pending_training = Some(task);
    }
}

fn status_from_capture(state: SplitCaptureState) -> TrainerStatus {
    match state {
        SplitCaptureState::Idle => TrainerStatus::Ready,
        SplitCaptureState::Armed => TrainerStatus::Armed,
        SplitCaptureState::WaitingForTransport => TrainerStatus::Waiting,
        SplitCaptureState::PreRoll { .. }
        | SplitCaptureState::Program { .. }
        | SplitCaptureState::Tail { .. }
        | SplitCaptureState::AlignmentMargin { .. } => TrainerStatus::Recording,
        SplitCaptureState::Ready => TrainerStatus::Captured,
        SplitCaptureState::Invalid(_) => TrainerStatus::Invalid,
    }
}

fn metadata_from_params(params: &MotTrainerParams) -> HardwareCaptureMetadata {
    let mut metadata = HardwareCaptureMetadata::uncalibrated_full_amp();
    metadata.target = if params.target.value_i32() == 1 {
        CaptureTarget::FullAmpUnfilteredLoad
    } else {
        CaptureTarget::SoftwarePluginChain
    };
    metadata.excitation_hash = CAPTURE_ASSET_SHA256.to_owned();
    metadata.amplifier = read_lock_string(&params.amplifier);
    metadata.amplifier_channel = read_lock_string(&params.amplifier_channel);
    metadata.control_positions = read_lock_string(&params.control_positions);
    metadata.interface_output = read_lock_string(&params.interface_output);
    metadata.interface_input = read_lock_string(&params.interface_input);
    metadata.reamp_box = read_lock_string(&params.reamp_box);
    metadata.reactive_load = read_lock_string(&params.reactive_load);
    metadata.load_impedance_ohms = u16::try_from(params.load_impedance_ohms.load())
        .ok()
        .filter(|value| *value > 0);
    metadata.return_gain_note = read_lock_string(&params.return_gain_note);
    metadata
}

fn train_and_publish(
    control: &TrainerControl,
    generation: u64,
    recording: CompletedTrainerRecording,
    max_epochs: u32,
    display_name: &str,
    mut capture_metadata: HardwareCaptureMetadata,
) -> Result<(), String> {
    ensure_training_active(control, generation, None)?;
    if !control.set_status_for_generation(generation, TrainerStatus::Aligning) {
        return Err("Training superseded before alignment".to_owned());
    }
    let library = ModelLibrary::for_current_user().map_err(|error| error.to_string())?;
    library
        .ensure_directories()
        .map_err(|error| error.to_string())?;
    let (model_id, timestamp) = unique_model_id();
    let capture_dir = library
        .paths()
        .plugin_root
        .join("Capture Records")
        .join(&model_id);
    fs::create_dir_all(&capture_dir).map_err(|error| error.to_string())?;
    write_mono_f32_wav(
        &capture_dir.join("raw-return.wav"),
        CAPTURE_SAMPLE_RATE_HZ,
        recording.audio(),
    )
    .map_err(|error| error.to_string())?;
    ensure_training_active(control, generation, Some(&capture_dir))?;

    if recording.peak_linear() > RETURN_CLIP_THRESHOLD_LINEAR {
        return Err(format!(
            "Return clipped at {:.2} dBFS; raw capture preserved in {}",
            linear_to_dbfs(recording.peak_linear()),
            capture_dir.display()
        ));
    }

    let program = load_default_capture_program()?;
    ensure_training_active(control, generation, Some(&capture_dir))?;
    let alignment = measure_alignment(&program, recording.audio(), recording.alignment_config())
        .map_err(|error| error.to_string())?;
    ensure_training_active(control, generation, Some(&capture_dir))?;
    let target = extract_aligned_excitation(&program, recording.audio(), alignment)
        .map_err(|error| error.to_string())?;
    let emitted = program.excitation().to_vec();
    write_mono_f32_wav(
        &capture_dir.join("emitted.wav"),
        CAPTURE_SAMPLE_RATE_HZ,
        &emitted,
    )
    .map_err(|error| error.to_string())?;
    write_mono_f32_wav(
        &capture_dir.join("aligned-return.wav"),
        CAPTURE_SAMPLE_RATE_HZ,
        &target,
    )
    .map_err(|error| error.to_string())?;
    ensure_training_active(control, generation, Some(&capture_dir))?;

    capture_metadata.measured_latency_samples = Some(alignment.fractional_latency_samples);
    capture_metadata.return_peak_dbfs = Some(linear_to_dbfs(recording.peak_linear()));
    capture_metadata.return_rms_dbfs = Some(linear_to_dbfs(recording.rms_linear()));

    if !control.set_status_for_generation(generation, TrainerStatus::Training) {
        return Err(format!(
            "Training superseded; capture preserved in {}",
            capture_dir.display()
        ));
    }
    let outcome = train_a2(
        A2TrainingData {
            input: &emitted,
            target: &target,
            sample_rate_hz: CAPTURE_SAMPLE_RATE_HZ,
        },
        A2TrainerConfig {
            max_epochs,
            ..A2TrainerConfig::default()
        },
        &control.cancellation,
        |progress| control.update_training(generation, progress),
    )
    .map_err(|error| error.to_string())?;

    write_capture_record(
        &capture_dir,
        &model_id,
        timestamp,
        alignment.fractional_latency_samples,
        alignment.normalized_correlation,
        &capture_metadata,
        &outcome,
    )?;

    if outcome.stop_reason == A2TrainingStopReason::Cancelled {
        return Err(format!(
            "Training cancelled; capture preserved in {}",
            capture_dir.display()
        ));
    }
    ensure_training_active(control, generation, Some(&capture_dir))?;
    if !outcome.quality.passes_publication_gate {
        return Err(format!(
            "Model rejected: validation ESR {:.6}; capture preserved in {}",
            outcome.quality.validation_esr,
            capture_dir.display()
        ));
    }

    let metadata = ModelMetadata {
        model_id: model_id.clone(),
        display_name: normalize_display_name(display_name, &model_id),
        architecture_id: A2_ARCHITECTURE_ID.to_owned(),
        architecture_version: A2_ARCHITECTURE_VERSION,
        sample_rate_hz: outcome.model.sample_rate_hz,
        causal: outcome.model.causal,
        lookahead_samples: outcome.model.lookahead_samples,
        runtime_latency_samples: outcome.model.runtime_latency_samples,
        estimated_macs_per_sample: u64::from(outcome.model.estimated_macs_per_sample),
    };
    ensure_training_active(control, generation, Some(&capture_dir))?;
    let payload = encode_a2_payload(&outcome.model).map_err(|error| error.to_string())?;
    let model = MotModel::new(metadata, payload).map_err(|error| error.to_string())?;
    let filename = format!("{model_id}.motmodel");
    let model_path = library.paths().models.join(&filename);
    ensure_training_active(control, generation, Some(&capture_dir))?;
    model
        .write_new(&model_path)
        .map_err(|error| error.to_string())?;
    if let Err(error) = ensure_training_active(control, generation, Some(&capture_dir)) {
        let _ = fs::remove_file(&model_path);
        return Err(error);
    }
    if let Ok(mut saved) = control.last_saved_model.write()
        && control.training_is_active(generation)
    {
        *saved = Some(model.model_ref(filename));
    }
    if !control.training_is_active(generation) {
        let _ = fs::remove_file(&model_path);
        return Err(format!(
            "Training cancelled before publication; capture preserved in {}",
            capture_dir.display()
        ));
    }
    control.set_status_for_generation(generation, TrainerStatus::ModelSaved);
    Ok(())
}

fn ensure_training_active(
    control: &TrainerControl,
    generation: u64,
    capture_dir: Option<&Path>,
) -> Result<(), String> {
    let preserved = capture_dir.map_or_else(String::new, |path| {
        format!("; capture preserved in {}", path.display())
    });
    if !control.is_training_generation_current(generation) {
        return Err(format!("Training superseded{preserved}"));
    }
    if control.cancellation.is_cancelled() {
        return Err(format!("Training cancelled{preserved}"));
    }
    Ok(())
}

fn write_capture_record(
    capture_dir: &Path,
    model_id: &str,
    timestamp: u64,
    latency_samples: f64,
    correlation: f64,
    metadata: &HardwareCaptureMetadata,
    outcome: &A2TrainingOutcome,
) -> Result<(), String> {
    let json = capture_record_json(
        model_id,
        timestamp,
        latency_samples,
        correlation,
        metadata,
        outcome,
    )?;
    fs::write(capture_dir.join("capture.json"), json).map_err(|error| error.to_string())
}

fn capture_record_json(
    model_id: &str,
    timestamp: u64,
    latency_samples: f64,
    correlation: f64,
    metadata: &HardwareCaptureMetadata,
    outcome: &A2TrainingOutcome,
) -> Result<String, String> {
    for (name, value) in [
        ("fractional_latency_samples", latency_samples),
        ("sync_correlation", correlation),
        ("validation_esr", outcome.quality.validation_esr),
        ("validation_esr_db", outcome.quality.validation_esr_db),
        (
            "original_target_rms_dbfs",
            outcome.quality.original_train_target_rms_dbfs,
        ),
        (
            "output_normalization_gain",
            outcome.quality.output_normalization_gain,
        ),
        (
            "mrstft_weight_applied",
            outcome.quality.mrstft_weight_applied,
        ),
        ("elapsed_seconds", outcome.elapsed_seconds),
    ] {
        if !value.is_finite() {
            return Err(format!("capture record field {name} is not finite"));
        }
    }
    let target = match metadata.target {
        CaptureTarget::SoftwarePluginChain => "software_plugin_chain",
        CaptureTarget::FullAmpUnfilteredLoad => "full_amp_unfiltered_load",
    };
    let calibration_status = match metadata.calibration_status {
        CalibrationStatus::Uncalibrated => "uncalibrated",
    };
    let stop_reason = match outcome.stop_reason {
        A2TrainingStopReason::MaximumEpochs => "maximum_passes",
        A2TrainingStopReason::ThresholdReached => "threshold_reached",
        A2TrainingStopReason::Cancelled => "cancelled",
    };
    let quality = match outcome.quality.quality {
        A2PublicationQuality::Great => "great",
        A2PublicationQuality::NotBad => "not_bad",
        A2PublicationQuality::MightSoundOkay => "might_sound_okay",
        A2PublicationQuality::ProbablyPoor => "probably_poor",
        A2PublicationQuality::Failed => "failed",
    };
    let return_peak_dbfs = json_optional_f32(metadata.return_peak_dbfs);
    let return_rms_dbfs = json_optional_f32(metadata.return_rms_dbfs);
    let input_level_dbu = json_optional_f32(metadata.input_level_dbu);
    let output_level_dbu = json_optional_f32(metadata.output_level_dbu);
    let load_impedance_ohms = metadata
        .load_impedance_ohms
        .map_or_else(|| "null".to_owned(), |value| value.to_string());
    let json = format!(
        concat!(
            "{{\n",
            "  \"schema_version\": 5,\n",
            "  \"capture_protocol_version\": {},\n",
            "  \"model_id\": {},\n",
            "  \"created_unix_seconds\": {},\n",
            "  \"target\": \"{}\",\n",
            "  \"sample_rate_hz\": {},\n",
            "  \"fractional_latency_samples\": {:.8},\n",
            "  \"sync_correlation\": {:.8},\n",
            "  \"return_peak_dbfs\": {},\n",
            "  \"return_rms_dbfs\": {},\n",
            "  \"excitation_sha256\": {},\n",
            "  \"calibration\": {{\n",
            "    \"status\": \"{}\",\n",
            "    \"input_level_dbu\": {},\n",
            "    \"output_level_dbu\": {}\n",
            "  }},\n",
            "  \"hardware\": {{\n",
            "    \"amplifier\": {},\n",
            "    \"amplifier_channel\": {},\n",
            "    \"control_positions\": {},\n",
            "    \"interface_output\": {},\n",
            "    \"interface_input\": {},\n",
            "    \"reamp_box\": {},\n",
            "    \"reactive_load\": {},\n",
            "    \"load_impedance_ohms\": {},\n",
            "    \"return_gain_note\": {}\n",
            "  }},\n",
            "  \"training\": {{\n",
            "    \"completed_passes\": {},\n",
            "    \"best_pass\": {},\n",
            "    \"stop_reason\": \"{}\",\n",
            "    \"validation_esr\": {:.9},\n",
            "    \"exported_runtime_validation_esr\": {:.9},\n",
            "    \"validation_esr_db\": {:.4},\n",
            "    \"quality\": \"{}\",\n",
            "    \"original_target_rms_dbfs\": {:.4},\n",
            "    \"output_normalization_gain\": {:.9},\n",
            "    \"mrstft_weight_applied\": {:.9},\n",
            "    \"elapsed_seconds\": {:.3},\n",
            "    \"publication_gate_passed\": {}\n",
            "  }}\n",
            "}}\n"
        ),
        CAPTURE_PROTOCOL_VERSION,
        json_string(model_id),
        timestamp,
        target,
        metadata.sample_rate_hz,
        latency_samples,
        correlation,
        return_peak_dbfs,
        return_rms_dbfs,
        json_string(&metadata.excitation_hash),
        calibration_status,
        input_level_dbu,
        output_level_dbu,
        json_string(&metadata.amplifier),
        json_string(&metadata.amplifier_channel),
        json_string(&metadata.control_positions),
        json_string(&metadata.interface_output),
        json_string(&metadata.interface_input),
        json_string(&metadata.reamp_box),
        json_string(&metadata.reactive_load),
        load_impedance_ohms,
        json_string(&metadata.return_gain_note),
        outcome.completed_epochs,
        outcome.best_epoch,
        stop_reason,
        outcome.quality.validation_esr,
        outcome.quality.exported_runtime_validation_esr,
        outcome.quality.validation_esr_db,
        quality,
        outcome.quality.original_train_target_rms_dbfs,
        outcome.quality.output_normalization_gain,
        outcome.quality.mrstft_weight_applied,
        outcome.elapsed_seconds,
        outcome.quality.passes_publication_gate,
    );
    Ok(json)
}

fn json_optional_f32(value: Option<f32>) -> String {
    value
        .filter(|value| value.is_finite())
        .map_or_else(|| "null".to_owned(), |value| format!("{value:.3}"))
}

fn json_string(value: &str) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";

    let mut escaped = String::with_capacity(value.len() + 2);
    escaped.push('"');
    for character in value.chars() {
        match character {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\u{08}' => escaped.push_str("\\b"),
            '\u{0c}' => escaped.push_str("\\f"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            control if control <= '\u{1f}' => {
                let code = control as usize;
                escaped.push_str("\\u00");
                escaped.push(char::from(HEX[(code >> 4) & 0x0f]));
                escaped.push(char::from(HEX[code & 0x0f]));
            }
            printable => escaped.push(printable),
        }
    }
    escaped.push('"');
    escaped
}

fn unique_model_id() -> (String, u64) {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs());
    let suffix = NEXT_CAPTURE_ID.fetch_add(1, Ordering::Relaxed).max(1);
    (format!("capture-{timestamp}-{suffix}"), timestamp)
}

fn normalize_display_name(requested: &str, fallback: &str) -> String {
    let trimmed = requested.trim();
    if trimmed.is_empty() {
        fallback.to_owned()
    } else {
        trimmed.chars().take(96).collect()
    }
}

pub(crate) fn read_lock_string(lock: &RwLock<String>) -> String {
    lock.read()
        .map_or_else(|_| String::new(), |value| value.clone())
}

pub(crate) fn write_lock_string(lock: &RwLock<String>, value: &str) {
    if let Ok(mut destination) = lock.write() {
        destination.clear();
        destination.push_str(value);
    }
}

truce::plugin! {
    logic: MotTrainer,
    params: MotTrainerParams,
    tasks: [PrepareTrainerTask, RetireRecorderTask, TrainRecordingTask],
}

truce::enable_rt_paranoid!();

#[cfg(test)]
mod tests {
    use super::*;
    use mot_core::capture::{CaptureInvalidation, CaptureProgram};

    fn test_program() -> Arc<CaptureProgram> {
        Arc::new(CaptureProgram::new(vec![0.25, -0.25], vec![0.1, -0.1, 0.2]).unwrap())
    }

    fn test_recorder(program: Arc<CaptureProgram>) -> Box<TrainerRecorder> {
        Box::new(
            TrainerRecorder::with_alignment_margin(program, CAPTURE_SAMPLE_RATE_HZ, 8).unwrap(),
        )
    }

    #[test]
    fn defaults_match_capture_contract() {
        let params = MotTrainerParams::new();
        assert_eq!(params.target.value_i32(), 0);
        assert_eq!(params.max_epochs.value_i32(), 400);
        assert_eq!(params.control.status(), TrainerStatus::Loading);
        assert!(
            params
                .param_infos()
                .iter()
                .all(|info| info.name != "Source Send Trim")
        );
        assert_eq!(params.meter_ids().len(), 3);
    }

    #[test]
    fn emitted_and_training_input_use_the_canonical_unity_excitation() {
        let program = test_program();
        let emitted = program.excitation().to_vec();

        assert_eq!(emitted, program.excitation());
    }

    #[test]
    fn wrapper_is_mono_and_reports_zero_latency() {
        let widths: Vec<_> = MotTrainer::bus_layouts()
            .iter()
            .map(|layout| {
                (
                    layout.total_input_channels(),
                    layout.total_output_channels(),
                )
            })
            .collect();
        assert_eq!(widths, vec![(1, 1)]);
        assert_eq!(MotTrainer::latency(&MotTrainer::default()), 0);
        assert_eq!(MotTrainer::tail(&MotTrainer::default()), 0);
    }

    #[test]
    fn capture_states_have_unambiguous_trainer_statuses() {
        assert_eq!(
            status_from_capture(SplitCaptureState::WaitingForTransport),
            TrainerStatus::Waiting
        );
        assert_eq!(
            status_from_capture(SplitCaptureState::Program {
                completed_samples: 1
            }),
            TrainerStatus::Recording
        );
        assert_eq!(
            status_from_capture(SplitCaptureState::Ready),
            TrainerStatus::Captured
        );
    }

    #[test]
    fn replacing_a_recorder_defers_its_drop_to_the_retirement_queue() {
        let params = MotTrainerParams::new();
        let old_program = test_program();
        let new_program = test_program();
        let mut state = MotTrainer {
            recorder: Some(test_recorder(Arc::clone(&old_program))),
            recorder_active: true,
            ..MotTrainer::default()
        };
        assert_eq!(Arc::strong_count(&old_program), 2);

        install_prepared_recorder(&mut state, &params, test_recorder(new_program));

        assert!(state.pending_retired.is_none());
        assert!(!state.recorder_active);
        assert_eq!(params.control.retired.len(), 1);
        assert_eq!(
            Arc::strong_count(&old_program),
            2,
            "the audio-side replacement must not destroy the old recorder"
        );
        params.control.drain_retired();
        assert_eq!(Arc::strong_count(&old_program), 1);
    }

    #[test]
    fn bypass_fail_closed_invalidates_an_armed_capture() {
        let mut recorder = test_recorder(test_program());
        recorder.arm(false).unwrap();
        let invalidated = fail_closed_capture_on_bypass(
            &mut recorder,
            TransportInfo {
                playing: false,
                timeline_sample: Some(0),
                sample_rate_hz: CAPTURE_SAMPLE_RATE_HZ,
                ..TransportInfo::default()
            },
        );

        assert!(invalidated);
        assert_eq!(
            recorder.state(),
            SplitCaptureState::Invalid(CaptureInvalidation::TransportDiscontinuity)
        );
    }

    #[test]
    fn handed_off_recorder_cannot_overwrite_worker_status() {
        let control = TrainerControl::default();
        control.set_status(TrainerStatus::Training);

        set_capture_status_if_authoritative(&control, false, SplitCaptureState::Ready);

        assert_eq!(control.status(), TrainerStatus::Training);
    }

    #[test]
    fn prepared_replacement_preserves_terminal_training_status() {
        let control = TrainerControl::default();
        for terminal in [TrainerStatus::ModelSaved, TrainerStatus::Error] {
            control.set_status(terminal);
            mark_initial_recorder_ready(&control);
            assert_eq!(control.status(), terminal);
        }

        control.set_status(TrainerStatus::Loading);
        mark_initial_recorder_ready(&control);
        assert_eq!(control.status(), TrainerStatus::Ready);
    }

    #[test]
    fn stale_training_generation_cannot_publish_status() {
        let control = TrainerControl::default();
        let stale = control.begin_training_generation();
        assert!(control.reset_training(stale, 400));
        control.set_status(TrainerStatus::Training);

        let current = control.begin_training_generation();
        control.set_status(TrainerStatus::Loading);

        assert!(!control.set_status_for_generation(stale, TrainerStatus::ModelSaved));
        assert_eq!(control.status(), TrainerStatus::Loading);
        assert!(!control.training_is_active(stale));
        assert!(ensure_training_active(&control, stale, None).is_err());
        assert!(control.reset_training(current, 400));
    }

    #[test]
    fn cancellation_is_honored_before_publication() {
        let control = TrainerControl::default();
        let generation = control.begin_training_generation();
        assert!(control.reset_training(generation, 400));

        control.cancel_training();

        assert!(!control.training_is_active(generation));
        assert!(
            ensure_training_active(&control, generation, None)
                .unwrap_err()
                .contains("cancelled")
        );
    }

    #[test]
    fn json_string_escapes_all_json_control_characters() {
        assert_eq!(
            json_string("a\"\\\n\r\t\u{08}\u{0c}\u{01}Ж"),
            "\"a\\\"\\\\\\n\\r\\t\\b\\f\\u0001Ж\""
        );
    }

    #[test]
    fn capture_record_contains_full_metadata_and_training_outcome() {
        let mut metadata = HardwareCaptureMetadata::uncalibrated_full_amp();
        metadata.return_peak_dbfs = Some(-2.25);
        metadata.return_rms_dbfs = Some(-18.75);
        metadata.excitation_hash = "asset-hash".to_owned();
        metadata.amplifier = "EVH \"Stealth\"\n100W".to_owned();
        metadata.amplifier_channel = "Blue\\Red".to_owned();
        metadata.control_positions = "Gain 6\tBass 4".to_owned();
        metadata.interface_output = "Out 3".to_owned();
        metadata.interface_input = "In 1".to_owned();
        metadata.reamp_box = "Reamp".to_owned();
        metadata.reactive_load = "Captor X".to_owned();
        metadata.load_impedance_ohms = Some(8);
        metadata.return_gain_note = "+12 dB".to_owned();

        let outcome = A2TrainingOutcome {
            model: mot_core::a2::A2Model::zeros(),
            completed_epochs: 400,
            best_epoch: 317,
            stop_reason: A2TrainingStopReason::MaximumEpochs,
            quality: mot_core::a2_train::A2QualityReport {
                validation_esr: 0.012_345_678,
                exported_runtime_validation_esr: 0.012_345_679,
                validation_esr_db: -19.0849,
                quality: A2PublicationQuality::NotBad,
                passes_publication_gate: true,
                original_train_target_rms_dbfs: -11.25,
                output_normalization_gain: 0.4625,
                mrstft_weight_applied: 0.0,
            },
            elapsed_seconds: 901.5,
        };

        let json = capture_record_json("capture-\"id", 123, 47.25, 0.987_654, &metadata, &outcome)
            .unwrap();

        for expected in [
            "\"schema_version\": 5",
            "\"capture_protocol_version\": 2",
            "\"model_id\": \"capture-\\\"id\"",
            "\"target\": \"full_amp_unfiltered_load\"",
            "\"input_level_dbu\": null",
            "\"output_level_dbu\": null",
            "\"amplifier\": \"EVH \\\"Stealth\\\"\\n100W\"",
            "\"amplifier_channel\": \"Blue\\\\Red\"",
            "\"control_positions\": \"Gain 6\\tBass 4\"",
            "\"interface_output\": \"Out 3\"",
            "\"interface_input\": \"In 1\"",
            "\"reamp_box\": \"Reamp\"",
            "\"reactive_load\": \"Captor X\"",
            "\"load_impedance_ohms\": 8",
            "\"return_gain_note\": \"+12 dB\"",
            "\"completed_passes\": 400",
            "\"best_pass\": 317",
            "\"stop_reason\": \"maximum_passes\"",
            "\"validation_esr\": 0.012345678",
            "\"exported_runtime_validation_esr\": 0.012345679",
            "\"validation_esr_db\": -19.0849",
            "\"quality\": \"not_bad\"",
            "\"original_target_rms_dbfs\": -11.2500",
            "\"output_normalization_gain\": 0.462500000",
            "\"elapsed_seconds\": 901.500",
            "\"publication_gate_passed\": true",
        ] {
            assert!(json.contains(expected), "missing {expected} in:\n{json}");
        }
        assert!(!json.contains("source_send_trim_db"));
    }

    #[test]
    fn editor_has_a_headless_render_path() {
        let params = Arc::new(MotTrainerParams::new());
        let mut editor = EguiEditor::with_ui(Arc::clone(&params), WINDOW_SIZE, MotTrainerUi);
        let erased: Arc<dyn truce::params::Params> = params;
        assert_eq!(Editor::size(&editor), WINDOW_SIZE);
        if let Some((_, width, height)) = Editor::screenshot(&mut editor, erased) {
            assert_eq!((width, height), (WINDOW_SIZE.0 * 2, WINDOW_SIZE.1 * 2));
        }
    }
}
