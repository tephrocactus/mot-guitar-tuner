mod editor;

use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use crossbeam_queue::ArrayQueue;
use mot_core::capture::{CAPTURE_SAMPLE_RATE_HZ, TransportInfo as CaptureTransportInfo};
pub use mot_core::capture_asset::{
    CAPTURE_ASSET_RELATIVE_PATH, CAPTURE_ASSET_SAMPLE_RATE_HZ, CAPTURE_ASSET_SAMPLES,
    CAPTURE_ASSET_SHA256, CAPTURE_PROTOCOL_VERSION, SYNC_HEADER_SAMPLES,
};
use mot_core::capture_asset::{capture_asset_path, load_default_capture_program};
use mot_core::split_capture::{GeneratorEngine, SplitCaptureState};
use truce::prelude::*;
use truce_egui::EguiEditor;

use editor::{GeneratorUi, WINDOW_SIZE};

pub const VERSION: &str = "0.4.0";
const READY_CAPACITY: usize = 2;
const RETIRED_CAPACITY: usize = 4;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum AssetLoadStatus {
    #[default]
    Idle = 0,
    Loading = 1,
    Ready = 2,
    Error = 3,
}

impl AssetLoadStatus {
    fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::Loading,
            2 => Self::Ready,
            3 => Self::Error,
            _ => Self::Idle,
        }
    }
}

#[derive(Debug)]
struct PreparedGenerator {
    generation: u64,
    engine: GeneratorEngine,
}

#[derive(Debug)]
pub struct AssetControl {
    ready: ArrayQueue<Box<PreparedGenerator>>,
    retired: ArrayQueue<Box<PreparedGenerator>>,
    status: AtomicU8,
    result_generation: AtomicU64,
    last_error: RwLock<String>,
}

impl Default for AssetControl {
    fn default() -> Self {
        Self {
            ready: ArrayQueue::new(READY_CAPACITY),
            retired: ArrayQueue::new(RETIRED_CAPACITY),
            status: AtomicU8::new(AssetLoadStatus::Idle as u8),
            result_generation: AtomicU64::new(0),
            last_error: RwLock::new(String::new()),
        }
    }
}

impl AssetControl {
    fn begin(&self, generation: u64) {
        self.result_generation.store(generation, Ordering::Release);
        self.status
            .store(AssetLoadStatus::Loading as u8, Ordering::Release);
        if let Ok(mut error) = self.last_error.write() {
            error.clear();
        }
    }

    fn publish(&self, prepared: Box<PreparedGenerator>) {
        self.result_generation
            .store(prepared.generation, Ordering::Release);
        let _ = self.ready.force_push(prepared);
        self.status
            .store(AssetLoadStatus::Ready as u8, Ordering::Release);
    }

    fn fail(&self, generation: u64, message: String) {
        self.result_generation.store(generation, Ordering::Release);
        if let Ok(mut error) = self.last_error.write() {
            *error = message;
        }
        self.status
            .store(AssetLoadStatus::Error as u8, Ordering::Release);
    }

    fn mark_ready(&self, generation: u64) {
        self.result_generation.store(generation, Ordering::Release);
        self.status
            .store(AssetLoadStatus::Ready as u8, Ordering::Release);
        if let Ok(mut error) = self.last_error.write() {
            error.clear();
        }
    }

    fn take_ready(&self) -> Option<Box<PreparedGenerator>> {
        self.ready.pop()
    }

    fn retire(&self, prepared: Box<PreparedGenerator>) -> Result<(), Box<PreparedGenerator>> {
        self.retired.push(prepared)
    }

    fn drain_retired(&self) {
        while self.retired.pop().is_some() {}
    }

    pub fn status(&self) -> AssetLoadStatus {
        AssetLoadStatus::from_u8(self.status.load(Ordering::Acquire))
    }

    pub fn last_error(&self) -> String {
        self.last_error.read().map_or_else(
            |_| "capture asset error lock poisoned".to_owned(),
            |error| error.clone(),
        )
    }
}

#[derive(Params)]
pub struct GeneratorParams {
    #[param(
        name = "Send Trim",
        range = "linear(-40, 0)",
        default = 0,
        flags = "automatable"
    )]
    pub send_trim: FloatParam,

    #[skip]
    pub arm_generation: AtomicU64,
    #[skip]
    pub asset_control: Arc<AssetControl>,

    #[meter]
    pub status: MeterSlot,
    #[meter]
    pub progress: MeterSlot,
    #[meter]
    pub exact_send_trim: MeterSlot,
}

pub(crate) use GeneratorParamsParamId as P;

#[derive(Clone, Copy, Debug)]
struct PrepareGeneratorTask {
    generation: u64,
    sample_rate_hz: u32,
    send_trim_db: f32,
}

impl BackgroundTask for PrepareGeneratorTask {
    type Params = GeneratorParams;
    const SERIALIZED: bool = true;

    fn run(self, params: &Self::Params) {
        let control = &params.asset_control;
        control.drain_retired();
        control.begin(self.generation);
        let result = (|| -> Result<PreparedGenerator, String> {
            if self.sample_rate_hz != CAPTURE_SAMPLE_RATE_HZ {
                return Err(format!(
                    "MOT GENERATOR requires 48000 Hz, host is {} Hz",
                    self.sample_rate_hz
                ));
            }
            let program = load_default_capture_program()?;
            let send_gain = db_to_gain(self.send_trim_db);
            let engine = GeneratorEngine::new(program, self.sample_rate_hz, send_gain)
                .map_err(|error| error.to_string())?;
            Ok(PreparedGenerator {
                generation: self.generation,
                engine,
            })
        })();
        match result {
            Ok(prepared) => control.publish(Box::new(prepared)),
            Err(error) => control.fail(self.generation, error),
        }
    }
}

pub struct MotGenerator {
    prepared: Option<Box<PreparedGenerator>>,
    pending_retired: Option<Box<PreparedGenerator>>,
    sample_rate_hz: u32,
    load_generation: u64,
    scheduled_generation: u64,
    observed_arm_generation: u64,
    pending_arm: bool,
    exact_send_trim_db: f32,
}

impl Default for MotGenerator {
    fn default() -> Self {
        Self {
            prepared: None,
            pending_retired: None,
            sample_rate_hz: CAPTURE_SAMPLE_RATE_HZ,
            load_generation: 1,
            scheduled_generation: 0,
            observed_arm_generation: 0,
            pending_arm: false,
            exact_send_trim_db: 0.0,
        }
    }
}

impl MotGenerator {
    #[inline]
    fn retire_pending(&mut self, params: &GeneratorParams) {
        let Some(retired) = self.pending_retired.take() else {
            return;
        };
        if let Err(returned) = params.asset_control.retire(retired) {
            self.pending_retired = Some(returned);
        }
    }

    #[inline]
    fn poll_prepared(&mut self, params: &GeneratorParams, transport_playing: bool) {
        self.retire_pending(params);
        while let Some(prepared) = params.asset_control.take_ready() {
            if prepared.generation != self.load_generation {
                if let Err(returned) = params.asset_control.retire(prepared) {
                    self.pending_retired = Some(returned);
                    break;
                }
                continue;
            }
            if let Some(previous) = self.prepared.replace(prepared)
                && let Err(returned) = params.asset_control.retire(previous)
            {
                self.pending_retired = Some(returned);
            }
            if self.pending_arm
                && let Some(prepared) = &mut self.prepared
            {
                let _ = prepared
                    .engine
                    .set_send_gain_linear(db_to_gain(self.exact_send_trim_db));
                prepared.engine.arm(transport_playing);
                self.pending_arm = false;
            }
        }
    }

    #[inline]
    fn schedule_load(&mut self, params: &GeneratorParams, context: &ProcessContext) {
        if self.sample_rate_hz != CAPTURE_SAMPLE_RATE_HZ
            || self.prepared.is_some()
            || self.scheduled_generation == self.load_generation
        {
            return;
        }
        let Some(spawner) = context.tasks::<PrepareGeneratorTask>() else {
            return;
        };
        let task = PrepareGeneratorTask {
            generation: self.load_generation,
            sample_rate_hz: self.sample_rate_hz,
            send_trim_db: params.send_trim.value(),
        };
        if spawner.try_spawn(task).is_ok() {
            self.scheduled_generation = self.load_generation;
        }
    }

    #[inline]
    fn observe_arm(&mut self, params: &GeneratorParams, transport_playing: bool) {
        let requested = params.arm_generation.load(Ordering::Acquire);
        if requested == self.observed_arm_generation {
            return;
        }
        self.observed_arm_generation = requested;
        self.exact_send_trim_db = params.send_trim.value().clamp(-40.0, 0.0);
        if let Some(prepared) = &mut self.prepared {
            let _ = prepared
                .engine
                .set_send_gain_linear(db_to_gain(self.exact_send_trim_db));
            prepared.engine.arm(transport_playing);
            self.pending_arm = false;
        } else {
            self.pending_arm = true;
        }
    }
}

impl PluginLogic for MotGenerator {
    type Params = GeneratorParams;
    type DspState = Self;

    fn init(params: &Self::Params, _context: &InitContext) -> Self::DspState {
        Self {
            observed_arm_generation: params.arm_generation.load(Ordering::Acquire),
            ..Self::default()
        }
    }

    fn supports_in_place() -> bool {
        false
    }

    fn bus_layouts() -> Vec<BusLayout> {
        vec![BusLayout::mono()]
    }

    fn reset(state: &mut Self::DspState, params: &Self::Params, config: &AudioConfig) {
        state.sample_rate_hz = config.sample_rate.round() as u32;
        state.observed_arm_generation = params.arm_generation.load(Ordering::Acquire);
        state.pending_arm = false;
        if let Some(prepared) = &mut state.prepared {
            prepared.engine.reset_off_thread();
        }
        if state.sample_rate_hz != CAPTURE_SAMPLE_RATE_HZ {
            state.load_generation = state.load_generation.wrapping_add(1).max(1);
            state.scheduled_generation = 0;
            params.asset_control.fail(
                state.load_generation,
                format!(
                    "MOT GENERATOR requires 48000 Hz, host is {} Hz",
                    state.sample_rate_hz
                ),
            );
        } else if let Some(prepared) = &state.prepared {
            params.asset_control.mark_ready(prepared.generation);
        }
    }

    fn process(
        state: &mut Self::DspState,
        params: &Self::Params,
        buffer: &mut AudioBuffer,
        _events: &EventList,
        context: &mut ProcessContext,
    ) -> ProcessStatus {
        state.poll_prepared(params, context.transport.playing);
        state.schedule_load(params, context);
        state.observe_arm(params, context.transport.playing);

        let samples = buffer.num_samples();
        if buffer.num_output_channels() > 0 {
            let output = buffer.output(0);
            output.fill(0.0);
            if state.sample_rate_hz == CAPTURE_SAMPLE_RATE_HZ
                && let Some(prepared) = &mut state.prepared
            {
                prepared.engine.process_block(
                    output,
                    CaptureTransportInfo {
                        playing: context.transport.playing,
                        recording: context.transport.recording,
                        timeline_sample: Some(context.transport.position_samples),
                        loop_active: context.transport.loop_active,
                        discontinuity: false,
                        sample_rate_hz: state.sample_rate_hz,
                    },
                );
            }
        }

        let (status, progress) = generator_meter_state(state, samples);
        context.set_meter(P::Status, f32::from(status) / 7.0);
        context.set_meter(P::Progress, progress);
        context.set_meter(
            P::ExactSendTrim,
            ((state.exact_send_trim_db + 40.0) / 40.0).clamp(0.0, 1.0),
        );
        ProcessStatus::Normal
    }

    fn latency(_state: &Self::DspState) -> u32 {
        0
    }

    fn tail(_state: &Self::DspState) -> u32 {
        0
    }

    fn editor(params: Arc<GeneratorParams>) -> Box<dyn Editor> {
        EguiEditor::with_ui(params, WINDOW_SIZE, GeneratorUi).into_editor()
    }
}

fn generator_meter_state(state: &MotGenerator, _block_samples: usize) -> (u8, f32) {
    if state.sample_rate_hz != CAPTURE_SAMPLE_RATE_HZ {
        return (7, 0.0);
    }
    let Some(prepared) = &state.prepared else {
        return (0, 0.0);
    };
    let total = prepared.engine.total_samples().max(1);
    let progress = prepared.engine.completed_samples() as f32 / total as f32;
    match prepared.engine.state() {
        SplitCaptureState::Idle => (1, 0.0),
        SplitCaptureState::Armed | SplitCaptureState::WaitingForTransport => (2, 0.0),
        SplitCaptureState::PreRoll { .. } => (3, progress),
        SplitCaptureState::Program { .. } => (4, progress),
        SplitCaptureState::Tail { .. } | SplitCaptureState::AlignmentMargin { .. } => (5, progress),
        SplitCaptureState::Ready => (6, 1.0),
        SplitCaptureState::Invalid(_) => (7, progress),
    }
}

pub(crate) fn generator_can_arm(load_status: AssetLoadStatus, normalized_status: f32) -> bool {
    if load_status != AssetLoadStatus::Ready || !normalized_status.is_finite() {
        return false;
    }
    let status = (normalized_status.clamp(0.0, 1.0) * 7.0).round() as u8;
    matches!(status, 0 | 1 | 6 | 7)
}

fn db_to_gain(db: f32) -> f32 {
    10.0_f32.powf(db.clamp(-40.0, 0.0) / 20.0)
}

pub(crate) fn canonical_asset_path_display() -> String {
    capture_asset_path().map_or_else(
        |_| format!("~/Library/Application Support/Plut&Mot/MOT Guitar Plugin/{CAPTURE_ASSET_RELATIVE_PATH}"),
        |path| path.display().to_string(),
    )
}

truce::plugin! {
    logic: MotGenerator,
    params: GeneratorParams,
    tasks: [PrepareGeneratorTask],
}

truce::enable_rt_paranoid!();

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sync_header_is_exact_and_faded() {
        let header = mot_core::capture_asset::generate_sync_header();
        assert_eq!(header.len(), SYNC_HEADER_SAMPLES);
        assert_eq!(header[0], 0.0);
        assert_eq!(header[SYNC_HEADER_SAMPLES - 1], 0.0);
        assert!(header.iter().all(|sample| sample.abs() <= 0.18));
    }

    #[test]
    fn canonical_asset_loads_with_expected_contract() {
        let program = load_default_capture_program().unwrap();
        assert_eq!(program.excitation().len(), CAPTURE_ASSET_SAMPLES);
        assert_eq!(program.sync_header().len(), SYNC_HEADER_SAMPLES);
    }

    #[test]
    fn arm_is_available_before_capture_and_after_complete_or_invalid() {
        for status in [0_u8, 1, 6, 7] {
            assert!(generator_can_arm(
                AssetLoadStatus::Ready,
                f32::from(status) / 7.0
            ));
        }
        for status in 2_u8..=5 {
            assert!(!generator_can_arm(
                AssetLoadStatus::Ready,
                f32::from(status) / 7.0
            ));
        }
        assert!(!generator_can_arm(AssetLoadStatus::Loading, 0.0));
        assert!(!generator_can_arm(AssetLoadStatus::Error, 1.0));
        assert!(!generator_can_arm(AssetLoadStatus::Ready, f32::NAN));
    }

    #[test]
    fn editor_has_a_headless_render_path() {
        let params = Arc::new(GeneratorParams::new());
        let mut editor = EguiEditor::with_ui(Arc::clone(&params), WINDOW_SIZE, GeneratorUi);
        let erased: Arc<dyn truce::params::Params> = params;
        assert_eq!(Editor::size(&editor), WINDOW_SIZE);
        if let Some((_, width, height)) = Editor::screenshot(&mut editor, erased) {
            assert_eq!((width, height), (WINDOW_SIZE.0 * 2, WINDOW_SIZE.1 * 2));
        }
    }
}
