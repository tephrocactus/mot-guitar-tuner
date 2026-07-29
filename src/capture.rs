//! Real-time-safe two-instance capture primitives.
//!
//! A [`CaptureEngine`] belongs to one plugin instance. `Source` emits an
//! immutable, exact capture program while `Return` records the signal arriving
//! on another track. The optional [`CaptureCoordinator`] only exchanges small
//! state values through atomics; audio never crosses between instances and no
//! lock, allocation, or file I/O is needed from the audio callback.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU32, AtomicU64, Ordering};

pub const CAPTURE_SAMPLE_RATE_HZ: u32 = 48_000;
pub const PRE_ROLL_SAMPLES: usize = CAPTURE_SAMPLE_RATE_HZ as usize;
pub const TAIL_SAMPLES: usize = CAPTURE_SAMPLE_RATE_HZ as usize * 2;
/// One second is long enough to expose the sustained peak/RMS of the routed
/// Return without making the safety check cumbersome.
pub const CHECK_LEVEL_DURATION_SAMPLES: usize = CAPTURE_SAMPLE_RATE_HZ as usize;
pub const RETURN_CLIP_THRESHOLD_DBFS: f32 = -1.0;
pub const RETURN_CLIP_THRESHOLD_LINEAR: f32 = 0.891_250_9;
pub const DEFAULT_MAX_ALIGNMENT_LAG_SAMPLES: usize = 24_000;

const NO_INSTANCE: u64 = 0;
const NO_SESSION: u64 = 0;
const NO_ABORT: u8 = 0;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum CaptureRole {
    #[default]
    Normal = 0,
    Source = 1,
    Return = 2,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CaptureSessionId(u64);

impl CaptureSessionId {
    pub fn new(value: u64) -> Option<Self> {
        (value != NO_SESSION).then_some(Self(value))
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum CapturePhase {
    Idle = 0,
    Armed = 1,
    WaitingForTransport = 2,
    PreRoll = 3,
    Capturing = 4,
    Tail = 5,
    Ready = 6,
    Invalid = 7,
}

impl CapturePhase {
    fn from_atomic(value: u8) -> Self {
        match value {
            1 => Self::Armed,
            2 => Self::WaitingForTransport,
            3 => Self::PreRoll,
            4 => Self::Capturing,
            5 => Self::Tail,
            6 => Self::Ready,
            7 => Self::Invalid,
            _ => Self::Idle,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum CaptureInvalidation {
    TransportStopped = 1,
    TimelineSeek = 2,
    LoopActive = 3,
    TransportDiscontinuity = 4,
    SampleRateChanged = 5,
    PairLost = 6,
    ReturnClipped = 7,
    BufferShapeChanged = 8,
    CoordinatorAbort = 9,
}

impl CaptureInvalidation {
    fn from_atomic(value: u8) -> Option<Self> {
        match value {
            1 => Some(Self::TransportStopped),
            2 => Some(Self::TimelineSeek),
            3 => Some(Self::LoopActive),
            4 => Some(Self::TransportDiscontinuity),
            5 => Some(Self::SampleRateChanged),
            6 => Some(Self::PairLost),
            7 => Some(Self::ReturnClipped),
            8 => Some(Self::BufferShapeChanged),
            9 => Some(Self::CoordinatorAbort),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CaptureState {
    Idle,
    Armed,
    WaitingForTransport,
    PreRoll { completed_samples: usize },
    Capturing { completed_samples: usize },
    Tail { completed_samples: usize },
    Ready,
    Invalid(CaptureInvalidation),
}

impl CaptureState {
    #[must_use]
    pub const fn phase(self) -> CapturePhase {
        match self {
            Self::Idle => CapturePhase::Idle,
            Self::Armed => CapturePhase::Armed,
            Self::WaitingForTransport => CapturePhase::WaitingForTransport,
            Self::PreRoll { .. } => CapturePhase::PreRoll,
            Self::Capturing { .. } => CapturePhase::Capturing,
            Self::Tail { .. } => CapturePhase::Tail,
            Self::Ready => CapturePhase::Ready,
            Self::Invalid(_) => CapturePhase::Invalid,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TransportInfo {
    pub playing: bool,
    pub recording: bool,
    pub timeline_sample: Option<i64>,
    pub loop_active: bool,
    pub discontinuity: bool,
    pub sample_rate_hz: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub enum CaptureProgramError {
    EmptySyncHeader,
    EmptyExcitation,
    NonFiniteSample,
    LengthOverflow,
}

impl fmt::Display for CaptureProgramError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySyncHeader => formatter.write_str("the sync header is empty"),
            Self::EmptyExcitation => formatter.write_str("the excitation is empty"),
            Self::NonFiniteSample => {
                formatter.write_str("capture audio contains a non-finite sample")
            }
            Self::LengthOverflow => formatter.write_str("the capture program is too long"),
        }
    }
}

impl std::error::Error for CaptureProgramError {}

/// Immutable audio shared by the Source and Return instances.
///
/// Construction and cloning happen outside the audio callback. The exact
/// emitted program is `sync_header` followed immediately by `excitation`.
#[derive(Clone, Debug)]
pub struct CaptureProgram {
    sync_header: Arc<[f32]>,
    excitation: Arc<[f32]>,
    check_level_probe: Arc<[f32]>,
    program_samples: usize,
    total_capture_samples: usize,
}

impl CaptureProgram {
    pub fn new(
        sync_header: impl Into<Arc<[f32]>>,
        excitation: impl Into<Arc<[f32]>>,
    ) -> Result<Self, CaptureProgramError> {
        let sync_header = sync_header.into();
        let excitation = excitation.into();
        if sync_header.is_empty() {
            return Err(CaptureProgramError::EmptySyncHeader);
        }
        if excitation.is_empty() {
            return Err(CaptureProgramError::EmptyExcitation);
        }
        if sync_header
            .iter()
            .chain(excitation.iter())
            .any(|sample| !sample.is_finite())
        {
            return Err(CaptureProgramError::NonFiniteSample);
        }
        let program_samples = sync_header
            .len()
            .checked_add(excitation.len())
            .ok_or(CaptureProgramError::LengthOverflow)?;
        let total_capture_samples = PRE_ROLL_SAMPLES
            .checked_add(program_samples)
            .and_then(|length| length.checked_add(TAIL_SAMPLES))
            .ok_or(CaptureProgramError::LengthOverflow)?;
        // This allocation and scan happen only while a program is prepared on
        // a loader thread. The audio callback later reads this immutable,
        // exactly-one-second probe without copying or searching.
        let check_level_probe = build_check_level_probe(&excitation).into();
        Ok(Self {
            sync_header,
            excitation,
            check_level_probe,
            program_samples,
            total_capture_samples,
        })
    }

    #[must_use]
    pub fn sync_header(&self) -> &[f32] {
        &self.sync_header
    }

    #[must_use]
    pub fn excitation(&self) -> &[f32] {
        &self.excitation
    }

    /// Precomputed one-second, maximum-energy excitation fragment used by
    /// pair-wide CHECK LEVEL. Short synthetic/test programs are repeated to
    /// the same fixed duration.
    #[must_use]
    pub fn check_level_probe(&self) -> &[f32] {
        &self.check_level_probe
    }

    #[must_use]
    pub const fn program_samples(&self) -> usize {
        self.program_samples
    }

    #[must_use]
    pub const fn total_capture_samples(&self) -> usize {
        self.total_capture_samples
    }

    #[must_use]
    pub const fn sync_start_sample(&self) -> usize {
        PRE_ROLL_SAMPLES
    }

    #[must_use]
    pub fn excitation_start_sample(&self) -> usize {
        PRE_ROLL_SAMPLES + self.sync_header.len()
    }

    #[inline]
    #[must_use]
    pub fn emitted_sample(&self, capture_sample: usize) -> f32 {
        let Some(program_sample) = capture_sample.checked_sub(PRE_ROLL_SAMPLES) else {
            return 0.0;
        };
        if program_sample < self.sync_header.len() {
            self.sync_header[program_sample]
        } else {
            self.excitation
                .get(program_sample - self.sync_header.len())
                .copied()
                .unwrap_or(0.0)
        }
    }

    /// Materializes the exact complete Source stream off the audio thread.
    #[must_use]
    pub fn exact_source_stream(&self) -> Vec<f32> {
        let mut stream = vec![0.0; self.total_capture_samples];
        let sync_start = self.sync_start_sample();
        let excitation_start = self.excitation_start_sample();
        stream[sync_start..excitation_start].copy_from_slice(&self.sync_header);
        stream[excitation_start..excitation_start + self.excitation.len()]
            .copy_from_slice(&self.excitation);
        stream
    }
}

fn build_check_level_probe(excitation: &[f32]) -> Vec<f32> {
    debug_assert!(!excitation.is_empty());
    if excitation.len() < CHECK_LEVEL_DURATION_SAMPLES {
        return (0..CHECK_LEVEL_DURATION_SAMPLES)
            .map(|index| excitation[index % excitation.len()])
            .collect();
    }

    let mut energy = excitation[..CHECK_LEVEL_DURATION_SAMPLES]
        .iter()
        .map(|sample| f64::from(*sample) * f64::from(*sample))
        .sum::<f64>();
    let mut maximum_energy = energy;
    let mut maximum_start = 0;
    for end in CHECK_LEVEL_DURATION_SAMPLES..excitation.len() {
        let entering = f64::from(excitation[end]);
        let leaving = f64::from(excitation[end - CHECK_LEVEL_DURATION_SAMPLES]);
        energy += entering * entering - leaving * leaving;
        let start = end + 1 - CHECK_LEVEL_DURATION_SAMPLES;
        if energy > maximum_energy {
            maximum_energy = energy;
            maximum_start = start;
        }
    }
    excitation[maximum_start..maximum_start + CHECK_LEVEL_DURATION_SAMPLES].to_vec()
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CaptureProgress {
    pub state: CaptureState,
    pub completed_samples: usize,
    pub total_samples: usize,
    pub return_peak_linear: f32,
}

/// Result accumulated by [`CheckLevelMeter`].
///
/// `CHECK LEVEL` is deliberately separate from the capture recorder. It can
/// be run and inspected before arming a pair, and therefore cannot leave a
/// partially-filled training buffer behind.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CheckLevelResult {
    pub measured_samples: usize,
    pub peak_linear: f32,
    pub peak_dbfs: f32,
    pub rms_dbfs: f32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CheckLevelFailure {
    ReturnClipped,
    NonFiniteSample,
    Interrupted,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CheckLevelState {
    Idle,
    Measuring { completed_samples: usize },
    Passed,
    Failed(CheckLevelFailure),
}

/// Pair-wide safety result. Unlike [`CheckLevelState`], this compact form is
/// stored in the lock-free session coordinator and can be observed by both
/// plugin instances.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum SessionCheckLevelState {
    #[default]
    Required = 0,
    Measuring = 1,
    Passed = 2,
    Failed = 3,
}

impl SessionCheckLevelState {
    fn from_atomic(value: u8) -> Self {
        match value {
            1 => Self::Measuring,
            2 => Self::Passed,
            3 => Self::Failed,
            _ => Self::Required,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SessionCheckLevelSnapshot {
    pub generation: u64,
    pub state: SessionCheckLevelState,
    pub progress: f32,
    pub peak_linear: f32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CheckLevelSampleRateError {
    pub received_hz: u32,
}

impl fmt::Display for CheckLevelSampleRateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "CHECK LEVEL requires 48000 Hz, received {} Hz",
            self.received_hz
        )
    }
}

impl std::error::Error for CheckLevelSampleRateError {}

/// Allocation-free Return level gate used before a real capture.
///
/// Feed the actual Return samples into [`Self::process_block`] while the
/// Source/check signal is routed through the intended software or hardware
/// chain. The check passes only after one complete second at 48 kHz and fails
/// immediately on a non-finite sample or a peak strictly above -1 dBFS.
#[derive(Clone, Copy, Debug)]
pub struct CheckLevelMeter {
    state: CheckLevelState,
    measured_samples: usize,
    peak_linear: f32,
    sum_squares: f64,
}

impl Default for CheckLevelMeter {
    fn default() -> Self {
        Self {
            state: CheckLevelState::Idle,
            measured_samples: 0,
            peak_linear: 0.0,
            sum_squares: 0.0,
        }
    }
}

impl CheckLevelMeter {
    pub fn start(&mut self, sample_rate_hz: u32) -> Result<(), CheckLevelSampleRateError> {
        if sample_rate_hz != CAPTURE_SAMPLE_RATE_HZ {
            return Err(CheckLevelSampleRateError {
                received_hz: sample_rate_hz,
            });
        }
        self.measured_samples = 0;
        self.peak_linear = 0.0;
        self.sum_squares = 0.0;
        self.state = CheckLevelState::Measuring {
            completed_samples: 0,
        };
        Ok(())
    }

    #[must_use]
    pub const fn state(&self) -> CheckLevelState {
        self.state
    }

    #[must_use]
    pub fn result(&self) -> CheckLevelResult {
        let rms_linear = if self.measured_samples == 0 {
            0.0
        } else {
            (self.sum_squares / self.measured_samples as f64).sqrt() as f32
        };
        CheckLevelResult {
            measured_samples: self.measured_samples,
            peak_linear: self.peak_linear,
            peak_dbfs: linear_to_dbfs(self.peak_linear),
            rms_dbfs: linear_to_dbfs(rms_linear),
        }
    }

    /// Marks an in-progress check invalid, for example after a transport stop,
    /// seek, loop, dropout, or routing change.
    pub fn interrupt(&mut self) {
        if matches!(self.state, CheckLevelState::Measuring { .. }) {
            self.state = CheckLevelState::Failed(CheckLevelFailure::Interrupted);
        }
    }

    #[inline]
    pub fn process_block(&mut self, input: &[f32]) {
        if !matches!(self.state, CheckLevelState::Measuring { .. }) {
            return;
        }
        for &sample in input {
            if self.measured_samples == CHECK_LEVEL_DURATION_SAMPLES {
                self.state = CheckLevelState::Passed;
                break;
            }
            if !sample.is_finite() {
                self.state = CheckLevelState::Failed(CheckLevelFailure::NonFiniteSample);
                break;
            }
            let magnitude = sample.abs();
            self.peak_linear = self.peak_linear.max(magnitude);
            self.sum_squares += f64::from(sample) * f64::from(sample);
            self.measured_samples += 1;
            if magnitude > RETURN_CLIP_THRESHOLD_LINEAR {
                self.state = CheckLevelState::Failed(CheckLevelFailure::ReturnClipped);
                break;
            }
        }
        if self.measured_samples == CHECK_LEVEL_DURATION_SAMPLES
            && matches!(self.state, CheckLevelState::Measuring { .. })
        {
            self.state = CheckLevelState::Passed;
        } else if matches!(self.state, CheckLevelState::Measuring { .. }) {
            self.state = CheckLevelState::Measuring {
                completed_samples: self.measured_samples,
            };
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CapturePrepareError {
    NormalRole,
    NotPrepared,
    CompletedReturnPending,
    ReturnStorageUnavailable,
    UnsupportedSampleRate(u32),
}

impl fmt::Display for CapturePrepareError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NormalRole => formatter.write_str("the Normal role cannot be armed for capture"),
            Self::NotPrepared => {
                formatter.write_str("a capture program must be prepared before arming")
            }
            Self::CompletedReturnPending => {
                formatter.write_str("take and recycle the completed Return before arming again")
            }
            Self::ReturnStorageUnavailable => {
                formatter.write_str("the preallocated Return handoff buffer is unavailable")
            }
            Self::UnsupportedSampleRate(rate) => {
                write!(formatter, "capture requires 48000 Hz, received {rate} Hz")
            }
        }
    }
}

impl std::error::Error for CapturePrepareError {}

/// Per-instance capture state. All audio storage is allocated by
/// [`CaptureEngine::prepare`] before the engine can be armed.
#[derive(Debug)]
pub struct CaptureEngine {
    role: CaptureRole,
    session_id: CaptureSessionId,
    program: Option<Arc<CaptureProgram>>,
    return_buffer: Vec<f32>,
    spare_return_buffer: Vec<f32>,
    completed_return: Option<CompletedReturn>,
    state: CaptureState,
    capture_position: usize,
    return_peak_linear: f32,
    return_sum_squares: f64,
    last_transport_playing: bool,
    expected_timeline_sample: Option<i64>,
    armed_generation: u64,
}

impl CaptureEngine {
    #[must_use]
    pub fn new(role: CaptureRole, session_id: CaptureSessionId) -> Self {
        Self {
            role,
            session_id,
            program: None,
            return_buffer: Vec::new(),
            spare_return_buffer: Vec::new(),
            completed_return: None,
            state: CaptureState::Idle,
            capture_position: 0,
            return_peak_linear: 0.0,
            return_sum_squares: 0.0,
            last_transport_playing: false,
            expected_timeline_sample: None,
            armed_generation: 0,
        }
    }

    /// Preallocates all audio storage. This must not be called from `process`.
    pub fn prepare(
        &mut self,
        program: Arc<CaptureProgram>,
        sample_rate_hz: u32,
    ) -> Result<(), CapturePrepareError> {
        if self.role == CaptureRole::Normal {
            return Err(CapturePrepareError::NormalRole);
        }
        if sample_rate_hz != CAPTURE_SAMPLE_RATE_HZ {
            return Err(CapturePrepareError::UnsupportedSampleRate(sample_rate_hz));
        }
        if self.role == CaptureRole::Return {
            self.return_buffer
                .resize(program.total_capture_samples(), 0.0);
            self.return_buffer.fill(0.0);
            self.spare_return_buffer
                .resize(program.total_capture_samples(), 0.0);
            self.spare_return_buffer.fill(0.0);
        } else {
            self.return_buffer.clear();
            self.spare_return_buffer.clear();
        }
        self.completed_return = None;
        self.program = Some(program);
        self.reset_capture_state();
        Ok(())
    }

    /// Arms this instance and requires a future stopped-to-playing edge.
    ///
    /// Pass the current host transport state so arming while playback is
    /// already running cannot accidentally start halfway through a block.
    pub fn arm(
        &mut self,
        transport_is_playing: bool,
        generation: u64,
    ) -> Result<(), CapturePrepareError> {
        if self.role == CaptureRole::Normal {
            return Err(CapturePrepareError::NormalRole);
        }
        if self.program.is_none() {
            return Err(CapturePrepareError::NotPrepared);
        }
        if self.role == CaptureRole::Return {
            if self.completed_return.is_some() {
                return Err(CapturePrepareError::CompletedReturnPending);
            }
            let expected_samples = self
                .program
                .as_ref()
                .map_or(0, |program| program.total_capture_samples());
            if self.return_buffer.len() != expected_samples
                || self.spare_return_buffer.len() != expected_samples
            {
                return Err(CapturePrepareError::ReturnStorageUnavailable);
            }
        }
        self.capture_position = 0;
        self.return_peak_linear = 0.0;
        self.return_sum_squares = 0.0;
        self.last_transport_playing = transport_is_playing;
        self.expected_timeline_sample = None;
        self.armed_generation = generation;
        self.state = CaptureState::Armed;
        Ok(())
    }

    #[must_use]
    pub const fn role(&self) -> CaptureRole {
        self.role
    }

    #[must_use]
    pub const fn session_id(&self) -> CaptureSessionId {
        self.session_id
    }

    #[must_use]
    pub const fn state(&self) -> CaptureState {
        self.state
    }

    #[must_use]
    pub const fn armed_generation(&self) -> u64 {
        self.armed_generation
    }

    #[must_use]
    pub fn progress(&self) -> CaptureProgress {
        let total_samples = self
            .program
            .as_ref()
            .map_or(0, |program| program.total_capture_samples());
        CaptureProgress {
            state: self.state,
            completed_samples: self.capture_position,
            total_samples,
            return_peak_linear: self.return_peak_linear,
        }
    }

    #[must_use]
    pub fn return_audio(&self) -> Option<&[f32]> {
        (self.role == CaptureRole::Return && self.state == CaptureState::Ready)
            .then(|| self.completed_return.as_ref().map(CompletedReturn::audio))
            .flatten()
    }

    /// Takes ownership of a completed Return without allocating or copying.
    ///
    /// Call this from the audio thread immediately after `process_block()`
    /// reports `Ready`, then move the result into a bounded worker queue. The
    /// engine already owns the preallocated spare and remains safe, but it
    /// cannot be armed again until [`CaptureEngine::recycle_completed_return`]
    /// returns this storage.
    #[must_use]
    pub fn take_completed_return(&mut self) -> Option<CompletedReturn> {
        self.completed_return.take()
    }

    /// Returns a consumed capture buffer to the engine's preallocated pool.
    ///
    /// This is a control/worker-thread operation between captures.
    pub fn recycle_completed_return(
        &mut self,
        completed: CompletedReturn,
    ) -> Result<(), (ReturnRecycleError, CompletedReturn)> {
        if self.role != CaptureRole::Return {
            return Err((ReturnRecycleError::WrongRole, completed));
        }
        if completed.session_id != self.session_id {
            return Err((ReturnRecycleError::WrongSession, completed));
        }
        let expected_samples = self
            .program
            .as_ref()
            .map_or(0, |program| program.total_capture_samples());
        if completed.audio.len() != expected_samples {
            return Err((
                ReturnRecycleError::WrongLength {
                    expected: expected_samples,
                    actual: completed.audio.len(),
                },
                completed,
            ));
        }
        if !self.spare_return_buffer.is_empty() {
            return Err((ReturnRecycleError::SpareAlreadyAvailable, completed));
        }
        self.spare_return_buffer = completed.audio;
        Ok(())
    }

    #[must_use]
    pub const fn return_peak_linear(&self) -> f32 {
        self.return_peak_linear
    }

    #[must_use]
    pub fn return_peak_dbfs(&self) -> f32 {
        linear_to_dbfs(self.return_peak_linear)
    }

    #[must_use]
    pub fn return_rms_dbfs(&self) -> f32 {
        if self.capture_position == 0 {
            f32::NEG_INFINITY
        } else {
            linear_to_dbfs((self.return_sum_squares / self.capture_position as f64).sqrt() as f32)
        }
    }

    pub fn invalidate(&mut self, reason: CaptureInvalidation) {
        if !matches!(
            self.state,
            CaptureState::Idle | CaptureState::Ready | CaptureState::Invalid(_)
        ) {
            self.state = CaptureState::Invalid(reason);
        }
    }

    /// Processes one mono host block.
    ///
    /// `Source` ignores `input` and emits the exact program. `Return` records
    /// `input` and deliberately emits silence to make accidental hardware
    /// feedback less likely. The `Normal` role is transparent.
    #[inline]
    pub fn process_block(&mut self, input: &[f32], output: &mut [f32], transport: TransportInfo) {
        if input.len() != output.len() {
            self.invalidate(CaptureInvalidation::BufferShapeChanged);
            output.fill(0.0);
            return;
        }
        match self.role {
            CaptureRole::Normal => {
                output.copy_from_slice(input);
                return;
            }
            CaptureRole::Source | CaptureRole::Return => output.fill(0.0),
        }

        if transport.sample_rate_hz != CAPTURE_SAMPLE_RATE_HZ {
            self.invalidate(CaptureInvalidation::SampleRateChanged);
            self.last_transport_playing = transport.playing;
            return;
        }
        if transport.loop_active {
            self.invalidate(CaptureInvalidation::LoopActive);
            self.last_transport_playing = transport.playing;
            return;
        }
        if transport.discontinuity {
            self.invalidate(CaptureInvalidation::TransportDiscontinuity);
            self.last_transport_playing = transport.playing;
            return;
        }
        if matches!(
            self.state,
            CaptureState::Idle | CaptureState::Ready | CaptureState::Invalid(_)
        ) {
            self.last_transport_playing = transport.playing;
            return;
        }

        let rising_edge = !self.last_transport_playing && transport.playing;
        match self.state {
            CaptureState::Armed | CaptureState::WaitingForTransport if !rising_edge => {
                self.state = CaptureState::WaitingForTransport;
                self.last_transport_playing = transport.playing;
                return;
            }
            CaptureState::Armed | CaptureState::WaitingForTransport => {
                self.state = CaptureState::PreRoll {
                    completed_samples: 0,
                };
                self.expected_timeline_sample = transport.timeline_sample;
            }
            _ => {}
        }

        if !transport.playing {
            self.invalidate(CaptureInvalidation::TransportStopped);
            self.last_transport_playing = false;
            return;
        }
        if let (Some(expected), Some(actual)) =
            (self.expected_timeline_sample, transport.timeline_sample)
            && expected != actual
        {
            self.invalidate(CaptureInvalidation::TimelineSeek);
            self.last_transport_playing = true;
            return;
        }

        let Some(program) = self.program.as_ref() else {
            self.invalidate(CaptureInvalidation::BufferShapeChanged);
            self.last_transport_playing = transport.playing;
            return;
        };
        let total_samples = program.total_capture_samples();
        for sample_index in 0..input.len() {
            if self.capture_position >= total_samples {
                self.state = CaptureState::Ready;
                break;
            }

            if self.role == CaptureRole::Source {
                output[sample_index] = program.emitted_sample(self.capture_position);
            } else {
                let sample = input[sample_index];
                self.return_buffer[self.capture_position] = sample;
                let magnitude = sample.abs();
                self.return_peak_linear = self.return_peak_linear.max(magnitude);
                self.return_sum_squares += f64::from(sample) * f64::from(sample);
                if magnitude > RETURN_CLIP_THRESHOLD_LINEAR {
                    debug_assert!(linear_to_dbfs(magnitude) > RETURN_CLIP_THRESHOLD_DBFS);
                    self.state = CaptureState::Invalid(CaptureInvalidation::ReturnClipped);
                    break;
                }
            }

            self.capture_position += 1;
            self.state = state_for_position(program, self.capture_position);
        }

        if self.role == CaptureRole::Return
            && self.state == CaptureState::Ready
            && self.completed_return.is_none()
        {
            self.publish_completed_return();
        }

        if transport.playing {
            self.expected_timeline_sample = transport
                .timeline_sample
                .and_then(|position| position.checked_add(input.len() as i64));
        }
        self.last_transport_playing = transport.playing;
    }

    /// Applies one lock-free coordinator snapshot from the audio callback.
    ///
    /// A new arm generation arms the local engine. A peer/coordinator abort
    /// invalidates an active capture. No allocation or lock is performed.
    #[inline]
    pub fn synchronize(&mut self, binding: &CaptureBinding) -> Result<(), CapturePrepareError> {
        let snapshot = binding.rt_snapshot();
        if snapshot.arm_generation != 0 && snapshot.arm_generation != self.armed_generation {
            self.arm(
                snapshot.transport_was_playing_when_armed,
                snapshot.arm_generation,
            )?;
        }
        if let Some(reason) = snapshot.abort_reason {
            self.invalidate(reason);
        }
        // A local safety/transport failure invalidates the entire logical
        // capture, not merely whichever DAW track happened to observe it.
        // A subsequent arm generation clears this atomic abort before either
        // engine is re-armed.
        if let CaptureState::Invalid(reason) = self.state {
            binding.abort_pair(reason);
        }
        binding.publish_phase(self.state.phase());
        Ok(())
    }

    fn reset_capture_state(&mut self) {
        self.capture_position = 0;
        self.return_peak_linear = 0.0;
        self.return_sum_squares = 0.0;
        self.last_transport_playing = false;
        self.expected_timeline_sample = None;
        self.armed_generation = 0;
        self.state = CaptureState::Idle;
    }

    /// Swaps a completed recording into an owned message. Both vectors were
    /// allocated by `prepare`; `Vec::new()` itself performs no allocation.
    #[inline]
    fn publish_completed_return(&mut self) {
        debug_assert_eq!(self.role, CaptureRole::Return);
        debug_assert!(self.completed_return.is_none());
        std::mem::swap(&mut self.return_buffer, &mut self.spare_return_buffer);
        let audio = std::mem::take(&mut self.spare_return_buffer);
        self.completed_return = Some(CompletedReturn {
            session_id: self.session_id,
            generation: self.armed_generation,
            peak_linear: self.return_peak_linear,
            rms_dbfs: self.return_rms_dbfs(),
            audio,
        });
    }
}

#[derive(Clone, Debug)]
pub struct CompletedReturn {
    pub session_id: CaptureSessionId,
    pub generation: u64,
    pub peak_linear: f32,
    pub rms_dbfs: f32,
    audio: Vec<f32>,
}

impl CompletedReturn {
    #[must_use]
    pub fn audio(&self) -> &[f32] {
        &self.audio
    }

    #[must_use]
    pub fn into_audio(self) -> Vec<f32> {
        self.audio
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReturnRecycleError {
    WrongRole,
    WrongSession,
    WrongLength { expected: usize, actual: usize },
    SpareAlreadyAvailable,
}

fn state_for_position(program: &CaptureProgram, position: usize) -> CaptureState {
    if position >= program.total_capture_samples() {
        CaptureState::Ready
    } else if position < PRE_ROLL_SAMPLES {
        CaptureState::PreRoll {
            completed_samples: position,
        }
    } else if position < PRE_ROLL_SAMPLES + program.program_samples() {
        CaptureState::Capturing {
            completed_samples: position - PRE_ROLL_SAMPLES,
        }
    } else {
        CaptureState::Tail {
            completed_samples: position - PRE_ROLL_SAMPLES - program.program_samples(),
        }
    }
}

#[must_use]
pub fn linear_to_dbfs(linear: f32) -> f32 {
    if linear <= 0.0 {
        f32::NEG_INFINITY
    } else {
        20.0 * linear.log10()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AlignmentConfig {
    pub maximum_lag_samples: usize,
    pub minimum_normalized_correlation: f64,
}

impl Default for AlignmentConfig {
    fn default() -> Self {
        Self {
            maximum_lag_samples: DEFAULT_MAX_ALIGNMENT_LAG_SAMPLES,
            minimum_normalized_correlation: 0.35,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AlignmentResult {
    pub integer_latency_samples: i64,
    pub fractional_latency_samples: f64,
    pub normalized_correlation: f64,
    pub polarity_inverted: bool,
    pub sync_start_in_return: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum AlignmentError {
    ReturnTooShort,
    SyncHeaderTooShort,
    NoFiniteCandidate,
    CorrelationTooLow { measured: f64, required: f64 },
}

impl fmt::Display for AlignmentError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ReturnTooShort => {
                formatter.write_str("the return capture is shorter than the sync header")
            }
            Self::SyncHeaderTooShort => {
                formatter.write_str("the sync header needs at least three samples")
            }
            Self::NoFiniteCandidate => {
                formatter.write_str("no finite alignment candidate was found")
            }
            Self::CorrelationTooLow { measured, required } => write!(
                formatter,
                "sync correlation {measured:.3} is below the required {required:.3}"
            ),
        }
    }
}

impl std::error::Error for AlignmentError {}

/// Finds round-trip latency from the sync header, then refines the peak with a
/// three-point parabolic interpolation. This is intentionally offline work.
pub fn measure_alignment(
    program: &CaptureProgram,
    return_audio: &[f32],
    config: AlignmentConfig,
) -> Result<AlignmentResult, AlignmentError> {
    let sync = program.sync_header();
    if sync.len() < 3 {
        return Err(AlignmentError::SyncHeaderTooShort);
    }
    if return_audio.len() < sync.len() {
        return Err(AlignmentError::ReturnTooShort);
    }

    let expected = program.sync_start_sample();
    let last_start = return_audio.len() - sync.len();
    let first = expected.saturating_sub(config.maximum_lag_samples);
    let last = expected
        .saturating_add(config.maximum_lag_samples)
        .min(last_start);
    if first > last {
        return Err(AlignmentError::ReturnTooShort);
    }

    let mut best_start = first;
    let mut best_signed_correlation = f64::NEG_INFINITY;
    let mut best_correlation_magnitude = f64::NEG_INFINITY;
    for start in first..=last {
        let correlation = normalized_correlation(sync, &return_audio[start..start + sync.len()]);
        let magnitude = correlation.abs();
        if correlation.is_finite() && magnitude > best_correlation_magnitude {
            best_start = start;
            best_signed_correlation = correlation;
            best_correlation_magnitude = magnitude;
        }
    }
    if !best_correlation_magnitude.is_finite() {
        return Err(AlignmentError::NoFiniteCandidate);
    }
    if best_correlation_magnitude < config.minimum_normalized_correlation {
        return Err(AlignmentError::CorrelationTooLow {
            measured: best_correlation_magnitude,
            required: config.minimum_normalized_correlation,
        });
    }

    let center = best_correlation_magnitude;
    let left = (best_start > first).then(|| {
        normalized_correlation(
            sync,
            &return_audio[best_start - 1..best_start - 1 + sync.len()],
        )
        .abs()
    });
    let right = (best_start < last).then(|| {
        normalized_correlation(
            sync,
            &return_audio[best_start + 1..best_start + 1 + sync.len()],
        )
        .abs()
    });
    let fractional_peak = match (left, right) {
        (Some(left), Some(right)) => {
            let denominator = left - 2.0 * center + right;
            if denominator.abs() > 1.0e-12 {
                (0.5 * (left - right) / denominator).clamp(-0.5, 0.5)
            } else {
                0.0
            }
        }
        _ => 0.0,
    };
    let integer_latency = best_start as i64 - expected as i64;
    Ok(AlignmentResult {
        integer_latency_samples: integer_latency,
        fractional_latency_samples: integer_latency as f64 + fractional_peak,
        normalized_correlation: best_correlation_magnitude,
        polarity_inverted: best_signed_correlation.is_sign_negative(),
        sync_start_in_return: best_start as f64 + fractional_peak,
    })
}

fn normalized_correlation(left: &[f32], right: &[f32]) -> f64 {
    debug_assert_eq!(left.len(), right.len());
    let count = left.len() as f64;
    let left_mean = left.iter().map(|sample| f64::from(*sample)).sum::<f64>() / count;
    let right_mean = right.iter().map(|sample| f64::from(*sample)).sum::<f64>() / count;
    let mut dot = 0.0;
    let mut left_energy = 0.0;
    let mut right_energy = 0.0;
    for (&left, &right) in left.iter().zip(right) {
        let left = f64::from(left) - left_mean;
        let right = f64::from(right) - right_mean;
        dot += left * right;
        left_energy += left * left;
        right_energy += right * right;
    }
    let denominator = (left_energy * right_energy).sqrt();
    if denominator > f64::MIN_POSITIVE {
        dot / denominator
    } else {
        f64::NEG_INFINITY
    }
}

/// Extracts the returned excitation after removing measured integer and
/// fractional round-trip latency. Linear interpolation is adequate here
/// because this is an offline dataset preparation step, not live DSP.
pub fn extract_aligned_excitation(
    program: &CaptureProgram,
    return_audio: &[f32],
    alignment: AlignmentResult,
) -> Result<Vec<f32>, AlignmentError> {
    let first = program.excitation_start_sample() as f64 + alignment.fractional_latency_samples;
    let last = first + program.excitation().len().saturating_sub(1) as f64;
    if first < 0.0 || last.ceil() as usize >= return_audio.len() {
        return Err(AlignmentError::ReturnTooShort);
    }
    let mut aligned = Vec::with_capacity(program.excitation().len());
    for index in 0..program.excitation().len() {
        let source_position = first + index as f64;
        let lower = source_position.floor() as usize;
        let fraction = (source_position - lower as f64) as f32;
        let lower_sample = return_audio[lower];
        let upper_sample = return_audio.get(lower + 1).copied().unwrap_or(lower_sample);
        aligned.push(lower_sample + (upper_sample - lower_sample) * fraction);
    }
    Ok(aligned)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CaptureTarget {
    SoftwarePluginChain,
    FullAmpUnfilteredLoad,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CalibrationStatus {
    Uncalibrated,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HardwareCaptureMetadata {
    pub target: CaptureTarget,
    pub calibration_status: CalibrationStatus,
    pub input_level_dbu: Option<f32>,
    pub output_level_dbu: Option<f32>,
    pub sample_rate_hz: u32,
    pub source_send_trim_db: f32,
    pub measured_latency_samples: Option<f64>,
    pub return_peak_dbfs: Option<f32>,
    pub return_rms_dbfs: Option<f32>,
    pub excitation_hash: String,
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

impl HardwareCaptureMetadata {
    #[must_use]
    pub fn uncalibrated_full_amp() -> Self {
        Self {
            target: CaptureTarget::FullAmpUnfilteredLoad,
            calibration_status: CalibrationStatus::Uncalibrated,
            input_level_dbu: None,
            output_level_dbu: None,
            sample_rate_hz: CAPTURE_SAMPLE_RATE_HZ,
            source_send_trim_db: 0.0,
            measured_latency_samples: None,
            return_peak_dbfs: None,
            return_rms_dbfs: None,
            excitation_hash: String::new(),
            amplifier: String::new(),
            amplifier_channel: String::new(),
            control_positions: String::new(),
            interface_output: String::new(),
            interface_input: String::new(),
            reamp_box: String::new(),
            reactive_load: String::new(),
            load_impedance_ohms: None,
            return_gain_note: String::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CoordinatorSnapshot {
    pub arm_generation: u64,
    pub transport_was_playing_when_armed: bool,
    pub peer_phase: CapturePhase,
    pub abort_reason: Option<CaptureInvalidation>,
}

#[derive(Debug)]
struct SessionSlot {
    session_id: AtomicU64,
    source_instance: AtomicU64,
    return_instance: AtomicU64,
    arm_generation: AtomicU64,
    arm_transport_playing: AtomicU8,
    abort_reason: AtomicU8,
    source_phase: AtomicU8,
    return_phase: AtomicU8,
    source_send_trim_bits: AtomicU32,
    check_level_generation: AtomicU64,
    check_level_state: AtomicU8,
    check_level_progress_bits: AtomicU32,
    check_level_peak_bits: AtomicU32,
}

impl SessionSlot {
    fn new() -> Self {
        Self {
            session_id: AtomicU64::new(NO_SESSION),
            source_instance: AtomicU64::new(NO_INSTANCE),
            return_instance: AtomicU64::new(NO_INSTANCE),
            arm_generation: AtomicU64::new(0),
            arm_transport_playing: AtomicU8::new(0),
            abort_reason: AtomicU8::new(NO_ABORT),
            source_phase: AtomicU8::new(CapturePhase::Idle as u8),
            return_phase: AtomicU8::new(CapturePhase::Idle as u8),
            source_send_trim_bits: AtomicU32::new((-20.0_f32).to_bits()),
            check_level_generation: AtomicU64::new(0),
            check_level_state: AtomicU8::new(SessionCheckLevelState::Required as u8),
            check_level_progress_bits: AtomicU32::new(0.0_f32.to_bits()),
            check_level_peak_bits: AtomicU32::new(0.0_f32.to_bits()),
        }
    }

    /// Invalidates any pass and advances the token so a stale Return callback
    /// cannot republish the result of a previous pair.
    fn reset_check_level(&self) -> u64 {
        self.check_level_state
            .store(SessionCheckLevelState::Required as u8, Ordering::Release);
        self.check_level_progress_bits
            .store(0.0_f32.to_bits(), Ordering::Release);
        self.check_level_peak_bits
            .store(0.0_f32.to_bits(), Ordering::Release);
        next_generation(&self.check_level_generation)
    }

    fn role_instance(&self, role: CaptureRole) -> Option<&AtomicU64> {
        match role {
            CaptureRole::Source => Some(&self.source_instance),
            CaptureRole::Return => Some(&self.return_instance),
            CaptureRole::Normal => None,
        }
    }

    fn role_phase(&self, role: CaptureRole) -> Option<&AtomicU8> {
        match role {
            CaptureRole::Source => Some(&self.source_phase),
            CaptureRole::Return => Some(&self.return_phase),
            CaptureRole::Normal => None,
        }
    }

    fn peer_phase(&self, role: CaptureRole) -> Option<&AtomicU8> {
        match role {
            CaptureRole::Source => Some(&self.return_phase),
            CaptureRole::Return => Some(&self.source_phase),
            CaptureRole::Normal => None,
        }
    }
}

/// Fixed-capacity, lock-free session registry for instances in one host
/// process. Binding/unbinding belongs on the UI or loader thread. The methods
/// called by [`CaptureEngine::synchronize`] use only atomic loads/stores.
#[derive(Debug)]
pub struct CaptureCoordinator {
    slots: Box<[SessionSlot]>,
}

impl CaptureCoordinator {
    #[must_use]
    pub fn with_capacity(maximum_sessions: usize) -> Arc<Self> {
        let slots = (0..maximum_sessions)
            .map(|_| SessionSlot::new())
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Arc::new(Self { slots })
    }

    pub fn bind(
        self: &Arc<Self>,
        session_id: CaptureSessionId,
        role: CaptureRole,
        instance_id: u64,
    ) -> Result<CaptureBinding, CoordinatorError> {
        if role == CaptureRole::Normal {
            return Err(CoordinatorError::NormalRole);
        }
        if instance_id == NO_INSTANCE {
            return Err(CoordinatorError::InvalidInstanceId);
        }

        let mut candidate = None;
        for (index, slot) in self.slots.iter().enumerate() {
            let current = slot.session_id.load(Ordering::Acquire);
            if current == session_id.get() {
                candidate = Some(index);
                break;
            }
            if current == NO_SESSION
                && slot
                    .session_id
                    .compare_exchange(
                        NO_SESSION,
                        session_id.get(),
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    )
                    .is_ok()
            {
                candidate = Some(index);
                break;
            }
        }
        let slot_index = candidate.ok_or(CoordinatorError::CapacityExhausted)?;
        let slot = &self.slots[slot_index];
        let role_instance = slot.role_instance(role).expect("normal role rejected");
        match role_instance.compare_exchange(
            NO_INSTANCE,
            instance_id,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => {
                // The physical/logical pair changed. A level check belongs to
                // exactly one pair and must never survive rebinding.
                slot.reset_check_level();
                Ok(CaptureBinding {
                    coordinator: Arc::clone(self),
                    slot_index,
                    session_id,
                    role,
                    instance_id,
                })
            }
            Err(existing) if existing == instance_id => Ok(CaptureBinding {
                coordinator: Arc::clone(self),
                slot_index,
                session_id,
                role,
                instance_id,
            }),
            Err(_) => Err(CoordinatorError::RoleAlreadyBound(role)),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CoordinatorError {
    NormalRole,
    InvalidInstanceId,
    CapacityExhausted,
    RoleAlreadyBound(CaptureRole),
    PairIncomplete,
    CheckLevelRequiresReturn,
    CheckLevelNotPassed(SessionCheckLevelState),
    CaptureActive,
}

impl fmt::Display for CoordinatorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NormalRole => {
                formatter.write_str("Normal instances do not join capture sessions")
            }
            Self::InvalidInstanceId => formatter.write_str("instance id zero is reserved"),
            Self::CapacityExhausted => formatter.write_str("capture session capacity is exhausted"),
            Self::RoleAlreadyBound(role) => write!(formatter, "the {role:?} role is already bound"),
            Self::PairIncomplete => {
                formatter.write_str("the capture session needs one Source and one Return")
            }
            Self::CheckLevelRequiresReturn => {
                formatter.write_str("CHECK LEVEL must be started by the Return instance")
            }
            Self::CheckLevelNotPassed(state) => {
                write!(
                    formatter,
                    "capture requires a passed Return CHECK LEVEL; current state is {state:?}"
                )
            }
            Self::CaptureActive => {
                formatter.write_str("CHECK LEVEL cannot start while capture is active")
            }
        }
    }
}

impl std::error::Error for CoordinatorError {}

fn next_generation(counter: &AtomicU64) -> u64 {
    let mut generation = counter.fetch_add(1, Ordering::AcqRel).wrapping_add(1);
    if generation == 0 {
        generation = 1;
        counter.store(generation, Ordering::Release);
    }
    generation
}

fn capture_phase_is_active(phase: CapturePhase) -> bool {
    matches!(
        phase,
        CapturePhase::Armed
            | CapturePhase::WaitingForTransport
            | CapturePhase::PreRoll
            | CapturePhase::Capturing
            | CapturePhase::Tail
    )
}

#[derive(Debug)]
pub struct CaptureBinding {
    coordinator: Arc<CaptureCoordinator>,
    slot_index: usize,
    session_id: CaptureSessionId,
    role: CaptureRole,
    instance_id: u64,
}

impl CaptureBinding {
    fn slot(&self) -> &SessionSlot {
        &self.coordinator.slots[self.slot_index]
    }

    #[must_use]
    pub fn pair_complete(&self) -> bool {
        let slot = self.slot();
        slot.source_instance.load(Ordering::Acquire) != NO_INSTANCE
            && slot.return_instance.load(Ordering::Acquire) != NO_INSTANCE
    }

    /// Arms both members through a monotonically increasing generation.
    pub fn arm_pair(&self, transport_is_playing: bool) -> Result<u64, CoordinatorError> {
        if !self.pair_complete() {
            return Err(CoordinatorError::PairIncomplete);
        }
        let slot = self.slot();
        let check_level_state =
            SessionCheckLevelState::from_atomic(slot.check_level_state.load(Ordering::Acquire));
        if check_level_state != SessionCheckLevelState::Passed {
            return Err(CoordinatorError::CheckLevelNotPassed(check_level_state));
        }
        slot.abort_reason.store(NO_ABORT, Ordering::Release);
        slot.arm_transport_playing
            .store(u8::from(transport_is_playing), Ordering::Release);
        slot.source_phase
            .store(CapturePhase::Armed as u8, Ordering::Release);
        slot.return_phase
            .store(CapturePhase::Armed as u8, Ordering::Release);
        Ok(next_generation(&slot.arm_generation))
    }

    pub fn abort_pair(&self, reason: CaptureInvalidation) {
        self.slot()
            .abort_reason
            .store(reason as u8, Ordering::Release);
    }

    /// Explicitly requires a new check, for example after host reset or a
    /// routing change. Advancing the generation rejects stale Return results.
    pub fn invalidate_check_level(&self) {
        self.slot().reset_check_level();
    }

    /// Starts a new pair-wide Return safety check. This is safe to call from
    /// the Return audio callback after observing the UI trigger generation.
    pub fn request_check_level(&self) -> Result<u64, CoordinatorError> {
        if self.role != CaptureRole::Return {
            return Err(CoordinatorError::CheckLevelRequiresReturn);
        }
        if !self.pair_complete() {
            return Err(CoordinatorError::PairIncomplete);
        }
        let slot = self.slot();
        let source_phase = CapturePhase::from_atomic(slot.source_phase.load(Ordering::Acquire));
        let return_phase = CapturePhase::from_atomic(slot.return_phase.load(Ordering::Acquire));
        if capture_phase_is_active(source_phase) || capture_phase_is_active(return_phase) {
            return Err(CoordinatorError::CaptureActive);
        }
        slot.check_level_progress_bits
            .store(0.0_f32.to_bits(), Ordering::Release);
        slot.check_level_peak_bits
            .store(0.0_f32.to_bits(), Ordering::Release);
        slot.check_level_state
            .store(SessionCheckLevelState::Measuring as u8, Ordering::Release);
        Ok(next_generation(&slot.check_level_generation))
    }

    /// Publishes Return measurement progress for the current generation.
    /// Stale callbacks are rejected after a new check or pair change.
    #[inline]
    pub fn publish_check_level(
        &self,
        generation: u64,
        state: SessionCheckLevelState,
        progress: f32,
        peak_linear: f32,
    ) -> bool {
        if self.role != CaptureRole::Return {
            return false;
        }
        let slot = self.slot();
        if generation == 0 || slot.check_level_generation.load(Ordering::Acquire) != generation {
            return false;
        }
        slot.check_level_progress_bits
            .store(progress.clamp(0.0, 1.0).to_bits(), Ordering::Release);
        slot.check_level_peak_bits
            .store(peak_linear.clamp(0.0, 1.0).to_bits(), Ordering::Release);
        slot.check_level_state.store(state as u8, Ordering::Release);
        true
    }

    #[inline]
    #[must_use]
    pub fn check_level_snapshot(&self) -> SessionCheckLevelSnapshot {
        let slot = self.slot();
        SessionCheckLevelSnapshot {
            generation: slot.check_level_generation.load(Ordering::Acquire),
            state: SessionCheckLevelState::from_atomic(
                slot.check_level_state.load(Ordering::Acquire),
            ),
            progress: f32::from_bits(slot.check_level_progress_bits.load(Ordering::Acquire))
                .clamp(0.0, 1.0),
            peak_linear: f32::from_bits(slot.check_level_peak_bits.load(Ordering::Acquire))
                .clamp(0.0, 1.0),
        }
    }

    /// Audio-thread-safe atomic snapshot.
    #[inline]
    #[must_use]
    pub fn rt_snapshot(&self) -> CoordinatorSnapshot {
        let slot = self.slot();
        let peer_phase = slot
            .peer_phase(self.role)
            .map_or(CapturePhase::Idle, |phase| {
                CapturePhase::from_atomic(phase.load(Ordering::Acquire))
            });
        CoordinatorSnapshot {
            arm_generation: slot.arm_generation.load(Ordering::Acquire),
            transport_was_playing_when_armed: slot.arm_transport_playing.load(Ordering::Acquire)
                != 0,
            peer_phase,
            abort_reason: CaptureInvalidation::from_atomic(
                slot.abort_reason.load(Ordering::Acquire),
            ),
        }
    }

    /// Audio-thread-safe phase publication.
    #[inline]
    pub fn publish_phase(&self, phase: CapturePhase) {
        if let Some(destination) = self.slot().role_phase(self.role) {
            destination.store(phase as u8, Ordering::Release);
        }
    }

    /// Publishes the exact gain used by the Source instance. The Return
    /// instance reads this value when it hands the aligned capture to the
    /// trainer, so the trainer's input is the exact emitted excitation rather
    /// than an independently configured approximation.
    #[inline]
    pub fn publish_source_send_trim_db(&self, trim_db: f32) {
        if self.role == CaptureRole::Source {
            let slot = self.slot();
            let new_bits = trim_db.clamp(-40.0, 0.0).to_bits();
            let previous_bits = slot.source_send_trim_bits.swap(new_bits, Ordering::AcqRel);
            if previous_bits != new_bits {
                // The safety result is tied to the exact level that reached
                // the routed chain. It cannot authorize capture at a later,
                // potentially much hotter Source setting.
                slot.reset_check_level();
            }
        }
    }

    /// Returns the Source send trim shared by the capture pair.
    #[inline]
    #[must_use]
    pub fn source_send_trim_db(&self) -> f32 {
        f32::from_bits(self.slot().source_send_trim_bits.load(Ordering::Acquire)).clamp(-40.0, 0.0)
    }
}

impl Drop for CaptureBinding {
    fn drop(&mut self) {
        let slot = self.slot();
        if let Some(role_instance) = slot.role_instance(self.role) {
            let _ = role_instance.compare_exchange(
                self.instance_id,
                NO_INSTANCE,
                Ordering::AcqRel,
                Ordering::Acquire,
            );
        }
        if let Some(role_phase) = slot.role_phase(self.role) {
            role_phase.store(CapturePhase::Idle as u8, Ordering::Release);
        }
        slot.reset_check_level();
        slot.abort_reason
            .store(CaptureInvalidation::PairLost as u8, Ordering::Release);

        if slot.source_instance.load(Ordering::Acquire) == NO_INSTANCE
            && slot.return_instance.load(Ordering::Acquire) == NO_INSTANCE
        {
            slot.arm_generation.store(0, Ordering::Release);
            slot.arm_transport_playing.store(0, Ordering::Release);
            slot.abort_reason.store(NO_ABORT, Ordering::Release);
            slot.source_send_trim_bits
                .store((-20.0_f32).to_bits(), Ordering::Release);
            let _ = slot.session_id.compare_exchange(
                self.session_id.get(),
                NO_SESSION,
                Ordering::AcqRel,
                Ordering::Acquire,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session_id() -> CaptureSessionId {
        CaptureSessionId::new(42).unwrap()
    }

    fn test_program() -> Arc<CaptureProgram> {
        let sync = (0..127)
            .map(|index| {
                let bit = ((index * 73 + 19) % 127) & 1;
                if bit == 0 { -0.35 } else { 0.35 }
            })
            .collect::<Vec<_>>();
        let excitation = (0..513)
            .map(|index| ((index as f32 * 0.071).sin() * 0.4).clamp(-0.5, 0.5))
            .collect::<Vec<_>>();
        Arc::new(CaptureProgram::new(sync, excitation).unwrap())
    }

    fn transport(playing: bool, position: i64) -> TransportInfo {
        TransportInfo {
            playing,
            recording: false,
            timeline_sample: Some(position),
            loop_active: false,
            discontinuity: false,
            sample_rate_hz: CAPTURE_SAMPLE_RATE_HZ,
        }
    }

    fn pass_pair_check(return_binding: &CaptureBinding) -> u64 {
        let generation = return_binding.request_check_level().unwrap();
        assert!(return_binding.publish_check_level(
            generation,
            SessionCheckLevelState::Passed,
            1.0,
            0.5,
        ));
        generation
    }

    #[test]
    fn source_emits_the_exact_program_across_odd_block_sizes() {
        let program = test_program();
        let expected = program.exact_source_stream();
        let mut source = CaptureEngine::new(CaptureRole::Source, session_id());
        source
            .prepare(Arc::clone(&program), CAPTURE_SAMPLE_RATE_HZ)
            .unwrap();
        source.arm(false, 1).unwrap();

        let mut emitted = Vec::new();
        let mut timeline = 1_000_i64;
        let mut size_index = 0;
        let sizes = [1, 7, 16, 31, 64, 257];
        while emitted.len() < expected.len() {
            let block_size = sizes[size_index % sizes.len()].min(expected.len() - emitted.len());
            size_index += 1;
            let input = vec![0.75; block_size];
            let mut output = vec![f32::NAN; block_size];
            source.process_block(&input, &mut output, transport(true, timeline));
            emitted.extend_from_slice(&output);
            timeline += block_size as i64;
        }
        assert_eq!(source.state(), CaptureState::Ready);
        assert_eq!(emitted, expected);
    }

    #[test]
    fn return_records_a_delayed_hardware_simulation_and_aligns_it() {
        let program = test_program();
        let source = program.exact_source_stream();
        let integer_delay = 137_usize;
        let fractional_delay = 0.25_f32;
        let mut hardware_return = vec![0.0; source.len()];
        for (output_index, output) in hardware_return.iter_mut().enumerate() {
            let source_position = output_index as f32 - integer_delay as f32 - fractional_delay;
            if source_position >= 0.0 {
                let lower = source_position.floor() as usize;
                let fraction = source_position - lower as f32;
                let a = source.get(lower).copied().unwrap_or(0.0);
                let b = source.get(lower + 1).copied().unwrap_or(a);
                // Mild nonlinear coloration keeps the sync realistic.
                let delayed = a + (b - a) * fraction;
                *output = (delayed * 1.15).tanh() * 0.55;
            }
        }

        let mut capture = CaptureEngine::new(CaptureRole::Return, session_id());
        capture
            .prepare(Arc::clone(&program), CAPTURE_SAMPLE_RATE_HZ)
            .unwrap();
        capture.arm(false, 1).unwrap();
        let mut timeline = 50_000_i64;
        for block in hardware_return.chunks(257) {
            let mut silent_output = vec![1.0; block.len()];
            capture.process_block(block, &mut silent_output, transport(true, timeline));
            assert!(silent_output.iter().all(|sample| *sample == 0.0));
            timeline += block.len() as i64;
        }
        assert_eq!(capture.state(), CaptureState::Ready);
        let alignment = measure_alignment(
            &program,
            capture.return_audio().unwrap(),
            AlignmentConfig {
                maximum_lag_samples: 512,
                minimum_normalized_correlation: 0.5,
            },
        )
        .unwrap();
        assert_eq!(alignment.integer_latency_samples, integer_delay as i64);
        assert!(!alignment.polarity_inverted);
        assert!(
            (alignment.fractional_latency_samples - integer_delay as f64 - 0.25).abs() < 0.35,
            "{alignment:?}"
        );
        let aligned =
            extract_aligned_excitation(&program, capture.return_audio().unwrap(), alignment)
                .unwrap();
        assert_eq!(aligned.len(), program.excitation().len());
    }

    #[test]
    fn completed_return_handoff_swaps_preallocated_storage_without_copying() {
        let program = test_program();
        let mut capture = CaptureEngine::new(CaptureRole::Return, session_id());
        capture
            .prepare(Arc::clone(&program), CAPTURE_SAMPLE_RATE_HZ)
            .unwrap();
        let recording_pointer = capture.return_buffer.as_ptr();
        let spare_pointer = capture.spare_return_buffer.as_ptr();
        capture.arm(false, 77).unwrap();

        let silent_return = vec![0.0; program.total_capture_samples()];
        let mut timeline = 0_i64;
        for block in silent_return.chunks(257) {
            let mut output = vec![1.0; block.len()];
            capture.process_block(block, &mut output, transport(true, timeline));
            timeline += block.len() as i64;
        }
        assert_eq!(capture.state(), CaptureState::Ready);
        assert_eq!(capture.return_buffer.as_ptr(), spare_pointer);
        assert_eq!(capture.return_audio().unwrap().as_ptr(), recording_pointer);

        let completed = capture.take_completed_return().unwrap();
        assert_eq!(completed.audio().as_ptr(), recording_pointer);
        assert_eq!(completed.generation, 77);
        assert_eq!(
            capture.arm(false, 78),
            Err(CapturePrepareError::ReturnStorageUnavailable)
        );
        capture.recycle_completed_return(completed).unwrap();
        capture.arm(false, 78).unwrap();
    }

    #[test]
    fn alignment_accepts_a_polarity_inverted_amplifier_return() {
        let program = test_program();
        let source = program.exact_source_stream();
        let delay = 91;
        let mut returned = vec![0.0; source.len()];
        for index in delay..returned.len() {
            returned[index] = -source[index - delay] * 0.7;
        }
        let alignment = measure_alignment(
            &program,
            &returned,
            AlignmentConfig {
                maximum_lag_samples: 256,
                minimum_normalized_correlation: 0.8,
            },
        )
        .unwrap();
        assert_eq!(alignment.integer_latency_samples, delay as i64);
        assert!(alignment.polarity_inverted);
        assert!(alignment.normalized_correlation > 0.99);
    }

    #[test]
    fn return_aborts_above_minus_one_dbfs() {
        let program = test_program();
        let mut capture = CaptureEngine::new(CaptureRole::Return, session_id());
        capture.prepare(program, CAPTURE_SAMPLE_RATE_HZ).unwrap();
        capture.arm(false, 1).unwrap();
        let input = [0.0, RETURN_CLIP_THRESHOLD_LINEAR + 0.001, 0.0];
        let mut output = [1.0; 3];
        capture.process_block(&input, &mut output, transport(true, 0));
        assert_eq!(
            capture.state(),
            CaptureState::Invalid(CaptureInvalidation::ReturnClipped)
        );
        assert_eq!(output, [0.0; 3]);
    }

    #[test]
    fn check_level_requires_one_safe_second_at_48_khz() {
        let mut check = CheckLevelMeter::default();
        assert_eq!(
            check.start(44_100),
            Err(CheckLevelSampleRateError {
                received_hz: 44_100
            })
        );
        check.start(CAPTURE_SAMPLE_RATE_HZ).unwrap();

        let safe_level = 0.5_f32;
        let input = vec![safe_level; CHECK_LEVEL_DURATION_SAMPLES + 17];
        for block in input.chunks(257) {
            check.process_block(block);
        }
        assert_eq!(check.state(), CheckLevelState::Passed);
        let result = check.result();
        assert_eq!(result.measured_samples, CHECK_LEVEL_DURATION_SAMPLES);
        assert_eq!(result.peak_linear, safe_level);
        assert!((result.peak_dbfs - linear_to_dbfs(safe_level)).abs() < 1.0e-6);
        assert!((result.rms_dbfs - linear_to_dbfs(safe_level)).abs() < 1.0e-6);
    }

    #[test]
    fn check_level_accepts_minus_one_dbfs_but_rejects_any_higher_peak() {
        let mut at_limit = CheckLevelMeter::default();
        at_limit.start(CAPTURE_SAMPLE_RATE_HZ).unwrap();
        at_limit.process_block(&vec![
            RETURN_CLIP_THRESHOLD_LINEAR;
            CHECK_LEVEL_DURATION_SAMPLES
        ]);
        assert_eq!(at_limit.state(), CheckLevelState::Passed);

        let mut above_limit = CheckLevelMeter::default();
        above_limit.start(CAPTURE_SAMPLE_RATE_HZ).unwrap();
        above_limit.process_block(&[0.0, RETURN_CLIP_THRESHOLD_LINEAR + 0.000_001, 0.0]);
        assert_eq!(
            above_limit.state(),
            CheckLevelState::Failed(CheckLevelFailure::ReturnClipped)
        );
        assert_eq!(above_limit.result().measured_samples, 2);
    }

    #[test]
    fn check_level_rejects_non_finite_audio_and_can_be_interrupted() {
        let mut non_finite = CheckLevelMeter::default();
        non_finite.start(CAPTURE_SAMPLE_RATE_HZ).unwrap();
        non_finite.process_block(&[0.1, f32::NAN, 0.1]);
        assert_eq!(
            non_finite.state(),
            CheckLevelState::Failed(CheckLevelFailure::NonFiniteSample)
        );
        assert_eq!(non_finite.result().measured_samples, 1);

        let mut interrupted = CheckLevelMeter::default();
        interrupted.start(CAPTURE_SAMPLE_RATE_HZ).unwrap();
        interrupted.process_block(&[0.1; 16]);
        interrupted.interrupt();
        assert_eq!(
            interrupted.state(),
            CheckLevelState::Failed(CheckLevelFailure::Interrupted)
        );
    }

    #[test]
    fn capture_program_precomputes_the_highest_energy_one_second_probe() {
        let sync = vec![0.1; 16];
        let mut excitation = vec![0.1; CHECK_LEVEL_DURATION_SAMPLES];
        excitation.extend(vec![0.7; CHECK_LEVEL_DURATION_SAMPLES]);
        let program = CaptureProgram::new(sync, excitation).unwrap();
        assert_eq!(
            program.check_level_probe().len(),
            CHECK_LEVEL_DURATION_SAMPLES
        );
        assert!(
            program
                .check_level_probe()
                .iter()
                .all(|sample| (*sample - 0.7).abs() < f32::EPSILON)
        );
    }

    #[test]
    fn timeline_jump_loop_stop_and_sample_rate_invalidate_capture() {
        let program = test_program();
        for (mut next_transport, expected) in [
            (
                TransportInfo {
                    playing: true,
                    timeline_sample: Some(999),
                    sample_rate_hz: CAPTURE_SAMPLE_RATE_HZ,
                    ..TransportInfo::default()
                },
                CaptureInvalidation::TimelineSeek,
            ),
            (
                TransportInfo {
                    playing: true,
                    timeline_sample: Some(8),
                    loop_active: true,
                    sample_rate_hz: CAPTURE_SAMPLE_RATE_HZ,
                    ..TransportInfo::default()
                },
                CaptureInvalidation::LoopActive,
            ),
            (
                TransportInfo {
                    playing: false,
                    timeline_sample: Some(8),
                    sample_rate_hz: CAPTURE_SAMPLE_RATE_HZ,
                    ..TransportInfo::default()
                },
                CaptureInvalidation::TransportStopped,
            ),
            (
                TransportInfo {
                    playing: true,
                    timeline_sample: Some(8),
                    sample_rate_hz: 44_100,
                    ..TransportInfo::default()
                },
                CaptureInvalidation::SampleRateChanged,
            ),
        ] {
            let mut engine = CaptureEngine::new(CaptureRole::Source, session_id());
            engine
                .prepare(Arc::clone(&program), CAPTURE_SAMPLE_RATE_HZ)
                .unwrap();
            engine.arm(false, 1).unwrap();
            let mut output = [0.0; 8];
            engine.process_block(&[0.0; 8], &mut output, transport(true, 0));
            if expected == CaptureInvalidation::TransportStopped {
                next_transport.timeline_sample = Some(8);
            }
            engine.process_block(&[0.0; 8], &mut output, next_transport);
            assert_eq!(engine.state(), CaptureState::Invalid(expected));
        }
    }

    #[test]
    fn coordinator_pairs_two_tracks_without_rt_locks() {
        let coordinator = CaptureCoordinator::with_capacity(2);
        let source = coordinator
            .bind(session_id(), CaptureRole::Source, 100)
            .unwrap();
        assert!(!source.pair_complete());
        assert_eq!(
            source.arm_pair(false),
            Err(CoordinatorError::PairIncomplete)
        );
        let returned = coordinator
            .bind(session_id(), CaptureRole::Return, 200)
            .unwrap();
        assert!(source.pair_complete());
        pass_pair_check(&returned);
        let generation = source.arm_pair(false).unwrap();
        assert_eq!(source.rt_snapshot().arm_generation, generation);
        assert_eq!(returned.rt_snapshot().arm_generation, generation);
        assert!(!source.rt_snapshot().transport_was_playing_when_armed);
        source.publish_phase(CapturePhase::Capturing);
        assert_eq!(returned.rt_snapshot().peer_phase, CapturePhase::Capturing);
        drop(source);
        assert_eq!(
            returned.rt_snapshot().abort_reason,
            Some(CaptureInvalidation::PairLost)
        );
    }

    #[test]
    fn coordinator_preserves_the_transport_edge_used_for_arming() {
        let coordinator = CaptureCoordinator::with_capacity(1);
        let source_binding = coordinator
            .bind(session_id(), CaptureRole::Source, 100)
            .unwrap();
        let _return_binding = coordinator
            .bind(session_id(), CaptureRole::Return, 200)
            .unwrap();
        pass_pair_check(&_return_binding);
        let program = test_program();
        let mut source = CaptureEngine::new(CaptureRole::Source, session_id());
        source
            .prepare(Arc::clone(&program), CAPTURE_SAMPLE_RATE_HZ)
            .unwrap();

        source_binding.arm_pair(false).unwrap();
        // Some hosts do not process the track while stopped. The first
        // callback may therefore already be the transport's rising edge.
        source.synchronize(&source_binding).unwrap();
        let mut output = [0.0; 8];
        source.process_block(&[0.0; 8], &mut output, transport(true, 10_000));
        assert!(matches!(source.state(), CaptureState::PreRoll { .. }));

        source_binding.arm_pair(true).unwrap();
        source.synchronize(&source_binding).unwrap();
        source.process_block(&[0.0; 8], &mut output, transport(true, 10_008));
        assert_eq!(source.state(), CaptureState::WaitingForTransport);
        source.process_block(&[0.0; 8], &mut output, transport(false, 10_016));
        assert_eq!(source.state(), CaptureState::WaitingForTransport);
        source.process_block(&[0.0; 8], &mut output, transport(true, 20_000));
        assert!(matches!(source.state(), CaptureState::PreRoll { .. }));
    }

    #[test]
    fn two_instances_capture_across_tracks_with_exact_stages_and_send_trim() {
        let coordinator = CaptureCoordinator::with_capacity(1);
        let source_binding = coordinator
            .bind(session_id(), CaptureRole::Source, 100)
            .unwrap();
        let return_binding = coordinator
            .bind(session_id(), CaptureRole::Return, 200)
            .unwrap();
        let program = test_program();
        let mut source = CaptureEngine::new(CaptureRole::Source, session_id());
        let mut returned = CaptureEngine::new(CaptureRole::Return, session_id());
        source
            .prepare(Arc::clone(&program), CAPTURE_SAMPLE_RATE_HZ)
            .unwrap();
        returned
            .prepare(Arc::clone(&program), CAPTURE_SAMPLE_RATE_HZ)
            .unwrap();

        let send_trim_db = -6.3_f32;
        let send_gain = 10.0_f32.powf(send_trim_db / 20.0);
        source_binding.publish_source_send_trim_db(send_trim_db);
        assert_eq!(return_binding.source_send_trim_db(), send_trim_db);
        pass_pair_check(&return_binding);
        let generation = source_binding.arm_pair(false).unwrap();

        let mut timeline = 80_000_i64;
        let mut captured_return = Vec::with_capacity(program.total_capture_samples());
        let block_sizes = [1_usize, 7, 16, 32, 64, 257, 512];
        let mut block_index = 0;
        while captured_return.len() < program.total_capture_samples() {
            source.synchronize(&source_binding).unwrap();
            returned.synchronize(&return_binding).unwrap();
            let block_size = block_sizes[block_index % block_sizes.len()]
                .min(program.total_capture_samples() - captured_return.len());
            block_index += 1;
            let silent = vec![0.0; block_size];
            let mut source_output = vec![f32::NAN; block_size];
            source.process_block(&silent, &mut source_output, transport(true, timeline));
            for sample in &mut source_output {
                *sample *= send_gain;
            }
            captured_return.extend_from_slice(&source_output);

            let mut return_output = vec![1.0; block_size];
            returned.process_block(
                &source_output,
                &mut return_output,
                transport(true, timeline),
            );
            assert!(return_output.iter().all(|sample| *sample == 0.0));
            timeline += block_size as i64;
        }

        assert_eq!(source.state(), CaptureState::Ready);
        assert_eq!(returned.state(), CaptureState::Ready);
        assert_eq!(source.armed_generation(), generation);
        assert_eq!(returned.armed_generation(), generation);
        assert_eq!(
            program.total_capture_samples(),
            PRE_ROLL_SAMPLES + program.program_samples() + TAIL_SAMPLES
        );
        assert!(
            captured_return[..PRE_ROLL_SAMPLES]
                .iter()
                .all(|sample| *sample == 0.0)
        );
        assert_eq!(
            captured_return[program.sync_start_sample()],
            program.sync_header()[0] * send_gain
        );
        let tail_start = PRE_ROLL_SAMPLES + program.program_samples();
        assert!(
            captured_return[tail_start..]
                .iter()
                .all(|sample| *sample == 0.0)
        );
        assert_eq!(returned.return_audio().unwrap(), captured_return);

        let completed = returned.take_completed_return().unwrap();
        assert_eq!(completed.session_id, session_id());
        assert_eq!(completed.generation, generation);
        returned.recycle_completed_return(completed).unwrap();
    }

    #[test]
    fn a_local_capture_failure_propagates_to_the_peer_instance() {
        let coordinator = CaptureCoordinator::with_capacity(1);
        let source_binding = coordinator
            .bind(session_id(), CaptureRole::Source, 100)
            .unwrap();
        let return_binding = coordinator
            .bind(session_id(), CaptureRole::Return, 200)
            .unwrap();
        pass_pair_check(&return_binding);
        let program = test_program();
        let mut source = CaptureEngine::new(CaptureRole::Source, session_id());
        let mut returned = CaptureEngine::new(CaptureRole::Return, session_id());
        source
            .prepare(Arc::clone(&program), CAPTURE_SAMPLE_RATE_HZ)
            .unwrap();
        returned.prepare(program, CAPTURE_SAMPLE_RATE_HZ).unwrap();
        source_binding.arm_pair(false).unwrap();
        source.synchronize(&source_binding).unwrap();
        returned.synchronize(&return_binding).unwrap();

        let mut source_output = [0.0; 8];
        source.process_block(&[0.0; 8], &mut source_output, transport(true, 12_000));
        let mut return_output = [0.0; 8];
        let mut clipped = [0.0; 8];
        clipped[3] = RETURN_CLIP_THRESHOLD_LINEAR + 0.01;
        returned.process_block(&clipped, &mut return_output, transport(true, 12_000));
        assert_eq!(
            returned.state(),
            CaptureState::Invalid(CaptureInvalidation::ReturnClipped)
        );

        // The failed side publishes the shared abort on its next lock-free
        // synchronization, then the other DAW track observes it.
        returned.synchronize(&return_binding).unwrap();
        source.synchronize(&source_binding).unwrap();
        assert_eq!(
            source.state(),
            CaptureState::Invalid(CaptureInvalidation::ReturnClipped)
        );
    }

    #[test]
    fn pair_cannot_arm_until_its_return_passes_a_fresh_level_check() {
        let coordinator = CaptureCoordinator::with_capacity(1);
        let source = coordinator
            .bind(session_id(), CaptureRole::Source, 100)
            .unwrap();
        let returned = coordinator
            .bind(session_id(), CaptureRole::Return, 200)
            .unwrap();

        assert_eq!(
            source.arm_pair(false),
            Err(CoordinatorError::CheckLevelNotPassed(
                SessionCheckLevelState::Required
            ))
        );
        assert_eq!(
            source.request_check_level(),
            Err(CoordinatorError::CheckLevelRequiresReturn)
        );

        let failed_generation = returned.request_check_level().unwrap();
        assert_eq!(
            source.check_level_snapshot(),
            SessionCheckLevelSnapshot {
                generation: failed_generation,
                state: SessionCheckLevelState::Measuring,
                progress: 0.0,
                peak_linear: 0.0,
            }
        );
        assert!(returned.publish_check_level(
            failed_generation,
            SessionCheckLevelState::Failed,
            0.25,
            RETURN_CLIP_THRESHOLD_LINEAR,
        ));
        assert_eq!(
            source.arm_pair(false),
            Err(CoordinatorError::CheckLevelNotPassed(
                SessionCheckLevelState::Failed
            ))
        );

        let passed_generation = returned.request_check_level().unwrap();
        assert_ne!(passed_generation, failed_generation);
        assert!(!returned.publish_check_level(
            failed_generation,
            SessionCheckLevelState::Passed,
            1.0,
            0.1,
        ));
        assert!(returned.publish_check_level(
            passed_generation,
            SessionCheckLevelState::Passed,
            1.0,
            0.5,
        ));
        assert_eq!(
            source.check_level_snapshot().state,
            SessionCheckLevelState::Passed
        );
        source.arm_pair(false).unwrap();
        assert_eq!(
            returned.request_check_level(),
            Err(CoordinatorError::CaptureActive)
        );
    }

    #[test]
    fn changing_the_bound_pair_invalidates_its_level_check_pass() {
        let coordinator = CaptureCoordinator::with_capacity(1);
        let source = coordinator
            .bind(session_id(), CaptureRole::Source, 100)
            .unwrap();
        let returned = coordinator
            .bind(session_id(), CaptureRole::Return, 200)
            .unwrap();
        pass_pair_check(&returned);
        assert_eq!(
            source.check_level_snapshot().state,
            SessionCheckLevelState::Passed
        );

        drop(source);
        assert_eq!(
            returned.check_level_snapshot().state,
            SessionCheckLevelState::Required
        );
        let replacement = coordinator
            .bind(session_id(), CaptureRole::Source, 300)
            .unwrap();
        assert_eq!(
            replacement.arm_pair(false),
            Err(CoordinatorError::CheckLevelNotPassed(
                SessionCheckLevelState::Required
            ))
        );
    }

    #[test]
    fn changing_source_send_trim_invalidates_a_pass_but_republishing_it_does_not() {
        let coordinator = CaptureCoordinator::with_capacity(1);
        let source = coordinator
            .bind(session_id(), CaptureRole::Source, 100)
            .unwrap();
        let returned = coordinator
            .bind(session_id(), CaptureRole::Return, 200)
            .unwrap();
        source.publish_source_send_trim_db(-12.0);
        pass_pair_check(&returned);
        let passed = source.check_level_snapshot();
        assert_eq!(passed.state, SessionCheckLevelState::Passed);

        source.publish_source_send_trim_db(-12.0);
        assert_eq!(source.check_level_snapshot(), passed);

        source.publish_source_send_trim_db(-6.0);
        let invalidated = source.check_level_snapshot();
        assert_eq!(invalidated.state, SessionCheckLevelState::Required);
        assert_ne!(invalidated.generation, passed.generation);
        assert_eq!(
            source.arm_pair(false),
            Err(CoordinatorError::CheckLevelNotPassed(
                SessionCheckLevelState::Required
            ))
        );
    }

    #[test]
    fn hardware_metadata_is_explicitly_uncalibrated() {
        let metadata = HardwareCaptureMetadata::uncalibrated_full_amp();
        assert_eq!(metadata.target, CaptureTarget::FullAmpUnfilteredLoad);
        assert_eq!(metadata.calibration_status, CalibrationStatus::Uncalibrated);
        assert_eq!(metadata.sample_rate_hz, CAPTURE_SAMPLE_RATE_HZ);
        assert_eq!(metadata.input_level_dbu, None);
        assert_eq!(metadata.output_level_dbu, None);
    }
}
