//! Transport-synchronized capture shared by the standalone Generator and
//! Trainer plugins.
//!
//! There is deliberately no process-global coordinator here. A Generator and
//! a Trainer each observe their own copy of the host transport and start on
//! the same stopped-to-playing edge. They only need identical immutable
//! [`CaptureProgram`] data; audio and state are never exchanged between plugin
//! instances.
//!
//! All storage is prepared before arming. [`GeneratorEngine::process_block`]
//! and [`TrainerRecorder::process_block`] perform no allocation, locking, I/O,
//! or ownership-count changes.

use std::fmt;
use std::sync::Arc;

use crate::capture::{
    AlignmentConfig, CAPTURE_SAMPLE_RATE_HZ, CaptureInvalidation, CaptureProgram,
    DEFAULT_MAX_ALIGNMENT_LAG_SAMPLES, PRE_ROLL_SAMPLES, TransportInfo,
};

/// Extra Return audio retained after the nominal two-second tail.
///
/// A hardware round trip is causal, so its sync header can only arrive later
/// than the Generator's header. Keeping the existing half-second alignment
/// search range after the nominal capture window lets the offline correlator
/// recover that delay without shifting either real-time instance.
pub const DEFAULT_ALIGNMENT_MARGIN_SAMPLES: usize = DEFAULT_MAX_ALIGNMENT_LAG_SAMPLES;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SplitCaptureState {
    Idle,
    Armed,
    WaitingForTransport,
    PreRoll { completed_samples: usize },
    Program { completed_samples: usize },
    Tail { completed_samples: usize },
    AlignmentMargin { completed_samples: usize },
    Ready,
    Invalid(CaptureInvalidation),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SplitCapturePrepareError {
    UnsupportedSampleRate(u32),
    RecordingLengthOverflow,
    CompletedRecordingPending,
    RecordingStorageUnavailable,
    WrongRecordingLength { expected: usize, actual: usize },
}

impl fmt::Display for SplitCapturePrepareError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedSampleRate(rate) => {
                write!(
                    formatter,
                    "split capture requires 48000 Hz, received {rate} Hz"
                )
            }
            Self::RecordingLengthOverflow => {
                formatter.write_str("the split-capture recording window is too long")
            }
            Self::CompletedRecordingPending => formatter
                .write_str("take and recycle the completed Trainer recording before arming again"),
            Self::RecordingStorageUnavailable => formatter
                .write_str("the preallocated Trainer recording handoff buffer is unavailable"),
            Self::WrongRecordingLength { expected, actual } => write!(
                formatter,
                "recording storage has {actual} samples, expected {expected}"
            ),
        }
    }
}

impl std::error::Error for SplitCapturePrepareError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ClockStatus {
    Idle,
    Armed,
    WaitingForTransport,
    Running,
    Ready,
    Invalid(CaptureInvalidation),
}

/// Per-instance transport clock. It contains no synchronization primitive:
/// host transport is the protocol clock.
#[derive(Clone, Copy, Debug)]
struct CaptureClock {
    status: ClockStatus,
    position: usize,
    last_transport_playing: bool,
    expected_timeline_sample: Option<i64>,
}

impl Default for CaptureClock {
    fn default() -> Self {
        Self {
            status: ClockStatus::Idle,
            position: 0,
            last_transport_playing: false,
            expected_timeline_sample: None,
        }
    }
}

impl CaptureClock {
    fn arm(&mut self, transport_is_playing: bool) {
        self.status = ClockStatus::Armed;
        self.position = 0;
        self.last_transport_playing = transport_is_playing;
        self.expected_timeline_sample = None;
    }

    fn reset(&mut self) {
        *self = Self::default();
    }

    fn invalidate(&mut self, reason: CaptureInvalidation) {
        if !matches!(
            self.status,
            ClockStatus::Idle | ClockStatus::Ready | ClockStatus::Invalid(_)
        ) {
            self.status = ClockStatus::Invalid(reason);
        }
    }

    #[inline]
    fn begin_block(&mut self, transport: TransportInfo) -> bool {
        if matches!(
            self.status,
            ClockStatus::Idle | ClockStatus::Ready | ClockStatus::Invalid(_)
        ) {
            self.last_transport_playing = transport.playing;
            return false;
        }

        if transport.sample_rate_hz != CAPTURE_SAMPLE_RATE_HZ {
            self.invalidate(CaptureInvalidation::SampleRateChanged);
            self.last_transport_playing = transport.playing;
            return false;
        }
        if transport.loop_active {
            self.invalidate(CaptureInvalidation::LoopActive);
            self.last_transport_playing = transport.playing;
            return false;
        }
        if transport.discontinuity {
            self.invalidate(CaptureInvalidation::TransportDiscontinuity);
            self.last_transport_playing = transport.playing;
            return false;
        }

        let rising_edge = !self.last_transport_playing && transport.playing;
        if matches!(
            self.status,
            ClockStatus::Armed | ClockStatus::WaitingForTransport
        ) {
            if !rising_edge {
                self.status = ClockStatus::WaitingForTransport;
                self.last_transport_playing = transport.playing;
                return false;
            }
            self.status = ClockStatus::Running;
            self.expected_timeline_sample = transport.timeline_sample;
        }

        if !transport.playing {
            self.invalidate(CaptureInvalidation::TransportStopped);
            self.last_transport_playing = false;
            return false;
        }
        if let (Some(expected), Some(actual)) =
            (self.expected_timeline_sample, transport.timeline_sample)
            && expected != actual
        {
            self.invalidate(CaptureInvalidation::TimelineSeek);
            self.last_transport_playing = true;
            return false;
        }
        matches!(self.status, ClockStatus::Running)
    }

    #[inline]
    fn end_block(&mut self, transport: TransportInfo, host_block_samples: usize) {
        if matches!(self.status, ClockStatus::Running) {
            self.expected_timeline_sample = transport
                .timeline_sample
                .and_then(|sample| sample.checked_add(host_block_samples as i64));
        }
        self.last_transport_playing = transport.playing;
    }

    fn complete(&mut self) {
        self.status = ClockStatus::Ready;
    }
}

/// Real-time engine for MOT Generator.
///
/// The complete emitted window is exactly:
///
/// `1 second silence -> sync header -> excitation -> 2 seconds silence
/// -> alignment-margin silence`.
#[derive(Debug)]
pub struct GeneratorEngine {
    program: Arc<CaptureProgram>,
    alignment_margin_samples: usize,
    generation_samples: usize,
    clock: CaptureClock,
}

impl GeneratorEngine {
    pub fn new(
        program: Arc<CaptureProgram>,
        sample_rate_hz: u32,
    ) -> Result<Self, SplitCapturePrepareError> {
        Self::with_alignment_margin(program, sample_rate_hz, DEFAULT_ALIGNMENT_MARGIN_SAMPLES)
    }

    pub fn with_alignment_margin(
        program: Arc<CaptureProgram>,
        sample_rate_hz: u32,
        alignment_margin_samples: usize,
    ) -> Result<Self, SplitCapturePrepareError> {
        validate_sample_rate(sample_rate_hz)?;
        let generation_samples = program
            .total_capture_samples()
            .checked_add(alignment_margin_samples)
            .ok_or(SplitCapturePrepareError::RecordingLengthOverflow)?;
        Ok(Self {
            program,
            alignment_margin_samples,
            generation_samples,
            clock: CaptureClock::default(),
        })
    }

    /// Requires a future stopped-to-playing edge.
    pub fn arm(&mut self, transport_is_playing: bool) {
        self.clock.arm(transport_is_playing);
    }

    /// Cancels an armed capture before the stopped-to-playing edge.
    ///
    /// Returns `true` when the pending capture was disarmed. Once generation
    /// has started, transport rules remain authoritative and this is a no-op.
    pub fn disarm(&mut self) -> bool {
        if matches!(
            self.clock.status,
            ClockStatus::Armed | ClockStatus::WaitingForTransport
        ) {
            self.clock.reset();
            true
        } else {
            false
        }
    }

    /// Control-thread reset. This does not allocate, but must not race with
    /// `process_block`.
    pub fn reset_off_thread(&mut self) {
        self.clock.reset();
    }

    #[must_use]
    pub fn state(&self) -> SplitCaptureState {
        state_for_clock(&self.clock, &self.program, self.alignment_margin_samples)
    }

    #[must_use]
    pub const fn completed_samples(&self) -> usize {
        self.clock.position
    }

    #[must_use]
    pub const fn total_samples(&self) -> usize {
        self.generation_samples
    }

    #[must_use]
    pub const fn alignment_margin_samples(&self) -> usize {
        self.alignment_margin_samples
    }

    #[must_use]
    pub fn program(&self) -> &Arc<CaptureProgram> {
        &self.program
    }

    /// Ignores track input and writes the mono Generator output.
    #[inline]
    pub fn process_block(&mut self, output: &mut [f32], transport: TransportInfo) {
        output.fill(0.0);
        if !self.clock.begin_block(transport) {
            return;
        }

        let total_samples = self.generation_samples;
        let remaining = total_samples.saturating_sub(self.clock.position);
        let processed = remaining.min(output.len());
        for (offset, output_sample) in output[..processed].iter_mut().enumerate() {
            *output_sample = self.program.emitted_sample(self.clock.position + offset);
        }
        self.clock.position += processed;
        if self.clock.position == total_samples {
            self.clock.complete();
        }
        self.clock.end_block(transport, output.len());
    }
}

/// Owned Trainer recording moved to an offline alignment/training worker.
#[derive(Debug)]
pub struct CompletedTrainerRecording {
    audio: Vec<f32>,
    alignment_margin_samples: usize,
    peak_linear: f32,
    rms_linear: f32,
}

impl CompletedTrainerRecording {
    #[must_use]
    pub fn audio(&self) -> &[f32] {
        &self.audio
    }

    #[must_use]
    pub fn into_audio(self) -> Vec<f32> {
        self.audio
    }

    #[must_use]
    pub const fn alignment_margin_samples(&self) -> usize {
        self.alignment_margin_samples
    }

    /// Builds the offline correlator contract from the exact margin that was
    /// used to allocate and record this take.
    #[must_use]
    pub fn alignment_config(&self) -> AlignmentConfig {
        AlignmentConfig {
            maximum_lag_samples: self.alignment_margin_samples,
            ..AlignmentConfig::default()
        }
    }

    #[must_use]
    pub const fn peak_linear(&self) -> f32 {
        self.peak_linear
    }

    #[must_use]
    pub const fn rms_linear(&self) -> f32 {
        self.rms_linear
    }
}

/// Real-time recorder for MOT Trainer.
///
/// It records its own track input from the same transport edge as the
/// Generator. No Generator reference, session registry, or cross-instance
/// audio transfer exists.
#[derive(Debug)]
pub struct TrainerRecorder {
    program: Arc<CaptureProgram>,
    alignment_margin_samples: usize,
    recording_samples: usize,
    recording_buffer: Vec<f32>,
    spare_buffer: Vec<f32>,
    completed: Option<CompletedTrainerRecording>,
    peak_linear: f32,
    sum_squares: f64,
    clock: CaptureClock,
}

impl TrainerRecorder {
    pub fn new(
        program: Arc<CaptureProgram>,
        sample_rate_hz: u32,
    ) -> Result<Self, SplitCapturePrepareError> {
        Self::with_alignment_margin(program, sample_rate_hz, DEFAULT_ALIGNMENT_MARGIN_SAMPLES)
    }

    pub fn with_alignment_margin(
        program: Arc<CaptureProgram>,
        sample_rate_hz: u32,
        alignment_margin_samples: usize,
    ) -> Result<Self, SplitCapturePrepareError> {
        validate_sample_rate(sample_rate_hz)?;
        let recording_samples = program
            .total_capture_samples()
            .checked_add(alignment_margin_samples)
            .ok_or(SplitCapturePrepareError::RecordingLengthOverflow)?;
        // Both vectors are allocated here, never in process_block or arm.
        let recording_buffer = vec![0.0; recording_samples];
        let spare_buffer = vec![0.0; recording_samples];
        Ok(Self {
            program,
            alignment_margin_samples,
            recording_samples,
            recording_buffer,
            spare_buffer,
            completed: None,
            peak_linear: 0.0,
            sum_squares: 0.0,
            clock: CaptureClock::default(),
        })
    }

    /// Arms the recorder without clearing or allocating its buffers. A
    /// complete run overwrites every sample before ownership is published.
    pub fn arm(&mut self, transport_is_playing: bool) -> Result<(), SplitCapturePrepareError> {
        if self.completed.is_some() {
            return Err(SplitCapturePrepareError::CompletedRecordingPending);
        }
        if self.recording_buffer.len() != self.recording_samples
            || self.spare_buffer.len() != self.recording_samples
        {
            return Err(SplitCapturePrepareError::RecordingStorageUnavailable);
        }
        self.peak_linear = 0.0;
        self.sum_squares = 0.0;
        self.clock.arm(transport_is_playing);
        Ok(())
    }

    /// Explicit control/worker-thread reset between attempts.
    ///
    /// Clearing a multi-minute capture buffer is intentionally excluded from
    /// `arm` and the audio callback.
    pub fn reset_off_thread(&mut self) -> Result<(), SplitCapturePrepareError> {
        if self.completed.is_some() {
            return Err(SplitCapturePrepareError::CompletedRecordingPending);
        }
        if self.recording_buffer.len() != self.recording_samples
            || self.spare_buffer.len() != self.recording_samples
        {
            return Err(SplitCapturePrepareError::RecordingStorageUnavailable);
        }
        self.recording_buffer.fill(0.0);
        self.spare_buffer.fill(0.0);
        self.peak_linear = 0.0;
        self.sum_squares = 0.0;
        self.clock.reset();
        Ok(())
    }

    #[must_use]
    pub fn state(&self) -> SplitCaptureState {
        state_for_clock(&self.clock, &self.program, self.alignment_margin_samples)
    }

    #[must_use]
    pub const fn completed_samples(&self) -> usize {
        self.clock.position
    }

    #[must_use]
    pub const fn total_samples(&self) -> usize {
        self.recording_samples
    }

    #[must_use]
    pub const fn alignment_margin_samples(&self) -> usize {
        self.alignment_margin_samples
    }

    #[must_use]
    pub const fn peak_linear(&self) -> f32 {
        self.peak_linear
    }

    #[must_use]
    pub fn rms_linear(&self) -> f32 {
        if self.clock.position == 0 {
            0.0
        } else {
            (self.sum_squares / self.clock.position as f64).sqrt() as f32
        }
    }

    #[must_use]
    pub fn program(&self) -> &Arc<CaptureProgram> {
        &self.program
    }

    /// Records this instance's mono input and produces no output.
    #[inline]
    pub fn process_block(&mut self, input: &[f32], transport: TransportInfo) {
        if !self.clock.begin_block(transport) {
            return;
        }

        let remaining = self.recording_samples.saturating_sub(self.clock.position);
        let processed = remaining.min(input.len());
        let destination =
            &mut self.recording_buffer[self.clock.position..self.clock.position + processed];
        destination.copy_from_slice(&input[..processed]);
        for &sample in &input[..processed] {
            let magnitude = sample.abs();
            self.peak_linear = self.peak_linear.max(magnitude);
            self.sum_squares += f64::from(sample) * f64::from(sample);
        }
        self.clock.position += processed;
        if self.clock.position == self.recording_samples {
            self.publish_completed();
            self.clock.complete();
        }
        self.clock.end_block(transport, input.len());
    }

    /// Constant-time ownership handoff; call immediately after `Ready`.
    #[must_use]
    pub fn take_completed(&mut self) -> Option<CompletedTrainerRecording> {
        self.completed.take()
    }

    /// Returns a worker-consumed allocation to the recorder's fixed pool.
    pub fn recycle(
        &mut self,
        completed: CompletedTrainerRecording,
    ) -> Result<(), (SplitCapturePrepareError, CompletedTrainerRecording)> {
        if completed.audio.len() != self.recording_samples {
            let actual = completed.audio.len();
            return Err((
                SplitCapturePrepareError::WrongRecordingLength {
                    expected: self.recording_samples,
                    actual,
                },
                completed,
            ));
        }
        if !self.spare_buffer.is_empty() {
            return Err((
                SplitCapturePrepareError::RecordingStorageUnavailable,
                completed,
            ));
        }
        self.spare_buffer = completed.audio;
        Ok(())
    }

    #[inline]
    fn publish_completed(&mut self) {
        debug_assert!(self.completed.is_none());
        std::mem::swap(&mut self.recording_buffer, &mut self.spare_buffer);
        let audio = std::mem::take(&mut self.spare_buffer);
        let rms_linear = if self.recording_samples == 0 {
            0.0
        } else {
            (self.sum_squares / self.recording_samples as f64).sqrt() as f32
        };
        self.completed = Some(CompletedTrainerRecording {
            audio,
            alignment_margin_samples: self.alignment_margin_samples,
            peak_linear: self.peak_linear,
            rms_linear,
        });
    }
}

fn validate_sample_rate(sample_rate_hz: u32) -> Result<(), SplitCapturePrepareError> {
    if sample_rate_hz == CAPTURE_SAMPLE_RATE_HZ {
        Ok(())
    } else {
        Err(SplitCapturePrepareError::UnsupportedSampleRate(
            sample_rate_hz,
        ))
    }
}

fn state_for_clock(
    clock: &CaptureClock,
    program: &CaptureProgram,
    alignment_margin_samples: usize,
) -> SplitCaptureState {
    match clock.status {
        ClockStatus::Idle => SplitCaptureState::Idle,
        ClockStatus::Armed => SplitCaptureState::Armed,
        ClockStatus::WaitingForTransport => SplitCaptureState::WaitingForTransport,
        ClockStatus::Ready => SplitCaptureState::Ready,
        ClockStatus::Invalid(reason) => SplitCaptureState::Invalid(reason),
        ClockStatus::Running => {
            let position = clock.position;
            if position < PRE_ROLL_SAMPLES {
                SplitCaptureState::PreRoll {
                    completed_samples: position,
                }
            } else if position < PRE_ROLL_SAMPLES + program.program_samples() {
                SplitCaptureState::Program {
                    completed_samples: position - PRE_ROLL_SAMPLES,
                }
            } else if position < program.total_capture_samples() {
                SplitCaptureState::Tail {
                    completed_samples: position - PRE_ROLL_SAMPLES - program.program_samples(),
                }
            } else if position < program.total_capture_samples() + alignment_margin_samples {
                SplitCaptureState::AlignmentMargin {
                    completed_samples: position - program.total_capture_samples(),
                }
            } else {
                SplitCaptureState::Ready
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capture::{
        AlignmentConfig, TAIL_SAMPLES, extract_aligned_excitation, measure_alignment,
    };

    const START_TIMELINE: i64 = 90_000;

    fn program() -> Arc<CaptureProgram> {
        let mut random = 0x3c6e_f372_u32;
        let sync = (0..521)
            .map(|_| {
                random ^= random << 13;
                random ^= random >> 17;
                random ^= random << 5;
                if random & 1 == 0 { -0.25 } else { 0.25 }
            })
            .collect::<Vec<_>>();
        let excitation = (0..4_099)
            .map(|index| {
                let phase = index as f32;
                0.22 * (phase * 0.037).sin()
                    + 0.11 * (phase * 0.113).sin()
                    + 0.04 * (phase * 0.271).sin()
            })
            .collect::<Vec<_>>();
        Arc::new(CaptureProgram::new(sync, excitation).unwrap())
    }

    fn transport(playing: bool, timeline_sample: i64) -> TransportInfo {
        TransportInfo {
            playing,
            timeline_sample: Some(timeline_sample),
            sample_rate_hz: CAPTURE_SAMPLE_RATE_HZ,
            ..TransportInfo::default()
        }
    }

    fn render_generator(
        program: Arc<CaptureProgram>,
        block_pattern: &[usize],
        sample_count: usize,
    ) -> Vec<f32> {
        let mut generator = GeneratorEngine::new(program, CAPTURE_SAMPLE_RATE_HZ).unwrap();
        generator.arm(false);
        let mut stopped = [1.0; 13];
        generator.process_block(&mut stopped, transport(false, START_TIMELINE));
        assert_eq!(generator.state(), SplitCaptureState::WaitingForTransport);
        assert!(stopped.iter().all(|sample| *sample == 0.0));

        let mut rendered = Vec::with_capacity(sample_count);
        let mut timeline = START_TIMELINE;
        let mut pattern_index = 0;
        while rendered.len() < sample_count {
            let block_size = block_pattern[pattern_index % block_pattern.len()]
                .min(sample_count - rendered.len());
            pattern_index += 1;
            let mut output = vec![f32::NAN; block_size];
            generator.process_block(&mut output, transport(true, timeline));
            rendered.extend_from_slice(&output);
            timeline += block_size as i64;
        }
        rendered
    }

    fn record_trainer(
        program: Arc<CaptureProgram>,
        input: &[f32],
        block_pattern: &[usize],
        alignment_margin: usize,
    ) -> CompletedTrainerRecording {
        let mut trainer = TrainerRecorder::with_alignment_margin(
            program,
            CAPTURE_SAMPLE_RATE_HZ,
            alignment_margin,
        )
        .unwrap();
        trainer.arm(false).unwrap();
        trainer.process_block(&[0.0; 11], transport(false, START_TIMELINE));
        assert_eq!(trainer.state(), SplitCaptureState::WaitingForTransport);

        let mut timeline = START_TIMELINE;
        let mut position = 0;
        let mut pattern_index = 0;
        while position < input.len() {
            let block_size =
                block_pattern[pattern_index % block_pattern.len()].min(input.len() - position);
            pattern_index += 1;
            trainer.process_block(
                &input[position..position + block_size],
                transport(true, timeline),
            );
            position += block_size;
            timeline += block_size as i64;
        }
        assert_eq!(trainer.state(), SplitCaptureState::Ready);
        trainer.take_completed().unwrap()
    }

    fn target_processor(sample: f32) -> f32 {
        (sample * 1.7).tanh() * 0.63
    }

    fn delayed_target(source: &[f32], delay: usize) -> Vec<f32> {
        (0..source.len())
            .map(|index| {
                index
                    .checked_sub(delay)
                    .and_then(|source_index| source.get(source_index))
                    .copied()
                    .map(target_processor)
                    .unwrap_or(0.0)
            })
            .collect()
    }

    fn assert_alignment_recovers_target(
        program: &CaptureProgram,
        recording: &CompletedTrainerRecording,
        delay: usize,
    ) {
        let alignment = measure_alignment(
            program,
            recording.audio(),
            AlignmentConfig {
                maximum_lag_samples: recording.alignment_margin_samples(),
                minimum_normalized_correlation: 0.8,
            },
        )
        .unwrap();
        assert_eq!(alignment.integer_latency_samples, delay as i64);
        assert!(!alignment.polarity_inverted);

        let aligned = extract_aligned_excitation(program, recording.audio(), alignment).unwrap();
        assert_eq!(aligned.len(), program.excitation().len());
        for (&actual, &excitation) in aligned.iter().zip(program.excitation()) {
            let expected = target_processor(excitation);
            assert!(
                (actual - expected).abs() < 2.0e-3,
                "actual {actual}, expected {expected}, alignment {alignment:?}"
            );
        }
    }

    #[test]
    fn same_odd_host_blocks_align_a_delayed_processed_target() {
        let program = program();
        let margin = 1_024;
        let total = program.total_capture_samples() + margin;
        let block_pattern = [1, 7, 31, 63, 257];
        let source = render_generator(Arc::clone(&program), &block_pattern, total);
        let returned = delayed_target(&source, 137);
        let recording = record_trainer(Arc::clone(&program), &returned, &block_pattern, margin);
        assert_alignment_recovers_target(&program, &recording, 137);
    }

    #[test]
    fn different_odd_host_blocks_remain_transport_synchronized() {
        let program = program();
        let margin = 2_048;
        let total = program.total_capture_samples() + margin;
        let source = render_generator(Arc::clone(&program), &[3, 17, 65, 129, 511], total);
        let returned = delayed_target(&source, 733);
        let recording = record_trainer(Arc::clone(&program), &returned, &[5, 11, 37, 251], margin);
        assert_alignment_recovers_target(&program, &recording, 733);
    }

    #[test]
    fn generator_window_has_exact_preroll_program_and_tail() {
        let program = program();
        let rendered = render_generator(
            Arc::clone(&program),
            &[1, 7, 16, 257],
            program.total_capture_samples(),
        );

        assert_eq!(rendered.len(), program.total_capture_samples());
        assert!(
            rendered[..PRE_ROLL_SAMPLES]
                .iter()
                .all(|sample| *sample == 0.0)
        );
        let program_start = PRE_ROLL_SAMPLES;
        for (&actual, expected) in rendered
            [program_start..program_start + program.program_samples()]
            .iter()
            .zip(program.sync_header().iter().chain(program.excitation()))
        {
            assert_eq!(actual, *expected);
        }
        assert!(
            rendered[rendered.len() - TAIL_SAMPLES..]
                .iter()
                .all(|sample| *sample == 0.0)
        );
    }

    #[test]
    fn generator_and_trainer_share_the_default_full_window() {
        let program = program();
        let mut generator =
            GeneratorEngine::new(Arc::clone(&program), CAPTURE_SAMPLE_RATE_HZ).unwrap();
        let trainer = TrainerRecorder::new(Arc::clone(&program), CAPTURE_SAMPLE_RATE_HZ).unwrap();

        assert_eq!(
            generator.alignment_margin_samples(),
            DEFAULT_ALIGNMENT_MARGIN_SAMPLES
        );
        assert_eq!(
            generator.alignment_margin_samples(),
            trainer.alignment_margin_samples()
        );
        assert_eq!(generator.total_samples(), trainer.total_samples());

        generator.arm(false);
        let nominal_samples = program.total_capture_samples();
        let mut nominal_window = vec![f32::NAN; nominal_samples];
        generator.process_block(&mut nominal_window, transport(true, START_TIMELINE));
        assert_eq!(
            generator.state(),
            SplitCaptureState::AlignmentMargin {
                completed_samples: 0
            }
        );

        let mut alignment_margin = vec![f32::NAN; DEFAULT_ALIGNMENT_MARGIN_SAMPLES];
        generator.process_block(
            &mut alignment_margin,
            transport(true, START_TIMELINE + nominal_samples as i64),
        );
        assert_eq!(generator.state(), SplitCaptureState::Ready);
        assert!(alignment_margin.iter().all(|sample| *sample == 0.0));
    }

    #[test]
    fn generator_can_rearm_after_transport_invalidation() {
        let program = program();
        let mut generator =
            GeneratorEngine::with_alignment_margin(program, CAPTURE_SAMPLE_RATE_HZ, 32).unwrap();
        generator.arm(false);
        generator.process_block(&mut [0.0; 16], transport(true, START_TIMELINE));
        generator.process_block(&mut [0.0; 16], transport(true, START_TIMELINE + 99));
        assert_eq!(
            generator.state(),
            SplitCaptureState::Invalid(CaptureInvalidation::TimelineSeek)
        );

        generator.arm(false);
        generator.process_block(&mut [0.0; 11], transport(false, START_TIMELINE));
        assert_eq!(generator.state(), SplitCaptureState::WaitingForTransport);
        let mut full_window = vec![f32::NAN; generator.total_samples()];
        generator.process_block(&mut full_window, transport(true, START_TIMELINE));
        assert_eq!(generator.state(), SplitCaptureState::Ready);
    }

    #[test]
    fn generator_can_disarm_before_transport_starts() {
        let mut generator = GeneratorEngine::new(program(), CAPTURE_SAMPLE_RATE_HZ).unwrap();
        generator.arm(false);
        generator.process_block(&mut [0.0; 11], transport(false, START_TIMELINE));
        assert_eq!(generator.state(), SplitCaptureState::WaitingForTransport);

        assert!(generator.disarm());
        assert_eq!(generator.state(), SplitCaptureState::Idle);

        let mut output = [f32::NAN; 17];
        generator.process_block(&mut output, transport(true, START_TIMELINE));
        assert!(output.iter().all(|sample| *sample == 0.0));
        assert_eq!(generator.state(), SplitCaptureState::Idle);
        assert!(!generator.disarm());
    }

    #[test]
    fn completed_recording_alignment_config_uses_its_actual_margin() {
        let program = program();
        let margin = 1_337;
        let input = vec![0.0; program.total_capture_samples() + margin];
        let recording = record_trainer(program, &input, &[257, 31, 7], margin);
        let config = recording.alignment_config();

        assert_eq!(config.maximum_lag_samples, margin);
        assert_eq!(
            config.minimum_normalized_correlation,
            AlignmentConfig::default().minimum_normalized_correlation
        );
    }

    #[test]
    fn active_capture_invalidates_on_transport_hazards() {
        let program = program();
        let cases = [
            (
                TransportInfo {
                    playing: false,
                    timeline_sample: Some(START_TIMELINE + 16),
                    sample_rate_hz: CAPTURE_SAMPLE_RATE_HZ,
                    ..TransportInfo::default()
                },
                CaptureInvalidation::TransportStopped,
            ),
            (
                TransportInfo {
                    playing: true,
                    timeline_sample: Some(START_TIMELINE + 99),
                    sample_rate_hz: CAPTURE_SAMPLE_RATE_HZ,
                    ..TransportInfo::default()
                },
                CaptureInvalidation::TimelineSeek,
            ),
            (
                TransportInfo {
                    playing: true,
                    timeline_sample: Some(START_TIMELINE + 16),
                    loop_active: true,
                    sample_rate_hz: CAPTURE_SAMPLE_RATE_HZ,
                    ..TransportInfo::default()
                },
                CaptureInvalidation::LoopActive,
            ),
            (
                TransportInfo {
                    playing: true,
                    timeline_sample: Some(START_TIMELINE + 16),
                    sample_rate_hz: 44_100,
                    ..TransportInfo::default()
                },
                CaptureInvalidation::SampleRateChanged,
            ),
        ];

        for (hazard, expected) in cases {
            let mut generator =
                GeneratorEngine::new(Arc::clone(&program), CAPTURE_SAMPLE_RATE_HZ).unwrap();
            generator.arm(false);
            generator.process_block(&mut [0.0; 16], transport(true, START_TIMELINE));
            generator.process_block(&mut [0.0; 16], hazard);
            assert_eq!(generator.state(), SplitCaptureState::Invalid(expected));

            let mut trainer = TrainerRecorder::with_alignment_margin(
                Arc::clone(&program),
                CAPTURE_SAMPLE_RATE_HZ,
                32,
            )
            .unwrap();
            trainer.arm(false).unwrap();
            trainer.process_block(&[0.0; 16], transport(true, START_TIMELINE));
            trainer.process_block(&[0.0; 16], hazard);
            assert_eq!(trainer.state(), SplitCaptureState::Invalid(expected));
        }
    }

    #[test]
    fn completed_recording_is_an_ownership_handoff() {
        let program = program();
        let margin = 32;
        let total = program.total_capture_samples() + margin;
        let mut trainer =
            TrainerRecorder::with_alignment_margin(program, CAPTURE_SAMPLE_RATE_HZ, margin)
                .unwrap();
        let recording_pointer = trainer.recording_buffer.as_ptr();
        let spare_pointer = trainer.spare_buffer.as_ptr();
        trainer.arm(false).unwrap();
        let mut timeline = START_TIMELINE;
        for block in vec![0.125_f32; total].chunks(257) {
            trainer.process_block(block, transport(true, timeline));
            timeline += block.len() as i64;
        }
        assert_eq!(trainer.recording_buffer.as_ptr(), spare_pointer);
        let completed = trainer.take_completed().unwrap();
        assert_eq!(completed.audio().as_ptr(), recording_pointer);
        assert_eq!(completed.audio().len(), total);
        trainer.recycle(completed).unwrap();
        trainer.arm(false).unwrap();
    }
}
