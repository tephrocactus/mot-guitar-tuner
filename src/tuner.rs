use std::f32::consts::TAU;

pub const STRING_COUNT: usize = 7;
pub const DEFAULT_TUNING: [u8; STRING_COUNT] = [35, 40, 45, 50, 55, 59, 64];

const AUTOCORRELATION_RING: usize = 16_384;
const AUTOCORRELATION_PERIODS: usize = 2;
const DETECTION_INTERVAL_SECONDS: f32 = 0.010;
const ACQUIRE_HOLD_SECONDS: f32 = 0.025;
const SWITCH_HOLD_SECONDS: f32 = 0.050;
const SILENCE_HOLD_SECONDS: f32 = 0.150;
const CENTS_SMOOTHING_SECONDS: f32 = 0.035;
const PITCH_SEARCH_CENTS: f32 = 70.0;
const SWITCH_RATIO: f32 = 2.25;
const MIN_SCORE: f32 = 0.14;
const MIN_LEVEL_DB: f32 = -72.0;
const MIN_GOERTZEL_WINDOW: usize = 64;
const MAX_GOERTZEL_WINDOW: usize = 4_096;
const GOERTZEL_CYCLES: f32 = 3.5;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PitchAnalysis {
    /// The nearest 12-TET MIDI note to the stable, actually sounding pitch.
    pub detected_note: Option<u8>,
    /// Smoothed raw input frequency. No per-note offset has been applied.
    pub detected_frequency_hz: f32,
    pub confidence: f32,
    pub level_db: f32,
}

#[inline]
#[must_use]
pub fn midi_to_hz(midi_note: u8) -> f32 {
    440.0 * 2.0_f32.powf((f32::from(midi_note) - 69.0) / 12.0)
}

#[inline]
#[must_use]
pub fn cents_ratio(cents: f32) -> f32 {
    2.0_f32.powf(cents / 1_200.0)
}

#[inline]
#[must_use]
pub fn chromatic_range(open_notes: &[u8; STRING_COUNT]) -> (u8, u8) {
    let mut lowest = u8::MAX;
    let mut highest = u8::MIN;
    for note in open_notes {
        lowest = lowest.min(*note);
        highest = highest.max(*note);
    }
    (
        lowest.saturating_sub(2),
        highest.saturating_add(24).min(127),
    )
}

pub struct TunerEngine {
    sample_rate: f32,
    range_min: u8,
    range_max: u8,
    selected_note: Option<u8>,
    candidate_note: Option<u8>,
    candidate_samples: usize,
    silent_samples: usize,
    analysis_elapsed_samples: usize,
    autocorrelation_ring: Box<[f32; AUTOCORRELATION_RING]>,
    autocorrelation_position: usize,
    autocorrelation_filled: usize,
    rms_squared: f32,
    smoothed_cents: f32,
    has_pitch: bool,
    latest: PitchAnalysis,
}

impl Default for TunerEngine {
    fn default() -> Self {
        let (range_min, range_max) = chromatic_range(&DEFAULT_TUNING);
        Self {
            sample_rate: 48_000.0,
            range_min,
            range_max,
            selected_note: None,
            candidate_note: None,
            candidate_samples: 0,
            silent_samples: 0,
            analysis_elapsed_samples: 0,
            autocorrelation_ring: Box::new([0.0; AUTOCORRELATION_RING]),
            autocorrelation_position: 0,
            autocorrelation_filled: 0,
            rms_squared: 0.0,
            smoothed_cents: 0.0,
            has_pitch: false,
            latest: PitchAnalysis::default(),
        }
    }
}

impl TunerEngine {
    pub fn reset(&mut self, sample_rate: f32, open_notes: &[u8; STRING_COUNT]) {
        self.sample_rate = sample_rate.max(1.0);
        (self.range_min, self.range_max) = chromatic_range(open_notes);
        self.selected_note = None;
        self.clear_candidate();
        self.silent_samples = 0;
        self.analysis_elapsed_samples = 0;
        self.autocorrelation_position = 0;
        self.autocorrelation_filled = 0;
        self.rms_squared = 0.0;
        self.smoothed_cents = 0.0;
        self.has_pitch = false;
        self.latest = PitchAnalysis::default();
    }

    pub fn configure_range(&mut self, open_notes: &[u8; STRING_COUNT]) {
        let (minimum, maximum) = chromatic_range(open_notes);
        if minimum == self.range_min && maximum == self.range_max {
            return;
        }

        self.range_min = minimum;
        self.range_max = maximum;
        if self
            .selected_note
            .is_some_and(|note| note < minimum || note > maximum)
        {
            self.selected_note = None;
            self.has_pitch = false;
        }
        if self
            .candidate_note
            .is_some_and(|note| note < minimum || note > maximum)
        {
            self.clear_candidate();
        }
    }

    #[inline]
    pub fn push_sample(&mut self, sample: f32) {
        let rms_alpha = 1.0 - (-1.0 / (self.sample_rate * 0.010)).exp();
        self.rms_squared += rms_alpha * (sample * sample - self.rms_squared);
        self.autocorrelation_ring[self.autocorrelation_position] = sample;
        self.autocorrelation_position =
            (self.autocorrelation_position + 1) & (AUTOCORRELATION_RING - 1);
        self.autocorrelation_filled = (self.autocorrelation_filled + 1).min(AUTOCORRELATION_RING);
        self.analysis_elapsed_samples = self.analysis_elapsed_samples.saturating_add(1);

        let analysis_interval = (self.sample_rate * DETECTION_INTERVAL_SECONDS)
            .round()
            .max(1.0) as usize;
        if self.analysis_elapsed_samples >= analysis_interval {
            let elapsed_samples = self.analysis_elapsed_samples;
            self.analysis_elapsed_samples = 0;
            self.latest.level_db = self.level_db();
            self.scan_chromatic(elapsed_samples);
            self.update_pitch(elapsed_samples);
        }
    }

    /// Returns the latest 10 ms analysis snapshot at the end of a host block.
    /// All expensive work is paced by samples in `push_sample`, so large host
    /// buffers cannot reduce the detector's update rate.
    pub fn finish_block(&mut self) -> PitchAnalysis {
        self.latest.level_db = self.level_db();
        self.latest
    }

    fn scan_chromatic(&mut self, elapsed_samples: usize) {
        let mut winner = None;
        let mut winner_score = 0.0_f32;
        let mut selected_score = 0.0_f32;
        let mut sum_scores = 0.0_f32;

        for note in self.range_min..=self.range_max {
            let frequency = midi_to_hz(note);
            let periodicity = self.periodicity_score(frequency).max(0.0);
            let fundamental_energy = self.fundamental_energy_score(frequency);
            // The autocorrelation establishes periodicity; direct energy at
            // the candidate fundamental rejects subharmonic/octave aliases.
            let score = periodicity * periodicity * (0.10 + 0.90 * fundamental_energy);
            sum_scores += score;
            if self.selected_note == Some(note) {
                selected_score = score;
            }
            if score > winner_score {
                winner = Some(note);
                winner_score = score;
            }
        }

        let valid = self.latest.level_db > MIN_LEVEL_DB && winner_score >= MIN_SCORE;
        if valid {
            self.silent_samples = 0;
            if let Some(winner_note) = winner {
                match self.selected_note {
                    None => {
                        self.track_candidate(winner_note, elapsed_samples);
                        if self.candidate_samples >= self.seconds_to_samples(ACQUIRE_HOLD_SECONDS) {
                            self.select(winner_note);
                        }
                    }
                    Some(current)
                        if current == winner_note
                            || selected_score >= winner_score / SWITCH_RATIO =>
                    {
                        self.clear_candidate();
                    }
                    Some(_) => {
                        self.track_candidate(winner_note, elapsed_samples);
                        if self.candidate_samples >= self.seconds_to_samples(SWITCH_HOLD_SECONDS) {
                            self.select(winner_note);
                        }
                    }
                }
            }
        } else {
            self.clear_candidate();
            self.silent_samples = self.silent_samples.saturating_add(elapsed_samples);
            if self.silent_samples >= self.seconds_to_samples(SILENCE_HOLD_SECONDS) {
                self.selected_note = None;
                self.has_pitch = false;
            }
        }

        self.latest.confidence = self.selected_note.map_or(0.0, |selected| {
            let score = if Some(selected) == winner {
                winner_score
            } else {
                selected_score
            };
            if sum_scores > 0.0 {
                (score / sum_scores).clamp(0.0, 1.0)
            } else {
                0.0
            }
        });
    }

    fn update_pitch(&mut self, elapsed_samples: usize) {
        let Some(note) = self.selected_note else {
            self.latest.detected_note = None;
            self.latest.detected_frequency_hz = 0.0;
            self.latest.confidence = 0.0;
            return;
        };

        let target_hz = midi_to_hz(note);
        if let Some(measured_hz) = self.estimate_frequency(target_hz) {
            let measured_cents = 1_200.0 * (measured_hz / target_hz).log2();
            if self.has_pitch {
                let elapsed_seconds = elapsed_samples as f32 / self.sample_rate;
                let alpha = 1.0 - (-elapsed_seconds / CENTS_SMOOTHING_SECONDS).exp();
                self.smoothed_cents += alpha * (measured_cents - self.smoothed_cents);
            } else {
                self.smoothed_cents = measured_cents;
                self.has_pitch = true;
            }
        }

        if self.has_pitch {
            self.latest.detected_note = Some(note);
            self.latest.detected_frequency_hz = target_hz * cents_ratio(self.smoothed_cents);
        }
    }

    fn select(&mut self, note: u8) {
        let changed = self.selected_note != Some(note);
        self.selected_note = Some(note);
        self.clear_candidate();
        if changed {
            self.has_pitch = false;
            self.smoothed_cents = 0.0;
        }
    }

    fn track_candidate(&mut self, note: u8, elapsed_samples: usize) {
        if self.candidate_note == Some(note) {
            self.candidate_samples = self.candidate_samples.saturating_add(elapsed_samples);
        } else {
            self.candidate_note = Some(note);
            self.candidate_samples = elapsed_samples;
        }
    }

    fn clear_candidate(&mut self) {
        self.candidate_note = None;
        self.candidate_samples = 0;
    }

    #[inline]
    fn seconds_to_samples(&self, seconds: f32) -> usize {
        (self.sample_rate * seconds).round() as usize
    }

    fn level_db(&self) -> f32 {
        let rms = self.rms_squared.max(0.0).sqrt();
        if rms > 1.0e-9 {
            20.0 * rms.log10()
        } else {
            -180.0
        }
    }

    fn periodicity_score(&self, target_hz: f32) -> f32 {
        debug_assert!(AUTOCORRELATION_RING.is_power_of_two());
        let lag = (self.sample_rate / target_hz.max(1.0)).round() as usize;
        if lag == 0 || lag >= AUTOCORRELATION_RING {
            return 0.0;
        }
        let span = (lag * AUTOCORRELATION_PERIODS)
            .max(128)
            .min(AUTOCORRELATION_RING - lag);
        if self.autocorrelation_filled < lag + span {
            return 0.0;
        }
        self.autocorrelation_at_lag(lag, span)
    }

    fn fundamental_energy_score(&self, target_hz: f32) -> f32 {
        let window = ((self.sample_rate / target_hz.max(1.0) * GOERTZEL_CYCLES).round() as usize)
            .clamp(MIN_GOERTZEL_WINDOW, MAX_GOERTZEL_WINDOW);
        if self.autocorrelation_filled < window {
            return 0.0;
        }

        let omega = TAU * target_hz / self.sample_rate;
        let coefficient = 2.0 * omega.cos();
        let mut s1 = 0.0_f64;
        let mut s2 = 0.0_f64;
        let mut sum_squared = 0.0_f64;
        for delay in 0..window {
            let sample = f64::from(self.autocorrelation_sample(delay));
            let s0 = sample + f64::from(coefficient) * s1 - s2;
            s2 = s1;
            s1 = s0;
            sum_squared += sample * sample;
        }

        let coefficient = f64::from(coefficient);
        let power = (s1 * s1 + s2 * s2 - coefficient * s1 * s2).max(0.0);
        let amplitude = 2.0 * power.sqrt() / window as f64;
        let rms = (sum_squared / window as f64).sqrt();
        if rms <= f64::EPSILON {
            0.0
        } else {
            (amplitude / (rms * 2.0_f64.sqrt())).clamp(0.0, 1.0) as f32
        }
    }

    fn estimate_frequency(&self, target_hz: f32) -> Option<f32> {
        let expected_lag = self.sample_rate / target_hz.max(1.0);
        let minimum_lag = (expected_lag / cents_ratio(PITCH_SEARCH_CENTS)).floor() as usize;
        let maximum_lag = (expected_lag / cents_ratio(-PITCH_SEARCH_CENTS)).ceil() as usize;
        let span = ((expected_lag.round() as usize) * AUTOCORRELATION_PERIODS)
            .max(128)
            .min(AUTOCORRELATION_RING.saturating_sub(maximum_lag + 1));
        if minimum_lag < 2
            || maximum_lag + 1 + span > AUTOCORRELATION_RING
            || self.autocorrelation_filled < maximum_lag + 1 + span
        {
            return None;
        }

        let mut best_lag = minimum_lag;
        let mut best_score = f32::NEG_INFINITY;
        for lag in minimum_lag..=maximum_lag {
            let score = self.autocorrelation_at_lag(lag, span);
            if score > best_score {
                best_score = score;
                best_lag = lag;
            }
        }
        if best_score < 0.35 {
            return None;
        }

        let left = self.autocorrelation_at_lag(best_lag - 1, span);
        let right = self.autocorrelation_at_lag(best_lag + 1, span);
        let denominator = left - 2.0 * best_score + right;
        let fractional = if denominator.abs() > 1.0e-6 {
            (0.5 * (left - right) / denominator).clamp(-1.0, 1.0)
        } else {
            0.0
        };
        let refined_lag = best_lag as f32 + fractional;
        let measured_hz = self.sample_rate / refined_lag;
        measured_hz.is_finite().then_some(measured_hz)
    }

    fn autocorrelation_at_lag(&self, lag: usize, span: usize) -> f32 {
        if lag == 0
            || span == 0
            || lag + span > self.autocorrelation_filled
            || lag + span > AUTOCORRELATION_RING
        {
            return 0.0;
        }

        let mut sum_x = 0.0_f64;
        let mut sum_y = 0.0_f64;
        let mut sum_xx = 0.0_f64;
        let mut sum_yy = 0.0_f64;
        let mut sum_xy = 0.0_f64;
        for offset in 0..span {
            let x = f64::from(self.autocorrelation_sample(offset));
            let y = f64::from(self.autocorrelation_sample(offset + lag));
            sum_x += x;
            sum_y += y;
            sum_xx += x * x;
            sum_yy += y * y;
            sum_xy += x * y;
        }

        let count = span as f64;
        let covariance = sum_xy - sum_x * sum_y / count;
        let variance_x = (sum_xx - sum_x * sum_x / count).max(0.0);
        let variance_y = (sum_yy - sum_y * sum_y / count).max(0.0);
        let denominator = (variance_x * variance_y).sqrt();
        if denominator <= f64::EPSILON {
            0.0
        } else {
            (covariance / denominator).clamp(-1.0, 1.0) as f32
        }
    }

    #[inline]
    fn autocorrelation_sample(&self, delay: usize) -> f32 {
        let index = (self.autocorrelation_position + AUTOCORRELATION_RING - 1 - delay)
            & (AUTOCORRELATION_RING - 1);
        self.autocorrelation_ring[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_signal(
        engine: &mut TunerEngine,
        sample_rate: f32,
        frequency: f32,
        seconds: f32,
        harmonic_rich: bool,
    ) -> PitchAnalysis {
        let samples = (sample_rate * seconds) as usize;
        let block = 64;
        let mut analysis = PitchAnalysis::default();
        for start in (0..samples).step_by(block) {
            let count = block.min(samples - start);
            for i in 0..count {
                let t = (start + i) as f32 / sample_rate;
                let phase = TAU * frequency * t;
                let sample = if harmonic_rich {
                    let envelope = (-1.2 * t).exp();
                    envelope
                        * (0.11 * phase.sin()
                            + 0.24 * (2.0 * phase + 0.2).sin()
                            + 0.17 * (3.0 * phase + 0.7).sin()
                            + 0.08 * (4.0 * phase + 1.1).sin())
                } else {
                    0.25 * phase.sin()
                };
                engine.push_sample(sample);
            }
            analysis = engine.finish_block();
        }
        analysis
    }

    #[test]
    fn default_dynamic_range_is_two_below_to_two_octaves_above() {
        assert_eq!(chromatic_range(&DEFAULT_TUNING), (33, 88));
        assert_eq!(chromatic_range(&[0, 0, 0, 0, 0, 0, 127]), (0, 127));
    }

    #[test]
    fn every_semitone_is_detected_at_common_sample_rates() {
        let (minimum, maximum) = chromatic_range(&DEFAULT_TUNING);
        for sample_rate in [44_100.0, 48_000.0, 96_000.0] {
            for note in minimum..=maximum {
                let mut engine = TunerEngine::default();
                engine.reset(sample_rate, &DEFAULT_TUNING);
                let analysis = run_signal(&mut engine, sample_rate, midi_to_hz(note), 0.16, false);
                assert_eq!(
                    analysis.detected_note,
                    Some(note),
                    "sample_rate={sample_rate}, expected MIDI {note}, got {analysis:?}"
                );
                let cents = 1_200.0 * (analysis.detected_frequency_hz / midi_to_hz(note)).log2();
                assert!(
                    cents.abs() < 0.8,
                    "sample_rate={sample_rate}, MIDI {note}, cents={cents}"
                );
            }
        }
    }

    #[test]
    fn every_semitone_survives_harmonic_rich_decay_without_octave_errors() {
        let (minimum, maximum) = chromatic_range(&DEFAULT_TUNING);
        for sample_rate in [44_100.0, 48_000.0, 96_000.0] {
            for note in minimum..=maximum {
                let mut engine = TunerEngine::default();
                engine.reset(sample_rate, &DEFAULT_TUNING);
                let analysis = run_signal(&mut engine, sample_rate, midi_to_hz(note), 0.24, true);
                assert_eq!(
                    analysis.detected_note,
                    Some(note),
                    "sample_rate={sample_rate}, expected MIDI {note}, got {analysis:?}"
                );
            }
        }
    }

    #[test]
    fn sharp_and_flat_signs_are_correct() {
        for cents in [-7.0, 7.0] {
            let mut engine = TunerEngine::default();
            engine.reset(48_000.0, &DEFAULT_TUNING);
            let frequency = midi_to_hz(55) * cents_ratio(cents);
            let analysis = run_signal(&mut engine, 48_000.0, frequency, 0.25, false);
            let measured = 1_200.0 * (analysis.detected_frequency_hz / midi_to_hz(55)).log2();
            assert_eq!(analysis.detected_note, Some(55));
            assert_eq!(measured.is_sign_positive(), cents.is_sign_positive());
            assert!((measured - cents).abs() < 1.0, "{measured} vs {cents}");
        }
    }

    #[test]
    fn low_and_high_notes_capture_inside_latency_budget() {
        for (note, budget_ms) in [(35, 100.0), (64, 70.0)] {
            let sample_rate = 48_000.0;
            let mut engine = TunerEngine::default();
            engine.reset(sample_rate, &DEFAULT_TUNING);
            let frequency = midi_to_hz(note);
            let mut captured_at = None;
            for sample_index in 0..(sample_rate * 0.12) as usize {
                let t = sample_index as f32 / sample_rate;
                engine.push_sample(0.25 * (TAU * frequency * t).sin());
                if sample_index % 64 == 63 && engine.finish_block().detected_note == Some(note) {
                    captured_at = Some(sample_index as f32 * 1_000.0 / sample_rate);
                    break;
                }
            }
            let captured_at = captured_at.expect("note must be captured");
            assert!(
                captured_at <= budget_ms,
                "MIDI {note} captured at {captured_at:.2} ms"
            );
        }
    }

    #[test]
    fn short_harmonic_burst_does_not_switch_the_latched_note() {
        let mut engine = TunerEngine::default();
        engine.reset(48_000.0, &DEFAULT_TUNING);
        let initial = run_signal(&mut engine, 48_000.0, midi_to_hz(35), 0.20, false);
        assert_eq!(initial.detected_note, Some(35));

        let brief = run_signal(&mut engine, 48_000.0, midi_to_hz(54), 0.035, false);
        assert_eq!(brief.detected_note, Some(35));

        let sustained = run_signal(&mut engine, 48_000.0, midi_to_hz(54), 0.16, false);
        assert_eq!(sustained.detected_note, Some(54));
    }
}
