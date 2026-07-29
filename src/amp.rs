use std::fmt;

use truce::prelude::AudioConfig;

pub const AMP_MODEL_FORMAT_VERSION: u32 = 1;
pub const AMP_MODEL_SAMPLE_RATE_HZ: u32 = 48_000;
pub const MAX_MODEL_UNITS: usize = 32;
pub const MAX_MODEL_MACS_PER_SAMPLE: u32 = 256;

const PARAMETER_SMOOTHING_SECONDS: f32 = 0.005;
const TIGHT_CORNER_HZ: f32 = 115.0;
const BITE_CORNER_HZ: f32 = 2_200.0;
const TIGHT_MAX_CUT: f32 = 0.88;
const BITE_MAX_BOOST: f32 = 1.25;
const DENORMAL_LIMIT: f32 = 1.0e-20;

/// User-facing, model-bound amplifier controls.
///
/// All controls are smoothed by [`AmpProcessor`]. `tight` and `bite` are
/// normalized to `0.0..=1.0`; input gain is clamped to `-24..=24 dB`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AmpControls {
    pub input_gain_db: f32,
    pub tight: f32,
    pub bite: f32,
}

impl Default for AmpControls {
    fn default() -> Self {
        Self {
            input_gain_db: 0.0,
            tight: 0.0,
            bite: 0.0,
        }
    }
}

impl AmpControls {
    #[must_use]
    pub fn sanitized(self) -> Self {
        Self {
            input_gain_db: finite_or(self.input_gain_db, 0.0).clamp(-24.0, 24.0),
            tight: finite_or(self.tight, 0.0).clamp(0.0, 1.0),
            bite: finite_or(self.bite, 0.0).clamp(0.0, 1.0),
        }
    }
}

/// A compact strictly-causal recurrent model.
///
/// This representation is deliberately fixed-size. It can be deserialized and
/// validated on a loader thread, then moved into a complete `AmpProcessor`
/// runtime without allocating in the audio callback. For sample `n`, every
/// unit reads the current input and its own state from sample `n - 1`; no
/// lookahead or block accumulation is possible.
///
/// A trainer may use fewer than [`MAX_MODEL_UNITS`] by setting `unit_count`.
/// Unused array entries are ignored.
#[derive(Clone, Copy, Debug)]
pub struct CompactCausalModel {
    pub format_version: u32,
    pub sample_rate_hz: u32,
    pub causal: bool,
    pub lookahead_samples: u32,
    pub runtime_latency_samples: u32,
    pub estimated_macs_per_sample: u32,
    pub unit_count: u8,
    pub dry_gain: f32,
    pub output_bias: f32,
    pub input_weights: [f32; MAX_MODEL_UNITS],
    pub recurrent_weights: [f32; MAX_MODEL_UNITS],
    pub biases: [f32; MAX_MODEL_UNITS],
    pub output_weights: [f32; MAX_MODEL_UNITS],
}

impl Default for CompactCausalModel {
    fn default() -> Self {
        Self::raw()
    }
}

impl CompactCausalModel {
    /// Bit-exact identity model. It is also the processor's startup model.
    #[must_use]
    pub const fn raw() -> Self {
        Self {
            format_version: AMP_MODEL_FORMAT_VERSION,
            sample_rate_hz: AMP_MODEL_SAMPLE_RATE_HZ,
            causal: true,
            lookahead_samples: 0,
            runtime_latency_samples: 0,
            estimated_macs_per_sample: 1,
            unit_count: 0,
            dry_gain: 1.0,
            output_bias: 0.0,
            input_weights: [0.0; MAX_MODEL_UNITS],
            recurrent_weights: [0.0; MAX_MODEL_UNITS],
            biases: [0.0; MAX_MODEL_UNITS],
            output_weights: [0.0; MAX_MODEL_UNITS],
        }
    }

    pub fn validate(&self) -> Result<(), AmpModelError> {
        if self.format_version != AMP_MODEL_FORMAT_VERSION {
            return Err(AmpModelError::UnsupportedFormatVersion(self.format_version));
        }
        if self.sample_rate_hz != AMP_MODEL_SAMPLE_RATE_HZ {
            return Err(AmpModelError::UnsupportedSampleRate(self.sample_rate_hz));
        }
        if !self.causal || self.lookahead_samples != 0 || self.runtime_latency_samples != 0 {
            return Err(AmpModelError::NonCausalOrLatent);
        }
        let unit_count = usize::from(self.unit_count);
        if unit_count > MAX_MODEL_UNITS {
            return Err(AmpModelError::TooManyUnits(self.unit_count));
        }
        if self.estimated_macs_per_sample > MAX_MODEL_MACS_PER_SAMPLE {
            return Err(AmpModelError::ComputeBudgetExceeded(
                self.estimated_macs_per_sample,
            ));
        }

        let scalar_values = [self.dry_gain, self.output_bias];
        if scalar_values.iter().any(|value| !value.is_finite())
            || self.input_weights[..unit_count]
                .iter()
                .chain(&self.recurrent_weights[..unit_count])
                .chain(&self.biases[..unit_count])
                .chain(&self.output_weights[..unit_count])
                .any(|value| !value.is_finite())
        {
            return Err(AmpModelError::NonFiniteCoefficient);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AmpModelError {
    UnsupportedFormatVersion(u32),
    UnsupportedSampleRate(u32),
    NonCausalOrLatent,
    TooManyUnits(u8),
    ComputeBudgetExceeded(u32),
    NonFiniteCoefficient,
    HostSampleRateMismatch { model: u32, host: u32 },
}

impl fmt::Display for AmpModelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedFormatVersion(version) => {
                write!(formatter, "unsupported amp model format version {version}")
            }
            Self::UnsupportedSampleRate(rate) => {
                write!(formatter, "unsupported amp model sample rate {rate} Hz")
            }
            Self::NonCausalOrLatent => {
                formatter.write_str("amp model must be causal with zero lookahead and latency")
            }
            Self::TooManyUnits(count) => write!(
                formatter,
                "amp model has {count} units; maximum is {MAX_MODEL_UNITS}"
            ),
            Self::ComputeBudgetExceeded(macs) => write!(
                formatter,
                "amp model requires {macs} MACs/sample; maximum is {MAX_MODEL_MACS_PER_SAMPLE}"
            ),
            Self::NonFiniteCoefficient => {
                formatter.write_str("amp model contains a non-finite coefficient")
            }
            Self::HostSampleRateMismatch { model, host } => write!(
                formatter,
                "amp model is {model} Hz but the host is running at {host} Hz"
            ),
        }
    }
}

impl std::error::Error for AmpModelError {}

#[derive(Clone, Copy, Debug)]
struct SmoothedValue {
    current: f32,
    target: f32,
    retention: f32,
}

impl SmoothedValue {
    const fn new(value: f32) -> Self {
        Self {
            current: value,
            target: value,
            retention: 0.0,
        }
    }

    fn set_sample_rate(&mut self, sample_rate: f32) {
        self.retention = (-1.0 / (sample_rate.max(1.0) * PARAMETER_SMOOTHING_SECONDS)).exp();
    }

    #[inline]
    fn next(&mut self) -> f32 {
        self.current = self.target + self.retention * (self.current - self.target);
        if (self.current - self.target).abs() < 1.0e-7 {
            self.current = self.target;
        }
        self.current
    }

    fn set_target(&mut self, target: f32) {
        self.target = target;
    }

    fn set_immediate(&mut self, value: f32) {
        self.current = value;
        self.target = value;
    }
}

/// Allocation-free, zero-latency amplifier runtime.
///
/// The default model and default controls are bit-exact passthrough. Model
/// loading/resetting is a control-thread operation; `process_block()` performs
/// no allocation, locking, I/O, or internal block buffering.
#[derive(Clone, Debug)]
pub struct AmpProcessor {
    sample_rate: f32,
    model: CompactCausalModel,
    model_state: [f32; MAX_MODEL_UNITS],
    input_gain: SmoothedValue,
    tight: SmoothedValue,
    bite: SmoothedValue,
    tight_low_state: f32,
    bite_low_state: f32,
    tight_low_coefficient: f32,
    bite_low_coefficient: f32,
}

impl Default for AmpProcessor {
    fn default() -> Self {
        let mut processor = Self {
            sample_rate: AMP_MODEL_SAMPLE_RATE_HZ as f32,
            model: CompactCausalModel::raw(),
            model_state: [0.0; MAX_MODEL_UNITS],
            input_gain: SmoothedValue::new(1.0),
            tight: SmoothedValue::new(0.0),
            bite: SmoothedValue::new(0.0),
            tight_low_state: 0.0,
            bite_low_state: 0.0,
            tight_low_coefficient: 0.0,
            bite_low_coefficient: 0.0,
        };
        processor.configure_rate_dependent_state();
        processor
    }
}

impl AmpProcessor {
    /// Clears causal state. The current parameter targets are applied
    /// immediately so transport resets never create a synthetic automation
    /// ramp.
    pub fn reset(&mut self, config: &AudioConfig) {
        self.sample_rate = (config.sample_rate as f32).max(1.0);
        self.model_state.fill(0.0);
        self.tight_low_state = 0.0;
        self.bite_low_state = 0.0;
        self.input_gain.current = self.input_gain.target;
        self.tight.current = self.tight.target;
        self.bite.current = self.bite.target;
        self.configure_rate_dependent_state();
    }

    fn configure_rate_dependent_state(&mut self) {
        self.input_gain.set_sample_rate(self.sample_rate);
        self.tight.set_sample_rate(self.sample_rate);
        self.bite.set_sample_rate(self.sample_rate);
        self.tight_low_coefficient =
            1.0 - (-std::f32::consts::TAU * TIGHT_CORNER_HZ / self.sample_rate).exp();
        self.bite_low_coefficient =
            1.0 - (-std::f32::consts::TAU * BITE_CORNER_HZ / self.sample_rate).exp();
    }

    /// Installs a validated model and clears its recurrent state.
    ///
    /// Construct a replacement `AmpProcessor` on the loader thread when a
    /// click-free whole-runtime swap is required.
    pub fn load_model(&mut self, model: CompactCausalModel) -> Result<(), AmpModelError> {
        model.validate()?;
        let host_rate = self.sample_rate.round() as u32;
        if model.unit_count != 0 && model.sample_rate_hz != host_rate {
            return Err(AmpModelError::HostSampleRateMismatch {
                model: model.sample_rate_hz,
                host: host_rate,
            });
        }
        self.model = model;
        self.model_state.fill(0.0);
        Ok(())
    }

    pub fn unload_model(&mut self) {
        self.model = CompactCausalModel::raw();
        self.model_state.fill(0.0);
    }

    #[must_use]
    pub const fn model(&self) -> &CompactCausalModel {
        &self.model
    }

    pub fn set_controls(&mut self, controls: AmpControls) {
        let controls = controls.sanitized();
        self.input_gain
            .set_target(db_to_gain(controls.input_gain_db));
        self.tight.set_target(controls.tight);
        self.bite.set_target(controls.bite);
    }

    /// Applies controls without smoothing. Intended for initialization, state
    /// restore before playback, and deterministic offline tests.
    pub fn set_controls_immediate(&mut self, controls: AmpControls) {
        let controls = controls.sanitized();
        self.input_gain
            .set_immediate(db_to_gain(controls.input_gain_db));
        self.tight.set_immediate(controls.tight);
        self.bite.set_immediate(controls.bite);
    }

    #[must_use]
    pub fn target_controls(&self) -> AmpControls {
        AmpControls {
            input_gain_db: gain_to_db(self.input_gain.target),
            tight: self.tight.target,
            bite: self.bite.target,
        }
    }

    #[inline]
    pub fn process_block(&mut self, input: &[f32], output: &mut [f32]) {
        debug_assert_eq!(input.len(), output.len());
        for (&input_sample, output_sample) in input.iter().zip(output) {
            let input_gain = self.input_gain.next();
            let tight = self.tight.next();
            let bite = self.bite.next();

            let gained = input_sample * input_gain;
            self.tight_low_state += self.tight_low_coefficient * (gained - self.tight_low_state);
            flush_denormal(&mut self.tight_low_state);
            let tightened = gained - TIGHT_MAX_CUT * tight * self.tight_low_state;

            let mut modeled = self.model.output_bias + self.model.dry_gain * tightened;
            for unit in 0..usize::from(self.model.unit_count) {
                let activation = self.model.input_weights[unit] * tightened
                    + self.model.recurrent_weights[unit] * self.model_state[unit]
                    + self.model.biases[unit];
                let state = activation.tanh();
                self.model_state[unit] = if state.abs() < DENORMAL_LIMIT {
                    0.0
                } else {
                    state
                };
                modeled += self.model.output_weights[unit] * state;
            }

            self.bite_low_state += self.bite_low_coefficient * (modeled - self.bite_low_state);
            flush_denormal(&mut self.bite_low_state);
            *output_sample = modeled + BITE_MAX_BOOST * bite * (modeled - self.bite_low_state);
        }
    }

    #[must_use]
    pub const fn latency_samples(&self) -> u32 {
        0
    }

    #[must_use]
    pub const fn tail_samples(&self) -> u32 {
        0
    }
}

#[inline]
fn db_to_gain(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}

#[inline]
fn gain_to_db(gain: f32) -> f32 {
    20.0 * gain.max(f32::MIN_POSITIVE).log10()
}

#[inline]
fn finite_or(value: f32, fallback: f32) -> f32 {
    if value.is_finite() { value } else { fallback }
}

#[inline]
fn flush_denormal(value: &mut f32) {
    if value.abs() < DENORMAL_LIMIT {
        *value = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_model() -> CompactCausalModel {
        let mut model = CompactCausalModel::raw();
        model.unit_count = 2;
        model.dry_gain = 0.35;
        model.input_weights[0] = 1.4;
        model.input_weights[1] = -0.7;
        model.recurrent_weights[0] = 0.62;
        model.recurrent_weights[1] = -0.27;
        model.biases[0] = 0.02;
        model.biases[1] = -0.04;
        model.output_weights[0] = 0.8;
        model.output_weights[1] = -0.3;
        model.estimated_macs_per_sample = 13;
        model
    }

    #[test]
    fn raw_amp_is_bit_exact_and_zero_latency() {
        let mut amp = AmpProcessor::default();
        amp.reset(&AudioConfig::new(96_000.0, 512));
        let input = [0.0, -1.0, 0.125, 0.5, 1.0];
        let mut output = [f32::NAN; 5];
        amp.process_block(&input, &mut output);
        assert_eq!(output, input);
        assert_eq!(amp.latency_samples(), 0);
        assert_eq!(amp.tail_samples(), 0);
    }

    #[test]
    fn recurrent_model_is_independent_of_host_block_partitioning() {
        let config = AudioConfig::new(48_000.0, 512);
        let controls = AmpControls {
            input_gain_db: 3.2,
            tight: 0.37,
            bite: 0.61,
        };
        let input: Vec<f32> = (0..777)
            .map(|index| ((index as f32 * 0.137).sin() * 0.7).clamp(-1.0, 1.0))
            .collect();

        let mut whole = AmpProcessor::default();
        whole.reset(&config);
        whole.load_model(test_model()).unwrap();
        whole.set_controls_immediate(controls);
        let mut whole_output = vec![0.0; input.len()];
        whole.process_block(&input, &mut whole_output);

        let mut partitioned = AmpProcessor::default();
        partitioned.reset(&config);
        partitioned.load_model(test_model()).unwrap();
        partitioned.set_controls_immediate(controls);
        let mut partitioned_output = vec![0.0; input.len()];
        let partitions = [1, 7, 32, 3, 64, 129, 5, 257];
        let mut start = 0;
        let mut partition_index = 0;
        while start < input.len() {
            let end = (start + partitions[partition_index % partitions.len()]).min(input.len());
            partitioned.process_block(&input[start..end], &mut partitioned_output[start..end]);
            start = end;
            partition_index += 1;
        }

        assert_eq!(whole_output, partitioned_output);
    }

    #[test]
    fn model_and_controls_preserve_sample_zero_onset() {
        let mut amp = AmpProcessor::default();
        amp.reset(&AudioConfig::new(48_000.0, 32));
        amp.load_model(test_model()).unwrap();
        amp.set_controls_immediate(AmpControls {
            input_gain_db: 6.0,
            tight: 1.0,
            bite: 1.0,
        });
        let input = [1.0, 0.0, 0.0, 0.0];
        let mut output = [0.0; 4];
        amp.process_block(&input, &mut output);
        assert_ne!(output[0], 0.0);
        assert_eq!(amp.latency_samples(), 0);
    }

    #[test]
    fn controls_are_clamped_and_smoothed() {
        let mut amp = AmpProcessor::default();
        amp.reset(&AudioConfig::new(48_000.0, 32));
        amp.set_controls(AmpControls {
            input_gain_db: 100.0,
            tight: -2.0,
            bite: 3.0,
        });
        assert_eq!(
            amp.target_controls(),
            AmpControls {
                input_gain_db: 24.0,
                tight: 0.0,
                bite: 1.0,
            }
        );

        let input = [1.0; 8];
        let mut output = [0.0; 8];
        amp.process_block(&input, &mut output);
        assert!(output.iter().all(|sample| sample.is_finite()));
        assert_ne!(output[0], output[7]);
    }

    #[test]
    fn loader_rejects_non_causal_or_invalid_models() {
        let mut amp = AmpProcessor::default();
        amp.reset(&AudioConfig::new(48_000.0, 32));
        let mut model = test_model();
        model.lookahead_samples = 1;
        assert_eq!(amp.load_model(model), Err(AmpModelError::NonCausalOrLatent));

        let mut model = test_model();
        model.input_weights[0] = f32::NAN;
        assert_eq!(
            amp.load_model(model),
            Err(AmpModelError::NonFiniteCoefficient)
        );

        let mut model = test_model();
        model.estimated_macs_per_sample = MAX_MODEL_MACS_PER_SAMPLE + 1;
        assert_eq!(
            amp.load_model(model),
            Err(AmpModelError::ComputeBudgetExceeded(
                MAX_MODEL_MACS_PER_SAMPLE + 1
            ))
        );
    }

    #[cfg(feature = "rt-paranoid")]
    #[test]
    fn processing_allocates_nothing() {
        let mut amp = AmpProcessor::default();
        amp.reset(&AudioConfig::new(48_000.0, 32));
        amp.load_model(test_model()).unwrap();
        amp.set_controls_immediate(AmpControls {
            input_gain_db: 2.0,
            tight: 0.4,
            bite: 0.3,
        });
        let input = [0.25; 257];
        let mut output = [0.0; 257];
        let (_, allocations) = truce::rt::audit(|| {
            let _section = truce::rt::RtSection::enter();
            amp.process_block(&input, &mut output);
        });
        assert_eq!(allocations, 0);
    }
}
