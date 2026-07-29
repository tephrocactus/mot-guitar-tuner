use truce::prelude::AudioConfig;

use crate::a2::{A2_MACS_PER_SAMPLE, A2_SAMPLE_RATE_HZ, A2Error, A2Model, A2Processor};

pub const MAX_MODEL_MACS_PER_SAMPLE: u32 = A2_MACS_PER_SAMPLE;

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
    a2: Option<A2Processor>,
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
            sample_rate: A2_SAMPLE_RATE_HZ as f32,
            a2: None,
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
        if let Some(a2) = &mut self.a2 {
            a2.reset();
        }
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

    /// Installs the exact fixed-shape NAM A2-C3 runtime used by captured
    /// `.motmodel` files.
    pub fn load_a2_model(&mut self, model: A2Model) -> Result<(), A2Error> {
        let host_rate = self.sample_rate.round() as u32;
        let processor = A2Processor::new_for_sample_rate(model, host_rate)?;
        self.a2 = Some(processor);
        Ok(())
    }

    pub fn unload_model(&mut self) {
        self.a2 = None;
    }

    #[must_use]
    pub const fn has_model(&self) -> bool {
        self.a2.is_some()
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

            let modeled = if let Some(a2) = &mut self.a2 {
                a2.process_sample(tightened)
            } else {
                tightened
            };

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
    use crate::a2::{A2_CHANNELS, A2_HEAD_KERNEL_SIZE, A2_HEAD_SCALE, A2_KERNEL_SIZES};

    /// A minimal valid A2 graph with a direct current-sample path. The public
    /// arrays use tap-major runtime order.
    fn current_sample_model() -> A2Model {
        let mut model = A2Model::zeros();
        model.weights.rechannel[0] = 1.0;
        let layer_current_tap = A2_KERNEL_SIZES[0] - 1;
        model.weights.layers[0].conv[layer_current_tap * A2_CHANNELS * A2_CHANNELS] = 1.0;
        let head_current_tap = A2_HEAD_KERNEL_SIZE - 1;
        model.weights.head[head_current_tap * A2_CHANNELS] = 1.0;
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
    fn a2_model_is_independent_of_host_block_partitioning() {
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
        whole.load_a2_model(current_sample_model()).unwrap();
        whole.set_controls_immediate(controls);
        let mut whole_output = vec![0.0; input.len()];
        whole.process_block(&input, &mut whole_output);

        let mut partitioned = AmpProcessor::default();
        partitioned.reset(&config);
        partitioned.load_a2_model(current_sample_model()).unwrap();
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
        amp.load_a2_model(current_sample_model()).unwrap();
        amp.set_controls_immediate(AmpControls {
            input_gain_db: 6.0,
            tight: 1.0,
            bite: 1.0,
        });
        let input = [1.0, 0.0, 0.0, 0.0];
        let mut output = [0.0; 4];
        amp.process_block(&input, &mut output);
        assert_ne!(output[0], 0.0);
        assert!(output[0].is_sign_positive());
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
    fn loader_rejects_invalid_or_wrong_rate_a2_models() {
        let mut amp = AmpProcessor::default();
        amp.reset(&AudioConfig::new(48_000.0, 32));
        let mut model = current_sample_model();
        model.lookahead_samples = 1;
        assert_eq!(amp.load_a2_model(model), Err(A2Error::NonCausalOrLatent));

        let mut model = current_sample_model();
        model.weights.head[0] = f32::NAN;
        assert_eq!(amp.load_a2_model(model), Err(A2Error::NonFiniteCoefficient));

        amp.reset(&AudioConfig::new(44_100.0, 32));
        assert_eq!(
            amp.load_a2_model(current_sample_model()),
            Err(A2Error::HostSampleRateMismatch {
                model: A2_SAMPLE_RATE_HZ,
                host: 44_100,
            })
        );
    }

    #[test]
    fn unloading_a2_restores_bit_exact_raw_path() {
        let mut amp = AmpProcessor::default();
        amp.reset(&AudioConfig::new(48_000.0, 32));
        amp.load_a2_model(current_sample_model()).unwrap();
        assert!(amp.has_model());
        amp.unload_model();
        assert!(!amp.has_model());

        let input = [-1.0, -0.25, 0.0, 0.125, 1.0];
        let mut output = [f32::NAN; 5];
        amp.process_block(&input, &mut output);
        assert_eq!(output, input);
    }

    #[test]
    fn direct_a2_model_has_expected_current_sample_gain() {
        let mut amp = AmpProcessor::default();
        amp.reset(&AudioConfig::new(48_000.0, 32));
        amp.load_a2_model(current_sample_model()).unwrap();
        let mut output = [0.0; 2];
        amp.process_block(&[1.0, 0.0], &mut output);
        assert_eq!(output[0].to_bits(), A2_HEAD_SCALE.to_bits());
    }

    #[cfg(feature = "rt-paranoid")]
    #[test]
    fn processing_allocates_nothing() {
        let mut amp = AmpProcessor::default();
        amp.reset(&AudioConfig::new(48_000.0, 32));
        amp.load_a2_model(current_sample_model()).unwrap();
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
