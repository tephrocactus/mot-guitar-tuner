use truce::prelude::AudioConfig;

use crate::amp::AmpProcessor;
use crate::cabinet::CabinetProcessor;

const MUTE_RAMP_SECONDS: f32 = 0.003;

/// The processed branch of the plugin.
///
/// Pitch analysis taps the dry mono signal before this type is called. The
/// public plugin deliberately supports mono-in/mono-out only; spatial effects
/// belong after it in the host.
#[derive(Default)]
pub struct GuitarSignalChain {
    amp: AmpProcessor,
    cabinet: CabinetProcessor,
    amp_scratch: Vec<f32>,
    #[cfg(test)]
    processed_samples: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct OutputMute {
    gain: f32,
    target: f32,
    step: f32,
    remaining_samples: usize,
    ramp_samples: usize,
}

impl Default for OutputMute {
    fn default() -> Self {
        Self {
            gain: 1.0,
            target: 1.0,
            step: 0.0,
            remaining_samples: 0,
            ramp_samples: 144,
        }
    }
}

impl OutputMute {
    pub fn reset(&mut self, sample_rate: f32, muted: bool) {
        self.ramp_samples = (sample_rate.max(1.0) * MUTE_RAMP_SECONDS).round().max(1.0) as usize;
        self.gain = if muted { 0.0 } else { 1.0 };
        self.target = self.gain;
        self.step = 0.0;
        self.remaining_samples = 0;
    }

    #[inline]
    pub fn next_gain(&mut self, muted: bool) -> f32 {
        let requested_target = if muted { 0.0 } else { 1.0 };
        if requested_target != self.target {
            self.target = requested_target;
            self.remaining_samples = self.ramp_samples;
            self.step = (self.target - self.gain) / self.remaining_samples as f32;
        }

        if self.remaining_samples > 0 {
            self.gain += self.step;
            self.remaining_samples -= 1;
            if self.remaining_samples == 0 {
                self.gain = self.target;
            }
        }
        self.gain
    }
}

impl GuitarSignalChain {
    pub fn reset(&mut self, config: &AudioConfig) {
        self.amp.reset(config);
        self.cabinet.reset(config);
        self.amp_scratch.resize(config.max_block_size, 0.0);
        self.amp_scratch.fill(0.0);
        #[cfg(test)]
        {
            self.processed_samples = 0;
        }
    }

    #[inline]
    pub fn process_block(&mut self, input: &[f32], output: &mut [f32]) {
        debug_assert_eq!(input.len(), output.len());
        debug_assert!(input.len() <= self.amp_scratch.len());
        #[cfg(test)]
        {
            self.processed_samples += input.len() as u64;
        }
        let amp_output = &mut self.amp_scratch[..input.len()];
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

    #[cfg(test)]
    pub fn processed_samples(&self) -> u64 {
        self.processed_samples
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholder_mono_chain_is_bit_exact() {
        let mut chain = GuitarSignalChain::default();
        chain.reset(&AudioConfig::new(48_000.0, 512));
        let input = [0.0, -1.0, 0.125, 0.5, 1.0];
        let mut output = [f32::NAN; 5];
        chain.process_block(&input, &mut output);
        assert_eq!(output, input);
        assert_eq!(chain.processed_samples(), input.len() as u64);
        assert_eq!(chain.latency_samples(), 0);
        assert_eq!(chain.tail_samples(), 0);
    }

    #[test]
    fn output_mute_reaches_both_targets_without_a_step() {
        let mut mute = OutputMute::default();
        mute.reset(48_000.0, false);
        let first_muted_gain = mute.next_gain(true);
        assert!(first_muted_gain < 1.0);
        assert!(first_muted_gain > 0.0);
        for _ in 1..144 {
            mute.next_gain(true);
        }
        assert_eq!(mute.next_gain(true), 0.0);

        let first_unmuted_gain = mute.next_gain(false);
        assert!(first_unmuted_gain > 0.0);
        assert!(first_unmuted_gain < 1.0);
        for _ in 1..144 {
            mute.next_gain(false);
        }
        assert_eq!(mute.next_gain(false), 1.0);
    }
}
