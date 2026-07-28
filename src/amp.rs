use truce::prelude::AudioConfig;

/// Real-time amplifier stage.
///
/// The first architecture pass is deliberately transparent. Keeping the
/// processor behind this small interface lets us develop and test the
/// nonlinear core without coupling it to the plugin wrapper or tuner.
#[derive(Clone, Copy, Debug)]
pub struct AmpProcessor {
    sample_rate: f32,
}

impl Default for AmpProcessor {
    fn default() -> Self {
        Self {
            sample_rate: 48_000.0,
        }
    }
}

impl AmpProcessor {
    pub fn reset(&mut self, config: &AudioConfig) {
        self.sample_rate = (config.sample_rate as f32).max(1.0);
    }

    #[inline]
    pub fn process_block(&mut self, input: &[f32], output: &mut [f32]) {
        // Transparent until the MOT amplifier core is introduced.
        debug_assert_eq!(input.len(), output.len());
        output.copy_from_slice(input);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholder_amp_is_bit_exact() {
        let mut amp = AmpProcessor::default();
        amp.reset(&AudioConfig::new(96_000.0, 512));
        let input = [0.0, -1.0, 0.125, 0.5, 1.0];
        let mut output = [f32::NAN; 5];
        amp.process_block(&input, &mut output);
        assert_eq!(output, input);
        assert_eq!(amp.latency_samples(), 0);
        assert_eq!(amp.tail_samples(), 0);
    }
}
