use truce::prelude::AudioConfig;

/// Real-time cabinet stage.
///
/// IR loading and partitioned convolution will live here. The initial stage is
/// transparent so the routing refactor cannot alter existing tuner sessions.
#[derive(Clone, Copy, Debug)]
pub struct CabinetProcessor {
    sample_rate: f32,
}

impl Default for CabinetProcessor {
    fn default() -> Self {
        Self {
            sample_rate: 48_000.0,
        }
    }
}

impl CabinetProcessor {
    pub fn reset(&mut self, config: &AudioConfig) {
        self.sample_rate = (config.sample_rate as f32).max(1.0);
    }

    #[inline]
    pub fn process_block(&mut self, input: &[f32], output: &mut [f32]) {
        // Transparent until the IR convolver is introduced.
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
    fn placeholder_cabinet_is_bit_exact() {
        let mut cabinet = CabinetProcessor::default();
        cabinet.reset(&AudioConfig::new(44_100.0, 512));
        let input = [0.0, -1.0, 0.125, 0.5, 1.0];
        let mut output = [f32::NAN; 5];
        cabinet.process_block(&input, &mut output);
        assert_eq!(output, input);
        assert_eq!(cabinet.latency_samples(), 0);
        assert_eq!(cabinet.tail_samples(), 0);
    }
}
