use truce::prelude::AudioConfig;

use crate::amp::AmpControls;
use crate::runtime::{PreparedRuntime, RuntimeMailbox, RuntimeUpdate};

const MUTE_RAMP_SECONDS: f32 = 0.003;
const RUNTIME_CROSSFADE_SAMPLES: usize = 480;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RuntimeApplyStatus {
    Ready { generation: u64 },
    SafeMuted { generation: u64 },
}

/// Zero-latency mono amp + cabinet branch.
///
/// A worker constructs complete runtimes. The callback takes one at a host
/// block boundary, evaluates old and new runtimes at the same sample positions,
/// and crossfades their outputs. There is no lookahead, internal quantum, or
/// block accumulation. Heap-backed retired runtimes travel back to the worker
/// and are never destroyed inside `process_block`.
pub struct GuitarSignalChain {
    current: Box<PreparedRuntime>,
    previous: Option<Box<PreparedRuntime>>,
    amp_scratch: Vec<f32>,
    previous_amp_scratch: Vec<f32>,
    previous_output: Vec<f32>,
    crossfade_gains: Vec<(f32, f32)>,
    crossfade_position: usize,
    safety_gain: f32,
    safety_target: f32,
    safety_step: f32,
    safety_remaining: usize,
    #[cfg(test)]
    processed_samples: u64,
}

impl Default for GuitarSignalChain {
    fn default() -> Self {
        Self {
            current: Box::new(PreparedRuntime::transparent()),
            previous: None,
            amp_scratch: Vec::new(),
            previous_amp_scratch: Vec::new(),
            previous_output: Vec::new(),
            crossfade_gains: Vec::new(),
            crossfade_position: RUNTIME_CROSSFADE_SAMPLES,
            safety_gain: 1.0,
            safety_target: 1.0,
            safety_step: 0.0,
            safety_remaining: 0,
            #[cfg(test)]
            processed_samples: 0,
        }
    }
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
        self.current.reset(config);
        // Reset is explicitly non-real-time, so a half-completed swap can be
        // destroyed here without violating the callback contract.
        self.previous = None;
        self.amp_scratch.resize(config.max_block_size, 0.0);
        self.previous_amp_scratch.resize(config.max_block_size, 0.0);
        self.previous_output.resize(config.max_block_size, 0.0);
        self.crossfade_gains
            .resize(RUNTIME_CROSSFADE_SAMPLES, (0.0, 0.0));
        for (index, (old_gain, new_gain)) in self.crossfade_gains.iter_mut().enumerate() {
            let phase =
                std::f32::consts::FRAC_PI_2 * (index + 1) as f32 / RUNTIME_CROSSFADE_SAMPLES as f32;
            let (new, old) = phase.sin_cos();
            // Normalize the equal-power-shaped curve for correlated material.
            // Two amp models fed by the same DI are highly correlated; raw
            // sin/cos gains would otherwise create a +3 dB midpoint bump.
            let unity = 1.0 / (old + new).max(f32::MIN_POSITIVE);
            *old_gain = old * unity;
            *new_gain = new * unity;
        }
        self.amp_scratch.fill(0.0);
        self.previous_amp_scratch.fill(0.0);
        self.previous_output.fill(0.0);
        self.crossfade_position = RUNTIME_CROSSFADE_SAMPLES;
        self.safety_gain = 1.0;
        self.safety_target = 1.0;
        self.safety_step = 0.0;
        self.safety_remaining = 0;
        #[cfg(test)]
        {
            self.processed_samples = 0;
        }
    }

    /// Applies the newest worker result at a host block boundary.
    ///
    /// A previous runtime whose crossfade has ended is first returned to the
    /// worker. If retirement is backpressured, this method simply waits and
    /// leaves the new update in its queue.
    #[inline]
    pub fn poll_runtime(&mut self, mailbox: &RuntimeMailbox) -> Option<RuntimeApplyStatus> {
        if self.crossfade_position >= RUNTIME_CROSSFADE_SAMPLES
            && let Some(previous) = self.previous.take()
            && let Err(returned) = mailbox.try_retire(previous)
        {
            self.previous = Some(returned);
            return None;
        }
        if self.previous.is_some() {
            return None;
        }
        let update = mailbox.take_latest()?;
        match update {
            RuntimeUpdate::Ready {
                generation,
                runtime,
            } => {
                let both_transparent = self.current.model_reference().is_none()
                    && self.current.ir_reference().is_none()
                    && runtime.model_reference().is_none()
                    && runtime.ir_reference().is_none();
                let previous = std::mem::replace(&mut self.current, runtime);
                self.previous = Some(previous);
                self.crossfade_position = if both_transparent {
                    RUNTIME_CROSSFADE_SAMPLES
                } else {
                    0
                };
                self.set_safety_target(1.0);
                Some(RuntimeApplyStatus::Ready { generation })
            }
            RuntimeUpdate::Mute { generation, .. } => {
                self.set_safety_target(0.0);
                Some(RuntimeApplyStatus::SafeMuted { generation })
            }
        }
    }

    pub fn set_controls(&mut self, controls: AmpControls) {
        self.current.set_controls(controls);
        if let Some(previous) = &mut self.previous {
            previous.set_controls(controls);
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

        self.current
            .process_block(input, &mut self.amp_scratch, output);
        if self.crossfade_position < RUNTIME_CROSSFADE_SAMPLES {
            if let Some(previous) = &mut self.previous {
                let old_output = &mut self.previous_output[..input.len()];
                previous.process_block(input, &mut self.previous_amp_scratch, old_output);
                for (new, old) in output.iter_mut().zip(old_output) {
                    let (old_gain, new_gain) = self.crossfade_gains[self.crossfade_position];
                    *new = old.mul_add(old_gain, *new * new_gain);
                    self.crossfade_position =
                        (self.crossfade_position + 1).min(RUNTIME_CROSSFADE_SAMPLES);
                }
            } else {
                self.crossfade_position = RUNTIME_CROSSFADE_SAMPLES;
            }
        }

        for sample in output {
            *sample *= self.next_safety_gain();
        }
    }

    fn set_safety_target(&mut self, target: f32) {
        let target = target.clamp(0.0, 1.0);
        if target != self.safety_target {
            self.safety_target = target;
            self.safety_remaining = RUNTIME_CROSSFADE_SAMPLES;
            self.safety_step =
                (self.safety_target - self.safety_gain) / RUNTIME_CROSSFADE_SAMPLES as f32;
        }
    }

    #[inline]
    fn next_safety_gain(&mut self) -> f32 {
        if self.safety_remaining > 0 {
            self.safety_gain += self.safety_step;
            self.safety_remaining -= 1;
            if self.safety_remaining == 0 {
                self.safety_gain = self.safety_target;
            }
        }
        self.safety_gain
    }

    #[must_use]
    pub fn latency_samples(&self) -> u32 {
        debug_assert_eq!(self.current.latency_samples(), 0);
        0
    }

    #[must_use]
    pub fn tail_samples(&self) -> u32 {
        self.previous.as_ref().map_or_else(
            || self.current.tail_samples(),
            |previous| previous.tail_samples().max(self.current.tail_samples()),
        )
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
    fn transparent_mono_chain_is_bit_exact() {
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
    fn arbitrary_host_block_sizes_do_not_change_transparent_output() {
        let source: Vec<f32> = (0..1024)
            .map(|index| ((index as f32 * 0.037).sin() * 0.75).clamp(-1.0, 1.0))
            .collect();
        for block_size in [1, 7, 16, 32, 64, 257, 512] {
            let mut chain = GuitarSignalChain::default();
            chain.reset(&AudioConfig::new(48_000.0, 512));
            let mut rendered = Vec::with_capacity(source.len());
            for block in source.chunks(block_size) {
                let mut output = vec![0.0; block.len()];
                chain.process_block(block, &mut output);
                rendered.extend(output);
            }
            assert_eq!(rendered, source, "block size {block_size}");
        }
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
