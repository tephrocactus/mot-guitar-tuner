//! Deterministic native-Rust trainer for the compact amplifier runtime.
//!
//! The runtime model lives in [`crate::amp`]. This module trains and exports
//! that exact equation: independent recurrent `tanh` units driven by the
//! current input, plus a dry path and linear output head. There is no Python,
//! process-global mutable state, or dependency on the audio callback.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(test)]
use crate::amp::AMP_MODEL_FORMAT_VERSION;
use crate::amp::{AMP_MODEL_SAMPLE_RATE_HZ, CompactCausalModel, MAX_MODEL_UNITS};

pub const TRAINING_SAMPLE_RATE_HZ: u32 = AMP_MODEL_SAMPLE_RATE_HZ;
pub const DEFAULT_MAX_PASSES: u16 = 400;
pub const MAX_PASSES_LIMIT: u16 = 400;
pub const DEFAULT_MODEL_UNITS: u8 = 16;
/// Must match `model::DIAGONAL_RNN_ARCHITECTURE_ID`.
pub const MODEL_ARCHITECTURE_ID: &str = "mot.diagonal-rnn-tanh";

/// Training is deliberately sample-budgeted. A 190-second capture therefore
/// does not turn 400 optimizer steps into 3.6 billion sequential samples.
/// Deterministic, stratified windows move on every pass and eventually cover
/// the whole capture.
pub const DEFAULT_TRAINING_SAMPLES_PER_PASS: usize = 32_768;
pub const DEFAULT_VALIDATION_SAMPLES_PER_PASS: usize = 16_384;
pub const DEFAULT_TRAINING_WINDOW_SAMPLES: usize = 2_048;
pub const DEFAULT_WARMUP_SAMPLES: usize = 256;

const TRAINABLE_PARAMETER_COUNT: usize = 2 + MAX_MODEL_UNITS * 4;
const PORTABLE_PAYLOAD_MAGIC: &[u8; 8] = b"MOTRNN01";
const PORTABLE_PAYLOAD_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrainerConfig {
    pub max_passes: u16,
    pub unit_count: u8,
    pub learning_rate: f32,
    pub validation_stride: usize,
    pub early_stopping_patience: u16,
    pub minimum_validation_improvement: f64,
    pub l2_regularization: f32,
    pub gradient_norm_limit: f32,
    pub training_samples_per_pass: usize,
    pub validation_samples_per_pass: usize,
    pub window_samples: usize,
    pub warmup_samples: usize,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            max_passes: DEFAULT_MAX_PASSES,
            unit_count: DEFAULT_MODEL_UNITS,
            learning_rate: 0.01,
            validation_stride: 10,
            early_stopping_patience: 40,
            minimum_validation_improvement: 1.0e-9,
            l2_regularization: 1.0e-7,
            gradient_norm_limit: 10.0,
            training_samples_per_pass: DEFAULT_TRAINING_SAMPLES_PER_PASS,
            validation_samples_per_pass: DEFAULT_VALIDATION_SAMPLES_PER_PASS,
            window_samples: DEFAULT_TRAINING_WINDOW_SAMPLES,
            warmup_samples: DEFAULT_WARMUP_SAMPLES,
        }
    }
}

impl TrainerConfig {
    pub fn validate(self) -> Result<Self, TrainingError> {
        if !(1..=MAX_PASSES_LIMIT).contains(&self.max_passes) {
            return Err(TrainingError::InvalidMaxPasses(self.max_passes));
        }
        if self.unit_count == 0 || usize::from(self.unit_count) > MAX_MODEL_UNITS {
            return Err(TrainingError::InvalidUnitCount(self.unit_count));
        }
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err(TrainingError::InvalidLearningRate);
        }
        if self.validation_stride < 2 {
            return Err(TrainingError::InvalidValidationStride);
        }
        if self.early_stopping_patience == 0 {
            return Err(TrainingError::InvalidPatience);
        }
        if !self.minimum_validation_improvement.is_finite()
            || self.minimum_validation_improvement < 0.0
        {
            return Err(TrainingError::InvalidMinimumImprovement);
        }
        if !self.l2_regularization.is_finite() || self.l2_regularization < 0.0 {
            return Err(TrainingError::InvalidRegularization);
        }
        if !self.gradient_norm_limit.is_finite() || self.gradient_norm_limit <= 0.0 {
            return Err(TrainingError::InvalidGradientLimit);
        }
        if self.training_samples_per_pass == 0
            || self.validation_samples_per_pass == 0
            || self.window_samples == 0
            || self.training_samples_per_pass < self.validation_stride
            || self.validation_samples_per_pass < self.validation_stride
            || self.window_samples < self.validation_stride
            || self.warmup_samples > self.window_samples.saturating_mul(8)
        {
            return Err(TrainingError::InvalidWindowBudget);
        }
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TrainingData<'audio> {
    pub input: &'audio [f32],
    pub target: &'audio [f32],
    pub sample_rate_hz: u32,
}

impl TrainingData<'_> {
    pub fn validate(&self, validation_stride: usize) -> Result<(), TrainingError> {
        if self.sample_rate_hz != TRAINING_SAMPLE_RATE_HZ {
            return Err(TrainingError::UnsupportedSampleRate(self.sample_rate_hz));
        }
        if self.input.len() != self.target.len() {
            return Err(TrainingError::LengthMismatch {
                input: self.input.len(),
                target: self.target.len(),
            });
        }
        if self.input.len() < validation_stride * 2 {
            return Err(TrainingError::DatasetTooShort);
        }
        if self
            .input
            .iter()
            .chain(self.target.iter())
            .any(|sample| !sample.is_finite())
        {
            return Err(TrainingError::NonFiniteAudio);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum TrainingError {
    InvalidMaxPasses(u16),
    InvalidUnitCount(u8),
    InvalidLearningRate,
    InvalidValidationStride,
    InvalidPatience,
    InvalidMinimumImprovement,
    InvalidRegularization,
    InvalidGradientLimit,
    InvalidWindowBudget,
    UnsupportedSampleRate(u32),
    LengthMismatch { input: usize, target: usize },
    DatasetTooShort,
    NonFiniteAudio,
    InvalidModelPayload,
}

impl fmt::Display for TrainingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMaxPasses(value) => {
                write!(
                    formatter,
                    "max_passes must be within 1..=400, received {value}"
                )
            }
            Self::InvalidUnitCount(value) => write!(
                formatter,
                "unit_count must be within 1..={MAX_MODEL_UNITS}, received {value}"
            ),
            Self::InvalidLearningRate => {
                formatter.write_str("learning_rate must be finite and positive")
            }
            Self::InvalidValidationStride => {
                formatter.write_str("validation_stride must be at least two")
            }
            Self::InvalidPatience => {
                formatter.write_str("early_stopping_patience must be at least one")
            }
            Self::InvalidMinimumImprovement => formatter
                .write_str("minimum_validation_improvement must be finite and non-negative"),
            Self::InvalidRegularization => {
                formatter.write_str("l2_regularization must be finite and non-negative")
            }
            Self::InvalidGradientLimit => {
                formatter.write_str("gradient_norm_limit must be finite and positive")
            }
            Self::InvalidWindowBudget => formatter.write_str(
                "training/validation/window budgets must be non-zero and warmup must be bounded",
            ),
            Self::UnsupportedSampleRate(rate) => {
                write!(formatter, "training requires 48000 Hz, received {rate} Hz")
            }
            Self::LengthMismatch { input, target } => {
                write!(
                    formatter,
                    "input length {input} does not match target length {target}"
                )
            }
            Self::DatasetTooShort => {
                formatter.write_str("the aligned training dataset is too short")
            }
            Self::NonFiniteAudio => {
                formatter.write_str("training audio contains a non-finite sample")
            }
            Self::InvalidModelPayload => {
                formatter.write_str("portable amplifier model payload is invalid")
            }
        }
    }
}

impl std::error::Error for TrainingError {}

#[derive(Clone, Debug, Default)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    pub fn reset(&self) {
        self.cancelled.store(false, Ordering::Release);
    }

    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TrainingStopReason {
    MaximumPasses,
    EarlyStopping,
    Cancelled,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrainingProgress {
    pub completed_passes: u16,
    pub maximum_passes: u16,
    pub training_loss: f64,
    pub validation_loss: f64,
    pub best_validation_loss: f64,
    pub best_pass: u16,
}

#[derive(Clone, Debug)]
pub struct TrainingOutcome {
    pub best_model: CompactCausalModel,
    pub completed_passes: u16,
    pub best_pass: u16,
    pub best_validation_loss: f64,
    pub stop_reason: TrainingStopReason,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RuntimeModelDescriptor {
    pub architecture_id: &'static str,
    pub architecture_version: u32,
    pub causal: bool,
    pub lookahead_samples: u32,
    pub runtime_latency_samples: u32,
    pub sample_rate_hz: u32,
    pub estimated_macs_per_sample: u32,
    pub unit_count: u8,
}

#[must_use]
pub fn model_descriptor(model: &CompactCausalModel) -> RuntimeModelDescriptor {
    RuntimeModelDescriptor {
        architecture_id: MODEL_ARCHITECTURE_ID,
        architecture_version: model.format_version,
        causal: model.causal,
        lookahead_samples: model.lookahead_samples,
        runtime_latency_samples: model.runtime_latency_samples,
        sample_rate_hz: model.sample_rate_hz,
        estimated_macs_per_sample: model.estimated_macs_per_sample,
        unit_count: model.unit_count,
    }
}

/// Trains the exact model consumed by `AmpProcessor`.
///
/// Every runtime coefficient is optimized. Because each recurrent unit is
/// diagonal, its exact causal sensitivities can be propagated online with
/// constant memory:
///
/// `dh[n]/da = (1-h[n]^2) * (x[n] + r * dh[n-1]/da)`.
///
/// The analogous equations for recurrent weight and bias make this an exact
/// forward-mode gradient for the model, rather than an output-head-only
/// reservoir fit. Long captures are sampled through deterministic stratified
/// windows, so 400 passes remain meaningful without scanning all 9.12 million
/// samples 400 times.
pub fn train_compact_model(
    data: TrainingData<'_>,
    config: TrainerConfig,
    cancellation: &CancellationToken,
    mut progress: impl FnMut(TrainingProgress),
) -> Result<TrainingOutcome, TrainingError> {
    let config = config.validate()?;
    data.validate(config.validation_stride)?;

    let mut model = initialized_reservoir(config.unit_count);
    let mut best_model = model;
    let mut first_moment = [0.0_f64; TRAINABLE_PARAMETER_COUNT];
    let mut second_moment = [0.0_f64; TRAINABLE_PARAMETER_COUNT];
    let mut best_validation_loss = validation_loss(&model, data, config);
    let mut best_pass = 0;
    let mut passes_without_improvement = 0_u16;
    let mut completed_passes = 0_u16;
    let mut stop_reason = TrainingStopReason::MaximumPasses;

    for pass in 1..=config.max_passes {
        if cancellation.is_cancelled() {
            stop_reason = TrainingStopReason::Cancelled;
            break;
        }

        let mut accumulator = GradientAccumulator::default();
        let mut cancelled_mid_pass = false;
        for_each_window(
            data.input.len(),
            config.training_samples_per_pass,
            config.window_samples,
            config.warmup_samples,
            u64::from(pass),
            0x5452_4149_4e49_4e47,
            |window| {
                if !cancelled_mid_pass {
                    cancelled_mid_pass = !accumulate_training_window(
                        &model,
                        data,
                        config.validation_stride,
                        window,
                        cancellation,
                        &mut accumulator,
                    );
                }
            },
        );
        if cancelled_mid_pass {
            stop_reason = TrainingStopReason::Cancelled;
            break;
        }

        let training_loss = accumulator.finish(
            &model,
            f64::from(config.l2_regularization),
            f64::from(config.gradient_norm_limit),
        );
        let learning_rate = scheduled_learning_rate(config, pass);
        adam_update_all_parameters(
            &mut model,
            &accumulator.gradient,
            &mut first_moment,
            &mut second_moment,
            pass,
            learning_rate,
        );

        let current_validation_loss = validation_loss(&model, data, config);
        completed_passes = pass;
        if best_validation_loss - current_validation_loss > config.minimum_validation_improvement {
            best_validation_loss = current_validation_loss;
            best_model = model;
            best_pass = pass;
            passes_without_improvement = 0;
        } else {
            passes_without_improvement = passes_without_improvement.saturating_add(1);
        }

        progress(TrainingProgress {
            completed_passes,
            maximum_passes: config.max_passes,
            training_loss,
            validation_loss: current_validation_loss,
            best_validation_loss,
            best_pass,
        });

        if passes_without_improvement >= config.early_stopping_patience {
            stop_reason = TrainingStopReason::EarlyStopping;
            break;
        }
    }

    Ok(TrainingOutcome {
        best_model,
        completed_passes,
        best_pass,
        best_validation_loss,
        stop_reason,
    })
}

/// Encodes the runtime-model payload. The outer `.motmodel` container is
/// responsible for IDs, metadata, checksums, and immutable asset versioning.
#[must_use]
pub fn encode_model_payload(model: &CompactCausalModel) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(48 + MAX_MODEL_UNITS * 16);
    bytes.extend_from_slice(PORTABLE_PAYLOAD_MAGIC);
    bytes.extend_from_slice(&PORTABLE_PAYLOAD_VERSION.to_le_bytes());
    bytes.extend_from_slice(&model.format_version.to_le_bytes());
    bytes.extend_from_slice(&model.sample_rate_hz.to_le_bytes());
    bytes.push(u8::from(model.causal));
    bytes.extend_from_slice(&[0; 3]);
    bytes.extend_from_slice(&model.lookahead_samples.to_le_bytes());
    bytes.extend_from_slice(&model.runtime_latency_samples.to_le_bytes());
    bytes.extend_from_slice(&model.estimated_macs_per_sample.to_le_bytes());
    bytes.push(model.unit_count);
    bytes.extend_from_slice(&[0; 3]);
    bytes.extend_from_slice(&model.dry_gain.to_le_bytes());
    bytes.extend_from_slice(&model.output_bias.to_le_bytes());
    for coefficients in [
        &model.input_weights,
        &model.recurrent_weights,
        &model.biases,
        &model.output_weights,
    ] {
        for coefficient in coefficients {
            bytes.extend_from_slice(&coefficient.to_le_bytes());
        }
    }
    bytes
}

pub fn decode_model_payload(bytes: &[u8]) -> Result<CompactCausalModel, TrainingError> {
    let expected_length = 48 + MAX_MODEL_UNITS * 16;
    if bytes.len() != expected_length || bytes.get(..8) != Some(PORTABLE_PAYLOAD_MAGIC.as_slice()) {
        return Err(TrainingError::InvalidModelPayload);
    }
    let mut cursor = 8;
    let payload_version = take_u32(bytes, &mut cursor)?;
    if payload_version != PORTABLE_PAYLOAD_VERSION {
        return Err(TrainingError::InvalidModelPayload);
    }
    let format_version = take_u32(bytes, &mut cursor)?;
    let sample_rate_hz = take_u32(bytes, &mut cursor)?;
    let causal = match bytes.get(cursor).copied() {
        Some(0) => false,
        Some(1) => true,
        _ => return Err(TrainingError::InvalidModelPayload),
    };
    cursor += 4;
    let lookahead_samples = take_u32(bytes, &mut cursor)?;
    let runtime_latency_samples = take_u32(bytes, &mut cursor)?;
    let estimated_macs_per_sample = take_u32(bytes, &mut cursor)?;
    let unit_count = *bytes
        .get(cursor)
        .ok_or(TrainingError::InvalidModelPayload)?;
    cursor += 4;
    let dry_gain = take_f32(bytes, &mut cursor)?;
    let output_bias = take_f32(bytes, &mut cursor)?;
    let mut input_weights = [0.0; MAX_MODEL_UNITS];
    let mut recurrent_weights = [0.0; MAX_MODEL_UNITS];
    let mut biases = [0.0; MAX_MODEL_UNITS];
    let mut output_weights = [0.0; MAX_MODEL_UNITS];
    for coefficients in [
        &mut input_weights,
        &mut recurrent_weights,
        &mut biases,
        &mut output_weights,
    ] {
        for coefficient in coefficients {
            *coefficient = take_f32(bytes, &mut cursor)?;
        }
    }
    let model = CompactCausalModel {
        format_version,
        sample_rate_hz,
        causal,
        lookahead_samples,
        runtime_latency_samples,
        estimated_macs_per_sample,
        unit_count,
        dry_gain,
        output_bias,
        input_weights,
        recurrent_weights,
        biases,
        output_weights,
    };
    model
        .validate()
        .map_err(|_| TrainingError::InvalidModelPayload)?;
    Ok(model)
}

fn initialized_reservoir(unit_count: u8) -> CompactCausalModel {
    let mut model = CompactCausalModel::raw();
    model.unit_count = unit_count;
    model.dry_gain = 1.0;
    model.output_bias = 0.0;
    model.estimated_macs_per_sample = 3 + u32::from(unit_count) * 7;

    // A reproducible spread of drive, memory, and operating points. Small
    // non-zero output weights are essential: they let the exact recurrent
    // sensitivities move the internal coefficients from the first pass.
    let mut seed = 0x6d2b_79f5_u32;
    for unit in 0..usize::from(unit_count) {
        let drive = 0.55 + random_unit(&mut seed) * 5.45;
        let polarity = if unit & 1 == 0 { 1.0 } else { -1.0 };
        model.input_weights[unit] = drive * polarity;
        model.recurrent_weights[unit] = -0.88 + random_unit(&mut seed) * 1.76;
        model.biases[unit] = -0.65 + random_unit(&mut seed) * 1.30;
        model.output_weights[unit] = (random_unit(&mut seed) - 0.5) * 0.03;
    }
    model
}

fn random_unit(seed: &mut u32) -> f32 {
    *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    (*seed >> 8) as f32 / 16_777_215.0
}

#[inline]
fn advance_states(model: &CompactCausalModel, input: f32, states: &mut [f32; MAX_MODEL_UNITS]) {
    for (unit, state) in states
        .iter_mut()
        .enumerate()
        .take(usize::from(model.unit_count))
    {
        *state = (model.input_weights[unit] * input
            + model.recurrent_weights[unit] * *state
            + model.biases[unit])
            .tanh();
    }
}

#[inline]
fn evaluate_output(model: &CompactCausalModel, input: f32, states: &[f32; MAX_MODEL_UNITS]) -> f32 {
    let mut output = model.output_bias + model.dry_gain * input;
    for (unit, state) in states
        .iter()
        .enumerate()
        .take(usize::from(model.unit_count))
    {
        output += model.output_weights[unit] * state;
    }
    output
}

#[derive(Clone, Copy, Debug)]
struct TrainingWindow {
    warmup_start: usize,
    training_start: usize,
    training_end: usize,
}

fn for_each_window(
    data_length: usize,
    sample_budget: usize,
    window_samples: usize,
    warmup_samples: usize,
    epoch: u64,
    salt: u64,
    mut callback: impl FnMut(TrainingWindow),
) {
    if sample_budget >= data_length {
        callback(TrainingWindow {
            warmup_start: 0,
            training_start: 0,
            training_end: data_length,
        });
        return;
    }

    let window_count = sample_budget.div_ceil(window_samples);
    let mut remaining = sample_budget;
    for ordinal in 0..window_count {
        let length = remaining.min(window_samples).min(data_length);
        remaining -= length;
        let possible_starts = data_length - length + 1;
        let bin_start = ordinal * possible_starts / window_count;
        let bin_end = ((ordinal + 1) * possible_starts / window_count).max(bin_start + 1);
        let hash = splitmix64(epoch ^ salt ^ (ordinal as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
        let training_start = bin_start + hash as usize % (bin_end - bin_start);
        callback(TrainingWindow {
            warmup_start: training_start.saturating_sub(warmup_samples),
            training_start,
            training_end: training_start + length,
        });
    }
}

#[inline]
fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[derive(Clone, Copy, Debug)]
struct OnlineSensitivities {
    input: [f32; MAX_MODEL_UNITS],
    recurrent: [f32; MAX_MODEL_UNITS],
    bias: [f32; MAX_MODEL_UNITS],
}

impl Default for OnlineSensitivities {
    fn default() -> Self {
        Self {
            input: [0.0; MAX_MODEL_UNITS],
            recurrent: [0.0; MAX_MODEL_UNITS],
            bias: [0.0; MAX_MODEL_UNITS],
        }
    }
}

#[inline]
fn advance_states_and_sensitivities(
    model: &CompactCausalModel,
    input: f32,
    states: &mut [f32; MAX_MODEL_UNITS],
    sensitivities: &mut OnlineSensitivities,
) {
    for (unit, state_slot) in states
        .iter_mut()
        .enumerate()
        .take(usize::from(model.unit_count))
    {
        let previous_state = *state_slot;
        let state = (model.input_weights[unit] * input
            + model.recurrent_weights[unit] * previous_state
            + model.biases[unit])
            .tanh();
        let slope = (1.0 - state * state).max(0.0);
        sensitivities.input[unit] =
            slope * (input + model.recurrent_weights[unit] * sensitivities.input[unit]);
        sensitivities.recurrent[unit] = slope
            * (previous_state + model.recurrent_weights[unit] * sensitivities.recurrent[unit]);
        sensitivities.bias[unit] =
            slope * (1.0 + model.recurrent_weights[unit] * sensitivities.bias[unit]);
        *state_slot = state;
    }
}

#[derive(Clone, Debug)]
struct GradientAccumulator {
    gradient: [f64; TRAINABLE_PARAMETER_COUNT],
    squared_error: f64,
    sample_count: usize,
}

impl Default for GradientAccumulator {
    fn default() -> Self {
        Self {
            gradient: [0.0; TRAINABLE_PARAMETER_COUNT],
            squared_error: 0.0,
            sample_count: 0,
        }
    }
}

impl GradientAccumulator {
    fn add_sample(
        &mut self,
        model: &CompactCausalModel,
        input: f32,
        states: &[f32; MAX_MODEL_UNITS],
        sensitivities: &OnlineSensitivities,
        target: f32,
    ) {
        let error = f64::from(evaluate_output(model, input, states)) - f64::from(target);
        let twice_error = error * 2.0;
        self.squared_error += error * error;
        self.sample_count += 1;
        self.gradient[0] += twice_error;
        self.gradient[1] += twice_error * f64::from(input);
        for (unit, state) in states
            .iter()
            .enumerate()
            .take(usize::from(model.unit_count))
        {
            let output_weight = f64::from(model.output_weights[unit]);
            self.gradient[output_weight_index(unit)] += twice_error * f64::from(*state);
            self.gradient[input_weight_index(unit)] +=
                twice_error * output_weight * f64::from(sensitivities.input[unit]);
            self.gradient[recurrent_weight_index(unit)] +=
                twice_error * output_weight * f64::from(sensitivities.recurrent[unit]);
            self.gradient[bias_index(unit)] +=
                twice_error * output_weight * f64::from(sensitivities.bias[unit]);
        }
    }

    fn finish(
        &mut self,
        model: &CompactCausalModel,
        l2_regularization: f64,
        gradient_norm_limit: f64,
    ) -> f64 {
        let reciprocal = 1.0 / self.sample_count.max(1) as f64;
        for value in &mut self.gradient {
            *value *= reciprocal;
        }
        add_l2_gradient(model, &mut self.gradient, l2_regularization);
        limit_gradient_norm(&mut self.gradient, gradient_norm_limit);
        self.squared_error * reciprocal
    }
}

fn accumulate_training_window(
    model: &CompactCausalModel,
    data: TrainingData<'_>,
    validation_stride: usize,
    window: TrainingWindow,
    cancellation: &CancellationToken,
    accumulator: &mut GradientAccumulator,
) -> bool {
    let mut states = [0.0_f32; MAX_MODEL_UNITS];
    let mut sensitivities = OnlineSensitivities::default();
    for index in window.warmup_start..window.training_end {
        let input = data.input[index];
        advance_states_and_sensitivities(model, input, &mut states, &mut sensitivities);
        if index >= window.training_start && index % validation_stride != 0 {
            accumulator.add_sample(model, input, &states, &sensitivities, data.target[index]);
        }
        if index & 0x0fff == 0 && cancellation.is_cancelled() {
            return false;
        }
    }
    true
}

fn validation_loss(
    model: &CompactCausalModel,
    data: TrainingData<'_>,
    config: TrainerConfig,
) -> f64 {
    let mut squared_error = 0.0_f64;
    let mut count = 0_usize;
    for_each_window(
        data.input.len(),
        config.validation_samples_per_pass,
        config.window_samples,
        config.warmup_samples,
        0,
        0x5641_4c49_4441_5445,
        |window| {
            let mut states = [0.0_f32; MAX_MODEL_UNITS];
            for index in window.warmup_start..window.training_end {
                let input = data.input[index];
                advance_states(model, input, &mut states);
                if index >= window.training_start && index % config.validation_stride == 0 {
                    let error = f64::from(evaluate_output(model, input, &states))
                        - f64::from(data.target[index]);
                    squared_error += error * error;
                    count += 1;
                }
            }
        },
    );
    squared_error / count.max(1) as f64
}

fn limit_gradient_norm(gradient: &mut [f64; TRAINABLE_PARAMETER_COUNT], limit: f64) {
    let norm = gradient
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    if norm > limit {
        let scale = limit / norm;
        for value in gradient {
            *value *= scale;
        }
    }
}

const fn output_weight_index(unit: usize) -> usize {
    2 + unit
}

const fn input_weight_index(unit: usize) -> usize {
    2 + MAX_MODEL_UNITS + unit
}

const fn recurrent_weight_index(unit: usize) -> usize {
    2 + MAX_MODEL_UNITS * 2 + unit
}

const fn bias_index(unit: usize) -> usize {
    2 + MAX_MODEL_UNITS * 3 + unit
}

fn add_l2_gradient(
    model: &CompactCausalModel,
    gradient: &mut [f64; TRAINABLE_PARAMETER_COUNT],
    strength: f64,
) {
    gradient[1] += strength * f64::from(model.dry_gain);
    for unit in 0..usize::from(model.unit_count) {
        gradient[output_weight_index(unit)] += strength * f64::from(model.output_weights[unit]);
        gradient[input_weight_index(unit)] += strength * f64::from(model.input_weights[unit]);
        gradient[recurrent_weight_index(unit)] +=
            strength * f64::from(model.recurrent_weights[unit]);
        gradient[bias_index(unit)] += strength * f64::from(model.biases[unit]);
    }
}

fn scheduled_learning_rate(config: TrainerConfig, pass: u16) -> f64 {
    let progress =
        f64::from(pass.saturating_sub(1)) / f64::from(config.max_passes.saturating_sub(1).max(1));
    let cosine = (std::f64::consts::PI * progress).cos();
    f64::from(config.learning_rate) * (0.1 + 0.9 * 0.5 * (1.0 + cosine))
}

fn adam_update_all_parameters(
    model: &mut CompactCausalModel,
    gradient: &[f64; TRAINABLE_PARAMETER_COUNT],
    first_moment: &mut [f64; TRAINABLE_PARAMETER_COUNT],
    second_moment: &mut [f64; TRAINABLE_PARAMETER_COUNT],
    pass: u16,
    learning_rate: f64,
) {
    const BETA_ONE: f64 = 0.9;
    const BETA_TWO: f64 = 0.999;
    let bias_one = 1.0 - BETA_ONE.powi(i32::from(pass));
    let bias_two = 1.0 - BETA_TWO.powi(i32::from(pass));
    update_parameter(
        model,
        0,
        adam_update(
            0,
            gradient,
            first_moment,
            second_moment,
            bias_one,
            bias_two,
            learning_rate,
        ),
    );
    update_parameter(
        model,
        1,
        adam_update(
            1,
            gradient,
            first_moment,
            second_moment,
            bias_one,
            bias_two,
            learning_rate,
        ),
    );
    for unit in 0..usize::from(model.unit_count) {
        for index in [
            output_weight_index(unit),
            input_weight_index(unit),
            recurrent_weight_index(unit),
            bias_index(unit),
        ] {
            let update = adam_update(
                index,
                gradient,
                first_moment,
                second_moment,
                bias_one,
                bias_two,
                learning_rate,
            );
            update_parameter(model, index, update);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn adam_update(
    index: usize,
    gradient: &[f64; TRAINABLE_PARAMETER_COUNT],
    first_moment: &mut [f64; TRAINABLE_PARAMETER_COUNT],
    second_moment: &mut [f64; TRAINABLE_PARAMETER_COUNT],
    bias_one: f64,
    bias_two: f64,
    learning_rate: f64,
) -> f32 {
    const BETA_ONE: f64 = 0.9;
    const BETA_TWO: f64 = 0.999;
    const EPSILON: f64 = 1.0e-8;
    first_moment[index] = BETA_ONE * first_moment[index] + (1.0 - BETA_ONE) * gradient[index];
    second_moment[index] =
        BETA_TWO * second_moment[index] + (1.0 - BETA_TWO) * gradient[index] * gradient[index];
    let corrected_first = first_moment[index] / bias_one;
    let corrected_second = second_moment[index] / bias_two;
    (learning_rate * corrected_first / (corrected_second.sqrt() + EPSILON)) as f32
}

fn update_parameter(model: &mut CompactCausalModel, index: usize, update: f32) {
    match index {
        0 => model.output_bias = (model.output_bias - update).clamp(-4.0, 4.0),
        1 => model.dry_gain = (model.dry_gain - update).clamp(-4.0, 4.0),
        index if index < 2 + MAX_MODEL_UNITS => {
            let unit = index - 2;
            model.output_weights[unit] = (model.output_weights[unit] - update).clamp(-4.0, 4.0);
        }
        index if index < 2 + MAX_MODEL_UNITS * 2 => {
            let unit = index - (2 + MAX_MODEL_UNITS);
            model.input_weights[unit] = (model.input_weights[unit] - update).clamp(-16.0, 16.0);
        }
        index if index < 2 + MAX_MODEL_UNITS * 3 => {
            let unit = index - (2 + MAX_MODEL_UNITS * 2);
            model.recurrent_weights[unit] =
                (model.recurrent_weights[unit] - update).clamp(-0.985, 0.985);
        }
        _ => {
            let unit = index - (2 + MAX_MODEL_UNITS * 3);
            model.biases[unit] = (model.biases[unit] - update).clamp(-4.0, 4.0);
        }
    }
}

fn take_u32(bytes: &[u8], cursor: &mut usize) -> Result<u32, TrainingError> {
    let chunk = bytes
        .get(*cursor..*cursor + 4)
        .ok_or(TrainingError::InvalidModelPayload)?;
    *cursor += 4;
    Ok(u32::from_le_bytes(
        chunk
            .try_into()
            .map_err(|_| TrainingError::InvalidModelPayload)?,
    ))
}

fn take_f32(bytes: &[u8], cursor: &mut usize) -> Result<f32, TrainingError> {
    Ok(f32::from_bits(take_u32(bytes, cursor)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deterministic_input(length: usize) -> Vec<f32> {
        let mut state = 0x1234_5678_u32;
        (0..length)
            .map(|index| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let noise = ((state >> 8) as f32 / 16_777_215.0) * 2.0 - 1.0;
                (noise * 0.55 + (index as f32 * 0.037).sin() * 0.2).clamp(-0.8, 0.8)
            })
            .collect()
    }

    fn render_model(model: &CompactCausalModel, input: &[f32]) -> Vec<f32> {
        let mut states = [0.0; MAX_MODEL_UNITS];
        input
            .iter()
            .map(|&input| {
                advance_states(model, input, &mut states);
                evaluate_output(model, input, &states)
            })
            .collect()
    }

    fn synthetic_target(input: &[f32], unit_count: u8) -> Vec<f32> {
        let mut target_model = initialized_reservoir(unit_count);
        target_model.dry_gain = 0.73;
        target_model.output_bias = 0.015;
        target_model.output_weights[0] = 0.19;
        target_model.output_weights[1] = -0.11;
        target_model.output_weights[3] = 0.07;
        render_model(&target_model, input)
    }

    fn nonlinear_dynamic_target(input: &[f32]) -> Vec<f32> {
        let mut target_model = CompactCausalModel::raw();
        target_model.unit_count = 4;
        target_model.estimated_macs_per_sample = 31;
        target_model.dry_gain = 0.18;
        target_model.output_bias = 0.012;
        target_model.input_weights[..4].copy_from_slice(&[2.4, -4.1, 6.2, -1.35]);
        target_model.recurrent_weights[..4].copy_from_slice(&[0.72, -0.48, 0.31, 0.88]);
        target_model.biases[..4].copy_from_slice(&[0.03, -0.16, 0.22, -0.04]);
        target_model.output_weights[..4].copy_from_slice(&[0.47, -0.21, 0.16, 0.09]);
        render_model(&target_model, input)
    }

    fn mean_squared_error(actual: &[f32], expected: &[f32]) -> f64 {
        actual
            .iter()
            .zip(expected)
            .map(|(&actual, &expected)| {
                let error = f64::from(actual) - f64::from(expected);
                error * error
            })
            .sum::<f64>()
            / actual.len() as f64
    }

    fn assert_same_model(left: &CompactCausalModel, right: &CompactCausalModel) {
        assert_eq!(encode_model_payload(left), encode_model_payload(right));
    }

    #[test]
    fn defaults_to_four_hundred_bounded_passes() {
        let config = TrainerConfig::default();
        assert_eq!(config.max_passes, 400);
        assert!(config.validate().is_ok());
        assert_eq!(
            TrainerConfig {
                max_passes: 0,
                ..config
            }
            .validate(),
            Err(TrainingError::InvalidMaxPasses(0))
        );
        assert_eq!(
            TrainerConfig {
                max_passes: 401,
                ..config
            }
            .validate(),
            Err(TrainingError::InvalidMaxPasses(401))
        );
    }

    #[test]
    fn training_is_deterministic_and_exports_best_checkpoint() {
        let input = deterministic_input(8_192);
        let config = TrainerConfig {
            max_passes: 220,
            early_stopping_patience: 60,
            ..TrainerConfig::default()
        };
        let target = synthetic_target(&input, config.unit_count);
        let data = TrainingData {
            input: &input,
            target: &target,
            sample_rate_hz: TRAINING_SAMPLE_RATE_HZ,
        };
        let token = CancellationToken::default();
        let first = train_compact_model(data, config, &token, |_| {}).unwrap();
        let second = train_compact_model(data, config, &token, |_| {}).unwrap();
        assert_same_model(&first.best_model, &second.best_model);
        assert_eq!(first.completed_passes, second.completed_passes);
        assert_eq!(first.best_pass, second.best_pass);
        assert_eq!(first.best_validation_loss, second.best_validation_loss);
        assert!(first.best_validation_loss < 1.0e-4, "{first:?}");
        assert!(first.best_pass <= first.completed_passes);
        first.best_model.validate().unwrap();
        let decoded = decode_model_payload(&encode_model_payload(&first.best_model)).unwrap();
        assert_same_model(&decoded, &first.best_model);
    }

    #[test]
    fn recurrent_coefficients_learn_a_nonlinear_dynamic_target() {
        let input = deterministic_input(12_288);
        let target = nonlinear_dynamic_target(&input);
        let initial = initialized_reservoir(4);
        let initial_loss = mean_squared_error(&render_model(&initial, &input), &target);
        let config = TrainerConfig {
            max_passes: 320,
            unit_count: 4,
            learning_rate: 0.008,
            early_stopping_patience: 80,
            ..TrainerConfig::default()
        };
        let trained = train_compact_model(
            TrainingData {
                input: &input,
                target: &target,
                sample_rate_hz: TRAINING_SAMPLE_RATE_HZ,
            },
            config,
            &CancellationToken::default(),
            |_| {},
        )
        .unwrap();
        let trained_loss = mean_squared_error(&render_model(&trained.best_model, &input), &target);
        assert!(
            trained_loss < initial_loss * 0.12,
            "initial={initial_loss:.8}, trained={trained_loss:.8}, outcome={trained:?}"
        );
        assert!((trained.best_model.input_weights[0] - initial.input_weights[0]).abs() > 1.0e-3);
        assert!(
            (trained.best_model.recurrent_weights[0] - initial.recurrent_weights[0]).abs() > 1.0e-3
        );
        assert!((trained.best_model.biases[0] - initial.biases[0]).abs() > 1.0e-3);
    }

    #[test]
    fn online_recurrent_sensitivities_match_finite_differences() {
        let input = deterministic_input(256);
        let target = nonlinear_dynamic_target(&input);
        let mut model = initialized_reservoir(1);
        model.output_weights[0] = 0.37;
        let data = TrainingData {
            input: &input,
            target: &target,
            sample_rate_hz: TRAINING_SAMPLE_RATE_HZ,
        };
        let window = TrainingWindow {
            warmup_start: 0,
            training_start: 0,
            training_end: input.len(),
        };
        let mut accumulator = GradientAccumulator::default();
        assert!(accumulate_training_window(
            &model,
            data,
            10,
            window,
            &CancellationToken::default(),
            &mut accumulator,
        ));
        accumulator.finish(&model, 0.0, f64::MAX);

        let finite_difference =
            |mut left: CompactCausalModel,
             mut right: CompactCausalModel,
             select: fn(&mut CompactCausalModel) -> &mut f32| {
                const EPSILON: f32 = 1.0e-3;
                *select(&mut left) -= EPSILON;
                *select(&mut right) += EPSILON;
                (training_loss_for_test(&right, data, 10) - training_loss_for_test(&left, data, 10))
                    / (2.0 * f64::from(EPSILON))
            };
        let checks = [
            (
                accumulator.gradient[input_weight_index(0)],
                finite_difference(model, model, |model| &mut model.input_weights[0]),
            ),
            (
                accumulator.gradient[recurrent_weight_index(0)],
                finite_difference(model, model, |model| &mut model.recurrent_weights[0]),
            ),
            (
                accumulator.gradient[bias_index(0)],
                finite_difference(model, model, |model| &mut model.biases[0]),
            ),
        ];
        for (analytic, numeric) in checks {
            assert!(
                (analytic - numeric).abs() < 2.0e-3,
                "analytic={analytic:.8}, numeric={numeric:.8}"
            );
        }
    }

    fn training_loss_for_test(
        model: &CompactCausalModel,
        data: TrainingData<'_>,
        validation_stride: usize,
    ) -> f64 {
        let mut states = [0.0_f32; MAX_MODEL_UNITS];
        let mut squared_error = 0.0_f64;
        let mut count = 0_usize;
        for (index, (&input, &target)) in data.input.iter().zip(data.target).enumerate() {
            advance_states(model, input, &mut states);
            if index % validation_stride != 0 {
                let error = f64::from(evaluate_output(model, input, &states)) - f64::from(target);
                squared_error += error * error;
                count += 1;
            }
        }
        squared_error / count as f64
    }

    #[test]
    fn long_capture_work_per_pass_is_bounded_and_stratified() {
        const CAPTURE_SAMPLES: usize = TRAINING_SAMPLE_RATE_HZ as usize * 190;
        let mut selected_samples = 0;
        let mut starts = Vec::new();
        for_each_window(
            CAPTURE_SAMPLES,
            DEFAULT_TRAINING_SAMPLES_PER_PASS,
            DEFAULT_TRAINING_WINDOW_SAMPLES,
            DEFAULT_WARMUP_SAMPLES,
            17,
            123,
            |window| {
                selected_samples += window.training_end - window.training_start;
                starts.push(window.training_start);
            },
        );
        assert_eq!(selected_samples, DEFAULT_TRAINING_SAMPLES_PER_PASS);
        assert_eq!(
            starts.len(),
            DEFAULT_TRAINING_SAMPLES_PER_PASS / DEFAULT_TRAINING_WINDOW_SAMPLES
        );
        assert!(starts.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(starts.last().copied().unwrap() > CAPTURE_SAMPLES * 9 / 10);
    }

    #[test]
    fn cancellation_returns_the_last_best_model() {
        let input = deterministic_input(2_048);
        let target = synthetic_target(&input, DEFAULT_MODEL_UNITS);
        let token = CancellationToken::default();
        let callback_token = token.clone();
        let result = train_compact_model(
            TrainingData {
                input: &input,
                target: &target,
                sample_rate_hz: TRAINING_SAMPLE_RATE_HZ,
            },
            TrainerConfig {
                max_passes: 50,
                ..TrainerConfig::default()
            },
            &token,
            move |progress| {
                if progress.completed_passes == 3 {
                    callback_token.cancel();
                }
            },
        )
        .unwrap();
        assert_eq!(result.stop_reason, TrainingStopReason::Cancelled);
        assert_eq!(result.completed_passes, 3);
        assert!(result.best_validation_loss.is_finite());
    }

    #[test]
    fn trained_model_contract_is_causal_and_zero_latency() {
        let model = initialized_reservoir(DEFAULT_MODEL_UNITS);
        let descriptor = model_descriptor(&model);
        assert_eq!(descriptor.architecture_version, AMP_MODEL_FORMAT_VERSION);
        assert!(descriptor.causal);
        assert_eq!(descriptor.lookahead_samples, 0);
        assert_eq!(descriptor.runtime_latency_samples, 0);
        assert_eq!(descriptor.sample_rate_hz, 48_000);

        let impulse = [1.0, 0.0, 0.0, 0.0];
        let output = render_model(&model, &impulse);
        assert_ne!(output[0], 0.0);
    }

    #[test]
    fn sample_rate_and_non_finite_audio_are_rejected() {
        let input = deterministic_input(128);
        let target = synthetic_target(&input, DEFAULT_MODEL_UNITS);
        let bad_rate = TrainingData {
            input: &input,
            target: &target,
            sample_rate_hz: 44_100,
        };
        assert_eq!(
            bad_rate.validate(10),
            Err(TrainingError::UnsupportedSampleRate(44_100))
        );
        let mut invalid = input;
        invalid[32] = f32::NAN;
        assert_eq!(
            TrainingData {
                input: &invalid,
                target: &target,
                sample_rate_hz: TRAINING_SAMPLE_RATE_HZ,
            }
            .validate(10),
            Err(TrainingError::NonFiniteAudio)
        );
    }

    /// Run with:
    /// `cargo test --release trainer_190_second_capture_budget -- --ignored --nocapture`
    #[test]
    #[ignore = "manual performance characterization on the target Mac"]
    fn trainer_190_second_capture_budget() {
        const SAMPLE_COUNT: usize = TRAINING_SAMPLE_RATE_HZ as usize * 190;
        let input = deterministic_input(SAMPLE_COUNT);
        let mut memory = 0.0_f32;
        let target: Vec<f32> = input
            .iter()
            .map(|&sample| {
                memory = (3.1 * sample + 0.76 * memory + 0.03).tanh();
                0.21 * sample + 0.62 * memory
            })
            .collect();
        let started = std::time::Instant::now();
        let outcome = train_compact_model(
            TrainingData {
                input: &input,
                target: &target,
                sample_rate_hz: TRAINING_SAMPLE_RATE_HZ,
            },
            TrainerConfig {
                early_stopping_patience: DEFAULT_MAX_PASSES,
                minimum_validation_improvement: 0.0,
                ..TrainerConfig::default()
            },
            &CancellationToken::default(),
            |_| {},
        )
        .unwrap();
        eprintln!(
            "190 s capture: {} passes in {:.3} s ({:.3} ms/pass), validation MSE {:.8}",
            outcome.completed_passes,
            started.elapsed().as_secs_f64(),
            started.elapsed().as_secs_f64() * 1_000.0 / f64::from(outcome.completed_passes.max(1)),
            outcome.best_validation_loss,
        );
        assert!(outcome.best_validation_loss.is_finite());
    }
}
