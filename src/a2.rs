//! Fixed-shape, allocation-free runtime for the NAM WaveNet A2 architecture.
//!
//! This module implements the single-array, three-channel A2 shape used by
//! Neural Amp Modeler.  The shape is deliberately not configurable at runtime:
//! model loading validates the exact architecture once, while the audio path
//! only executes fixed-size causal state transitions.
//!
//! For sample `n`, layer `l` evaluates
//!
//! ```text
//! z_l[n] = b_conv_l
//!          + sum_k W_conv_l[k] r_l[n - (K_l - 1 - k) D_l]
//!          + W_mixin_l x[n]
//! a_l[n] = LeakyReLU_0.01(z_l[n])
//! h[n]  += a_l[n]
//! r_l+1[n] = r_l[n] + b_res_l + W_res_l a_l[n]
//! ```
//!
//! where `r_0[n] = W_in x[n]`.  The output head is another causal convolution:
//!
//! ```text
//! y[n] = s_head * (b_head
//!          + sum_k W_head[k] h[n - (15 - k)])
//! ```
//!
//! Kernel tap order matches PyTorch `Conv1d` and the official NAM weight
//! stream: the final tap is the current sample.  Consequently this runtime has
//! a receptive field of 6,347 samples but adds zero samples of latency.  The
//! exact A2 training config initializes/exports `s_head = 0.01`, while the
//! serialized scale remains part of the official weight stream.

use std::fmt;

pub const A2_MODEL_FORMAT_VERSION: u32 = 1;
pub const A2_SAMPLE_RATE_HZ: u32 = 48_000;
pub const A2_CHANNELS: usize = 3;
pub const A2_LAYER_COUNT: usize = 23;
pub const A2_HEAD_KERNEL_SIZE: usize = 16;
pub const A2_MAX_LAYER_KERNEL_SIZE: usize = 15;
pub const A2_LEAKY_RELU_SLOPE: f32 = 0.01;
pub const A2_HEAD_SCALE: f32 = 0.01;
pub const A2_RECEPTIVE_FIELD_SAMPLES: usize = 6_347;
pub const A2_WEIGHT_COUNT: usize = 1_871;
pub const A2_MACS_PER_SAMPLE: u32 = 1_731;

pub const A2_KERNEL_SIZES: [usize; A2_LAYER_COUNT] = [
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 15, 15, 6, 6, 6, 6, 6, 6, 6,
];

pub const A2_DILATIONS: [usize; A2_LAYER_COUNT] = [
    1, 3, 7, 17, 41, 101, 239, 1, 3, 7, 17, 41, 101, 239, 1, 13, 1, 3, 7, 17, 41, 101, 239,
];

const A2_PAYLOAD_MAGIC: &[u8; 8] = b"MOTA2_01";
const A2_PAYLOAD_VERSION: u32 = 1;
const A2_PAYLOAD_HEADER_BYTES: usize = 52;
pub const A2_PAYLOAD_BYTES: usize = A2_PAYLOAD_HEADER_BYTES + A2_WEIGHT_COUNT * 4;

const LAYER_CONV_WEIGHT_CAPACITY: usize = A2_MAX_LAYER_KERNEL_SIZE * A2_CHANNELS * A2_CHANNELS;
const RESIDUAL_WEIGHT_COUNT: usize = A2_CHANNELS * A2_CHANNELS;
const HEAD_WEIGHT_COUNT: usize = A2_HEAD_KERNEL_SIZE * A2_CHANNELS;

const fn layer_history_sample_count() -> usize {
    let mut total = 0;
    let mut layer = 0;
    while layer < A2_LAYER_COUNT {
        total += (A2_KERNEL_SIZES[layer] - 1) * A2_DILATIONS[layer] + 1;
        layer += 1;
    }
    total
}

const fn layer_history_offsets() -> [usize; A2_LAYER_COUNT] {
    let mut offsets = [0; A2_LAYER_COUNT];
    let mut total = 0;
    let mut layer = 0;
    while layer < A2_LAYER_COUNT {
        offsets[layer] = total;
        total += (A2_KERNEL_SIZES[layer] - 1) * A2_DILATIONS[layer] + 1;
        layer += 1;
    }
    offsets
}

pub const A2_LAYER_HISTORY_SAMPLES: usize = layer_history_sample_count();
pub const A2_STATE_FLOATS: usize =
    A2_LAYER_HISTORY_SAMPLES * A2_CHANNELS + A2_HEAD_KERNEL_SIZE * A2_CHANNELS;
const A2_LAYER_HISTORY_OFFSETS: [usize; A2_LAYER_COUNT] = layer_history_offsets();

const _: () = assert!(
    A2_RECEPTIVE_FIELD_SAMPLES
        == 1 + (A2_HEAD_KERNEL_SIZE - 1) + {
            let mut lookback = 0;
            let mut layer = 0;
            while layer < A2_LAYER_COUNT {
                lookback += (A2_KERNEL_SIZES[layer] - 1) * A2_DILATIONS[layer];
                layer += 1;
            }
            lookback
        }
);

const _: () = assert!(
    A2_WEIGHT_COUNT
        == A2_CHANNELS
            + 21 * (6 * A2_CHANNELS * A2_CHANNELS
                + A2_CHANNELS
                + A2_CHANNELS
                + RESIDUAL_WEIGHT_COUNT
                + A2_CHANNELS)
            + 2 * (15 * A2_CHANNELS * A2_CHANNELS
                + A2_CHANNELS
                + A2_CHANNELS
                + RESIDUAL_WEIGHT_COUNT
                + A2_CHANNELS)
            + HEAD_WEIGHT_COUNT
            + 1
            + 1
);

/// Static cost and memory description for the exact A2 C=3 runtime.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct A2RuntimeCost {
    pub channels: usize,
    pub layer_count: usize,
    pub receptive_field_samples: usize,
    pub parameter_count: usize,
    /// Matrix/filter weight multiply-accumulates.  Bias additions, the
    /// conditional LeakyReLU multiply, and the final head-scale multiply are
    /// reported separately.
    pub macs_per_sample: u32,
    pub maximum_leaky_relu_multiplies_per_sample: u32,
    pub head_scale_multiplies_per_sample: u32,
    pub state_floats: usize,
    pub state_bytes: usize,
}

pub const A2_RUNTIME_COST: A2RuntimeCost = A2RuntimeCost {
    channels: A2_CHANNELS,
    layer_count: A2_LAYER_COUNT,
    receptive_field_samples: A2_RECEPTIVE_FIELD_SAMPLES,
    parameter_count: A2_WEIGHT_COUNT,
    macs_per_sample: A2_MACS_PER_SAMPLE,
    maximum_leaky_relu_multiplies_per_sample: (A2_LAYER_COUNT * A2_CHANNELS) as u32,
    head_scale_multiplies_per_sample: 1,
    state_floats: A2_STATE_FLOATS,
    state_bytes: A2_STATE_FLOATS * std::mem::size_of::<f32>()
        + (A2_LAYER_COUNT + 1) * std::mem::size_of::<usize>(),
};

/// Coefficients for one fixed-width WaveNet residual layer.
///
/// `conv` is tap-major, then input-channel-major, then output-channel-major:
/// `conv[tap * 9 + input * 3 + output]`.  `residual` is input-major then
/// output-major: `residual[input * 3 + output]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct A2LayerWeights {
    pub conv: [f32; LAYER_CONV_WEIGHT_CAPACITY],
    pub conv_bias: [f32; A2_CHANNELS],
    pub input_mixin: [f32; A2_CHANNELS],
    pub residual: [f32; RESIDUAL_WEIGHT_COUNT],
    pub residual_bias: [f32; A2_CHANNELS],
}

impl Default for A2LayerWeights {
    fn default() -> Self {
        Self {
            conv: [0.0; LAYER_CONV_WEIGHT_CAPACITY],
            conv_bias: [0.0; A2_CHANNELS],
            input_mixin: [0.0; A2_CHANNELS],
            residual: [0.0; RESIDUAL_WEIGHT_COUNT],
            residual_bias: [0.0; A2_CHANNELS],
        }
    }
}

/// All trainable/exported coefficients for the exact A2 C=3 shape.
///
/// `head` is tap-major then input-channel-major:
/// `head[tap * 3 + input]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct A2Weights {
    pub rechannel: [f32; A2_CHANNELS],
    pub layers: [A2LayerWeights; A2_LAYER_COUNT],
    pub head: [f32; HEAD_WEIGHT_COUNT],
    pub head_bias: f32,
    pub head_scale: f32,
}

impl Default for A2Weights {
    fn default() -> Self {
        Self {
            rechannel: [0.0; A2_CHANNELS],
            layers: [A2LayerWeights::default(); A2_LAYER_COUNT],
            head: [0.0; HEAD_WEIGHT_COUNT],
            head_bias: 0.0,
            head_scale: A2_HEAD_SCALE,
        }
    }
}

impl A2Weights {
    /// Reorders the canonical NAM A2 float stream into the runtime's
    /// tap-major fixed arrays.
    ///
    /// The input order is:
    ///
    /// ```text
    /// rechannel;
    /// each layer: conv[out,in,tap], conv_bias, mixin,
    ///             residual[out,in], residual_bias;
    /// head[out=0,in,tap], head_bias, head_scale
    /// ```
    pub fn from_official_weight_slice(coefficients: &[f32]) -> Result<Self, A2Error> {
        if coefficients.len() != A2_WEIGHT_COUNT {
            return Err(A2Error::InvalidWeightCount(
                u32::try_from(coefficients.len()).unwrap_or(u32::MAX),
            ));
        }
        let mut cursor = 0;
        let mut take = || {
            let coefficient = coefficients[cursor];
            cursor += 1;
            coefficient
        };
        let mut weights = Self::default();
        for coefficient in &mut weights.rechannel {
            *coefficient = take();
        }
        for (layer_index, layer) in weights.layers.iter_mut().enumerate() {
            for output in 0..A2_CHANNELS {
                for input in 0..A2_CHANNELS {
                    for tap in 0..A2_KERNEL_SIZES[layer_index] {
                        layer.conv[conv_index(tap, input, output)] = take();
                    }
                }
            }
            for coefficient in &mut layer.conv_bias {
                *coefficient = take();
            }
            for coefficient in &mut layer.input_mixin {
                *coefficient = take();
            }
            for output in 0..A2_CHANNELS {
                for input in 0..A2_CHANNELS {
                    layer.residual[matrix_index(input, output)] = take();
                }
            }
            for coefficient in &mut layer.residual_bias {
                *coefficient = take();
            }
        }
        for input in 0..A2_CHANNELS {
            for tap in 0..A2_HEAD_KERNEL_SIZE {
                weights.head[head_index(tap, input)] = take();
            }
        }
        weights.head_bias = take();
        weights.head_scale = take();
        debug_assert_eq!(cursor, A2_WEIGHT_COUNT);

        // Reuse full model validation so this adapter never hands the runtime
        // non-finite coefficients.
        A2Model::from_weights(weights).validate()?;
        Ok(weights)
    }

    /// Emits the canonical NAM A2 float stream. This is a loader/trainer
    /// helper and is never called from the audio thread.
    #[must_use]
    pub fn to_official_weight_vec(&self) -> Vec<f32> {
        let mut coefficients = Vec::with_capacity(A2_WEIGHT_COUNT);
        coefficients.extend_from_slice(&self.rechannel);
        for (layer_index, layer) in self.layers.iter().enumerate() {
            for output in 0..A2_CHANNELS {
                for input in 0..A2_CHANNELS {
                    for tap in 0..A2_KERNEL_SIZES[layer_index] {
                        coefficients.push(layer.conv[conv_index(tap, input, output)]);
                    }
                }
            }
            coefficients.extend_from_slice(&layer.conv_bias);
            coefficients.extend_from_slice(&layer.input_mixin);
            for output in 0..A2_CHANNELS {
                for input in 0..A2_CHANNELS {
                    coefficients.push(layer.residual[matrix_index(input, output)]);
                }
            }
            coefficients.extend_from_slice(&layer.residual_bias);
        }
        for input in 0..A2_CHANNELS {
            for tap in 0..A2_HEAD_KERNEL_SIZE {
                coefficients.push(self.head[head_index(tap, input)]);
            }
        }
        coefficients.push(self.head_bias);
        coefficients.push(self.head_scale);
        debug_assert_eq!(coefficients.len(), A2_WEIGHT_COUNT);
        coefficients
    }
}

/// Validated model descriptor plus its fixed-size weights.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct A2Model {
    pub format_version: u32,
    pub sample_rate_hz: u32,
    pub causal: bool,
    pub lookahead_samples: u32,
    pub runtime_latency_samples: u32,
    pub estimated_macs_per_sample: u32,
    pub weights: A2Weights,
}

impl Default for A2Model {
    fn default() -> Self {
        Self::zeros()
    }
}

impl A2Model {
    /// A valid A2 model whose output is silence.
    #[must_use]
    pub const fn zeros() -> Self {
        Self {
            format_version: A2_MODEL_FORMAT_VERSION,
            sample_rate_hz: A2_SAMPLE_RATE_HZ,
            causal: true,
            lookahead_samples: 0,
            runtime_latency_samples: 0,
            estimated_macs_per_sample: A2_MACS_PER_SAMPLE,
            weights: A2Weights {
                rechannel: [0.0; A2_CHANNELS],
                layers: [A2LayerWeights {
                    conv: [0.0; LAYER_CONV_WEIGHT_CAPACITY],
                    conv_bias: [0.0; A2_CHANNELS],
                    input_mixin: [0.0; A2_CHANNELS],
                    residual: [0.0; RESIDUAL_WEIGHT_COUNT],
                    residual_bias: [0.0; A2_CHANNELS],
                }; A2_LAYER_COUNT],
                head: [0.0; HEAD_WEIGHT_COUNT],
                head_bias: 0.0,
                head_scale: A2_HEAD_SCALE,
            },
        }
    }

    #[must_use]
    pub const fn from_weights(weights: A2Weights) -> Self {
        Self {
            weights,
            ..Self::zeros()
        }
    }

    pub fn validate(&self) -> Result<(), A2Error> {
        if self.format_version != A2_MODEL_FORMAT_VERSION {
            return Err(A2Error::UnsupportedModelFormat(self.format_version));
        }
        if self.sample_rate_hz != A2_SAMPLE_RATE_HZ {
            return Err(A2Error::UnsupportedSampleRate(self.sample_rate_hz));
        }
        if !self.causal || self.lookahead_samples != 0 || self.runtime_latency_samples != 0 {
            return Err(A2Error::NonCausalOrLatent);
        }
        if self.estimated_macs_per_sample != A2_MACS_PER_SAMPLE {
            return Err(A2Error::RuntimeCostMismatch(self.estimated_macs_per_sample));
        }
        if self
            .weights
            .rechannel
            .iter()
            .chain(self.weights.head.iter())
            .chain([self.weights.head_bias, self.weights.head_scale].iter())
            .any(|coefficient| !coefficient.is_finite())
        {
            return Err(A2Error::NonFiniteCoefficient);
        }

        for (layer_index, weights) in self.weights.layers.iter().enumerate() {
            if weights
                .conv
                .iter()
                .chain(weights.conv_bias.iter())
                .chain(weights.input_mixin.iter())
                .chain(weights.residual.iter())
                .chain(weights.residual_bias.iter())
                .any(|coefficient| !coefficient.is_finite())
            {
                return Err(A2Error::NonFiniteCoefficient);
            }

            let active_conv_weights = A2_KERNEL_SIZES[layer_index] * A2_CHANNELS * A2_CHANNELS;
            if weights.conv[active_conv_weights..]
                .iter()
                .any(|coefficient| coefficient.to_bits() != 0.0_f32.to_bits())
            {
                return Err(A2Error::NonZeroUnusedCoefficient(layer_index));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum A2Error {
    InvalidPayload,
    UnsupportedPayloadVersion(u32),
    UnsupportedModelFormat(u32),
    UnsupportedSampleRate(u32),
    NonCausalOrLatent,
    RuntimeCostMismatch(u32),
    InvalidShape,
    InvalidWeightCount(u32),
    NonFiniteCoefficient,
    NonZeroUnusedCoefficient(usize),
    HostSampleRateMismatch { model: u32, host: u32 },
}

impl fmt::Display for A2Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPayload => formatter.write_str("invalid A2 model payload"),
            Self::UnsupportedPayloadVersion(version) => {
                write!(formatter, "unsupported A2 payload version {version}")
            }
            Self::UnsupportedModelFormat(version) => {
                write!(formatter, "unsupported A2 model format {version}")
            }
            Self::UnsupportedSampleRate(rate) => {
                write!(
                    formatter,
                    "A2 model sample rate must be 48000 Hz, found {rate}"
                )
            }
            Self::NonCausalOrLatent => {
                formatter.write_str("A2 model must be causal with zero lookahead and latency")
            }
            Self::RuntimeCostMismatch(macs) => write!(
                formatter,
                "A2 model declares {macs} MACs/sample; exact shape requires {A2_MACS_PER_SAMPLE}"
            ),
            Self::InvalidShape => formatter.write_str("payload is not the exact A2 C=3 shape"),
            Self::InvalidWeightCount(count) => write!(
                formatter,
                "A2 payload contains {count} weights; expected {A2_WEIGHT_COUNT}"
            ),
            Self::NonFiniteCoefficient => {
                formatter.write_str("A2 model contains a non-finite coefficient")
            }
            Self::NonZeroUnusedCoefficient(layer) => write!(
                formatter,
                "A2 layer {layer} contains data outside its active kernel"
            ),
            Self::HostSampleRateMismatch { model, host } => write!(
                formatter,
                "A2 model is {model} Hz but the host is running at {host} Hz"
            ),
        }
    }
}

impl std::error::Error for A2Error {}

/// Encodes a validated A2 model in official NAM weight-stream order.
///
/// The outer `.motmodel` container remains responsible for immutable model
/// identity and SHA-256.  This payload carries enough fixed-shape metadata to
/// reject accidental architecture or sample-rate mismatches before playback.
pub fn encode_a2_payload(model: &A2Model) -> Result<Vec<u8>, A2Error> {
    model.validate()?;
    let mut bytes = Vec::with_capacity(A2_PAYLOAD_BYTES);
    bytes.extend_from_slice(A2_PAYLOAD_MAGIC);
    push_u32(&mut bytes, A2_PAYLOAD_VERSION);
    push_u32(&mut bytes, model.format_version);
    push_u32(&mut bytes, model.sample_rate_hz);
    bytes.push(u8::from(model.causal));
    bytes.extend_from_slice(&[0; 3]);
    push_u32(&mut bytes, model.lookahead_samples);
    push_u32(&mut bytes, model.runtime_latency_samples);
    push_u32(&mut bytes, model.estimated_macs_per_sample);
    push_u32(&mut bytes, A2_CHANNELS as u32);
    push_u32(&mut bytes, A2_LAYER_COUNT as u32);
    push_u32(&mut bytes, A2_RECEPTIVE_FIELD_SAMPLES as u32);
    push_u32(&mut bytes, A2_WEIGHT_COUNT as u32);

    for coefficient in model.weights.rechannel {
        push_f32(&mut bytes, coefficient);
    }
    for (layer_index, layer) in model.weights.layers.iter().enumerate() {
        let kernel_size = A2_KERNEL_SIZES[layer_index];
        // Official order is output, input, tap.
        for output in 0..A2_CHANNELS {
            for input in 0..A2_CHANNELS {
                for tap in 0..kernel_size {
                    push_f32(&mut bytes, layer.conv[conv_index(tap, input, output)]);
                }
            }
        }
        for coefficient in layer.conv_bias {
            push_f32(&mut bytes, coefficient);
        }
        for coefficient in layer.input_mixin {
            push_f32(&mut bytes, coefficient);
        }
        // Official Conv1x1 order is output, input.
        for output in 0..A2_CHANNELS {
            for input in 0..A2_CHANNELS {
                push_f32(&mut bytes, layer.residual[matrix_index(input, output)]);
            }
        }
        for coefficient in layer.residual_bias {
            push_f32(&mut bytes, coefficient);
        }
    }
    // Official head order is output(only one), input, tap.
    for input in 0..A2_CHANNELS {
        for tap in 0..A2_HEAD_KERNEL_SIZE {
            push_f32(&mut bytes, model.weights.head[head_index(tap, input)]);
        }
    }
    push_f32(&mut bytes, model.weights.head_bias);
    push_f32(&mut bytes, model.weights.head_scale);

    debug_assert_eq!(bytes.len(), A2_PAYLOAD_BYTES);
    Ok(bytes)
}

pub fn decode_a2_payload(bytes: &[u8]) -> Result<A2Model, A2Error> {
    if bytes.len() != A2_PAYLOAD_BYTES || bytes.get(..8) != Some(A2_PAYLOAD_MAGIC.as_slice()) {
        return Err(A2Error::InvalidPayload);
    }
    let mut cursor = 8;
    let payload_version = take_u32(bytes, &mut cursor)?;
    if payload_version != A2_PAYLOAD_VERSION {
        return Err(A2Error::UnsupportedPayloadVersion(payload_version));
    }
    let format_version = take_u32(bytes, &mut cursor)?;
    let sample_rate_hz = take_u32(bytes, &mut cursor)?;
    let causal = match bytes.get(cursor).copied() {
        Some(0) => false,
        Some(1) => true,
        _ => return Err(A2Error::InvalidPayload),
    };
    cursor += 4;
    let lookahead_samples = take_u32(bytes, &mut cursor)?;
    let runtime_latency_samples = take_u32(bytes, &mut cursor)?;
    let estimated_macs_per_sample = take_u32(bytes, &mut cursor)?;
    let channels = take_u32(bytes, &mut cursor)?;
    let layers = take_u32(bytes, &mut cursor)?;
    let receptive_field = take_u32(bytes, &mut cursor)?;
    let weight_count = take_u32(bytes, &mut cursor)?;
    if channels != A2_CHANNELS as u32
        || layers != A2_LAYER_COUNT as u32
        || receptive_field != A2_RECEPTIVE_FIELD_SAMPLES as u32
    {
        return Err(A2Error::InvalidShape);
    }
    if weight_count != A2_WEIGHT_COUNT as u32 {
        return Err(A2Error::InvalidWeightCount(weight_count));
    }

    let mut weights = A2Weights::default();
    for coefficient in &mut weights.rechannel {
        *coefficient = take_f32(bytes, &mut cursor)?;
    }
    for (layer_index, layer) in weights.layers.iter_mut().enumerate() {
        let kernel_size = A2_KERNEL_SIZES[layer_index];
        for output in 0..A2_CHANNELS {
            for input in 0..A2_CHANNELS {
                for tap in 0..kernel_size {
                    layer.conv[conv_index(tap, input, output)] = take_f32(bytes, &mut cursor)?;
                }
            }
        }
        for coefficient in &mut layer.conv_bias {
            *coefficient = take_f32(bytes, &mut cursor)?;
        }
        for coefficient in &mut layer.input_mixin {
            *coefficient = take_f32(bytes, &mut cursor)?;
        }
        for output in 0..A2_CHANNELS {
            for input in 0..A2_CHANNELS {
                layer.residual[matrix_index(input, output)] = take_f32(bytes, &mut cursor)?;
            }
        }
        for coefficient in &mut layer.residual_bias {
            *coefficient = take_f32(bytes, &mut cursor)?;
        }
    }
    for input in 0..A2_CHANNELS {
        for tap in 0..A2_HEAD_KERNEL_SIZE {
            weights.head[head_index(tap, input)] = take_f32(bytes, &mut cursor)?;
        }
    }
    weights.head_bias = take_f32(bytes, &mut cursor)?;
    weights.head_scale = take_f32(bytes, &mut cursor)?;
    if cursor != bytes.len() {
        return Err(A2Error::InvalidPayload);
    }

    let model = A2Model {
        format_version,
        sample_rate_hz,
        causal,
        lookahead_samples,
        runtime_latency_samples,
        estimated_macs_per_sample,
        weights,
    };
    model.validate()?;
    Ok(model)
}

#[derive(Clone, Debug)]
struct A2State {
    layer_history: [f32; A2_LAYER_HISTORY_SAMPLES * A2_CHANNELS],
    layer_write_positions: [usize; A2_LAYER_COUNT],
    head_history: [f32; HEAD_WEIGHT_COUNT],
    head_write_position: usize,
}

impl Default for A2State {
    fn default() -> Self {
        Self {
            layer_history: [0.0; A2_LAYER_HISTORY_SAMPLES * A2_CHANNELS],
            layer_write_positions: [0; A2_LAYER_COUNT],
            head_history: [0.0; HEAD_WEIGHT_COUNT],
            head_write_position: 0,
        }
    }
}

/// Fixed-state streaming A2 processor.
///
/// Construction/model replacement is a control-thread operation.
/// [`process_block`](Self::process_block) performs no allocation, locking,
/// I/O, block accumulation, or lookahead.
#[derive(Clone, Debug)]
pub struct A2Processor {
    model: A2Model,
    state: A2State,
}

impl A2Processor {
    pub fn new(model: A2Model) -> Result<Self, A2Error> {
        model.validate()?;
        Ok(Self {
            model,
            state: A2State::default(),
        })
    }

    pub fn new_for_sample_rate(model: A2Model, host_sample_rate_hz: u32) -> Result<Self, A2Error> {
        model.validate()?;
        if host_sample_rate_hz != model.sample_rate_hz {
            return Err(A2Error::HostSampleRateMismatch {
                model: model.sample_rate_hz,
                host: host_sample_rate_hz,
            });
        }
        Self::new(model)
    }

    #[must_use]
    pub const fn model(&self) -> &A2Model {
        &self.model
    }

    pub fn replace_model(&mut self, model: A2Model) -> Result<(), A2Error> {
        model.validate()?;
        self.model = model;
        self.reset();
        Ok(())
    }

    pub fn reset(&mut self) {
        self.state.layer_history.fill(0.0);
        self.state.layer_write_positions.fill(0);
        self.state.head_history.fill(0.0);
        self.state.head_write_position = 0;
    }

    #[inline]
    pub fn process_block(&mut self, input: &[f32], output: &mut [f32]) {
        debug_assert_eq!(input.len(), output.len());
        for (&input_sample, output_sample) in input.iter().zip(output) {
            *output_sample = self.process_sample(input_sample);
        }
    }

    #[inline]
    pub(crate) fn process_sample(&mut self, input: f32) -> f32 {
        let mut residual = [0.0; A2_CHANNELS];
        for (channel, residual_sample) in residual.iter_mut().enumerate() {
            *residual_sample = self.model.weights.rechannel[channel] * input;
        }
        let mut head_sum = [0.0; A2_CHANNELS];

        for layer_index in 0..A2_LAYER_COUNT {
            let kernel_size = A2_KERNEL_SIZES[layer_index];
            let dilation = A2_DILATIONS[layer_index];
            let history_samples = (kernel_size - 1) * dilation + 1;
            let history_offset = A2_LAYER_HISTORY_OFFSETS[layer_index] * A2_CHANNELS;
            let write_position = self.state.layer_write_positions[layer_index];

            let current_offset = history_offset + write_position * A2_CHANNELS;
            self.state.layer_history[current_offset..current_offset + A2_CHANNELS]
                .copy_from_slice(&residual);

            let layer = &self.model.weights.layers[layer_index];
            let mut activation = layer.conv_bias;
            for (channel, activation_sample) in activation.iter_mut().enumerate() {
                *activation_sample += layer.input_mixin[channel] * input;
            }

            for tap in 0..kernel_size {
                let delay = (kernel_size - 1 - tap) * dilation;
                let read_position = if write_position >= delay {
                    write_position - delay
                } else {
                    write_position + history_samples - delay
                };
                let source_offset = history_offset + read_position * A2_CHANNELS;
                for input_channel in 0..A2_CHANNELS {
                    let source = self.state.layer_history[source_offset + input_channel];
                    for (output_channel, activation_sample) in activation.iter_mut().enumerate() {
                        *activation_sample +=
                            layer.conv[conv_index(tap, input_channel, output_channel)] * source;
                    }
                }
            }

            for value in &mut activation {
                if *value < 0.0 {
                    *value *= A2_LEAKY_RELU_SLOPE;
                }
            }
            for (head_sample, activation_sample) in head_sum.iter_mut().zip(activation) {
                *head_sample += activation_sample;
            }

            let mut next_residual = residual;
            for (output_channel, next_sample) in next_residual.iter_mut().enumerate() {
                *next_sample += layer.residual_bias[output_channel];
                for (input_channel, activation_sample) in activation.iter().enumerate() {
                    *next_sample += layer.residual[matrix_index(input_channel, output_channel)]
                        * activation_sample;
                }
            }
            residual = next_residual;
            self.state.layer_write_positions[layer_index] = if write_position + 1 == history_samples
            {
                0
            } else {
                write_position + 1
            };
        }

        let head_write_position = self.state.head_write_position;
        let head_current_offset = head_write_position * A2_CHANNELS;
        self.state.head_history[head_current_offset..head_current_offset + A2_CHANNELS]
            .copy_from_slice(&head_sum);

        let mut output = self.model.weights.head_bias;
        for tap in 0..A2_HEAD_KERNEL_SIZE {
            let delay = A2_HEAD_KERNEL_SIZE - 1 - tap;
            let read_position = if head_write_position >= delay {
                head_write_position - delay
            } else {
                head_write_position + A2_HEAD_KERNEL_SIZE - delay
            };
            let source_offset = read_position * A2_CHANNELS;
            for channel in 0..A2_CHANNELS {
                output += self.model.weights.head[head_index(tap, channel)]
                    * self.state.head_history[source_offset + channel];
            }
        }
        self.state.head_write_position = if head_write_position + 1 == A2_HEAD_KERNEL_SIZE {
            0
        } else {
            head_write_position + 1
        };
        output * self.model.weights.head_scale
    }

    #[must_use]
    pub const fn latency_samples(&self) -> u32 {
        0
    }

    #[must_use]
    pub const fn receptive_field_samples(&self) -> usize {
        A2_RECEPTIVE_FIELD_SAMPLES
    }

    #[must_use]
    pub const fn runtime_cost(&self) -> A2RuntimeCost {
        A2_RUNTIME_COST
    }
}

#[inline(always)]
const fn conv_index(tap: usize, input: usize, output: usize) -> usize {
    tap * A2_CHANNELS * A2_CHANNELS + input * A2_CHANNELS + output
}

#[inline(always)]
const fn matrix_index(input: usize, output: usize) -> usize {
    input * A2_CHANNELS + output
}

#[inline(always)]
const fn head_index(tap: usize, input: usize) -> usize {
    tap * A2_CHANNELS + input
}

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_f32(bytes: &mut Vec<u8>, value: f32) {
    push_u32(bytes, value.to_bits());
}

fn take_u32(bytes: &[u8], cursor: &mut usize) -> Result<u32, A2Error> {
    let chunk = bytes
        .get(*cursor..cursor.saturating_add(4))
        .ok_or(A2Error::InvalidPayload)?;
    *cursor += 4;
    Ok(u32::from_le_bytes(
        chunk.try_into().map_err(|_| A2Error::InvalidPayload)?,
    ))
}

fn take_f32(bytes: &[u8], cursor: &mut usize) -> Result<f32, A2Error> {
    Ok(f32::from_bits(take_u32(bytes, cursor)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn next_signed(seed: &mut u32) -> f32 {
        *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((*seed >> 8) as f32 / 16_777_215.0) * 2.0 - 1.0
    }

    fn deterministic_model() -> A2Model {
        let mut model = A2Model::zeros();
        let mut seed = 0x5a17_9c3d;
        for value in &mut model.weights.rechannel {
            *value = next_signed(&mut seed) * 0.5;
        }
        for (layer_index, layer) in model.weights.layers.iter_mut().enumerate() {
            let active = A2_KERNEL_SIZES[layer_index] * A2_CHANNELS * A2_CHANNELS;
            for value in &mut layer.conv[..active] {
                *value = next_signed(&mut seed) * 0.025;
            }
            for value in &mut layer.conv_bias {
                *value = next_signed(&mut seed) * 0.003;
            }
            for value in &mut layer.input_mixin {
                *value = next_signed(&mut seed) * 0.12;
            }
            for value in &mut layer.residual {
                *value = next_signed(&mut seed) * 0.035;
            }
            for value in &mut layer.residual_bias {
                *value = next_signed(&mut seed) * 0.002;
            }
        }
        for value in &mut model.weights.head {
            *value = next_signed(&mut seed) * 0.08;
        }
        model.weights.head_bias = -0.0017;
        model
    }

    fn deterministic_input(length: usize) -> Vec<f32> {
        let mut seed = 0x1234_5678;
        (0..length)
            .map(|index| {
                let noise = next_signed(&mut seed) * 0.2;
                noise + (index as f32 * 0.071).sin() * 0.45
            })
            .collect()
    }

    /// Intentionally straightforward full-sequence definition, independent of
    /// the runtime ring-buffer implementation.
    #[allow(clippy::needless_range_loop)]
    fn reference_render(model: &A2Model, input: &[f32]) -> Vec<f32> {
        let sample_count = input.len();
        let mut layer_inputs = vec![vec![[0.0; A2_CHANNELS]; sample_count]; A2_LAYER_COUNT];
        let mut head_sums = vec![[0.0; A2_CHANNELS]; sample_count];
        let mut output = vec![0.0; sample_count];

        for sample in 0..sample_count {
            let condition = input[sample];
            let mut residual = [0.0; A2_CHANNELS];
            for channel in 0..A2_CHANNELS {
                residual[channel] = model.weights.rechannel[channel] * condition;
            }

            for layer_index in 0..A2_LAYER_COUNT {
                layer_inputs[layer_index][sample] = residual;
                let layer = &model.weights.layers[layer_index];
                let kernel_size = A2_KERNEL_SIZES[layer_index];
                let dilation = A2_DILATIONS[layer_index];
                let mut activation = layer.conv_bias;
                for channel in 0..A2_CHANNELS {
                    activation[channel] += layer.input_mixin[channel] * condition;
                }
                for tap in 0..kernel_size {
                    let delay = (kernel_size - 1 - tap) * dilation;
                    let Some(source_sample) = sample.checked_sub(delay) else {
                        continue;
                    };
                    for input_channel in 0..A2_CHANNELS {
                        let source = layer_inputs[layer_index][source_sample][input_channel];
                        for output_channel in 0..A2_CHANNELS {
                            activation[output_channel] +=
                                layer.conv[conv_index(tap, input_channel, output_channel)] * source;
                        }
                    }
                }
                for value in &mut activation {
                    if *value < 0.0 {
                        *value *= A2_LEAKY_RELU_SLOPE;
                    }
                }
                for channel in 0..A2_CHANNELS {
                    head_sums[sample][channel] += activation[channel];
                }
                let mut next_residual = residual;
                for output_channel in 0..A2_CHANNELS {
                    next_residual[output_channel] += layer.residual_bias[output_channel];
                    for input_channel in 0..A2_CHANNELS {
                        next_residual[output_channel] += layer.residual
                            [matrix_index(input_channel, output_channel)]
                            * activation[input_channel];
                    }
                }
                residual = next_residual;
            }

            let mut head = model.weights.head_bias;
            for tap in 0..A2_HEAD_KERNEL_SIZE {
                let delay = A2_HEAD_KERNEL_SIZE - 1 - tap;
                let Some(source_sample) = sample.checked_sub(delay) else {
                    continue;
                };
                for channel in 0..A2_CHANNELS {
                    head += model.weights.head[head_index(tap, channel)]
                        * head_sums[source_sample][channel];
                }
            }
            output[sample] = head * model.weights.head_scale;
        }
        output
    }

    #[test]
    fn constants_describe_official_a2_shape() {
        assert_eq!(A2_KERNEL_SIZES[..14], [6; 14]);
        assert_eq!(A2_KERNEL_SIZES[14..16], [15; 2]);
        assert_eq!(A2_KERNEL_SIZES[16..], [6; 7]);
        assert_eq!(A2_RECEPTIVE_FIELD_SAMPLES, 6_347);
        assert_eq!(A2_RUNTIME_COST.parameter_count, 1_871);
        assert_eq!(A2_RUNTIME_COST.macs_per_sample, 1_731);
        assert_eq!(A2_RUNTIME_COST.state_floats, A2_STATE_FLOATS);
    }

    #[test]
    fn payload_round_trip_preserves_every_exported_bit() {
        let model = deterministic_model();
        let official = model.weights.to_official_weight_vec();
        assert_eq!(official.len(), A2_WEIGHT_COUNT);
        let reordered = A2Weights::from_official_weight_slice(&official).unwrap();
        assert_eq!(reordered, model.weights);

        let payload = encode_a2_payload(&model).unwrap();
        assert_eq!(payload.len(), A2_PAYLOAD_BYTES);
        let decoded = decode_a2_payload(&payload).unwrap();
        assert_eq!(encode_a2_payload(&decoded).unwrap(), payload);
        assert_eq!(decoded, model);
    }

    #[test]
    fn streaming_runtime_matches_direct_causal_equations() {
        let model = deterministic_model();
        let input = deterministic_input(8_111);
        let expected = reference_render(&model, &input);
        let mut processor = A2Processor::new(model).unwrap();
        let mut actual = vec![0.0; input.len()];
        processor.process_block(&input, &mut actual);
        assert_eq!(actual, expected);
    }

    #[test]
    fn output_is_bit_exact_across_host_block_sizes() {
        let model = deterministic_model();
        let input = deterministic_input(2_047);
        let mut whole = A2Processor::new(model).unwrap();
        let mut whole_output = vec![0.0; input.len()];
        whole.process_block(&input, &mut whole_output);

        let block_sizes = [1, 7, 16, 32, 64, 257, 512];
        let mut partitioned = A2Processor::new(model).unwrap();
        let mut partitioned_output = vec![0.0; input.len()];
        let mut start = 0;
        let mut block = 0;
        while start < input.len() {
            let end = (start + block_sizes[block % block_sizes.len()]).min(input.len());
            partitioned.process_block(&input[start..end], &mut partitioned_output[start..end]);
            start = end;
            block += 1;
        }
        assert_eq!(partitioned_output, whole_output);
    }

    #[test]
    fn current_sample_path_has_onset_at_sample_zero() {
        let mut model = A2Model::zeros();
        model.weights.rechannel[0] = 1.0;
        // First layer, current-sample tap, channel 0 -> channel 0.
        let current_tap = A2_KERNEL_SIZES[0] - 1;
        model.weights.layers[0].conv[conv_index(current_tap, 0, 0)] = 1.0;
        // Head current-sample tap, channel 0.
        model.weights.head[head_index(A2_HEAD_KERNEL_SIZE - 1, 0)] = 1.0;

        let mut processor = A2Processor::new(model).unwrap();
        let mut output = [0.0; 8];
        processor.process_block(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &mut output);
        assert_eq!(output[0].to_bits(), A2_HEAD_SCALE.to_bits());
        assert_eq!(processor.latency_samples(), 0);
    }

    #[test]
    fn first_kernel_tap_is_the_oldest_sample() {
        let mut model = A2Model::zeros();
        model.weights.rechannel[0] = 1.0;
        // Layer 0 is K=6, D=1: tap zero must read five samples back.
        model.weights.layers[0].conv[conv_index(0, 0, 0)] = 1.0;
        model.weights.head[head_index(A2_HEAD_KERNEL_SIZE - 1, 0)] = 1.0;
        model.weights.head_scale = 1.0;

        let mut processor = A2Processor::new(model).unwrap();
        let mut output = [0.0; 8];
        processor.process_block(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &mut output);
        assert_eq!(output[..5], [0.0; 5]);
        assert_eq!(output[5], 1.0);
    }

    #[test]
    fn validator_rejects_latency_non_finite_scale_and_unused_coefficients() {
        let mut model = A2Model::zeros();
        model.lookahead_samples = 1;
        assert_eq!(model.validate(), Err(A2Error::NonCausalOrLatent));

        let mut model = A2Model::zeros();
        model.weights.head_scale = f32::NAN;
        assert_eq!(model.validate(), Err(A2Error::NonFiniteCoefficient));

        let mut model = A2Model::zeros();
        let first_unused = A2_KERNEL_SIZES[0] * A2_CHANNELS * A2_CHANNELS;
        model.weights.layers[0].conv[first_unused] = 1.0;
        assert_eq!(model.validate(), Err(A2Error::NonZeroUnusedCoefficient(0)));
    }

    #[cfg(feature = "rt-paranoid")]
    #[test]
    fn process_block_allocates_nothing() {
        let input = [0.25; 257];
        let mut output = [0.0; 257];
        let mut processor = A2Processor::new(deterministic_model()).unwrap();
        let (_, allocations) = truce::rt::audit(|| {
            let _section = truce::rt::RtSection::enter();
            processor.process_block(&input, &mut output);
        });
        assert_eq!(allocations, 0);
    }
}
