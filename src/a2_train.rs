//! Native offline training for the fixed NAM WaveNet A2 (C=3) model.
//!
//! This is intentionally separate from the real-time runtime.  The training
//! path may allocate and block, while the model it exports remains a fixed,
//! causal, zero-latency [`A2Model`].
//!
//! The network and data recipe mirror NAM's canonical A2 submodel:
//!
//! - 23 residual layers, three channels, LeakyReLU with slope 0.01;
//! - the exact A2 kernel/dilation schedule and 6,347-sample receptive field;
//! - valid convolutions with the target starting 6,346 samples after input;
//! - v3 crop: training `[480_000, len - 432_000)`, validation the final
//!   432,000 samples;
//! - output-only joint normalization to -18 dBFS, undone in the exported
//!   `head_scale`;
//! - shuffled batches of 16, 8,192 output samples, AdamW at 0.004 with
//!   weight decay 3.17e-7, and an exponential 0.994 learning-rate schedule;
//! - best checkpoint selected by validation error-signal ratio (ESR).
//!
//! NAM also adds multi-resolution STFT loss at weight 0.0005.  Candle does not
//! yet expose the exact auraloss-compatible primitive used by NAM, so that
//! term is deliberately isolated behind [`A2_MRSTFT_IMPLEMENTED`] and is not
//! silently approximated.  The time-domain MSE trainer is complete and
//! functional; adding the spectral term does not change the dataset, model,
//! optimizer, checkpoint, or export code.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Conv1d, Conv1dConfig, Init, Module, Optimizer, ParamsAdamW, VarMap};

use crate::a2::{
    A2_CHANNELS, A2_DILATIONS, A2_HEAD_KERNEL_SIZE, A2_HEAD_SCALE, A2_KERNEL_SIZES, A2_LAYER_COUNT,
    A2_LEAKY_RELU_SLOPE, A2_RECEPTIVE_FIELD_SAMPLES, A2_SAMPLE_RATE_HZ, A2Model, A2Weights,
};

pub const A2_TRAIN_START_SAMPLE: usize = 480_000;
pub const A2_VALIDATION_SAMPLE_COUNT: usize = 432_000;
pub const A2_DEFAULT_BATCH_SIZE: usize = 16;
pub const A2_DEFAULT_OUTPUT_SAMPLES: usize = 8_192;
pub const A2_DEFAULT_MAX_EPOCHS: u32 = 400;
pub const A2_INITIAL_LEARNING_RATE: f64 = 0.004;
pub const A2_ADAMW_WEIGHT_DECAY: f64 = 3.17e-7;
pub const A2_LEARNING_RATE_GAMMA: f64 = 0.994;
pub const A2_OUTPUT_NORMALIZATION_DBFS: f64 = -18.0;
pub const A2_MRSTFT_WEIGHT: f64 = 0.0005;
pub const A2_MRSTFT_IMPLEMENTED: bool = false;
pub const A2_TRAINABLE_PARAMETER_COUNT: usize = 1_870;

const A2_RECEPTIVE_LOOKBACK: usize = A2_RECEPTIVE_FIELD_SAMPLES - 1;
const A2_RECEPTIVE_FIELD_WITHOUT_HEAD: usize =
    A2_RECEPTIVE_FIELD_SAMPLES - (A2_HEAD_KERNEL_SIZE - 1);

/// Requested offline compute backend.
///
/// This affects only model training. Exported models always run through the
/// independent native CPU implementation in [`crate::a2::A2Processor`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum A2TrainingDevice {
    #[default]
    Auto,
    Metal,
    Cpu,
}

impl A2TrainingDevice {
    /// Resolve the requested backend before allocating the training graph.
    ///
    /// `Auto` prefers Metal device 0 and records a visible CPU-fallback status
    /// when Metal cannot be initialized. Explicit `Metal` never falls back:
    /// callers receive a diagnostic explaining why the requested device is
    /// unavailable.
    pub fn resolve(self) -> Result<A2ResolvedTrainingDevice, A2TrainingError> {
        match self {
            Self::Auto => match initialize_metal_device() {
                Ok(device) => Ok(A2ResolvedTrainingDevice {
                    device,
                    requested: self,
                    status: A2TrainingDeviceStatus::Metal,
                    fallback_reason: None,
                }),
                Err(error) => Ok(A2ResolvedTrainingDevice {
                    device: Device::Cpu,
                    requested: self,
                    status: A2TrainingDeviceStatus::CpuAutoFallback,
                    fallback_reason: Some(error.to_string()),
                }),
            },
            Self::Metal => initialize_metal_device().map_or_else(
                |reason| Err(A2TrainingError::MetalUnavailable { reason }),
                |device| {
                    Ok(A2ResolvedTrainingDevice {
                        device,
                        requested: self,
                        status: A2TrainingDeviceStatus::Metal,
                        fallback_reason: None,
                    })
                },
            ),
            Self::Cpu => Ok(A2ResolvedTrainingDevice {
                device: Device::Cpu,
                requested: self,
                status: A2TrainingDeviceStatus::Cpu,
                fallback_reason: None,
            }),
        }
    }

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Auto => "AUTO (METAL → CPU)",
            Self::Metal => "METAL",
            Self::Cpu => "CPU",
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn initialize_metal_device() -> Result<Device, String> {
    // Candle 0.11 indexes `Device::all()` with `swap_remove(ordinal)`
    // without checking the length. Sandboxed hosts may expose no Metal
    // devices, which would otherwise panic the entire DAW instead of letting
    // Auto fall back to CPU.
    if candle_metal_kernels::metal::Device::all().is_empty() {
        return Err("no Metal device is visible to this process".to_owned());
    }
    std::panic::catch_unwind(|| Device::new_metal(0))
        .map_err(|payload| {
            let detail = payload.downcast_ref::<&str>().map_or_else(
                || {
                    payload
                        .downcast_ref::<String>()
                        .map_or("unknown panic", String::as_str)
                },
                |message| *message,
            );
            format!("Candle panicked while initializing Metal device 0: {detail}")
        })?
        .map_err(|error| error.to_string())
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn initialize_metal_device() -> Result<Device, String> {
    Device::new_metal(0).map_err(|error| error.to_string())
}

/// Backend selected for one offline training run.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum A2TrainingDeviceStatus {
    Metal,
    Cpu,
    CpuAutoFallback,
}

impl A2TrainingDeviceStatus {
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Metal => "METAL",
            Self::Cpu => "CPU",
            Self::CpuAutoFallback => "CPU (AUTO FALLBACK)",
        }
    }
}

/// Concrete Candle device together with a UI/log-friendly resolution status.
#[derive(Debug)]
pub struct A2ResolvedTrainingDevice {
    device: Device,
    requested: A2TrainingDevice,
    status: A2TrainingDeviceStatus,
    fallback_reason: Option<String>,
}

impl A2ResolvedTrainingDevice {
    #[must_use]
    pub const fn requested(&self) -> A2TrainingDevice {
        self.requested
    }

    #[must_use]
    pub const fn status(&self) -> A2TrainingDeviceStatus {
        self.status
    }

    #[must_use]
    pub const fn label(&self) -> &'static str {
        self.status.label()
    }

    /// Candle's initialization diagnostic when `Auto` selected CPU.
    #[must_use]
    pub fn fallback_reason(&self) -> Option<&str> {
        self.fallback_reason.as_deref()
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct A2TrainerConfig {
    pub max_epochs: u32,
    pub batch_size: usize,
    pub output_samples: usize,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub learning_rate_gamma: f64,
    /// The only opt-in early-exit mechanism.  `None` means every requested
    /// epoch is completed; lack of improvement never stops training.
    pub threshold_esr: Option<f64>,
    pub seed: u64,
    pub validation_chunk_samples: usize,
    pub device: A2TrainingDevice,
}

impl Default for A2TrainerConfig {
    fn default() -> Self {
        Self {
            max_epochs: A2_DEFAULT_MAX_EPOCHS,
            batch_size: A2_DEFAULT_BATCH_SIZE,
            output_samples: A2_DEFAULT_OUTPUT_SAMPLES,
            learning_rate: A2_INITIAL_LEARNING_RATE,
            weight_decay: A2_ADAMW_WEIGHT_DECAY,
            learning_rate_gamma: A2_LEARNING_RATE_GAMMA,
            threshold_esr: None,
            seed: 0,
            validation_chunk_samples: A2_DEFAULT_OUTPUT_SAMPLES,
            device: A2TrainingDevice::Auto,
        }
    }
}

impl A2TrainerConfig {
    pub fn validate(self) -> Result<Self, A2TrainingError> {
        if self.max_epochs == 0 {
            return Err(A2TrainingError::InvalidConfig(
                "max_epochs must be positive",
            ));
        }
        if self.batch_size == 0 {
            return Err(A2TrainingError::InvalidConfig(
                "batch_size must be positive",
            ));
        }
        if self.output_samples == 0 {
            return Err(A2TrainingError::InvalidConfig(
                "output_samples must be positive",
            ));
        }
        if self.validation_chunk_samples == 0 {
            return Err(A2TrainingError::InvalidConfig(
                "validation_chunk_samples must be positive",
            ));
        }
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err(A2TrainingError::InvalidConfig(
                "learning_rate must be finite and positive",
            ));
        }
        if !self.weight_decay.is_finite() || self.weight_decay < 0.0 {
            return Err(A2TrainingError::InvalidConfig(
                "weight_decay must be finite and non-negative",
            ));
        }
        if !self.learning_rate_gamma.is_finite()
            || !(0.0..=1.0).contains(&self.learning_rate_gamma)
            || self.learning_rate_gamma == 0.0
        {
            return Err(A2TrainingError::InvalidConfig(
                "learning_rate_gamma must be in (0, 1]",
            ));
        }
        if self
            .threshold_esr
            .is_some_and(|threshold| !threshold.is_finite() || threshold <= 0.0)
        {
            return Err(A2TrainingError::InvalidConfig(
                "threshold_esr must be finite and positive",
            ));
        }
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct A2TrainingData<'a> {
    pub input: &'a [f32],
    pub target: &'a [f32],
    pub sample_rate_hz: u32,
}

impl A2TrainingData<'_> {
    fn validate(self) -> Result<(), A2TrainingError> {
        if self.sample_rate_hz != A2_SAMPLE_RATE_HZ {
            return Err(A2TrainingError::UnsupportedSampleRate(self.sample_rate_hz));
        }
        if self.input.len() != self.target.len() {
            return Err(A2TrainingError::LengthMismatch {
                input: self.input.len(),
                target: self.target.len(),
            });
        }
        let minimum =
            A2_TRAIN_START_SAMPLE + A2_VALIDATION_SAMPLE_COUNT + A2_RECEPTIVE_FIELD_SAMPLES;
        if self.input.len() < minimum {
            return Err(A2TrainingError::DatasetTooShort {
                found: self.input.len(),
                minimum,
            });
        }
        if self
            .input
            .iter()
            .chain(self.target.iter())
            .any(|sample| !sample.is_finite())
        {
            return Err(A2TrainingError::NonFiniteAudio);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct A2DatasetPlan {
    pub total_samples: usize,
    pub train_start_sample: usize,
    pub train_stop_sample: usize,
    pub validation_start_sample: usize,
    pub validation_stop_sample: usize,
    pub train_window_count: usize,
    pub full_batches_per_epoch: usize,
    pub output_samples_per_window: usize,
    pub input_samples_per_window: usize,
}

impl A2DatasetPlan {
    pub fn canonical(
        total_samples: usize,
        config: A2TrainerConfig,
    ) -> Result<Self, A2TrainingError> {
        let config = config.validate()?;
        let minimum =
            A2_TRAIN_START_SAMPLE + A2_VALIDATION_SAMPLE_COUNT + A2_RECEPTIVE_FIELD_SAMPLES;
        if total_samples < minimum {
            return Err(A2TrainingError::DatasetTooShort {
                found: total_samples,
                minimum,
            });
        }
        let train_stop_sample = total_samples - A2_VALIDATION_SAMPLE_COUNT;
        let train_samples = train_stop_sample - A2_TRAIN_START_SAMPLE;
        let single_sample_pairs = train_samples - A2_RECEPTIVE_LOOKBACK;
        let train_window_count = single_sample_pairs / config.output_samples;
        let full_batches_per_epoch = train_window_count / config.batch_size;
        if full_batches_per_epoch == 0 {
            return Err(A2TrainingError::NoCompleteTrainingBatch);
        }
        Ok(Self {
            total_samples,
            train_start_sample: A2_TRAIN_START_SAMPLE,
            train_stop_sample,
            validation_start_sample: train_stop_sample,
            validation_stop_sample: total_samples,
            train_window_count,
            full_batches_per_epoch,
            output_samples_per_window: config.output_samples,
            input_samples_per_window: A2_RECEPTIVE_LOOKBACK + config.output_samples,
        })
    }
}

#[derive(Clone, Default)]
pub struct A2CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl A2CancellationToken {
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
pub enum A2PublicationQuality {
    /// Official NAM wording: "Great!"
    Great,
    /// Official NAM wording: "Not bad!"
    NotBad,
    /// Official NAM wording: "This might sound ok."
    MightSoundOkay,
    /// Official NAM wording: "This probably won't sound great."
    ProbablyPoor,
    /// Official NAM wording: "Something seems to have gone wrong."
    Failed,
}

impl A2PublicationQuality {
    #[must_use]
    pub fn from_esr(esr: f64) -> Self {
        if esr < 0.01 {
            Self::Great
        } else if esr < 0.035 {
            Self::NotBad
        } else if esr < 0.1 {
            Self::MightSoundOkay
        } else if esr < 0.3 {
            Self::ProbablyPoor
        } else {
            Self::Failed
        }
    }

    /// The agreed capture-library gate rejects only NAM's explicit failure
    /// band (`ESR >= 0.30`).  The finer quality label remains available to the
    /// UI and metadata.
    #[must_use]
    pub const fn passes_publication_gate(self) -> bool {
        !matches!(self, Self::Failed)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct A2TrainingProgress {
    pub completed_epochs: u32,
    pub maximum_epochs: u32,
    pub epoch_training_mse: f64,
    pub validation_esr: f64,
    pub best_validation_esr: f64,
    pub best_epoch: u32,
    pub learning_rate: f64,
    pub epoch_seconds: f64,
    pub elapsed_seconds: f64,
    pub output_samples_per_second: f64,
    pub quality: A2PublicationQuality,
    pub device_status: A2TrainingDeviceStatus,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum A2TrainingStopReason {
    MaximumEpochs,
    ThresholdReached,
    Cancelled,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct A2QualityReport {
    pub validation_esr: f64,
    pub validation_esr_db: f64,
    pub quality: A2PublicationQuality,
    pub passes_publication_gate: bool,
    pub original_train_target_rms_dbfs: f64,
    pub output_normalization_gain: f64,
    pub mrstft_weight_applied: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct A2TrainingOutcome {
    pub model: A2Model,
    pub completed_epochs: u32,
    pub best_epoch: u32,
    pub stop_reason: A2TrainingStopReason,
    pub quality: A2QualityReport,
    pub elapsed_seconds: f64,
}

#[derive(Debug)]
pub enum A2TrainingError {
    InvalidConfig(&'static str),
    UnsupportedSampleRate(u32),
    LengthMismatch { input: usize, target: usize },
    DatasetTooShort { found: usize, minimum: usize },
    NoCompleteTrainingBatch,
    NonFiniteAudio,
    SilentTrainingTarget,
    CancelledBeforeFirstCheckpoint,
    MetalUnavailable { reason: String },
    InvalidExport(String),
    Candle(candle_core::Error),
}

impl fmt::Display for A2TrainingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => {
                write!(formatter, "invalid A2 trainer config: {message}")
            }
            Self::UnsupportedSampleRate(rate) => {
                write!(formatter, "A2 training requires 48000 Hz, found {rate} Hz")
            }
            Self::LengthMismatch { input, target } => {
                write!(
                    formatter,
                    "input/target length mismatch: {input} != {target}"
                )
            }
            Self::DatasetTooShort { found, minimum } => write!(
                formatter,
                "A2 v3 dataset is too short: {found} samples, requires at least {minimum}"
            ),
            Self::NoCompleteTrainingBatch => {
                formatter.write_str("A2 dataset does not contain one complete training batch")
            }
            Self::NonFiniteAudio => {
                formatter.write_str("A2 training audio contains NaN or infinity")
            }
            Self::SilentTrainingTarget => formatter
                .write_str("A2 training target is silent; output normalization is undefined"),
            Self::CancelledBeforeFirstCheckpoint => formatter
                .write_str("A2 training was cancelled before the first validation checkpoint"),
            Self::MetalUnavailable { reason } => write!(
                formatter,
                "Metal A2 training was explicitly requested, but Metal device 0 could not be initialized: {reason}"
            ),
            Self::InvalidExport(message) => write!(formatter, "invalid A2 export: {message}"),
            Self::Candle(error) => write!(formatter, "Candle A2 trainer error: {error}"),
        }
    }
}

impl std::error::Error for A2TrainingError {}

impl From<candle_core::Error> for A2TrainingError {
    fn from(error: candle_core::Error) -> Self {
        Self::Candle(error)
    }
}

struct A2TrainLayer {
    conv: Conv1d,
    input_mixin: Conv1d,
    residual: Conv1d,
}

struct A2TrainNetwork {
    rechannel: Conv1d,
    layers: Vec<A2TrainLayer>,
    head: Conv1d,
}

impl A2TrainNetwork {
    fn new(
        var_map: &mut VarMap,
        device: &Device,
        rng: &mut XorShift64,
    ) -> Result<Self, A2TrainingError> {
        let rechannel = make_conv1d(
            var_map,
            "a2.rechannel",
            1,
            A2_CHANNELS,
            1,
            1,
            false,
            device,
            rng,
        )?;
        let mut layers = Vec::with_capacity(A2_LAYER_COUNT);
        for layer_index in 0..A2_LAYER_COUNT {
            let prefix = format!("a2.layers.{layer_index:02}");
            let conv = make_conv1d(
                var_map,
                &format!("{prefix}.conv"),
                A2_CHANNELS,
                A2_CHANNELS,
                A2_KERNEL_SIZES[layer_index],
                A2_DILATIONS[layer_index],
                true,
                device,
                rng,
            )?;
            let input_mixin = make_conv1d(
                var_map,
                &format!("{prefix}.input_mixin"),
                1,
                A2_CHANNELS,
                1,
                1,
                false,
                device,
                rng,
            )?;
            let residual = make_conv1d(
                var_map,
                &format!("{prefix}.residual"),
                A2_CHANNELS,
                A2_CHANNELS,
                1,
                1,
                true,
                device,
                rng,
            )?;
            layers.push(A2TrainLayer {
                conv,
                input_mixin,
                residual,
            });
        }
        let head = make_conv1d(
            var_map,
            "a2.head",
            A2_CHANNELS,
            1,
            A2_HEAD_KERNEL_SIZE,
            1,
            true,
            device,
            rng,
        )?;

        let trainable_parameter_count: usize = var_map
            .all_vars()
            .iter()
            .map(|variable| variable.elem_count())
            .sum();
        if trainable_parameter_count != A2_TRAINABLE_PARAMETER_COUNT {
            return Err(A2TrainingError::InvalidExport(format!(
                "A2 trainable parameter count is {trainable_parameter_count}, expected {A2_TRAINABLE_PARAMETER_COUNT}"
            )));
        }

        Ok(Self {
            rechannel,
            layers,
            head,
        })
    }

    /// Input shape `(batch, 1, receptive_field + output_samples - 1)`;
    /// output shape `(batch, output_samples)`.
    fn forward(&self, input: &Tensor) -> Result<Tensor, A2TrainingError> {
        let input_length = input.dim(2)?;
        if input_length < A2_RECEPTIVE_FIELD_SAMPLES {
            return Err(A2TrainingError::InvalidConfig(
                "network input is shorter than the A2 receptive field",
            ));
        }
        let head_input_length = input_length - (A2_RECEPTIVE_FIELD_WITHOUT_HEAD - 1);
        let condition = input;
        let condition_length = condition.dim(2)?;
        let mut residual = self.rechannel.forward(input)?;
        let mut head_sum: Option<Tensor> = None;

        for layer in &self.layers {
            let convolved = layer.conv.forward(&residual)?;
            let convolved_length = convolved.dim(2)?;
            let mixed_full = layer.input_mixin.forward(condition)?;
            let mixed =
                mixed_full.narrow(2, condition_length - convolved_length, convolved_length)?;
            let activation = candle_nn::ops::leaky_relu(
                &(&convolved + &mixed)?,
                f64::from(A2_LEAKY_RELU_SLOPE),
            )?;

            let residual_delta = layer.residual.forward(&activation)?;
            let residual_length = residual_delta.dim(2)?;
            residual =
                (&residual.narrow(2, residual.dim(2)? - residual_length, residual_length)?
                    + &residual_delta)?;

            let activation_length = activation.dim(2)?;
            let head_term =
                activation.narrow(2, activation_length - head_input_length, head_input_length)?;
            head_sum = Some(match head_sum {
                Some(accumulator) => (&accumulator + &head_term)?,
                None => head_term,
            });
        }

        let output = self
            .head
            .forward(&head_sum.expect("A2 always contains residual layers"))?
            .affine(f64::from(A2_HEAD_SCALE), 0.0)?;
        Ok(output.squeeze(1)?)
    }

    fn export(&self, output_normalization_gain: f64) -> Result<A2Model, A2TrainingError> {
        if !output_normalization_gain.is_finite() || output_normalization_gain <= 0.0 {
            return Err(A2TrainingError::InvalidExport(
                "output normalization gain is invalid".to_owned(),
            ));
        }

        // Build the canonical official NAM stream in PyTorch Conv1d storage
        // order.  The runtime-owned adapter performs the single authoritative
        // conversion into its tap-major fixed arrays.
        let mut coefficients = Vec::with_capacity(A2_TRAINABLE_PARAMETER_COUNT + 1);
        coefficients.extend(tensor_flat(self.rechannel.weight())?);
        for layer in &self.layers {
            coefficients.extend(tensor_flat(layer.conv.weight())?);
            coefficients.extend(tensor_bias(layer.conv.bias())?);
            coefficients.extend(tensor_flat(layer.input_mixin.weight())?);
            coefficients.extend(tensor_flat(layer.residual.weight())?);
            coefficients.extend(tensor_bias(layer.residual.bias())?);
        }
        coefficients.extend(tensor_flat(self.head.weight())?);
        coefficients.extend(tensor_bias(self.head.bias())?);
        // Training predicts a target multiplied by normalization_gain.  NAM's
        // export hook divides the final head scale by the same gain.
        coefficients.push((f64::from(A2_HEAD_SCALE) / output_normalization_gain) as f32);
        let weights = A2Weights::from_official_weight_slice(&coefficients)
            .map_err(|error| A2TrainingError::InvalidExport(error.to_string()))?;

        let model = A2Model::from_weights(weights);
        model
            .validate()
            .map_err(|error| A2TrainingError::InvalidExport(error.to_string()))?;
        Ok(model)
    }
}

fn make_conv1d(
    var_map: &mut VarMap,
    prefix: &str,
    input_channels: usize,
    output_channels: usize,
    kernel_size: usize,
    dilation: usize,
    bias: bool,
    device: &Device,
    rng: &mut XorShift64,
) -> Result<Conv1d, A2TrainingError> {
    let fan_in = input_channels * kernel_size;
    let bound = 1.0 / (fan_in as f64).sqrt();
    let weight_name = format!("{prefix}.weight");
    let weight_shape = (output_channels, input_channels, kernel_size);
    let weight_count = output_channels * input_channels * kernel_size;
    let initial_weight = Tensor::from_vec(
        random_uniform_f32(rng, weight_count, -bound, bound),
        weight_shape,
        device,
    )?;
    let _ = var_map.get(
        weight_shape,
        &weight_name,
        Init::Const(0.0),
        DType::F32,
        device,
    )?;
    var_map.set_one(&weight_name, &initial_weight)?;
    let weight = var_map.get(
        weight_shape,
        &weight_name,
        Init::Const(0.0),
        DType::F32,
        device,
    )?;

    let bias_tensor = if bias {
        let bias_name = format!("{prefix}.bias");
        let initial_bias = Tensor::from_vec(
            random_uniform_f32(rng, output_channels, -bound, bound),
            output_channels,
            device,
        )?;
        let _ = var_map.get(
            output_channels,
            &bias_name,
            Init::Const(0.0),
            DType::F32,
            device,
        )?;
        var_map.set_one(&bias_name, &initial_bias)?;
        Some(var_map.get(
            output_channels,
            &bias_name,
            Init::Const(0.0),
            DType::F32,
            device,
        )?)
    } else {
        None
    };

    Ok(Conv1d::new(
        weight,
        bias_tensor,
        Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation,
            groups: 1,
            cudnn_fwd_algo: None,
        },
    ))
}

fn tensor_flat(tensor: &Tensor) -> Result<Vec<f32>, A2TrainingError> {
    Ok(tensor.flatten_all()?.to_vec1::<f32>()?)
}

fn tensor_bias(bias: Option<&Tensor>) -> Result<Vec<f32>, A2TrainingError> {
    let bias = bias
        .ok_or_else(|| A2TrainingError::InvalidExport("expected a convolution bias".to_owned()))?;
    tensor_flat(bias)
}

#[derive(Clone)]
struct VariableSnapshot {
    entries: Vec<(String, Vec<usize>, Vec<f32>)>,
}

impl VariableSnapshot {
    fn capture(var_map: &VarMap) -> Result<Self, A2TrainingError> {
        let variables = var_map.data().lock().expect("Candle VarMap mutex poisoned");
        let mut names: Vec<&String> = variables.keys().collect();
        names.sort_unstable();
        let mut entries = Vec::with_capacity(names.len());
        for name in names {
            let variable = variables
                .get(name)
                .expect("name was collected from the same VarMap");
            entries.push((
                name.clone(),
                variable.dims().to_vec(),
                variable.flatten_all()?.to_vec1::<f32>()?,
            ));
        }
        Ok(Self { entries })
    }

    fn restore(&self, var_map: &VarMap, device: &Device) -> Result<(), A2TrainingError> {
        let variables = var_map.data().lock().expect("Candle VarMap mutex poisoned");
        for (name, shape, values) in &self.entries {
            let variable = variables.get(name).ok_or_else(|| {
                A2TrainingError::InvalidExport(format!(
                    "best checkpoint variable {name} is missing"
                ))
            })?;
            variable.set(&Tensor::from_vec(values.clone(), shape.as_slice(), device)?)?;
        }
        Ok(())
    }
}

struct NormalizedDataset<'a> {
    input: &'a [f32],
    target: &'a [f32],
    plan: A2DatasetPlan,
    normalization_gain: f64,
    original_train_target_rms_dbfs: f64,
}

impl<'a> NormalizedDataset<'a> {
    fn new(data: A2TrainingData<'a>, config: A2TrainerConfig) -> Result<Self, A2TrainingError> {
        data.validate()?;
        let plan = A2DatasetPlan::canonical(data.input.len(), config)?;
        let train_target = &data.target[plan.train_start_sample..plan.train_stop_sample];
        let sum_squares = train_target
            .iter()
            .map(|sample| {
                let sample = f64::from(*sample);
                sample * sample
            })
            .sum::<f64>();
        if sum_squares == 0.0 {
            return Err(A2TrainingError::SilentTrainingTarget);
        }
        let original_rms = (sum_squares / train_target.len() as f64).sqrt();
        let desired_rms = 10.0_f64.powf(A2_OUTPUT_NORMALIZATION_DBFS / 20.0);
        let normalization_gain = desired_rms / original_rms;
        if !normalization_gain.is_finite() || normalization_gain <= 0.0 {
            return Err(A2TrainingError::SilentTrainingTarget);
        }
        Ok(Self {
            input: data.input,
            target: data.target,
            plan,
            normalization_gain,
            original_train_target_rms_dbfs: 20.0 * original_rms.log10(),
        })
    }

    fn training_batch(
        &self,
        window_indices: &[usize],
        device: &Device,
    ) -> Result<(Tensor, Tensor), A2TrainingError> {
        let input_samples = self.plan.input_samples_per_window;
        let output_samples = self.plan.output_samples_per_window;
        let mut input = Vec::with_capacity(window_indices.len() * input_samples);
        let mut target = Vec::with_capacity(window_indices.len() * output_samples);

        for &window_index in window_indices {
            let input_start = self.plan.train_start_sample + window_index * output_samples;
            let input_stop = input_start + input_samples;
            let target_start = input_start + A2_RECEPTIVE_LOOKBACK;
            let target_stop = target_start + output_samples;
            input.extend_from_slice(&self.input[input_start..input_stop]);
            target.extend(
                self.target[target_start..target_stop]
                    .iter()
                    .map(|sample| (f64::from(*sample) * self.normalization_gain) as f32),
            );
        }

        Ok((
            Tensor::from_vec(input, (window_indices.len(), 1, input_samples), device)?,
            Tensor::from_vec(target, (window_indices.len(), output_samples), device)?,
        ))
    }

    fn validation_esr(
        &self,
        network: &A2TrainNetwork,
        chunk_samples: usize,
        device: &Device,
        cancellation: &A2CancellationToken,
    ) -> Result<Option<f64>, A2TrainingError> {
        let split_samples = self.plan.validation_stop_sample - self.plan.validation_start_sample;
        let output_samples = split_samples - A2_RECEPTIVE_LOOKBACK;
        let mut error_sum_squares = 0.0_f64;
        let mut target_sum_squares = 0.0_f64;
        let mut output_offset = 0;

        while output_offset < output_samples {
            if cancellation.is_cancelled() {
                return Ok(None);
            }
            let chunk = chunk_samples.min(output_samples - output_offset);
            let input_start = self.plan.validation_start_sample + output_offset;
            let input_stop = input_start + A2_RECEPTIVE_LOOKBACK + chunk;
            let target_start = input_start + A2_RECEPTIVE_LOOKBACK;
            let target_stop = target_start + chunk;
            let input = Tensor::from_slice(
                &self.input[input_start..input_stop],
                (1, 1, A2_RECEPTIVE_LOOKBACK + chunk),
                device,
            )?;
            let target_values: Vec<f32> = self.target[target_start..target_stop]
                .iter()
                .map(|sample| (f64::from(*sample) * self.normalization_gain) as f32)
                .collect();
            let target = Tensor::from_slice(&target_values, (1, chunk), device)?;
            let prediction = network.forward(&input)?;
            let error = (&prediction - &target)?;
            error_sum_squares += error.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
            target_sum_squares += target.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
            output_offset += chunk;
        }
        if target_sum_squares <= 0.0 {
            return Err(A2TrainingError::SilentTrainingTarget);
        }
        Ok(Some(error_sum_squares / target_sum_squares))
    }
}

/// Train the official fixed A2 C=3 network and return the best validation
/// checkpoint, not merely the weights from the final epoch.
///
/// The progress callback runs once per fully validated epoch.  There is no
/// patience-based early stop.  Cancellation is observed between minibatches
/// and validation chunks.
pub fn train_a2(
    data: A2TrainingData<'_>,
    config: A2TrainerConfig,
    cancellation: &A2CancellationToken,
    mut progress: impl FnMut(A2TrainingProgress),
) -> Result<A2TrainingOutcome, A2TrainingError> {
    let config = config.validate()?;
    let dataset = NormalizedDataset::new(data, config)?;
    let resolved_device = config.device.resolve()?;
    let device_status = resolved_device.status();
    let device = resolved_device.device();
    let mut rng = XorShift64::new(config.seed);
    let mut var_map = VarMap::new();
    let network = A2TrainNetwork::new(&mut var_map, device, &mut rng)?;
    let mut optimizer = AdamW::new(
        var_map.all_vars(),
        ParamsAdamW {
            lr: config.learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: config.weight_decay,
        },
    )?;

    let mut indices: Vec<usize> = (0..dataset.plan.train_window_count).collect();
    let total_started = Instant::now();
    let mut completed_epochs = 0;
    let mut best_epoch = 0;
    let mut best_validation_esr = f64::INFINITY;
    let mut best_snapshot: Option<VariableSnapshot> = None;
    let mut stop_reason = A2TrainingStopReason::MaximumEpochs;

    for epoch_index in 0..config.max_epochs {
        if cancellation.is_cancelled() {
            stop_reason = A2TrainingStopReason::Cancelled;
            break;
        }
        let epoch_started = Instant::now();
        shuffle(&mut indices, &mut rng);
        // Keep loss accounting on the selected device. Reading every
        // minibatch scalar with `to_scalar()` serializes the Metal command
        // queue and largely defeats GPU training. Detached scalar additions
        // do not retain the autograd graph; one readback at epoch end is
        // sufficient for progress reporting and the non-finite guard.
        let mut epoch_mse_sum: Option<Tensor> = None;
        let mut processed_batches = 0_usize;

        for batch_indices in indices
            .chunks_exact(config.batch_size)
            .take(dataset.plan.full_batches_per_epoch)
        {
            if cancellation.is_cancelled() {
                stop_reason = A2TrainingStopReason::Cancelled;
                break;
            }
            let (input, target) = dataset.training_batch(batch_indices, device)?;
            let prediction = network.forward(&input)?;
            let mse = (&prediction - &target)?.sqr()?.mean_all()?;
            let detached_mse = mse.detach();
            epoch_mse_sum = Some(match epoch_mse_sum {
                Some(sum) => (&sum + &detached_mse)?,
                None => detached_mse,
            });

            // TODO(A2/MRSTFT): add exact auraloss-compatible multi-resolution
            // STFT loss at A2_MRSTFT_WEIGHT.  Do not substitute a different
            // spectral loss under the same name.
            optimizer.backward_step(&mse)?;
            processed_batches += 1;
        }

        if cancellation.is_cancelled() {
            stop_reason = A2TrainingStopReason::Cancelled;
            break;
        }
        let epoch_mse_sum = epoch_mse_sum
            .expect("canonical A2 plan guarantees at least one full batch")
            .to_scalar::<f32>()? as f64;
        if !epoch_mse_sum.is_finite() {
            return Err(A2TrainingError::InvalidExport(
                "training loss became non-finite".to_owned(),
            ));
        }

        let Some(validation_esr) = dataset.validation_esr(
            &network,
            config.validation_chunk_samples,
            device,
            cancellation,
        )?
        else {
            stop_reason = A2TrainingStopReason::Cancelled;
            break;
        };
        completed_epochs = epoch_index + 1;
        if validation_esr < best_validation_esr {
            best_validation_esr = validation_esr;
            best_epoch = completed_epochs;
            best_snapshot = Some(VariableSnapshot::capture(&var_map)?);
        }

        let epoch_seconds = epoch_started.elapsed().as_secs_f64();
        let processed_output_samples = processed_batches
            .saturating_mul(config.batch_size)
            .saturating_mul(config.output_samples);
        let epoch_training_mse = epoch_mse_sum / processed_batches.max(1) as f64;
        progress(A2TrainingProgress {
            completed_epochs,
            maximum_epochs: config.max_epochs,
            epoch_training_mse,
            validation_esr,
            best_validation_esr,
            best_epoch,
            learning_rate: optimizer.learning_rate(),
            epoch_seconds,
            elapsed_seconds: total_started.elapsed().as_secs_f64(),
            output_samples_per_second: processed_output_samples as f64
                / epoch_seconds.max(f64::EPSILON),
            quality: A2PublicationQuality::from_esr(validation_esr),
            device_status,
        });

        if config
            .threshold_esr
            .is_some_and(|threshold| validation_esr <= threshold)
        {
            stop_reason = A2TrainingStopReason::ThresholdReached;
            break;
        }
        optimizer.set_learning_rate(optimizer.learning_rate() * config.learning_rate_gamma);
    }

    let best_snapshot = best_snapshot.ok_or(A2TrainingError::CancelledBeforeFirstCheckpoint)?;
    best_snapshot.restore(&var_map, device)?;
    let model = network.export(dataset.normalization_gain)?;
    let quality = A2PublicationQuality::from_esr(best_validation_esr);
    Ok(A2TrainingOutcome {
        model,
        completed_epochs,
        best_epoch,
        stop_reason,
        quality: A2QualityReport {
            validation_esr: best_validation_esr,
            validation_esr_db: 10.0 * best_validation_esr.log10(),
            quality,
            passes_publication_gate: quality.passes_publication_gate(),
            original_train_target_rms_dbfs: dataset.original_train_target_rms_dbfs,
            output_normalization_gain: dataset.normalization_gain,
            mrstft_weight_applied: 0.0,
        },
        elapsed_seconds: total_started.elapsed().as_secs_f64(),
    })
}

#[derive(Clone, Copy)]
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x9e37_79b9_7f4a_7c15
            } else {
                seed
            },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut value = self.state;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        self.state = value;
        value
    }

    fn unit_f64(&mut self) -> f64 {
        let mantissa = self.next_u64() >> 11;
        mantissa as f64 * (1.0 / ((1_u64 << 53) as f64))
    }

    fn index(&mut self, exclusive_upper_bound: usize) -> usize {
        (self.next_u64() as usize) % exclusive_upper_bound
    }
}

fn random_uniform_f32(rng: &mut XorShift64, count: usize, minimum: f64, maximum: f64) -> Vec<f32> {
    (0..count)
        .map(|_| (minimum + (maximum - minimum) * rng.unit_f64()) as f32)
        .collect()
}

fn shuffle(values: &mut [usize], rng: &mut XorShift64) {
    for index in (1..values.len()).rev() {
        values.swap(index, rng.index(index + 1));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn training_device_defaults_to_auto_and_cpu_is_explicitly_resolvable() {
        assert_eq!(A2TrainingDevice::default(), A2TrainingDevice::Auto);
        assert_eq!(A2TrainerConfig::default().device, A2TrainingDevice::Auto);

        let resolved = A2TrainingDevice::Cpu.resolve().unwrap();
        assert_eq!(resolved.requested(), A2TrainingDevice::Cpu);
        assert_eq!(resolved.status(), A2TrainingDeviceStatus::Cpu);
        assert_eq!(resolved.label(), "CPU");
        assert!(resolved.fallback_reason().is_none());
        assert!(resolved.device().is_cpu());
    }

    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    #[test]
    fn auto_reports_cpu_fallback_when_metal_is_not_compiled() {
        let resolved = A2TrainingDevice::Auto.resolve().unwrap();
        assert_eq!(resolved.status(), A2TrainingDeviceStatus::CpuAutoFallback);
        assert_eq!(resolved.label(), "CPU (AUTO FALLBACK)");
        assert!(resolved.fallback_reason().is_some());

        let error = A2TrainingDevice::Metal.resolve().unwrap_err();
        assert!(matches!(error, A2TrainingError::MetalUnavailable { .. }));
        assert!(
            error
                .to_string()
                .contains("Metal device 0 could not be initialized")
        );
    }

    #[test]
    fn canonical_a2_shape_has_1870_trainable_parameters() {
        let device = Device::Cpu;
        let mut var_map = VarMap::new();
        let mut rng = XorShift64::new(7);
        let network = A2TrainNetwork::new(&mut var_map, &device, &mut rng).unwrap();
        let count: usize = var_map
            .all_vars()
            .iter()
            .map(|variable| variable.elem_count())
            .sum();
        assert_eq!(count, A2_TRAINABLE_PARAMETER_COUNT);
        assert_eq!(network.layers.len(), 23);
        assert_eq!(A2_RECEPTIVE_FIELD_SAMPLES, 6_347);
    }

    #[test]
    fn synthetic_batch_runs_forward_backward_and_exports_runtime_weights() {
        let device = Device::Cpu;
        let mut var_map = VarMap::new();
        let mut rng = XorShift64::new(19);
        let network = A2TrainNetwork::new(&mut var_map, &device, &mut rng).unwrap();
        let output_samples = 8;
        let input_samples = A2_RECEPTIVE_LOOKBACK + output_samples;
        let source: Vec<f32> = (0..input_samples)
            .map(|index| ((index as f32 * 0.013).sin() * 0.2) + 0.01)
            .collect();
        let target: Vec<f32> = source[A2_RECEPTIVE_LOOKBACK..]
            .iter()
            .map(|sample| (sample * 4.0).tanh())
            .collect();
        let input = Tensor::from_vec(source, (1, 1, input_samples), &device).unwrap();
        let target = Tensor::from_vec(target, (1, output_samples), &device).unwrap();
        let before = network.forward(&input).unwrap();
        assert_eq!(before.dims(), &[1, output_samples]);
        let loss = (&before - &target)
            .unwrap()
            .sqr()
            .unwrap()
            .mean_all()
            .unwrap();
        assert!(loss.to_scalar::<f32>().unwrap().is_finite());

        let mut optimizer = AdamW::new(
            var_map.all_vars(),
            ParamsAdamW {
                lr: A2_INITIAL_LEARNING_RATE,
                weight_decay: A2_ADAMW_WEIGHT_DECAY,
                ..ParamsAdamW::default()
            },
        )
        .unwrap();
        optimizer.backward_step(&loss).unwrap();

        let exported = network.export(2.0).unwrap();
        assert_eq!(exported.format_version, crate::a2::A2_MODEL_FORMAT_VERSION);
        assert_eq!(exported.weights.head_scale, A2_HEAD_SCALE / 2.0);
        exported.validate().unwrap();
    }

    #[test]
    fn candle_training_graph_and_streaming_runtime_are_numerically_equivalent() {
        let device = Device::Cpu;
        let mut var_map = VarMap::new();
        let mut rng = XorShift64::new(31);
        let network = A2TrainNetwork::new(&mut var_map, &device, &mut rng).unwrap();
        let output_samples = 16;
        let input_samples = A2_RECEPTIVE_LOOKBACK + output_samples;
        let source: Vec<f32> = (0..input_samples)
            .map(|index| {
                let phase = index as f32;
                0.17 * (phase * 0.019).sin() + 0.07 * (phase * 0.071).cos()
            })
            .collect();
        let input = Tensor::from_slice(&source, (1, 1, input_samples), &device).unwrap();
        let candle = network
            .forward(&input)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        let model = network.export(1.0).unwrap();
        let mut runtime = crate::a2::A2Processor::new(model).unwrap();
        let mut streamed = vec![0.0; source.len()];
        runtime.process_block(&source, &mut streamed);
        let streamed = &streamed[A2_RECEPTIVE_LOOKBACK..];
        assert_eq!(candle.len(), streamed.len());
        for (index, (expected, actual)) in candle.iter().zip(streamed).enumerate() {
            let error = (expected - actual).abs();
            assert!(
                error <= 2.0e-5,
                "sample {index}: Candle {expected}, runtime {actual}, error {error}"
            );
        }
    }

    #[test]
    fn output_normalization_is_reversed_only_in_exported_head_scale() {
        let device = Device::Cpu;
        let mut var_map = VarMap::new();
        let mut rng = XorShift64::new(23);
        let network = A2TrainNetwork::new(&mut var_map, &device, &mut rng).unwrap();
        let gain = 3.25;
        let model = network.export(gain).unwrap();
        assert_eq!(
            model.weights.head_scale,
            (f64::from(A2_HEAD_SCALE) / gain) as f32
        );
    }

    #[test]
    fn canonical_plan_matches_nam_v3_crop_and_drop_last_batches() {
        let total_samples = 9_120_000;
        let config = A2TrainerConfig {
            device: A2TrainingDevice::Cpu,
            ..A2TrainerConfig::default()
        };
        let plan = A2DatasetPlan::canonical(total_samples, config).unwrap();
        assert_eq!(plan.train_start_sample, 480_000);
        assert_eq!(plan.train_stop_sample, 8_688_000);
        assert_eq!(plan.validation_start_sample, 8_688_000);
        assert_eq!(plan.validation_stop_sample, 9_120_000);
        assert_eq!(
            plan.train_window_count,
            (8_208_000 - A2_RECEPTIVE_LOOKBACK) / 8_192
        );
        assert_eq!(
            plan.full_batches_per_epoch,
            plan.train_window_count / A2_DEFAULT_BATCH_SIZE
        );
        assert_eq!(
            plan.input_samples_per_window,
            A2_RECEPTIVE_LOOKBACK + A2_DEFAULT_OUTPUT_SAMPLES
        );
    }

    #[test]
    fn publication_quality_thresholds_match_official_nam_comments() {
        assert_eq!(
            A2PublicationQuality::from_esr(0.009),
            A2PublicationQuality::Great
        );
        assert_eq!(
            A2PublicationQuality::from_esr(0.02),
            A2PublicationQuality::NotBad
        );
        assert_eq!(
            A2PublicationQuality::from_esr(0.05),
            A2PublicationQuality::MightSoundOkay
        );
        assert_eq!(
            A2PublicationQuality::from_esr(0.2),
            A2PublicationQuality::ProbablyPoor
        );
        assert_eq!(
            A2PublicationQuality::from_esr(0.4),
            A2PublicationQuality::Failed
        );
        assert!(A2PublicationQuality::ProbablyPoor.passes_publication_gate());
        assert!(!A2PublicationQuality::Failed.passes_publication_gate());
    }

    #[test]
    fn cancellation_token_is_reusable() {
        let token = A2CancellationToken::default();
        assert!(!token.is_cancelled());
        token.cancel();
        assert!(token.is_cancelled());
        token.reset();
        assert!(!token.is_cancelled());
    }

    /// Opt-in because creating a Metal command queue and compiling kernels is
    /// comparatively expensive and requires a physical Apple GPU.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    #[ignore = "requires Apple Metal; run explicitly for backend smoke/parity"]
    fn metal_forward_backward_smoke_and_cpu_parity() {
        fn one_step(
            device: &Device,
            source: &[f32],
            target_values: &[f32],
        ) -> (Vec<f32>, Vec<f32>) {
            let mut var_map = VarMap::new();
            let mut rng = XorShift64::new(0x4d45_5441_4c);
            let network = A2TrainNetwork::new(&mut var_map, device, &mut rng).unwrap();
            let output_samples = target_values.len();
            let input = Tensor::from_slice(
                source,
                (1, 1, A2_RECEPTIVE_LOOKBACK + output_samples),
                device,
            )
            .unwrap();
            let target = Tensor::from_slice(target_values, (1, output_samples), device).unwrap();
            let before = network.forward(&input).unwrap();
            let before_values = before.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let loss = (&before - &target)
                .unwrap()
                .sqr()
                .unwrap()
                .mean_all()
                .unwrap();
            assert!(loss.to_scalar::<f32>().unwrap().is_finite());

            let mut optimizer = AdamW::new(
                var_map.all_vars(),
                ParamsAdamW {
                    lr: A2_INITIAL_LEARNING_RATE,
                    weight_decay: A2_ADAMW_WEIGHT_DECAY,
                    ..ParamsAdamW::default()
                },
            )
            .unwrap();
            optimizer.backward_step(&loss).unwrap();
            let after_values = network
                .forward(&input)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert!(after_values.iter().all(|sample| sample.is_finite()));
            (before_values, after_values)
        }

        let resolved = A2TrainingDevice::Auto.resolve().unwrap();
        if resolved.status() == A2TrainingDeviceStatus::CpuAutoFallback {
            eprintln!(
                "Metal smoke skipped because this process cannot access a GPU: {}",
                resolved.fallback_reason().unwrap_or("unknown reason")
            );
            return;
        }
        assert_eq!(resolved.status(), A2TrainingDeviceStatus::Metal);
        assert!(resolved.fallback_reason().is_none());
        let metal = resolved.device();
        let cpu = Device::Cpu;
        let output_samples = 16;
        let source: Vec<f32> = (0..A2_RECEPTIVE_LOOKBACK + output_samples)
            .map(|index| {
                let phase = index as f32;
                0.13 * (phase * 0.017).sin() + 0.05 * (phase * 0.059).cos()
            })
            .collect();
        let target: Vec<f32> = source[A2_RECEPTIVE_LOOKBACK..]
            .iter()
            .map(|sample| (sample * 3.5).tanh())
            .collect();
        let (cpu_before, cpu_after) = one_step(&cpu, &source, &target);
        let (metal_before, metal_after) = one_step(metal, &source, &target);

        for (index, (cpu_sample, metal_sample)) in cpu_before.iter().zip(&metal_before).enumerate()
        {
            assert!(
                (cpu_sample - metal_sample).abs() <= 5.0e-4,
                "pre-update sample {index}: CPU {cpu_sample}, Metal {metal_sample}"
            );
        }
        for (index, (cpu_sample, metal_sample)) in cpu_after.iter().zip(&metal_after).enumerate() {
            assert!(
                (cpu_sample - metal_sample).abs() <= 5.0e-3,
                "post-update sample {index}: CPU {cpu_sample}, Metal {metal_sample}"
            );
        }
    }
}
