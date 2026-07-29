//! Strict, offline conversion from Neural Amp Modeler `.nam` JSON to the
//! immutable MOT A2 container.
//!
//! A `.nam` file is not a single architecture.  The live MOT runtime currently
//! implements only the exact, zero-latency WaveNet A2 shape with three
//! channels.  Import therefore validates the complete mathematical shape
//! before reordering the official weight stream.  It never approximates,
//! retrains, or silently substitutes an unsupported model.

use std::fmt;
use std::path::Path;

use serde_json::{Map, Value};

use crate::a2::{
    A2_CHANNELS, A2_DILATIONS, A2_KERNEL_SIZES, A2_LAYER_COUNT, A2_LEAKY_RELU_SLOPE,
    A2_MACS_PER_SAMPLE, A2_SAMPLE_RATE_HZ, A2Error, A2Model, A2Weights, encode_a2_payload,
};
use crate::model::{
    A2_ARCHITECTURE_ID, A2_ARCHITECTURE_VERSION, ModelError, ModelMetadata, MotModel, Sha256Digest,
    sha256,
};

/// A conservative ceiling for a JSON source read by the loader worker.
///
/// Exact C3 A2 models are far smaller than this.  The ceiling keeps malformed
/// or unrelated files from making an editor-triggered import consume
/// unbounded memory.
pub const MAX_NAM_SOURCE_BYTES: usize = 16 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NamImportSelection {
    DirectWaveNet,
    SlimmableContainer {
        selected_submodel_index: usize,
        submodel_count: usize,
    },
}

impl NamImportSelection {
    #[must_use]
    pub fn notice(self) -> Option<String> {
        match self {
            Self::DirectWaveNet => None,
            Self::SlimmableContainer { .. } => Some(
                "Imported the compatible C3 / Nano submodel; a NAM host may use the container's C8 model by default.".to_owned(),
            ),
        }
    }

    #[must_use]
    fn display_name_suffix(self) -> Option<&'static str> {
        match self {
            Self::DirectWaveNet => None,
            Self::SlimmableContainer { .. } => Some(" — C3 Nano"),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConvertedNam {
    pub model: MotModel,
    pub source_sha256: Sha256Digest,
    pub selection: NamImportSelection,
    pub provenance: NamImportProvenance,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NamImportProvenance {
    pub source_format_version: String,
    pub source_architecture: String,
    pub source_metadata: Option<Value>,
    pub selected_model_metadata: Option<Value>,
}

impl ConvertedNam {
    /// Encodes immutable source provenance next to the converted model.
    ///
    /// NAM calibration fields are retained verbatim in the source metadata.
    /// They are deliberately not folded into the model weights or INPUT GAIN:
    /// the Player has no physical dBu calibration contract yet.
    pub fn provenance_json(&self) -> Result<Vec<u8>, serde_json::Error> {
        let selection = match self.selection {
            NamImportSelection::DirectWaveNet => serde_json::json!({
                "kind": "direct_wavenet",
                "runtime_variant": "A2 C3 Nano"
            }),
            NamImportSelection::SlimmableContainer {
                selected_submodel_index,
                submodel_count,
            } => serde_json::json!({
                "kind": "slimmable_container_submodel",
                "selected_submodel_index": selected_submodel_index,
                "submodel_count": submodel_count,
                "runtime_variant": "A2 C3 Nano"
            }),
        };
        serde_json::to_vec_pretty(&serde_json::json!({
            "schema_version": 1,
            "source_format": "Neural Amp Modeler .nam",
            "source_sha256": self.source_sha256.to_string(),
            "source_format_version": self.provenance.source_format_version,
            "source_architecture": self.provenance.source_architecture,
            "selection": selection,
            "source_metadata": self.provenance.source_metadata,
            "selected_model_metadata": self.provenance.selected_model_metadata,
            "calibration_policy": "metadata retained; dBu calibration is not applied automatically"
        }))
    }

    #[must_use]
    pub fn has_calibration_metadata(&self) -> bool {
        [
            &self.provenance.selected_model_metadata,
            &self.provenance.source_metadata,
        ]
        .into_iter()
        .flatten()
        .filter_map(Value::as_object)
        .any(|metadata| {
            metadata.contains_key("input_level_dbu") || metadata.contains_key("output_level_dbu")
        })
    }
}

#[derive(Debug)]
pub enum NamImportError {
    SourceTooLarge(usize),
    InvalidJson(serde_json::Error),
    InvalidDocument(String),
    UnsupportedFormatVersion(String),
    UnsupportedArchitecture(String),
    UnsupportedSampleRate { found: f64, required: u32 },
    UnsupportedA2Configuration(String),
    NoCompatibleContainerSubmodel,
    AmbiguousContainerSubmodels(usize),
    InvalidWeights(String),
    A2(A2Error),
    Model(ModelError),
}

impl fmt::Display for NamImportError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SourceTooLarge(bytes) => write!(
                formatter,
                "NAM source is {bytes} bytes; maximum is {MAX_NAM_SOURCE_BYTES}"
            ),
            Self::InvalidJson(error) => write!(formatter, "cannot parse NAM JSON: {error}"),
            Self::InvalidDocument(message) => write!(formatter, "invalid NAM document: {message}"),
            Self::UnsupportedFormatVersion(version) => write!(
                formatter,
                "NAM format version {version:?} is not supported; expected 0.5.x through 0.7.x"
            ),
            Self::UnsupportedArchitecture(architecture) => write!(
                formatter,
                "NAM architecture {architecture:?} is not supported; this Player currently accepts only exact WaveNet A2/C3 models"
            ),
            Self::UnsupportedSampleRate { found, required } => write!(
                formatter,
                "NAM sample rate is {found} Hz; this Player requires {required} Hz"
            ),
            Self::UnsupportedA2Configuration(message) => {
                write!(
                    formatter,
                    "NAM WaveNet is not the exact A2/C3 shape: {message}"
                )
            }
            Self::NoCompatibleContainerSubmodel => formatter.write_str(
                "NAM SlimmableContainer contains no exact 48 kHz WaveNet A2/C3 submodel",
            ),
            Self::AmbiguousContainerSubmodels(count) => write!(
                formatter,
                "NAM SlimmableContainer contains {count} compatible A2/C3 submodels; automatic selection would be ambiguous"
            ),
            Self::InvalidWeights(message) => {
                write!(formatter, "invalid NAM A2/C3 weight stream: {message}")
            }
            Self::A2(error) => write!(formatter, "{error}"),
            Self::Model(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for NamImportError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidJson(error) => Some(error),
            Self::A2(error) => Some(error),
            Self::Model(error) => Some(error),
            _ => None,
        }
    }
}

impl From<A2Error> for NamImportError {
    fn from(error: A2Error) -> Self {
        Self::A2(error)
    }
}

impl From<ModelError> for NamImportError {
    fn from(error: ModelError) -> Self {
        Self::Model(error)
    }
}

/// Converts one complete `.nam` byte stream without touching the source.
///
/// `source_filename` is used only as a display-name fallback when the NAM
/// metadata has no non-empty `name`; that fallback display name then becomes
/// part of model identity.  The source directory and all other filesystem
/// location details are ignored.
pub fn convert_nam(
    source_bytes: &[u8],
    source_filename: &str,
) -> Result<ConvertedNam, NamImportError> {
    if source_bytes.len() > MAX_NAM_SOURCE_BYTES {
        return Err(NamImportError::SourceTooLarge(source_bytes.len()));
    }
    let root: Value = serde_json::from_slice(source_bytes).map_err(NamImportError::InvalidJson)?;
    let root = as_object(&root, "root")?;
    validate_nam_version(root)?;
    let source_format_version = required_string(root, "version", "root")?.to_owned();
    let architecture = required_string(root, "architecture", "root")?;
    let source_sha256 = sha256(source_bytes);
    let top_level_name = metadata_name(root);
    let source_metadata = root.get("metadata").cloned();

    let (model_object, selection) = match architecture {
        "WaveNet" => {
            validate_sample_rate(root)?;
            validate_exact_a2_c3(root)?;
            (root, NamImportSelection::DirectWaveNet)
        }
        "SlimmableContainer" => {
            validate_sample_rate(root)?;
            select_container_submodel(root)?
        }
        other => return Err(NamImportError::UnsupportedArchitecture(other.to_owned())),
    };

    let weights = parse_weights(model_object)?;
    let a2_weights = A2Weights::from_official_weight_slice(&weights)?;
    let a2_model = A2Model::from_weights(a2_weights);
    let payload = encode_a2_payload(&a2_model)?;
    let mut display_name = metadata_name(model_object)
        .or(top_level_name)
        .unwrap_or_else(|| fallback_display_name(source_filename, source_sha256));
    if let Some(suffix) = selection.display_name_suffix() {
        display_name = append_display_name_suffix(&display_name, suffix);
    }
    let mut identity_material = Vec::with_capacity(source_bytes.len() + 1 + display_name.len());
    identity_material.extend_from_slice(source_bytes);
    identity_material.push(0);
    identity_material.extend_from_slice(display_name.as_bytes());
    let model_identity = sha256(&identity_material);
    let metadata = ModelMetadata {
        model_id: format!("nam-{model_identity}"),
        display_name,
        architecture_id: A2_ARCHITECTURE_ID.to_owned(),
        architecture_version: A2_ARCHITECTURE_VERSION,
        sample_rate_hz: A2_SAMPLE_RATE_HZ,
        causal: true,
        lookahead_samples: 0,
        runtime_latency_samples: 0,
        estimated_macs_per_sample: u64::from(A2_MACS_PER_SAMPLE),
    };
    let model = MotModel::new(metadata, payload)?;
    Ok(ConvertedNam {
        model,
        source_sha256,
        selection,
        provenance: NamImportProvenance {
            source_format_version,
            source_architecture: architecture.to_owned(),
            source_metadata,
            selected_model_metadata: model_object.get("metadata").cloned(),
        },
    })
}

fn select_container_submodel(
    root: &Map<String, Value>,
) -> Result<(&Map<String, Value>, NamImportSelection), NamImportError> {
    let config = required_object(root, "config", "SlimmableContainer")?;
    let submodels = required_array(config, "submodels", "SlimmableContainer config")?;
    if submodels.is_empty() {
        return Err(NamImportError::InvalidDocument(
            "SlimmableContainer config.submodels cannot be empty".to_owned(),
        ));
    }

    let mut compatible = Vec::new();
    for (index, entry) in submodels.iter().enumerate() {
        let Ok(entry) = entry.as_object().ok_or(()) else {
            continue;
        };
        let Some(model) = entry.get("model").and_then(Value::as_object) else {
            continue;
        };
        if model.get("architecture").and_then(Value::as_str) != Some("WaveNet") {
            continue;
        }
        if validate_nam_version(model).is_err() {
            continue;
        }
        if parse_sample_rate(model).ok() != Some(A2_SAMPLE_RATE_HZ) {
            continue;
        }
        if validate_exact_a2_c3(model).is_ok() {
            // Once the mathematical shape matches, malformed weights are an
            // invalid source rather than a reason to choose another submodel.
            let _ = parse_weights(model)?;
            compatible.push((index, model));
        }
    }

    match compatible.len() {
        0 => Err(NamImportError::NoCompatibleContainerSubmodel),
        1 => {
            let (selected_submodel_index, model) = compatible[0];
            Ok((
                model,
                NamImportSelection::SlimmableContainer {
                    selected_submodel_index,
                    submodel_count: submodels.len(),
                },
            ))
        }
        count => Err(NamImportError::AmbiguousContainerSubmodels(count)),
    }
}

fn validate_exact_a2_c3(model: &Map<String, Value>) -> Result<(), NamImportError> {
    if model.get("architecture").and_then(Value::as_str) != Some("WaveNet") {
        return Err(NamImportError::UnsupportedA2Configuration(
            "architecture is not WaveNet".to_owned(),
        ));
    }
    validate_sample_rate(model)?;
    let config = required_object(model, "config", "WaveNet")?;
    let layers = required_array(config, "layers", "WaveNet config")?;
    if layers.len() != 1 {
        return config_mismatch("config.layers must contain exactly one layer array");
    }
    if config.get("head").is_some_and(|head| !head.is_null()) {
        return config_mismatch("post-stack config.head must be null or absent");
    }
    if config
        .get("condition_dsp")
        .is_some_and(|condition| !condition.is_null())
    {
        return config_mismatch("config.condition_dsp must be null or absent");
    }
    let _ = required_f32(config, "head_scale", "WaveNet config")?;
    if optional_integer(config, "in_channels", 1, "WaveNet config")? != 1 {
        return config_mismatch("config.in_channels must be 1");
    }

    let layer = layers[0].as_object().ok_or_else(|| {
        NamImportError::UnsupportedA2Configuration("config.layers[0] must be an object".to_owned())
    })?;
    require_integer_value(layer, "input_size", 1, "layer")?;
    require_integer_value(layer, "condition_size", 1, "layer")?;
    require_integer_value(layer, "channels", A2_CHANNELS as u64, "layer")?;
    require_integer_value(layer, "bottleneck", A2_CHANNELS as u64, "layer")?;
    require_integer_array(
        layer,
        "kernel_sizes",
        &A2_KERNEL_SIZES.map(|value| value as u64),
        "layer",
    )?;
    require_integer_array(
        layer,
        "dilations",
        &A2_DILATIONS.map(|value| value as u64),
        "layer",
    )?;
    validate_activations(layer)?;
    validate_gating(layer)?;
    validate_secondary_activations(layer)?;
    validate_layer_head(layer)?;
    validate_one_by_one_convolutions(layer)?;
    validate_film(layer)?;
    if optional_integer(layer, "groups_input", 1, "layer")? != 1 {
        return config_mismatch("layer.groups_input must be 1");
    }
    if optional_integer(layer, "groups_input_mixin", 1, "layer")? != 1 {
        return config_mismatch("layer.groups_input_mixin must be 1");
    }
    if layer.get("slimmable").is_some_and(|value| !value.is_null()) {
        return config_mismatch("layer.slimmable must be null or absent");
    }
    Ok(())
}

fn validate_activations(layer: &Map<String, Value>) -> Result<(), NamImportError> {
    let activations = required_array(layer, "activation", "layer")?;
    if activations.len() != A2_LAYER_COUNT {
        return config_mismatch(format!(
            "layer.activation must contain {A2_LAYER_COUNT} entries"
        ));
    }
    for (index, activation) in activations.iter().enumerate() {
        let activation = activation.as_object().ok_or_else(|| {
            NamImportError::UnsupportedA2Configuration(format!(
                "layer.activation[{index}] must be an object"
            ))
        })?;
        if activation.get("type").and_then(Value::as_str) != Some("LeakyReLU") {
            return config_mismatch(format!("layer.activation[{index}].type must be LeakyReLU"));
        }
        let slope = required_f32(activation, "negative_slope", "activation")?;
        if (slope - A2_LEAKY_RELU_SLOPE).abs() > 1.0e-7 {
            return config_mismatch(format!(
                "layer.activation[{index}].negative_slope must be {A2_LEAKY_RELU_SLOPE}"
            ));
        }
    }
    Ok(())
}

fn validate_gating(layer: &Map<String, Value>) -> Result<(), NamImportError> {
    if let Some(gating) = layer.get("gating_mode").filter(|value| !value.is_null()) {
        let entries = gating.as_array().ok_or_else(|| {
            NamImportError::UnsupportedA2Configuration(
                "layer.gating_mode must be an array or null".to_owned(),
            )
        })?;
        if entries.len() != A2_LAYER_COUNT
            || entries.iter().any(|entry| entry.as_str() != Some("none"))
        {
            return config_mismatch(format!(
                "layer.gating_mode must contain {A2_LAYER_COUNT} \"none\" entries"
            ));
        }
    }
    if layer.get("gated").and_then(Value::as_bool) == Some(true) {
        return config_mismatch("legacy layer.gated must not be true");
    }
    Ok(())
}

fn validate_secondary_activations(layer: &Map<String, Value>) -> Result<(), NamImportError> {
    if let Some(secondary) = layer
        .get("secondary_activation")
        .filter(|value| !value.is_null())
    {
        let entries = secondary.as_array().ok_or_else(|| {
            NamImportError::UnsupportedA2Configuration(
                "layer.secondary_activation must be an array or null".to_owned(),
            )
        })?;
        if entries.len() != A2_LAYER_COUNT || entries.iter().any(|entry| !entry.is_null()) {
            return config_mismatch(format!(
                "layer.secondary_activation must contain {A2_LAYER_COUNT} null entries"
            ));
        }
    }
    Ok(())
}

fn validate_layer_head(layer: &Map<String, Value>) -> Result<(), NamImportError> {
    let head = required_object(layer, "head", "layer")?;
    require_integer_value(head, "out_channels", 1, "layer.head")?;
    require_integer_value(head, "kernel_size", 16, "layer.head")?;
    if optional_integer(head, "head_dilation", 1, "layer.head")? != 1 {
        return config_mismatch("layer.head.head_dilation must be 1");
    }
    if head.get("bias").and_then(Value::as_bool) != Some(true) {
        return config_mismatch("layer.head.bias must be true");
    }
    Ok(())
}

fn validate_one_by_one_convolutions(layer: &Map<String, Value>) -> Result<(), NamImportError> {
    if let Some(head) = layer.get("head1x1") {
        let head = head.as_object().ok_or_else(|| {
            NamImportError::UnsupportedA2Configuration("layer.head1x1 must be an object".to_owned())
        })?;
        if head.get("active").and_then(Value::as_bool).unwrap_or(false) {
            return config_mismatch("layer.head1x1 must be inactive");
        }
    }
    let residual = required_object(layer, "layer1x1", "layer")?;
    if residual.get("active").and_then(Value::as_bool) != Some(true) {
        return config_mismatch("layer.layer1x1.active must be true");
    }
    if optional_integer(residual, "groups", 1, "layer.layer1x1")? != 1 {
        return config_mismatch("layer.layer1x1.groups must be 1");
    }
    Ok(())
}

fn validate_film(layer: &Map<String, Value>) -> Result<(), NamImportError> {
    const FILM_FIELDS: [&str; 8] = [
        "conv_pre_film",
        "conv_post_film",
        "input_mixin_pre_film",
        "input_mixin_post_film",
        "activation_pre_film",
        "activation_post_film",
        "layer1x1_post_film",
        "head1x1_post_film",
    ];
    for field in FILM_FIELDS {
        let Some(value) = layer.get(field) else {
            continue;
        };
        let inactive = match value {
            Value::Null => true,
            Value::Bool(active) => !active,
            Value::Object(config) => !config
                .get("active")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            _ => false,
        };
        if !inactive {
            return config_mismatch(format!("layer.{field} must be inactive"));
        }
    }
    Ok(())
}

fn parse_weights(model: &Map<String, Value>) -> Result<Vec<f32>, NamImportError> {
    let weights = required_array(model, "weights", "WaveNet")?;
    let mut coefficients = Vec::with_capacity(weights.len());
    for (index, value) in weights.iter().enumerate() {
        let number = value.as_f64().ok_or_else(|| {
            NamImportError::InvalidWeights(format!("weights[{index}] is not a number"))
        })?;
        let coefficient = number as f32;
        if !number.is_finite() || !coefficient.is_finite() {
            return Err(NamImportError::InvalidWeights(format!(
                "weights[{index}] is not representable as a finite f32"
            )));
        }
        coefficients.push(coefficient);
    }
    if coefficients.len() != crate::a2::A2_WEIGHT_COUNT {
        return Err(NamImportError::InvalidWeights(format!(
            "expected {} coefficients, found {}",
            crate::a2::A2_WEIGHT_COUNT,
            coefficients.len()
        )));
    }
    Ok(coefficients)
}

fn validate_sample_rate(model: &Map<String, Value>) -> Result<(), NamImportError> {
    let sample_rate = parse_sample_rate(model)?;
    if sample_rate != A2_SAMPLE_RATE_HZ {
        return Err(NamImportError::UnsupportedSampleRate {
            found: model
                .get("sample_rate")
                .and_then(Value::as_f64)
                .unwrap_or(f64::from(sample_rate)),
            required: A2_SAMPLE_RATE_HZ,
        });
    }
    Ok(())
}

fn parse_sample_rate(model: &Map<String, Value>) -> Result<u32, NamImportError> {
    let Some(value) = model.get("sample_rate") else {
        // The NAM file-format contract defines 48 kHz for legacy files that
        // predate the explicit field.
        return Ok(A2_SAMPLE_RATE_HZ);
    };
    let value = value.as_f64().ok_or_else(|| {
        NamImportError::InvalidDocument("sample_rate must be a number".to_owned())
    })?;
    if !value.is_finite() || value.fract() != 0.0 || !(0.0..=f64::from(u32::MAX)).contains(&value) {
        return Err(NamImportError::InvalidDocument(
            "sample_rate must be a positive integer".to_owned(),
        ));
    }
    Ok(value as u32)
}

fn validate_nam_version(model: &Map<String, Value>) -> Result<(), NamImportError> {
    let version = required_non_empty_string(model, "version", "NAM model")?;
    let core = version.split(['-', '+']).next().unwrap_or(version);
    let mut components = core.split('.');
    let major = components
        .next()
        .and_then(|value| value.parse::<u32>().ok());
    let minor = components
        .next()
        .and_then(|value| value.parse::<u32>().ok());
    let patch = components
        .next()
        .and_then(|value| value.parse::<u32>().ok());
    if major != Some(0)
        || !matches!(minor, Some(5..=7))
        || patch.is_none()
        || components.next().is_some()
    {
        return Err(NamImportError::UnsupportedFormatVersion(version.to_owned()));
    }
    Ok(())
}

fn metadata_name(model: &Map<String, Value>) -> Option<String> {
    model
        .get("metadata")
        .and_then(Value::as_object)
        .and_then(|metadata| metadata.get("name"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(truncate_display_name)
}

fn fallback_display_name(source_filename: &str, digest: Sha256Digest) -> String {
    let stem = Path::new(source_filename)
        .file_stem()
        .and_then(std::ffi::OsStr::to_str)
        .map(str::trim)
        .filter(|name| !name.is_empty());
    stem.map(truncate_display_name)
        .unwrap_or_else(|| format!("Imported NAM {}", &digest.to_string()[..12]))
}

fn truncate_display_name(value: &str) -> String {
    const MAX_DISPLAY_NAME_BYTES: usize = 512;
    let normalized = value
        .chars()
        .map(|character| {
            if character.is_control() {
                ' '
            } else {
                character
            }
        })
        .collect::<String>();
    let normalized = normalized.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.len() <= MAX_DISPLAY_NAME_BYTES {
        return normalized;
    }
    let mut end = MAX_DISPLAY_NAME_BYTES;
    while !normalized.is_char_boundary(end) {
        end -= 1;
    }
    normalized[..end].trim_end().to_owned()
}

fn append_display_name_suffix(value: &str, suffix: &str) -> String {
    const MAX_DISPLAY_NAME_BYTES: usize = 512;
    let maximum_base_bytes = MAX_DISPLAY_NAME_BYTES.saturating_sub(suffix.len());
    let mut end = value.len().min(maximum_base_bytes);
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    let mut result = value[..end].trim_end().to_owned();
    result.push_str(suffix);
    result
}

fn config_mismatch<T>(message: impl Into<String>) -> Result<T, NamImportError> {
    Err(NamImportError::UnsupportedA2Configuration(message.into()))
}

fn as_object<'a>(value: &'a Value, field: &str) -> Result<&'a Map<String, Value>, NamImportError> {
    value
        .as_object()
        .ok_or_else(|| NamImportError::InvalidDocument(format!("{field} must be an object")))
}

fn required_object<'a>(
    object: &'a Map<String, Value>,
    key: &str,
    context: &str,
) -> Result<&'a Map<String, Value>, NamImportError> {
    object.get(key).and_then(Value::as_object).ok_or_else(|| {
        NamImportError::InvalidDocument(format!("{context}.{key} must be an object"))
    })
}

fn required_array<'a>(
    object: &'a Map<String, Value>,
    key: &str,
    context: &str,
) -> Result<&'a Vec<Value>, NamImportError> {
    object
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| NamImportError::InvalidDocument(format!("{context}.{key} must be an array")))
}

fn required_string<'a>(
    object: &'a Map<String, Value>,
    key: &str,
    context: &str,
) -> Result<&'a str, NamImportError> {
    object
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| NamImportError::InvalidDocument(format!("{context}.{key} must be a string")))
}

fn required_non_empty_string<'a>(
    object: &'a Map<String, Value>,
    key: &str,
    context: &str,
) -> Result<&'a str, NamImportError> {
    let value = required_string(object, key, context)?;
    if value.trim().is_empty() {
        return Err(NamImportError::InvalidDocument(format!(
            "{context}.{key} cannot be empty"
        )));
    }
    Ok(value)
}

fn required_f32(
    object: &Map<String, Value>,
    key: &str,
    context: &str,
) -> Result<f32, NamImportError> {
    let value = object.get(key).and_then(Value::as_f64).ok_or_else(|| {
        NamImportError::UnsupportedA2Configuration(format!("{context}.{key} must be a number"))
    })?;
    let value_f32 = value as f32;
    if !value.is_finite() || !value_f32.is_finite() {
        return config_mismatch(format!("{context}.{key} must be a finite f32"));
    }
    Ok(value_f32)
}

fn integer_value(value: &Value) -> Option<u64> {
    if let Some(value) = value.as_u64() {
        return Some(value);
    }
    let value = value.as_f64()?;
    (value.is_finite() && value.fract() == 0.0 && (0.0..=u64::MAX as f64).contains(&value))
        .then_some(value as u64)
}

fn optional_integer(
    object: &Map<String, Value>,
    key: &str,
    default: u64,
    context: &str,
) -> Result<u64, NamImportError> {
    match object.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(value) => integer_value(value).ok_or_else(|| {
            NamImportError::UnsupportedA2Configuration(format!(
                "{context}.{key} must be an unsigned integer"
            ))
        }),
    }
}

fn require_integer_value(
    object: &Map<String, Value>,
    key: &str,
    expected: u64,
    context: &str,
) -> Result<(), NamImportError> {
    if optional_integer(object, key, u64::MAX, context)? != expected {
        return config_mismatch(format!("{context}.{key} must be {expected}"));
    }
    Ok(())
}

fn require_integer_array(
    object: &Map<String, Value>,
    key: &str,
    expected: &[u64],
    context: &str,
) -> Result<(), NamImportError> {
    let values = required_array(object, key, context)?;
    if values.len() != expected.len() {
        return config_mismatch(format!(
            "{context}.{key} must contain {} entries",
            expected.len()
        ));
    }
    for (index, (value, expected)) in values.iter().zip(expected).enumerate() {
        if integer_value(value) != Some(*expected) {
            return config_mismatch(format!("{context}.{key}[{index}] must be {expected}"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    use crate::a2::{A2_WEIGHT_COUNT, decode_a2_payload};

    fn exact_config(channels: usize) -> Value {
        json!({
            "layers": [{
                "input_size": 1,
                "condition_size": 1,
                "head": {
                    "out_channels": 1,
                    "kernel_size": 16,
                    "bias": true
                },
                "channels": channels,
                "kernel_sizes": A2_KERNEL_SIZES,
                "dilations": A2_DILATIONS,
                "activation": (0..A2_LAYER_COUNT)
                    .map(|_| json!({
                        "type": "LeakyReLU",
                        "negative_slope": A2_LEAKY_RELU_SLOPE
                    }))
                    .collect::<Vec<_>>(),
                "bottleneck": channels,
                "head1x1": {"active": false, "out_channels": 1, "groups": 1},
                "layer1x1": {"active": true, "groups": 1},
                "groups_input": 1,
                "groups_input_mixin": 1,
                "gating_mode": vec!["none"; A2_LAYER_COUNT],
                "secondary_activation": vec![Value::Null; A2_LAYER_COUNT],
                "slimmable": null
            }],
            "head": null,
            "head_scale": 0.01
        })
    }

    fn direct_document(name: Option<&str>) -> Value {
        let mut model = A2Model::zeros();
        model.weights.rechannel = [0.25, -0.5, 0.75];
        model.weights.layers[0].conv[0] = -0.125;
        model.weights.head_bias = 0.03125;
        json!({
            "version": "0.7.0",
            "metadata": {"name": name},
            "architecture": "WaveNet",
            "config": exact_config(A2_CHANNELS),
            "weights": model.weights.to_official_weight_vec(),
            "sample_rate": 48000.0
        })
    }

    fn encode(value: &Value) -> Vec<u8> {
        serde_json::to_vec(value).unwrap()
    }

    #[test]
    fn direct_a2_c3_conversion_preserves_the_official_weight_stream() {
        let mut document = direct_document(Some("Imported A2"));
        document["metadata"]["input_level_dbu"] = json!(12.4);
        document["metadata"]["output_level_dbu"] = json!(18.0);
        let source = encode(&document);
        let converted = convert_nam(&source, "ignored.nam").unwrap();

        assert_eq!(converted.selection, NamImportSelection::DirectWaveNet);
        assert_eq!(converted.source_sha256, sha256(&source));
        assert!(converted.has_calibration_metadata());
        assert_eq!(converted.model.metadata().display_name, "Imported A2");
        let mut identity_material = source.clone();
        identity_material.push(0);
        identity_material.extend_from_slice(b"Imported A2");
        assert_eq!(
            converted.model.metadata().model_id,
            format!("nam-{}", sha256(&identity_material))
        );
        assert_eq!(
            converted.model.metadata().architecture_id,
            A2_ARCHITECTURE_ID
        );
        let decoded = decode_a2_payload(converted.model.payload()).unwrap();
        let expected: Vec<f32> = document["weights"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap() as f32)
            .collect();
        assert_eq!(decoded.weights.to_official_weight_vec(), expected);
        let provenance: Value =
            serde_json::from_slice(&converted.provenance_json().unwrap()).unwrap();
        assert_eq!(
            provenance["selected_model_metadata"]["input_level_dbu"],
            json!(12.4)
        );
    }

    #[test]
    fn filename_fallback_is_part_of_model_identity() {
        let document = direct_document(None);
        let source = encode(&document);
        let first = convert_nam(&source, "First Name.nam").unwrap();
        let second = convert_nam(&source, "Second Name.nam").unwrap();

        assert_eq!(first.source_sha256, second.source_sha256);
        assert_eq!(first.model.metadata().display_name, "First Name");
        assert_eq!(second.model.metadata().display_name, "Second Name");
        assert_ne!(
            first.model.metadata().model_id,
            second.model.metadata().model_id
        );
        assert_ne!(first.model.content_sha256(), second.model.content_sha256());
    }

    #[test]
    fn current_slimmable_container_extracts_the_unique_c3_submodel() {
        let c3 = direct_document(None);
        let mut c8 = direct_document(None);
        c8["config"] = exact_config(8);
        c8["weights"] = Value::Array(vec![json!(0.0); 12_146]);
        let document = json!({
            "version": "0.7.0",
            "architecture": "SlimmableContainer",
            "config": {
                "submodels": [
                    {"max_value": 0.5, "model": c3},
                    {"max_value": 1.0, "model": c8}
                ]
            },
            "weights": [],
            "sample_rate": 48000
        });
        let source = encode(&document);
        let converted = convert_nam(&source, "Tone 3000 A2.nam").unwrap();

        assert_eq!(
            converted.selection,
            NamImportSelection::SlimmableContainer {
                selected_submodel_index: 0,
                submodel_count: 2,
            }
        );
        assert_eq!(
            converted.selection.notice().as_deref(),
            Some(
                "Imported the compatible C3 / Nano submodel; a NAM host may use the container's C8 model by default."
            )
        );
        assert_eq!(
            converted.model.metadata().display_name,
            "Tone 3000 A2 — C3 Nano"
        );
        let provenance: Value =
            serde_json::from_slice(&converted.provenance_json().unwrap()).unwrap();
        assert_eq!(provenance["source_sha256"], sha256(&source).to_string());
        assert_eq!(provenance["selection"]["selected_submodel_index"], json!(0));
        assert_eq!(
            provenance["selection"]["runtime_variant"],
            json!("A2 C3 Nano")
        );
    }

    #[test]
    fn container_rejects_ambiguous_or_missing_c3_submodels() {
        let c3 = direct_document(Some("C3"));
        let ambiguous = json!({
            "version": "0.7.0",
            "architecture": "SlimmableContainer",
            "config": {"submodels": [
                {"max_value": 0.5, "model": c3.clone()},
                {"max_value": 1.0, "model": c3}
            ]},
            "weights": [],
            "sample_rate": 48000
        });
        assert!(matches!(
            convert_nam(&encode(&ambiguous), "ambiguous.nam"),
            Err(NamImportError::AmbiguousContainerSubmodels(2))
        ));

        let missing = json!({
            "version": "0.7.0",
            "architecture": "SlimmableContainer",
            "config": {"submodels": [{
                "max_value": 1.0,
                "model": {
                    "version": "0.7.0",
                    "architecture": "LSTM",
                    "config": {},
                    "weights": [],
                    "sample_rate": 48000
                }
            }]},
            "weights": [],
            "sample_rate": 48000
        });
        assert!(matches!(
            convert_nam(&encode(&missing), "missing.nam"),
            Err(NamImportError::NoCompatibleContainerSubmodel)
        ));
    }

    #[test]
    fn unsupported_sample_rate_architecture_and_shape_fail_closed() {
        let mut sample_rate = direct_document(Some("Wrong rate"));
        sample_rate["sample_rate"] = json!(44_100);
        assert!(matches!(
            convert_nam(&encode(&sample_rate), "rate.nam"),
            Err(NamImportError::UnsupportedSampleRate { .. })
        ));

        let lstm = json!({
            "version": "0.5.4",
            "architecture": "LSTM",
            "config": {"input_size": 1, "hidden_size": 16, "num_layers": 1},
            "weights": [],
            "sample_rate": 48000
        });
        assert!(matches!(
            convert_nam(&encode(&lstm), "lstm.nam"),
            Err(NamImportError::UnsupportedArchitecture(_))
        ));

        let mut wrong_shape = direct_document(Some("Wrong shape"));
        wrong_shape["config"]["layers"][0]["activation"][7]["negative_slope"] = json!(0.02);
        assert!(matches!(
            convert_nam(&encode(&wrong_shape), "shape.nam"),
            Err(NamImportError::UnsupportedA2Configuration(_))
        ));

        let mut future = direct_document(Some("Future"));
        future["version"] = json!("0.8.0");
        assert!(matches!(
            convert_nam(&encode(&future), "future.nam"),
            Err(NamImportError::UnsupportedFormatVersion(_))
        ));
    }

    #[test]
    fn wrong_weight_count_is_rejected_and_exported_head_scale_wins() {
        let mut wrong_count = direct_document(Some("Wrong count"));
        wrong_count["weights"] = Value::Array(vec![json!(0.0); A2_WEIGHT_COUNT - 1]);
        assert!(matches!(
            convert_nam(&encode(&wrong_count), "count.nam"),
            Err(NamImportError::InvalidWeights(_))
        ));

        let mut different_config_scale = direct_document(Some("Config scale differs"));
        different_config_scale["config"]["head_scale"] = json!(0.02);
        let converted =
            convert_nam(&encode(&different_config_scale), "scale.nam").expect("valid NAM");
        let decoded = decode_a2_payload(converted.model.payload()).unwrap();
        assert_eq!(decoded.weights.head_scale.to_bits(), 0.01_f32.to_bits());
    }
}
