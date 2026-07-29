//! Immutable, versioned MOT neural-model container.
//!
//! A `.motmodel` is deliberately small and deterministic:
//!
//! ```text
//! magic | format version | reserved | metadata length | payload length
//! metadata | opaque model payload | SHA-256(all previous bytes)
//! ```
//!
//! The digest makes the model identity independent from its filename.  Model
//! files are written with create-new semantics so a trained model is never
//! silently changed in place.

use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

pub const MODEL_FORMAT_VERSION: u16 = 1;
pub const REQUIRED_SAMPLE_RATE_HZ: u32 = 48_000;
pub const MAX_MODEL_PAYLOAD_BYTES: usize = 256 * 1024 * 1024;
pub const DIAGONAL_RNN_ARCHITECTURE_ID: &str = "mot.diagonal-rnn-tanh";
pub const DIAGONAL_RNN_ARCHITECTURE_VERSION: u32 = 1;

const MODEL_MAGIC: &[u8; 8] = b"MOTMODL\0";
const FIXED_HEADER_BYTES: usize = 8 + 2 + 2 + 4 + 8;
const SHA256_BYTES: usize = 32;
const MAX_METADATA_BYTES: usize = 64 * 1024;
const MAX_TEXT_BYTES: usize = u16::MAX as usize;
static TEMP_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Sha256Digest([u8; SHA256_BYTES]);

impl Sha256Digest {
    #[must_use]
    pub const fn from_bytes(bytes: [u8; SHA256_BYTES]) -> Self {
        Self(bytes)
    }

    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; SHA256_BYTES] {
        &self.0
    }

    #[must_use]
    pub fn to_hex(self) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut output = String::with_capacity(SHA256_BYTES * 2);
        for byte in self.0 {
            output.push(HEX[usize::from(byte >> 4)] as char);
            output.push(HEX[usize::from(byte & 0x0f)] as char);
        }
        output
    }
}

impl fmt::Display for Sha256Digest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.to_hex())
    }
}

impl FromStr for Sha256Digest {
    type Err = ModelError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value.len() != SHA256_BYTES * 2 {
            return Err(ModelError::InvalidSha256);
        }
        let mut bytes = [0_u8; SHA256_BYTES];
        let encoded = value.as_bytes();
        for (index, byte) in bytes.iter_mut().enumerate() {
            let high = decode_hex(encoded[index * 2]).ok_or(ModelError::InvalidSha256)?;
            let low = decode_hex(encoded[index * 2 + 1]).ok_or(ModelError::InvalidSha256)?;
            *byte = (high << 4) | low;
        }
        Ok(Self(bytes))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelMetadata {
    pub model_id: String,
    pub display_name: String,
    pub architecture_id: String,
    pub architecture_version: u32,
    pub sample_rate_hz: u32,
    pub causal: bool,
    pub lookahead_samples: u32,
    pub runtime_latency_samples: u32,
    pub estimated_macs_per_sample: u64,
}

impl ModelMetadata {
    pub fn validate_zero_latency(&self) -> Result<(), ModelError> {
        validate_identifier("model_id", &self.model_id)?;
        validate_non_empty_text("display_name", &self.display_name)?;
        validate_identifier("architecture_id", &self.architecture_id)?;

        if self.architecture_version == 0 {
            return Err(ModelError::InvalidArchitectureVersion);
        }
        if !self.causal {
            return Err(ModelError::NonCausalModel);
        }
        if self.lookahead_samples != 0 {
            return Err(ModelError::NonZeroLookahead(self.lookahead_samples));
        }
        if self.runtime_latency_samples != 0 {
            return Err(ModelError::NonZeroRuntimeLatency(
                self.runtime_latency_samples,
            ));
        }
        if self.sample_rate_hz != REQUIRED_SAMPLE_RATE_HZ {
            return Err(ModelError::UnsupportedSampleRate {
                found: self.sample_rate_hz,
                required: REQUIRED_SAMPLE_RATE_HZ,
            });
        }
        if self.estimated_macs_per_sample == 0 {
            return Err(ModelError::InvalidEstimatedCost);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SupportedArchitecture {
    pub architecture_id: String,
    pub minimum_version: u32,
    pub maximum_version: u32,
}

impl SupportedArchitecture {
    #[must_use]
    pub fn exact(architecture_id: impl Into<String>, version: u32) -> Self {
        Self {
            architecture_id: architecture_id.into(),
            minimum_version: version,
            maximum_version: version,
        }
    }

    #[must_use]
    pub fn supports(&self, architecture_id: &str, version: u32) -> bool {
        self.architecture_id == architecture_id
            && (self.minimum_version..=self.maximum_version).contains(&version)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelRuntimeLimits {
    pub required_sample_rate_hz: u32,
    pub maximum_macs_per_sample: u64,
    pub supported_architectures: Vec<SupportedArchitecture>,
}

impl ModelRuntimeLimits {
    #[must_use]
    pub fn new(
        maximum_macs_per_sample: u64,
        supported_architectures: Vec<SupportedArchitecture>,
    ) -> Self {
        Self {
            required_sample_rate_hz: REQUIRED_SAMPLE_RATE_HZ,
            maximum_macs_per_sample,
            supported_architectures,
        }
    }

    pub fn validate(&self, metadata: &ModelMetadata) -> Result<(), ModelError> {
        metadata.validate_zero_latency()?;
        if metadata.sample_rate_hz != self.required_sample_rate_hz {
            return Err(ModelError::UnsupportedSampleRate {
                found: metadata.sample_rate_hz,
                required: self.required_sample_rate_hz,
            });
        }
        if metadata.estimated_macs_per_sample > self.maximum_macs_per_sample {
            return Err(ModelError::ModelTooExpensive {
                estimated_macs_per_sample: metadata.estimated_macs_per_sample,
                maximum_macs_per_sample: self.maximum_macs_per_sample,
            });
        }
        if !self.supported_architectures.iter().any(|support| {
            support.supports(&metadata.architecture_id, metadata.architecture_version)
        }) {
            return Err(ModelError::UnsupportedArchitecture {
                architecture_id: metadata.architecture_id.clone(),
                architecture_version: metadata.architecture_version,
            });
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelRef {
    pub model_id: String,
    pub sha256: Sha256Digest,
    pub filename_hint: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MotModel {
    metadata: ModelMetadata,
    payload: Vec<u8>,
    content_sha256: Sha256Digest,
}

impl MotModel {
    pub fn new(metadata: ModelMetadata, payload: Vec<u8>) -> Result<Self, ModelError> {
        metadata.validate_zero_latency()?;
        if payload.len() > MAX_MODEL_PAYLOAD_BYTES {
            return Err(ModelError::PayloadTooLarge(payload.len()));
        }
        let content = encode_content(&metadata, &payload)?;
        let content_sha256 = sha256(&content);
        Ok(Self {
            metadata,
            payload,
            content_sha256,
        })
    }

    #[must_use]
    pub const fn content_sha256(&self) -> Sha256Digest {
        self.content_sha256
    }

    #[must_use]
    pub const fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    #[must_use]
    pub fn payload(&self) -> &[u8] {
        &self.payload
    }

    #[must_use]
    pub fn model_ref(&self, filename_hint: impl Into<String>) -> ModelRef {
        ModelRef {
            model_id: self.metadata.model_id.clone(),
            sha256: self.content_sha256,
            filename_hint: filename_hint.into(),
        }
    }

    pub fn validate_for_runtime(&self, limits: &ModelRuntimeLimits) -> Result<(), ModelError> {
        limits.validate(&self.metadata)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, ModelError> {
        let mut content = encode_content(&self.metadata, &self.payload)?;
        let digest = sha256(&content);
        content.extend_from_slice(digest.as_bytes());
        Ok(content)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ModelError> {
        if bytes.len() < FIXED_HEADER_BYTES + SHA256_BYTES {
            return Err(ModelError::Truncated);
        }
        let (content, encoded_digest) = bytes.split_at(bytes.len() - SHA256_BYTES);
        let expected_digest = sha256(content);
        if encoded_digest != expected_digest.as_bytes() {
            return Err(ModelError::DigestMismatch);
        }

        let mut cursor = ByteCursor::new(content);
        if cursor.read_exact(MODEL_MAGIC.len())? != MODEL_MAGIC {
            return Err(ModelError::InvalidMagic);
        }
        let format_version = cursor.read_u16()?;
        if format_version != MODEL_FORMAT_VERSION {
            return Err(ModelError::UnsupportedFormatVersion(format_version));
        }
        let reserved = cursor.read_u16()?;
        if reserved != 0 {
            return Err(ModelError::NonZeroReservedField);
        }
        let metadata_len = usize::try_from(cursor.read_u32()?)
            .map_err(|_| ModelError::MetadataTooLarge(usize::MAX))?;
        if metadata_len > MAX_METADATA_BYTES {
            return Err(ModelError::MetadataTooLarge(metadata_len));
        }
        let payload_len = usize::try_from(cursor.read_u64()?)
            .map_err(|_| ModelError::PayloadTooLarge(usize::MAX))?;
        if payload_len > MAX_MODEL_PAYLOAD_BYTES {
            return Err(ModelError::PayloadTooLarge(payload_len));
        }

        let metadata_bytes = cursor.read_exact(metadata_len)?;
        let metadata = decode_metadata(metadata_bytes)?;
        metadata.validate_zero_latency()?;
        let payload = cursor.read_exact(payload_len)?.to_vec();
        if !cursor.is_finished() {
            return Err(ModelError::TrailingBytes);
        }

        Ok(Self {
            metadata,
            payload,
            content_sha256: expected_digest,
        })
    }

    pub fn read(path: impl AsRef<Path>) -> Result<Self, ModelError> {
        let bytes = fs::read(path).map_err(ModelError::Io)?;
        Self::from_bytes(&bytes)
    }

    /// Atomically creates an immutable model file and refuses to replace an
    /// existing destination.
    pub fn write_new(&self, path: impl AsRef<Path>) -> Result<(), ModelError> {
        let path = path.as_ref();
        let parent = path.parent().ok_or(ModelError::MissingParentDirectory)?;
        fs::create_dir_all(parent).map_err(ModelError::Io)?;
        let bytes = self.to_bytes()?;
        atomic_create_new(path, &bytes)
    }
}

#[derive(Debug)]
pub enum ModelError {
    Io(io::Error),
    InvalidMagic,
    UnsupportedFormatVersion(u16),
    NonZeroReservedField,
    Truncated,
    TrailingBytes,
    DigestMismatch,
    InvalidSha256,
    InvalidUtf8(&'static str),
    EmptyField(&'static str),
    InvalidIdentifier(&'static str),
    TextTooLong(&'static str),
    InvalidArchitectureVersion,
    NonCausalModel,
    NonZeroLookahead(u32),
    NonZeroRuntimeLatency(u32),
    UnsupportedSampleRate {
        found: u32,
        required: u32,
    },
    InvalidEstimatedCost,
    ModelTooExpensive {
        estimated_macs_per_sample: u64,
        maximum_macs_per_sample: u64,
    },
    UnsupportedArchitecture {
        architecture_id: String,
        architecture_version: u32,
    },
    MetadataTooLarge(usize),
    PayloadTooLarge(usize),
    MissingParentDirectory,
}

impl fmt::Display for ModelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "{error}"),
            Self::InvalidMagic => formatter.write_str("not a MOT model file"),
            Self::UnsupportedFormatVersion(version) => {
                write!(formatter, "unsupported MOT model format version {version}")
            }
            Self::NonZeroReservedField => formatter.write_str("reserved model header is non-zero"),
            Self::Truncated => formatter.write_str("truncated MOT model"),
            Self::TrailingBytes => formatter.write_str("unexpected bytes in MOT model"),
            Self::DigestMismatch => formatter.write_str("MOT model SHA-256 mismatch"),
            Self::InvalidSha256 => formatter.write_str("invalid SHA-256 digest"),
            Self::InvalidUtf8(field) => write!(formatter, "{field} is not valid UTF-8"),
            Self::EmptyField(field) => write!(formatter, "{field} cannot be empty"),
            Self::InvalidIdentifier(field) => write!(formatter, "{field} is not a safe identifier"),
            Self::TextTooLong(field) => write!(formatter, "{field} is too long"),
            Self::InvalidArchitectureVersion => {
                formatter.write_str("architecture version must be greater than zero")
            }
            Self::NonCausalModel => formatter.write_str("model must be strictly causal"),
            Self::NonZeroLookahead(samples) => {
                write!(formatter, "model declares {samples} lookahead samples")
            }
            Self::NonZeroRuntimeLatency(samples) => {
                write!(
                    formatter,
                    "model declares {samples} runtime latency samples"
                )
            }
            Self::UnsupportedSampleRate { found, required } => {
                write!(
                    formatter,
                    "model sample rate is {found} Hz; {required} Hz is required"
                )
            }
            Self::InvalidEstimatedCost => {
                formatter.write_str("estimated MACs per sample must be greater than zero")
            }
            Self::ModelTooExpensive {
                estimated_macs_per_sample,
                maximum_macs_per_sample,
            } => write!(
                formatter,
                "model costs {estimated_macs_per_sample} MACs/sample; runtime limit is {maximum_macs_per_sample}"
            ),
            Self::UnsupportedArchitecture {
                architecture_id,
                architecture_version,
            } => write!(
                formatter,
                "unsupported architecture {architecture_id} version {architecture_version}"
            ),
            Self::MetadataTooLarge(bytes) => {
                write!(formatter, "metadata is too large: {bytes} bytes")
            }
            Self::PayloadTooLarge(bytes) => {
                write!(formatter, "model payload is too large: {bytes} bytes")
            }
            Self::MissingParentDirectory => {
                formatter.write_str("model path has no parent directory")
            }
        }
    }
}

impl std::error::Error for ModelError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for ModelError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

fn validate_identifier(field: &'static str, value: &str) -> Result<(), ModelError> {
    validate_non_empty_text(field, value)?;
    if value.len() > 128
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err(ModelError::InvalidIdentifier(field));
    }
    Ok(())
}

fn validate_non_empty_text(field: &'static str, value: &str) -> Result<(), ModelError> {
    if value.trim().is_empty() {
        return Err(ModelError::EmptyField(field));
    }
    if value.len() > MAX_TEXT_BYTES {
        return Err(ModelError::TextTooLong(field));
    }
    Ok(())
}

fn encode_content(metadata: &ModelMetadata, payload: &[u8]) -> Result<Vec<u8>, ModelError> {
    metadata.validate_zero_latency()?;
    if payload.len() > MAX_MODEL_PAYLOAD_BYTES {
        return Err(ModelError::PayloadTooLarge(payload.len()));
    }
    let metadata_bytes = encode_metadata(metadata)?;
    let capacity = FIXED_HEADER_BYTES
        .checked_add(metadata_bytes.len())
        .and_then(|size| size.checked_add(payload.len()))
        .ok_or(ModelError::PayloadTooLarge(payload.len()))?;
    let mut bytes = Vec::with_capacity(capacity);
    bytes.extend_from_slice(MODEL_MAGIC);
    bytes.extend_from_slice(&MODEL_FORMAT_VERSION.to_le_bytes());
    bytes.extend_from_slice(&0_u16.to_le_bytes());
    bytes.extend_from_slice(
        &u32::try_from(metadata_bytes.len())
            .map_err(|_| ModelError::MetadataTooLarge(metadata_bytes.len()))?
            .to_le_bytes(),
    );
    bytes.extend_from_slice(
        &u64::try_from(payload.len())
            .map_err(|_| ModelError::PayloadTooLarge(payload.len()))?
            .to_le_bytes(),
    );
    bytes.extend_from_slice(&metadata_bytes);
    bytes.extend_from_slice(payload);
    Ok(bytes)
}

fn encode_metadata(metadata: &ModelMetadata) -> Result<Vec<u8>, ModelError> {
    let mut bytes = Vec::new();
    write_string(&mut bytes, "model_id", &metadata.model_id)?;
    write_string(&mut bytes, "display_name", &metadata.display_name)?;
    write_string(&mut bytes, "architecture_id", &metadata.architecture_id)?;
    bytes.extend_from_slice(&metadata.architecture_version.to_le_bytes());
    bytes.extend_from_slice(&metadata.sample_rate_hz.to_le_bytes());
    bytes.push(u8::from(metadata.causal));
    bytes.extend_from_slice(&metadata.lookahead_samples.to_le_bytes());
    bytes.extend_from_slice(&metadata.runtime_latency_samples.to_le_bytes());
    bytes.extend_from_slice(&metadata.estimated_macs_per_sample.to_le_bytes());
    if bytes.len() > MAX_METADATA_BYTES {
        return Err(ModelError::MetadataTooLarge(bytes.len()));
    }
    Ok(bytes)
}

fn decode_metadata(bytes: &[u8]) -> Result<ModelMetadata, ModelError> {
    let mut cursor = ByteCursor::new(bytes);
    let metadata = ModelMetadata {
        model_id: cursor.read_string("model_id")?,
        display_name: cursor.read_string("display_name")?,
        architecture_id: cursor.read_string("architecture_id")?,
        architecture_version: cursor.read_u32()?,
        sample_rate_hz: cursor.read_u32()?,
        causal: match cursor.read_u8()? {
            0 => false,
            1 => true,
            _ => return Err(ModelError::NonCausalModel),
        },
        lookahead_samples: cursor.read_u32()?,
        runtime_latency_samples: cursor.read_u32()?,
        estimated_macs_per_sample: cursor.read_u64()?,
    };
    if !cursor.is_finished() {
        return Err(ModelError::TrailingBytes);
    }
    Ok(metadata)
}

fn write_string(bytes: &mut Vec<u8>, field: &'static str, value: &str) -> Result<(), ModelError> {
    validate_non_empty_text(field, value)?;
    let len = u16::try_from(value.len()).map_err(|_| ModelError::TextTooLong(field))?;
    bytes.extend_from_slice(&len.to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
    Ok(())
}

struct ByteCursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> ByteCursor<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn read_exact(&mut self, length: usize) -> Result<&'a [u8], ModelError> {
        let end = self
            .offset
            .checked_add(length)
            .ok_or(ModelError::Truncated)?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or(ModelError::Truncated)?;
        self.offset = end;
        Ok(value)
    }

    fn read_u8(&mut self) -> Result<u8, ModelError> {
        Ok(self.read_exact(1)?[0])
    }

    fn read_u16(&mut self) -> Result<u16, ModelError> {
        let bytes: [u8; 2] = self
            .read_exact(2)?
            .try_into()
            .map_err(|_| ModelError::Truncated)?;
        Ok(u16::from_le_bytes(bytes))
    }

    fn read_u32(&mut self) -> Result<u32, ModelError> {
        let bytes: [u8; 4] = self
            .read_exact(4)?
            .try_into()
            .map_err(|_| ModelError::Truncated)?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn read_u64(&mut self) -> Result<u64, ModelError> {
        let bytes: [u8; 8] = self
            .read_exact(8)?
            .try_into()
            .map_err(|_| ModelError::Truncated)?;
        Ok(u64::from_le_bytes(bytes))
    }

    fn read_string(&mut self, field: &'static str) -> Result<String, ModelError> {
        let length = usize::from(self.read_u16()?);
        let bytes = self.read_exact(length)?;
        let value = std::str::from_utf8(bytes).map_err(|_| ModelError::InvalidUtf8(field))?;
        validate_non_empty_text(field, value)?;
        Ok(value.to_owned())
    }

    const fn is_finished(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

fn atomic_create_new(path: &Path, bytes: &[u8]) -> Result<(), ModelError> {
    let parent = path.parent().ok_or(ModelError::MissingParentDirectory)?;
    let temporary = unique_temporary_path(parent, path.file_name());
    let result = (|| -> io::Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;

        // `hard_link` is an atomic no-clobber publication on the same volume.
        // It keeps immutable model files immutable even under concurrent saves.
        fs::hard_link(&temporary, path)?;
        fs::remove_file(&temporary)?;
        sync_directory(parent);
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result.map_err(ModelError::Io)
}

fn unique_temporary_path(parent: &Path, filename: Option<&std::ffi::OsStr>) -> PathBuf {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos());
    let sequence = TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let filename = filename
        .and_then(std::ffi::OsStr::to_str)
        .unwrap_or("model");
    parent.join(format!(
        ".{filename}.{}.{}.tmp",
        std::process::id(),
        timestamp ^ u128::from(sequence)
    ))
}

fn sync_directory(path: &Path) {
    if let Ok(directory) = File::open(path) {
        let _ = directory.sync_all();
    }
}

const fn decode_hex(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

/// Compact SHA-256 used for immutable model/IR identity.
#[must_use]
pub fn sha256(bytes: &[u8]) -> Sha256Digest {
    const INITIAL: [u32; 8] = [
        0x6a09_e667,
        0xbb67_ae85,
        0x3c6e_f372,
        0xa54f_f53a,
        0x510e_527f,
        0x9b05_688c,
        0x1f83_d9ab,
        0x5be0_cd19,
    ];
    const K: [u32; 64] = [
        0x428a_2f98,
        0x7137_4491,
        0xb5c0_fbcf,
        0xe9b5_dba5,
        0x3956_c25b,
        0x59f1_11f1,
        0x923f_82a4,
        0xab1c_5ed5,
        0xd807_aa98,
        0x1283_5b01,
        0x2431_85be,
        0x550c_7dc3,
        0x72be_5d74,
        0x80de_b1fe,
        0x9bdc_06a7,
        0xc19b_f174,
        0xe49b_69c1,
        0xefbe_4786,
        0x0fc1_9dc6,
        0x240c_a1cc,
        0x2de9_2c6f,
        0x4a74_84aa,
        0x5cb0_a9dc,
        0x76f9_88da,
        0x983e_5152,
        0xa831_c66d,
        0xb003_27c8,
        0xbf59_7fc7,
        0xc6e0_0bf3,
        0xd5a7_9147,
        0x06ca_6351,
        0x1429_2967,
        0x27b7_0a85,
        0x2e1b_2138,
        0x4d2c_6dfc,
        0x5338_0d13,
        0x650a_7354,
        0x766a_0abb,
        0x81c2_c92e,
        0x9272_2c85,
        0xa2bf_e8a1,
        0xa81a_664b,
        0xc24b_8b70,
        0xc76c_51a3,
        0xd192_e819,
        0xd699_0624,
        0xf40e_3585,
        0x106a_a070,
        0x19a4_c116,
        0x1e37_6c08,
        0x2748_774c,
        0x34b0_bcb5,
        0x391c_0cb3,
        0x4ed8_aa4a,
        0x5b9c_ca4f,
        0x682e_6ff3,
        0x748f_82ee,
        0x78a5_636f,
        0x84c8_7814,
        0x8cc7_0208,
        0x90be_fffa,
        0xa450_6ceb,
        0xbef9_a3f7,
        0xc671_78f2,
    ];

    let bit_length = (bytes.len() as u64).wrapping_mul(8);
    let padded_length = (bytes.len() + 1 + 8).div_ceil(64) * 64;
    let mut state = INITIAL;
    let mut schedule = [0_u32; 64];
    let mut block = [0_u8; 64];

    for block_index in 0..(padded_length / 64) {
        block.fill(0);
        let block_start = block_index * 64;
        for (index, target) in block.iter_mut().enumerate() {
            let source_index = block_start + index;
            if source_index < bytes.len() {
                *target = bytes[source_index];
            } else if source_index == bytes.len() {
                *target = 0x80;
            }
        }
        if block_start + 64 == padded_length {
            block[56..64].copy_from_slice(&bit_length.to_be_bytes());
        }

        for (index, word) in schedule.iter_mut().take(16).enumerate() {
            *word = u32::from_be_bytes([
                block[index * 4],
                block[index * 4 + 1],
                block[index * 4 + 2],
                block[index * 4 + 3],
            ]);
        }
        for index in 16..64 {
            let s0 = schedule[index - 15].rotate_right(7)
                ^ schedule[index - 15].rotate_right(18)
                ^ (schedule[index - 15] >> 3);
            let s1 = schedule[index - 2].rotate_right(17)
                ^ schedule[index - 2].rotate_right(19)
                ^ (schedule[index - 2] >> 10);
            schedule[index] = schedule[index - 16]
                .wrapping_add(s0)
                .wrapping_add(schedule[index - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state;
        for index in 0..64 {
            let sum1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choice = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(sum1)
                .wrapping_add(choice)
                .wrapping_add(K[index])
                .wrapping_add(schedule[index]);
            let sum0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = sum0.wrapping_add(majority);

            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        state[0] = state[0].wrapping_add(a);
        state[1] = state[1].wrapping_add(b);
        state[2] = state[2].wrapping_add(c);
        state[3] = state[3].wrapping_add(d);
        state[4] = state[4].wrapping_add(e);
        state[5] = state[5].wrapping_add(f);
        state[6] = state[6].wrapping_add(g);
        state[7] = state[7].wrapping_add(h);
    }

    let mut digest = [0_u8; SHA256_BYTES];
    for (index, word) in state.iter().enumerate() {
        digest[index * 4..index * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    Sha256Digest(digest)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metadata() -> ModelMetadata {
        ModelMetadata {
            model_id: "pasadena-red-001".to_owned(),
            display_name: "Pasadena Red".to_owned(),
            architecture_id: "mot-causal-tcn".to_owned(),
            architecture_version: 1,
            sample_rate_hz: REQUIRED_SAMPLE_RATE_HZ,
            causal: true,
            lookahead_samples: 0,
            runtime_latency_samples: 0,
            estimated_macs_per_sample: 24_000,
        }
    }

    #[test]
    fn sha256_matches_standard_vectors() {
        assert_eq!(
            sha256(b"").to_string(),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256(b"abc").to_string(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn digest_hex_round_trips_and_accepts_uppercase() {
        let digest = sha256(b"MOT");
        assert_eq!(digest.to_string().parse::<Sha256Digest>().unwrap(), digest);
        assert_eq!(
            digest
                .to_string()
                .to_uppercase()
                .parse::<Sha256Digest>()
                .unwrap(),
            digest
        );
    }

    #[test]
    fn model_round_trip_is_deterministic() {
        let model = MotModel::new(metadata(), vec![1, 2, 3, 4, 5]).unwrap();
        let first = model.to_bytes().unwrap();
        let second = model.to_bytes().unwrap();
        assert_eq!(first, second);

        let decoded = MotModel::from_bytes(&first).unwrap();
        assert_eq!(decoded, model);
        assert_eq!(decoded.content_sha256(), model.content_sha256());
    }

    #[test]
    fn one_changed_payload_byte_fails_integrity() {
        let model = MotModel::new(metadata(), vec![1, 2, 3, 4, 5]).unwrap();
        let mut bytes = model.to_bytes().unwrap();
        let payload_index = bytes.len() - SHA256_BYTES - 1;
        bytes[payload_index] ^= 0x01;
        assert!(matches!(
            MotModel::from_bytes(&bytes),
            Err(ModelError::DigestMismatch)
        ));
    }

    #[test]
    fn zero_latency_contract_is_mandatory() {
        let mut value = metadata();
        value.causal = false;
        assert!(matches!(
            value.validate_zero_latency(),
            Err(ModelError::NonCausalModel)
        ));

        let mut value = metadata();
        value.lookahead_samples = 1;
        assert!(matches!(
            value.validate_zero_latency(),
            Err(ModelError::NonZeroLookahead(1))
        ));

        let mut value = metadata();
        value.runtime_latency_samples = 32;
        assert!(matches!(
            value.validate_zero_latency(),
            Err(ModelError::NonZeroRuntimeLatency(32))
        ));

        let mut value = metadata();
        value.sample_rate_hz = 96_000;
        assert!(matches!(
            value.validate_zero_latency(),
            Err(ModelError::UnsupportedSampleRate { .. })
        ));
    }

    #[test]
    fn runtime_rejects_unsupported_or_overweight_models() {
        let limits = ModelRuntimeLimits::new(
            30_000,
            vec![SupportedArchitecture::exact("mot-causal-tcn", 1)],
        );
        assert!(limits.validate(&metadata()).is_ok());

        let mut overweight = metadata();
        overweight.estimated_macs_per_sample = 30_001;
        assert!(matches!(
            limits.validate(&overweight),
            Err(ModelError::ModelTooExpensive { .. })
        ));

        let mut unknown = metadata();
        unknown.architecture_id = "other".to_owned();
        assert!(matches!(
            limits.validate(&unknown),
            Err(ModelError::UnsupportedArchitecture { .. })
        ));
    }

    #[test]
    fn model_id_cannot_escape_the_library_directory() {
        let mut value = metadata();
        value.model_id = "../outside".to_owned();
        assert!(matches!(
            value.validate_zero_latency(),
            Err(ModelError::InvalidIdentifier("model_id"))
        ));
    }
}
