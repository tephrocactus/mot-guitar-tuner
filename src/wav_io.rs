//! Minimal WAV I/O for local capture assets and cabinet impulses.
//!
//! Keeping this small avoids pulling a file-format parser into the audio
//! callback. Every function in this module is intended for loader/trainer
//! threads only.

use std::fmt;
use std::fs::{self, File};
#[cfg(test)]
use std::io::Read;
use std::io::{self, BufWriter, Write};
use std::path::Path;

#[derive(Clone, Debug, PartialEq)]
pub struct MonoWav {
    /// Channel count in the source file. `samples` is always a mono stream;
    /// multichannel sources are averaged during decoding.
    pub channels: u16,
    pub sample_rate: u32,
    pub samples: Vec<f32>,
}

#[derive(Debug)]
pub enum WavError {
    Io(io::Error),
    Invalid(&'static str),
    Unsupported(String),
}

impl fmt::Display for WavError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "{error}"),
            Self::Invalid(message) => write!(f, "invalid WAV: {message}"),
            Self::Unsupported(message) => write!(f, "unsupported WAV: {message}"),
        }
    }
}

impl std::error::Error for WavError {}

impl From<io::Error> for WavError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

fn le_u16(bytes: &[u8]) -> Result<u16, WavError> {
    Ok(u16::from_le_bytes(
        bytes
            .try_into()
            .map_err(|_| WavError::Invalid("truncated u16"))?,
    ))
}

fn le_u32(bytes: &[u8]) -> Result<u32, WavError> {
    Ok(u32::from_le_bytes(
        bytes
            .try_into()
            .map_err(|_| WavError::Invalid("truncated u32"))?,
    ))
}

/// Read a RIFF/WAVE file and return a mono floating-point stream.
///
/// Multichannel files are averaged. PCM 16/24/32-bit and IEEE float32 are
/// supported, which covers the NAM excitation and common cabinet IR files.
#[cfg(test)]
pub fn read_mono_wav(path: &Path) -> Result<MonoWav, WavError> {
    let mut bytes = Vec::new();
    File::open(path)?.read_to_end(&mut bytes)?;
    decode_mono_wav(&bytes)
}

/// Decode a complete RIFF/WAVE byte stream.
///
/// Import code uses this form so validation, hashing, and immutable archival
/// all refer to the exact same source snapshot even if the original file is
/// changed concurrently.
pub fn decode_mono_wav(bytes: &[u8]) -> Result<MonoWav, WavError> {
    if bytes.len() < 12 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(WavError::Invalid("missing RIFF/WAVE header"));
    }

    let mut format = None;
    let mut data = None;
    let mut cursor = 12usize;
    while cursor + 8 <= bytes.len() {
        let id = &bytes[cursor..cursor + 4];
        let length = le_u32(&bytes[cursor + 4..cursor + 8])? as usize;
        let start = cursor + 8;
        let end = start
            .checked_add(length)
            .ok_or(WavError::Invalid("chunk size overflow"))?;
        if end > bytes.len() {
            return Err(WavError::Invalid("truncated chunk"));
        }
        match id {
            b"fmt " => format = Some(&bytes[start..end]),
            b"data" => data = Some(&bytes[start..end]),
            _ => {}
        }
        cursor = end + (length & 1);
    }

    let format = format.ok_or(WavError::Invalid("missing fmt chunk"))?;
    let data = data.ok_or(WavError::Invalid("missing data chunk"))?;
    if format.len() < 16 {
        return Err(WavError::Invalid("short fmt chunk"));
    }
    let format_tag = le_u16(&format[0..2])?;
    let source_channels = le_u16(&format[2..4])?;
    let channels = usize::from(source_channels);
    let sample_rate = le_u32(&format[4..8])?;
    let block_align = usize::from(le_u16(&format[12..14])?);
    let bits = usize::from(le_u16(&format[14..16])?);
    if channels == 0 || block_align == 0 || data.len() % block_align != 0 {
        return Err(WavError::Invalid(
            "invalid channel count or block alignment",
        ));
    }
    let bytes_per_sample = bits.div_ceil(8);
    if bytes_per_sample * channels > block_align {
        return Err(WavError::Invalid("sample width exceeds block alignment"));
    }

    let decode = |sample: &[u8]| -> Result<f32, WavError> {
        match (format_tag, bits) {
            (1, 16) => Ok(f32::from(i16::from_le_bytes(
                sample
                    .try_into()
                    .map_err(|_| WavError::Invalid("short PCM16 sample"))?,
            )) / 32_768.0),
            (1, 24) => {
                if sample.len() != 3 {
                    return Err(WavError::Invalid("short PCM24 sample"));
                }
                let raw = i32::from(sample[0])
                    | (i32::from(sample[1]) << 8)
                    | (i32::from(sample[2]) << 16);
                let signed = (raw << 8) >> 8;
                Ok(signed as f32 / 8_388_608.0)
            }
            (1, 32) => Ok(i32::from_le_bytes(
                sample
                    .try_into()
                    .map_err(|_| WavError::Invalid("short PCM32 sample"))?,
            ) as f32
                / 2_147_483_648.0),
            (3, 32) => Ok(f32::from_le_bytes(
                sample
                    .try_into()
                    .map_err(|_| WavError::Invalid("short float32 sample"))?,
            )),
            _ => Err(WavError::Unsupported(format!(
                "format tag {format_tag}, {bits}-bit"
            ))),
        }
    };

    let frames = data.len() / block_align;
    let mut samples = Vec::with_capacity(frames);
    for frame in data.chunks_exact(block_align) {
        let mut mono = 0.0f32;
        for channel in 0..channels {
            let start = channel * bytes_per_sample;
            mono += decode(&frame[start..start + bytes_per_sample])?;
        }
        samples.push(mono / channels as f32);
    }
    Ok(MonoWav {
        channels: source_channels,
        sample_rate,
        samples,
    })
}

/// Atomically write mono IEEE-float WAV data.
pub fn write_mono_f32_wav(path: &Path, sample_rate: u32, samples: &[f32]) -> Result<(), WavError> {
    let parent = path
        .parent()
        .ok_or(WavError::Invalid("output path has no parent"))?;
    fs::create_dir_all(parent)?;
    let temporary = path.with_extension("wav.tmp");
    let mut writer = BufWriter::new(File::create(&temporary)?);
    let data_bytes = samples
        .len()
        .checked_mul(4)
        .and_then(|n| u32::try_from(n).ok())
        .ok_or(WavError::Invalid("WAV data exceeds RIFF size"))?;
    let riff_size = 36u32
        .checked_add(data_bytes)
        .ok_or(WavError::Invalid("RIFF size overflow"))?;

    writer.write_all(b"RIFF")?;
    writer.write_all(&riff_size.to_le_bytes())?;
    writer.write_all(b"WAVEfmt ")?;
    writer.write_all(&16u32.to_le_bytes())?;
    writer.write_all(&3u16.to_le_bytes())?; // IEEE float
    writer.write_all(&1u16.to_le_bytes())?;
    writer.write_all(&sample_rate.to_le_bytes())?;
    writer.write_all(&(sample_rate * 4).to_le_bytes())?;
    writer.write_all(&4u16.to_le_bytes())?;
    writer.write_all(&32u16.to_le_bytes())?;
    writer.write_all(b"data")?;
    writer.write_all(&data_bytes.to_le_bytes())?;
    for sample in samples {
        writer.write_all(&sample.to_le_bytes())?;
    }
    writer.flush()?;
    drop(writer);
    fs::rename(temporary, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_path(name: &str) -> std::path::PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        std::env::temp_dir().join(format!("mot-wav-{nonce}-{name}"))
    }

    #[test]
    fn float_wav_round_trip_is_exact() {
        let path = temp_path("roundtrip.wav");
        let expected = [0.0, -1.0, 0.125, 0.75, 1.0];
        write_mono_f32_wav(&path, 48_000, &expected).expect("write");
        let decoded = read_mono_wav(&path).expect("read");
        assert_eq!(decoded.channels, 1);
        assert_eq!(decoded.sample_rate, 48_000);
        assert_eq!(decoded.samples, expected);
        let _ = fs::remove_file(path);
    }
}
