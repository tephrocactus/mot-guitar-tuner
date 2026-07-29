//! Canonical excitation asset shared by MOT Generator and MOT Trainer.

use std::fs;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::{Arc, OnceLock};

use crate::capture::CaptureProgram;
use crate::model::{Sha256Digest, sha256};
use crate::model_library::ModelLibrary;
use crate::wav_io::decode_mono_wav;

pub const CAPTURE_ASSET_RELATIVE_PATH: &str = "Capture Assets/input.wav";
pub const CAPTURE_ASSET_SHA256: &str =
    "70f8ec7f25686a1bd77f25973de8e51a6721e957e81eec121822e5e53366bc41";
pub const CAPTURE_ASSET_SAMPLE_RATE_HZ: u32 = 48_000;
pub const CAPTURE_ASSET_SAMPLES: usize = 9_120_000;
pub const SYNC_HEADER_SAMPLES: usize = 4_096;
pub const CAPTURE_PROTOCOL_VERSION: u32 = 1;

static CAPTURE_PROGRAM: OnceLock<Arc<CaptureProgram>> = OnceLock::new();

pub fn capture_asset_path() -> Result<PathBuf, String> {
    let library = ModelLibrary::for_current_user().map_err(|error| error.to_string())?;
    Ok(library
        .paths()
        .plugin_root
        .join(CAPTURE_ASSET_RELATIVE_PATH))
}

pub fn load_default_capture_program() -> Result<Arc<CaptureProgram>, String> {
    if let Some(program) = CAPTURE_PROGRAM.get() {
        return Ok(Arc::clone(program));
    }
    let program = load_capture_program(&capture_asset_path()?)?;
    let _ = CAPTURE_PROGRAM.set(Arc::clone(&program));
    Ok(CAPTURE_PROGRAM.get().map_or(program, Arc::clone))
}

pub fn load_capture_program(path: &Path) -> Result<Arc<CaptureProgram>, String> {
    let bytes = fs::read(path)
        .map_err(|error| format!("cannot read capture asset {}: {error}", path.display()))?;
    let expected = Sha256Digest::from_str(CAPTURE_ASSET_SHA256).map_err(|e| e.to_string())?;
    let actual = sha256(&bytes);
    if actual != expected {
        return Err(format!(
            "capture asset SHA-256 mismatch: expected {expected}, found {actual}"
        ));
    }

    let wav = decode_mono_wav(&bytes).map_err(|error| error.to_string())?;
    if wav.sample_rate != CAPTURE_ASSET_SAMPLE_RATE_HZ {
        return Err(format!(
            "capture asset must be {CAPTURE_ASSET_SAMPLE_RATE_HZ} Hz, found {} Hz",
            wav.sample_rate
        ));
    }
    if wav.samples.len() != CAPTURE_ASSET_SAMPLES {
        return Err(format!(
            "capture asset must contain {CAPTURE_ASSET_SAMPLES} samples, found {}",
            wav.samples.len()
        ));
    }

    CaptureProgram::new(generate_sync_header(), wav.samples)
        .map(Arc::new)
        .map_err(|error| error.to_string())
}

pub fn generate_sync_header() -> Vec<f32> {
    let mut state = 0x00c0_ffee_u32;
    let mut header = Vec::with_capacity(SYNC_HEADER_SAMPLES);
    for index in 0..SYNC_HEADER_SAMPLES {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let sign = if state & 1 == 0 { -1.0 } else { 1.0 };
        let edge = index.min(SYNC_HEADER_SAMPLES - 1 - index).min(127) as f32 / 127.0;
        header.push(sign * 0.18 * edge);
    }
    header
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sync_header_is_deterministic_and_faded() {
        let left = generate_sync_header();
        let right = generate_sync_header();
        assert_eq!(left, right);
        assert_eq!(left.len(), SYNC_HEADER_SAMPLES);
        assert_eq!(left[0], 0.0);
        assert_eq!(*left.last().expect("header sample"), 0.0);
    }
}
