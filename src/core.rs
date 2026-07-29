//! Format-agnostic DSP and persistence shared by the MOT guitar plug-ins.
//!
//! This crate deliberately exports no VST3 factory. The three products under
//! `plugins/` each own one wrapper, one parameter schema, and one editor.

pub mod a2;
#[cfg(feature = "training")]
pub mod a2_train;
#[cfg(test)]
mod acceptance;
pub mod amp;
pub mod cabinet;
pub mod capture;
pub mod capture_asset;
pub mod model;
pub mod model_library;
pub mod runtime;
pub mod signal_chain;
pub mod split_capture;
pub mod tuner;
pub mod wav_io;
