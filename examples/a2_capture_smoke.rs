//! Read-only convergence smoke test for a preserved MOT Trainer capture.
//!
//! This intentionally rebuilds the training target from `raw-return.wav` and
//! keeps the resulting A2 model in memory. It never publishes a model or
//! writes into the capture directory.

use std::env;
use std::error::Error;
use std::ffi::OsString;
use std::fmt::Write as _;
use std::fs;
use std::io;
use std::path::PathBuf;
use std::process::ExitCode;

use mot_core::a2_train::{
    A2CancellationToken, A2TrainerConfig, A2TrainingData, A2TrainingDevice, train_a2,
};
use mot_core::capture::{
    AlignmentConfig, CaptureTarget, extract_excitation_at_latency, measure_alignment,
    resolve_training_alignment,
};
use mot_core::capture_asset::load_default_capture_program;
use mot_core::wav_io::decode_mono_wav;

const DEFAULT_PASSES: u32 = 10;
const MAX_PASSES: u32 = 400;

#[derive(Debug)]
struct Arguments {
    capture_dir: PathBuf,
    passes: u32,
    device: A2TrainingDevice,
}

enum ParseResult {
    Run(Arguments),
    Help,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("a2_capture_smoke: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let arguments = match parse_arguments(env::args_os().skip(1))? {
        ParseResult::Run(arguments) => arguments,
        ParseResult::Help => {
            print_usage();
            return Ok(());
        }
    };

    let raw_return_path = arguments.capture_dir.join("raw-return.wav");
    let raw_return_bytes = fs::read(&raw_return_path).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!("cannot read {}: {error}", raw_return_path.display()),
        )
    })?;
    let raw_return = decode_mono_wav(&raw_return_bytes)?;
    let program = load_default_capture_program().map_err(io::Error::other)?;

    if raw_return.sample_rate != 48_000 {
        return Err(format!(
            "{} must be 48000 Hz, found {} Hz",
            raw_return_path.display(),
            raw_return.sample_rate
        )
        .into());
    }

    let measured = measure_alignment(&program, &raw_return.samples, AlignmentConfig::default())?;
    let training_alignment =
        resolve_training_alignment(CaptureTarget::SoftwarePluginChain, measured)?;
    let target = extract_excitation_at_latency(
        &program,
        &raw_return.samples,
        training_alignment.applied_training_shift_samples,
    )?;
    let emitted = program.excitation();

    println!("MOT A2 CAPTURE SMOKE");
    println!("capture       {}", arguments.capture_dir.display());
    println!(
        "raw return    {} ch, {} Hz, {} samples",
        raw_return.channels,
        raw_return.sample_rate,
        raw_return.samples.len()
    );
    println!("training pair {} samples", emitted.len());
    println!(
        "sync peak     {:+.3} samples, correlation {:.6}, polarity {}",
        training_alignment.measured.fractional_latency_samples,
        training_alignment.measured.normalized_correlation,
        if training_alignment.measured.polarity_inverted {
            "inverted"
        } else {
            "normal"
        }
    );
    println!(
        "applied shift {:+.3} samples (software-chain causality policy)",
        training_alignment.applied_training_shift_samples
    );
    println!(
        "training      {} passes, requested backend {}",
        arguments.passes,
        arguments.device.label()
    );

    let cancellation = A2CancellationToken::default();
    let outcome = train_a2(
        A2TrainingData {
            input: emitted,
            target: &target,
            sample_rate_hz: raw_return.sample_rate,
        },
        A2TrainerConfig {
            max_epochs: arguments.passes,
            device: arguments.device,
            ..A2TrainerConfig::default()
        },
        &cancellation,
        |progress| {
            let average_epoch_seconds =
                progress.elapsed_seconds / f64::from(progress.completed_epochs.max(1));
            let remaining_epochs = progress
                .maximum_epochs
                .saturating_sub(progress.completed_epochs);
            let eta_seconds = average_epoch_seconds * f64::from(remaining_epochs);
            println!(
                "pass {:>3}/{:<3} | ESR {:.6} (best {:.6} @ {}) | MSE {:.6e} | \
                 LR {:.8} | {:<19} | elapsed {} | ETA {}",
                progress.completed_epochs,
                progress.maximum_epochs,
                progress.validation_esr,
                progress.best_validation_esr,
                progress.best_epoch,
                progress.epoch_training_mse,
                progress.learning_rate,
                progress.device_status.label(),
                format_duration(progress.elapsed_seconds),
                format_duration(eta_seconds),
            );
        },
    )?;

    println!(
        "complete      {} passes, best pass {}, ESR {:.6}, runtime ESR {:.6}, elapsed {}",
        outcome.completed_epochs,
        outcome.best_epoch,
        outcome.quality.validation_esr,
        outcome.quality.exported_runtime_validation_esr,
        format_duration(outcome.elapsed_seconds),
    );
    println!("read-only run: no model or capture artifact was written");
    Ok(())
}

fn parse_arguments(arguments: impl IntoIterator<Item = OsString>) -> Result<ParseResult, String> {
    let mut arguments = arguments.into_iter();
    let mut capture_dir = None;
    let mut passes = DEFAULT_PASSES;
    let mut device = A2TrainingDevice::Auto;

    while let Some(argument) = arguments.next() {
        let flag = argument.to_string_lossy();
        match flag.as_ref() {
            "-h" | "--help" => return Ok(ParseResult::Help),
            "--capture-dir" => {
                capture_dir = Some(PathBuf::from(
                    arguments
                        .next()
                        .ok_or("--capture-dir requires a directory")?,
                ));
            }
            "--passes" => {
                let value = arguments.next().ok_or("--passes requires a number")?;
                passes = value
                    .to_string_lossy()
                    .parse::<u32>()
                    .map_err(|_| "--passes must be an integer")?;
                if !(1..=MAX_PASSES).contains(&passes) {
                    return Err(format!("--passes must be in 1..={MAX_PASSES}"));
                }
            }
            "--device" => {
                let value = arguments.next().ok_or("--device requires a value")?;
                device = match value.to_string_lossy().to_ascii_lowercase().as_str() {
                    "metal" => A2TrainingDevice::Metal,
                    "auto" => A2TrainingDevice::Auto,
                    "cpu" => A2TrainingDevice::Cpu,
                    _ => return Err("--device must be metal, auto, or cpu".to_owned()),
                };
            }
            _ => return Err(format!("unknown argument: {flag}")),
        }
    }

    let capture_dir = capture_dir.ok_or("--capture-dir is required")?;
    Ok(ParseResult::Run(Arguments {
        capture_dir,
        passes,
        device,
    }))
}

fn format_duration(seconds: f64) -> String {
    let total_seconds = seconds.max(0.0).round() as u64;
    let hours = total_seconds / 3_600;
    let minutes = (total_seconds % 3_600) / 60;
    let seconds = total_seconds % 60;
    let mut formatted = String::new();
    if hours > 0 {
        let _ = write!(formatted, "{hours}:{minutes:02}:{seconds:02}");
    } else {
        let _ = write!(formatted, "{minutes}:{seconds:02}");
    }
    formatted
}

fn print_usage() {
    println!(
        "Usage: cargo run --release --features training --example a2_capture_smoke -- \\\n\
         \t--capture-dir <directory> [--passes <1..=400>] [--device metal|auto|cpu]\n\n\
         Defaults: --passes {DEFAULT_PASSES} --device auto"
    );
}
