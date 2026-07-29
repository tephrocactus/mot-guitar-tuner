//! Release-gate acceptance tests for the zero-latency live path.
//!
//! The regular tests are deterministic functional checks. The ignored
//! microbenchmarks are intentionally run by hand on the target M3 Pro:
//!
//! `cargo test --release --all-features acceptance::m3_pro_48k_32_sample_runtime_budget -- --ignored --nocapture`
//! `cargo test --release --all-features acceptance::m3_pro_48k_32_sample_30_minute_soak -- --ignored --nocapture`
//! `cargo test --release --all-features acceptance::native_48k_aliasing_proxy_release_gate -- --ignored --nocapture`
//!
//! The aliasing test is a deterministic native-rate regression proxy, not a
//! claim of psychoacoustic transparency. A 48 kHz render cannot distinguish
//! every folded component from intended nonlinear products without a trusted
//! oversampled reference. The gate therefore combines three conservative,
//! reproducible indicators and leaves the final listening decision to the
//! user.

#![cfg(test)]

use std::fs;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
#[cfg(any(target_os = "android", target_os = "linux", target_os = "macos"))]
use std::{ffi::c_int, ffi::c_long};

use truce::prelude::AudioConfig;

use crate::a2::{
    A2_CHANNELS, A2_HEAD_KERNEL_SIZE, A2_HEAD_SCALE, A2_KERNEL_SIZES, A2_MACS_PER_SAMPLE, A2Model,
    encode_a2_payload,
};
use crate::model::{
    A2_ARCHITECTURE_ID, A2_ARCHITECTURE_VERSION, ModelMetadata, ModelRef, MotModel,
    REQUIRED_SAMPLE_RATE_HZ, sha256,
};
use crate::model_library::{
    IrProcessingMode, IrReference, ModelLibrary, ModelLibraryPaths, TONE_SETTINGS_VERSION,
    ToneSettings,
};
use crate::runtime::{
    PreparedRuntime, RuntimeLoadRequest, RuntimeLoader, RuntimeMailbox, RuntimeMuteReason,
    RuntimeUpdate,
};
use crate::signal_chain::{GuitarSignalChain, RuntimeApplyStatus};
use crate::wav_io::write_mono_f32_wav;

const TARGET_BLOCK_SIZE: usize = 32;
const MAX_IR_SAMPLES: usize = 8_192;
const CROSSFADE_SAMPLES: usize = 480;
const STEADY_CALLBACKS: usize = 20_000;
const SWAP_SAMPLES: usize = 64;
const SOAK_SECONDS: usize = 30 * 60;
const SOAK_CALLBACKS: usize = REQUIRED_SAMPLE_RATE_HZ as usize * SOAK_SECONDS / TARGET_BLOCK_SIZE;
const CALLBACK_DEADLINE: Duration = Duration::from_micros(667);
const ALIAS_SWEEP_MAX_RESIDUAL_DB: f64 = -18.0;
const ALIAS_MULTITONE_MAX_FOLD_DB: f64 = -55.0;
const ALIAS_PALM_MAX_RESIDUAL_DB: f64 = -4.0;
const ALIAS_PALM_MAX_NYQUIST_PRESSURE_DB: f64 = -58.0;

static TEST_DIRECTORY_COUNTER: AtomicU64 = AtomicU64::new(0);

#[cfg(any(target_os = "android", target_os = "linux", target_os = "macos"))]
#[repr(C)]
struct NativeTimespec {
    seconds: c_long,
    nanoseconds: c_long,
}

#[cfg(target_os = "macos")]
const CLOCK_THREAD_CPUTIME_ID: c_int = 16;
#[cfg(any(target_os = "android", target_os = "linux"))]
const CLOCK_THREAD_CPUTIME_ID: c_int = 3;

#[cfg(any(target_os = "android", target_os = "linux", target_os = "macos"))]
unsafe extern "C" {
    fn clock_gettime(clock_id: c_int, time: *mut NativeTimespec) -> c_int;
}

/// Current thread's consumed CPU time, excluding time descheduled by the OS.
///
/// macOS and Linux/Android expose this through a monotonic POSIX thread clock.
/// Other targets return `None`; their acceptance run deliberately falls back
/// to the stricter wall-clock gate rather than guessing at scheduler time.
#[cfg(any(target_os = "android", target_os = "linux", target_os = "macos"))]
fn thread_cpu_time() -> Option<Duration> {
    let mut native = NativeTimespec {
        seconds: 0,
        nanoseconds: 0,
    };
    // SAFETY: `native` is valid writable storage for the C timespec layout,
    // and the platform-specific clock id is defined by the target's libc ABI.
    let status = unsafe { clock_gettime(CLOCK_THREAD_CPUTIME_ID, &raw mut native) };
    if status != 0 || native.seconds < 0 || !(0..1_000_000_000).contains(&native.nanoseconds) {
        return None;
    }
    Some(Duration::new(
        u64::try_from(native.seconds).ok()?,
        u32::try_from(native.nanoseconds).ok()?,
    ))
}

#[cfg(not(any(target_os = "android", target_os = "linux", target_os = "macos")))]
fn thread_cpu_time() -> Option<Duration> {
    None
}

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new() -> Self {
        let unique = format!(
            "mot-acceptance-{}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system clock")
                .as_nanos(),
            TEST_DIRECTORY_COUNTER.fetch_add(1, Ordering::Relaxed)
        );
        let path = std::env::temp_dir().join(unique);
        fs::create_dir(&path).expect("create acceptance directory");
        Self(path)
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn maximum_tracking_model() -> A2Model {
    let mut model = A2Model::zeros();
    model.weights.rechannel = [0.20, -0.20, 0.000_1];

    for (layer_index, layer) in model.weights.layers.iter_mut().enumerate() {
        let active_conv_weights = A2_KERNEL_SIZES[layer_index] * A2_CHANNELS * A2_CHANNELS;
        for (coefficient_index, coefficient) in
            layer.conv.iter_mut().take(active_conv_weights).enumerate()
        {
            let magnitude =
                0.000_001 + 0.000_001 * ((coefficient_index + layer_index * 3) % 7) as f32;
            *coefficient = if (coefficient_index + layer_index) % 2 == 0 {
                magnitude
            } else {
                -magnitude
            };
        }

        let layer_scale = 0.92 + layer_index as f32 * 0.004;
        layer.input_mixin = [
            0.20 * layer_scale,
            -0.20 * layer_scale,
            0.000_1 * layer_scale,
        ];
        for (coefficient_index, coefficient) in layer.residual.iter_mut().enumerate() {
            let input_channel = coefficient_index / A2_CHANNELS;
            let output_channel = coefficient_index % A2_CHANNELS;
            *coefficient = if input_channel == output_channel {
                0.000_03
            } else if (input_channel + output_channel + layer_index) % 2 == 0 {
                0.000_01
            } else {
                -0.000_01
            };
        }
    }

    populate_acceptance_head(&mut model);

    // The final three coefficients are the current-sample head taps. Keep
    // them strong enough to make sample-zero processing observable while all
    // preceding taps remain populated for the maximum A2 runtime exercise.
    model.validate().expect("maximum model must be valid");
    model
}

fn populate_acceptance_head(model: &mut A2Model) {
    for tap in 0..A2_HEAD_KERNEL_SIZE {
        let offset = tap * A2_CHANNELS;
        let magnitude = 0.001_5 + 0.000_25 * (tap % 5) as f32;
        model.weights.head[offset] = magnitude;
        model.weights.head[offset + 1] = -magnitude;
        model.weights.head[offset + 2] = 0.000_01 * (tap + 1) as f32;
    }
    let current_head_tap = (A2_HEAD_KERNEL_SIZE - 1) * A2_CHANNELS;
    model.weights.head[current_head_tap..current_head_tap + A2_CHANNELS]
        .copy_from_slice(&[0.12, -0.12, 0.000_16]);
    // Assign this explicitly so the payload round-trip exercises the
    // serialized head-scale field instead of relying on its default.
    model.weights.head_scale = A2_HEAD_SCALE;
}

fn linear_reference_model() -> A2Model {
    let mut model = A2Model::zeros();
    for (layer_index, layer) in model.weights.layers.iter_mut().enumerate() {
        // LeakyReLU(x) - LeakyReLU(-x) is exactly linear. Using a channel
        // pair preserves a deterministic linear baseline for the nonlinear
        // aliasing proxy while exercising the same fixed A2 runtime shape.
        let layer_scale = 0.92 + layer_index as f32 * 0.004;
        layer.input_mixin = [0.20 * layer_scale, -0.20 * layer_scale, 0.0];
    }
    populate_acceptance_head(&mut model);
    model
        .validate()
        .expect("linear A2 reference model must be valid");
    model
}

fn maximum_raw_ir() -> Vec<f32> {
    let mut ir = vec![0.0; MAX_IR_SAMPLES];
    ir[0] = 0.72;
    for (index, tap) in ir.iter_mut().enumerate().skip(1) {
        let time = index as f32;
        *tap =
            (-time / 1_450.0).exp() * ((time * 0.071).sin() + 0.31 * (time * 0.193).cos()) * 0.035;
    }
    ir
}

fn metadata_for(model: &A2Model, model_id: &str) -> ModelMetadata {
    model
        .validate()
        .expect("acceptance metadata requires a valid A2 model");
    ModelMetadata {
        model_id: model_id.to_owned(),
        display_name: format!("Acceptance {model_id}"),
        architecture_id: A2_ARCHITECTURE_ID.to_owned(),
        architecture_version: A2_ARCHITECTURE_VERSION,
        sample_rate_hz: model.sample_rate_hz,
        causal: model.causal,
        lookahead_samples: model.lookahead_samples,
        runtime_latency_samples: model.runtime_latency_samples,
        estimated_macs_per_sample: u64::from(A2_MACS_PER_SAMPLE),
    }
}

fn write_ir(path: &Path, samples: &[f32]) -> IrReference {
    write_mono_f32_wav(path, REQUIRED_SAMPLE_RATE_HZ, samples).expect("write acceptance IR");
    let bytes = fs::read(path).expect("read acceptance IR");
    IrReference {
        ir_id: "acceptance-max-ir".to_owned(),
        sha256: sha256(&bytes),
        filename_hint: path
            .file_name()
            .expect("IR filename")
            .to_string_lossy()
            .into_owned(),
        processing: IrProcessingMode::Raw,
    }
}

fn build_maximum_runtime() -> PreparedRuntime {
    build_acceptance_runtime(maximum_tracking_model(), "maximum-runtime")
}

fn build_linear_reference_runtime() -> PreparedRuntime {
    build_acceptance_runtime(linear_reference_model(), "linear-reference")
}

fn build_acceptance_runtime(model: A2Model, model_id: &str) -> PreparedRuntime {
    let directory = TestDirectory::new();
    let library = ModelLibrary::new(ModelLibraryPaths::from_plugin_root(
        directory.0.join("library"),
    ));
    library
        .ensure_directories()
        .expect("create acceptance model library");

    let container = MotModel::new(
        metadata_for(&model, model_id),
        encode_a2_payload(&model).expect("encode acceptance A2 payload"),
    )
    .expect("model container");
    let model_filename = format!("acceptance-{model_id}.motmodel");
    container
        .write_new(library.paths().models.join(&model_filename))
        .expect("write acceptance model");
    let model_reference: ModelRef = container.model_ref(&model_filename);

    let ir_path = directory.0.join("acceptance-max-ir.wav");
    let ir_reference = write_ir(&ir_path, &maximum_raw_ir());
    let tone = ToneSettings {
        schema_version: TONE_SETTINGS_VERSION,
        model_id: model_reference.model_id.clone(),
        model_sha256: model_reference.sha256,
        input_gain_db: 0.0,
        tight_percent: 0.0,
        bite_percent: 0.0,
        ir: Some(ir_reference),
    };
    let mut request = RuntimeLoadRequest::new(1, model_reference);
    request.tone = Some(tone);
    request.ir_path = Some(ir_path);
    request.host_sample_rate_hz = REQUIRED_SAMPLE_RATE_HZ;
    request.host_max_block_size = TARGET_BLOCK_SIZE;

    match RuntimeLoader::new(library).load(request).update {
        RuntimeUpdate::Ready { runtime, .. } => *runtime,
        RuntimeUpdate::Mute { reason, .. } => {
            panic!("maximum acceptance runtime unexpectedly muted: {reason:?}")
        }
    }
}

fn render_partitioned(
    base: &PreparedRuntime,
    input: &[f32],
    partition_pattern: &[usize],
) -> Vec<f32> {
    let mut runtime = base.clone();
    runtime.reset(&AudioConfig::new(
        f64::from(REQUIRED_SAMPLE_RATE_HZ),
        input.len(),
    ));
    let mut scratch = vec![0.0; input.len()];
    let mut output = vec![0.0; input.len()];
    let mut cursor = 0;
    let mut partition = 0;
    while cursor < input.len() {
        let end =
            (cursor + partition_pattern[partition % partition_pattern.len()]).min(input.len());
        runtime.process_block(
            &input[cursor..end],
            &mut scratch[cursor..end],
            &mut output[cursor..end],
        );
        cursor = end;
        partition += 1;
    }
    output
}

fn logarithmic_sine_sweep(sample_count: usize) -> Vec<f32> {
    let sample_rate = REQUIRED_SAMPLE_RATE_HZ as f64;
    let start_hz = 35.0_f64;
    let end_hz = 20_000.0_f64;
    let duration = sample_count as f64 / sample_rate;
    let sweep_rate = (end_hz / start_hz).ln() / duration;
    let phase_scale = std::f64::consts::TAU * start_hz / sweep_rate;

    (0..sample_count)
        .map(|sample| {
            let time = sample as f64 / sample_rate;
            // Starting on cosine gives the test an unambiguous sample-zero
            // onset while preserving the instantaneous-frequency law.
            (0.32 * (phase_scale * ((sweep_rate * time).exp() - 1.0)).cos()) as f32
        })
        .collect()
}

fn deterministic_multitone(sample_count: usize) -> Vec<f32> {
    const TONES_HZ: [f64; 8] = [
        73.0, 147.0, 293.0, 587.0, 1_171.0, 2_339.0, 4_677.0, 9_349.0,
    ];
    let sample_rate = REQUIRED_SAMPLE_RATE_HZ as f64;
    (0..sample_count)
        .map(|sample| {
            let time = sample as f64 / sample_rate;
            let sum: f64 = TONES_HZ
                .iter()
                .enumerate()
                .map(|(index, frequency)| {
                    let phase = index as f64 * 0.371;
                    (std::f64::consts::TAU * frequency * time + phase).cos()
                })
                .sum();
            (sum * (0.48 / TONES_HZ.len() as f64)) as f32
        })
        .collect()
}

fn synthetic_palm_mute_di(sample_count: usize) -> Vec<f32> {
    const FUNDAMENTAL_HZ: f64 = 77.78;
    const CHUG_INTERVAL: usize = 4_000;

    let sample_rate = REQUIRED_SAMPLE_RATE_HZ as f64;
    let mut noise_state = 0x6d2b_79f5_u32;
    let mut output = Vec::with_capacity(sample_count);
    for sample in 0..sample_count {
        noise_state = noise_state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        let noise = (f64::from(noise_state >> 8) / f64::from(1_u32 << 24)) * 2.0 - 1.0;

        let within_chug = sample % CHUG_INTERVAL;
        let time = within_chug as f64 / sample_rate;
        let string = (1..=9)
            .map(|harmonic| {
                let harmonic = harmonic as f64;
                let decay_seconds = 0.070 / harmonic.sqrt();
                let envelope = (-time / decay_seconds).exp();
                envelope
                    * (std::f64::consts::TAU * FUNDAMENTAL_HZ * harmonic * time + harmonic * 0.17)
                        .cos()
                    / harmonic.powf(0.78)
            })
            .sum::<f64>();
        let pick = noise * (-time / 0.0025).exp();
        output.push((0.19 * string + 0.055 * pick) as f32);
    }
    output
}

fn assert_quality_render(
    name: &str,
    runtime: &PreparedRuntime,
    input: &[f32],
    absolute_bound: f32,
) {
    assert!(
        input.first().is_some_and(|sample| sample.abs() > 1.0e-4),
        "{name} must have an explicit sample-zero onset"
    );

    let whole = render_partitioned(runtime, input, &[input.len()]);
    let arbitrary = render_partitioned(runtime, input, &[1, 7, 16, 32, 64, 257, 512]);
    assert_eq!(
        arbitrary, whole,
        "{name} changed with arbitrary host block partitioning"
    );
    assert!(
        whole.iter().all(|sample| sample.is_finite()),
        "{name} produced a non-finite sample"
    );
    let peak = whole
        .iter()
        .fold(0.0_f32, |peak, sample| peak.max(sample.abs()));
    assert!(
        peak <= absolute_bound,
        "{name} exceeded the conservative output bound: peak={peak}, bound={absolute_bound}"
    );
    assert!(
        whole[0].abs() > 1.0e-5,
        "{name} onset was moved after sample zero: first output={}",
        whole[0]
    );
    let output_energy: f64 = whole
        .iter()
        .map(|sample| f64::from(*sample) * f64::from(*sample))
        .sum();
    assert!(
        output_energy > 1.0e-8,
        "{name} unexpectedly rendered silence"
    );
}

#[test]
fn maximum_model_and_ir_have_zero_reported_latency_and_sample_zero_onset() {
    let mut runtime = build_maximum_runtime();
    runtime.reset(&AudioConfig::new(
        f64::from(REQUIRED_SAMPLE_RATE_HZ),
        TARGET_BLOCK_SIZE,
    ));
    assert_eq!(runtime.latency_samples(), 0);
    assert_eq!(runtime.tail_samples(), (MAX_IR_SAMPLES - 1) as u32);

    let mut impulse = [0.0; TARGET_BLOCK_SIZE];
    impulse[0] = 1.0;
    let mut scratch = [0.0; TARGET_BLOCK_SIZE];
    let mut output = [0.0; TARGET_BLOCK_SIZE];
    runtime.process_block(&impulse, &mut scratch, &mut output);
    assert!(
        output[0].abs() > 1.0e-4,
        "model+IR onset moved after sample zero: {output:?}"
    );
}

#[test]
fn maximum_model_and_ir_are_invariant_to_arbitrary_host_blocks() {
    let runtime = build_maximum_runtime();
    let input: Vec<f32> = (0..12_000)
        .map(|sample| {
            let time = sample as f32;
            (0.31 * (time * 0.031).sin() + 0.17 * (time * 0.117).cos()).clamp(-1.0, 1.0)
        })
        .collect();
    let whole = render_partitioned(&runtime, &input, &[input.len()]);
    let split = render_partitioned(&runtime, &input, &[1, 7, 16, 32, 64, 257, 512]);
    assert_eq!(split, whole);
}

#[test]
fn maximum_runtime_quality_stimuli_are_finite_bounded_and_block_invariant() {
    const QUALITY_SAMPLES: usize = 48_000;
    const CONSERVATIVE_OUTPUT_BOUND: f32 = 16.0;

    let runtime = build_maximum_runtime();
    let stimuli = [
        (
            "logarithmic sine sweep",
            logarithmic_sine_sweep(QUALITY_SAMPLES),
        ),
        (
            "deterministic multitone",
            deterministic_multitone(QUALITY_SAMPLES),
        ),
        (
            "synthetic palm-mute DI",
            synthetic_palm_mute_di(QUALITY_SAMPLES),
        ),
    ];
    for (name, input) in &stimuli {
        assert_quality_render(name, &runtime, input.as_slice(), CONSERVATIVE_OUTPUT_BOUND);
    }
}

#[test]
fn safe_mute_is_click_free_and_reaches_exact_silence() {
    let mailbox = RuntimeMailbox::new(2);
    let mut chain = GuitarSignalChain::default();
    chain.reset(&AudioConfig::new(
        f64::from(REQUIRED_SAMPLE_RATE_HZ),
        CROSSFADE_SAMPLES,
    ));
    mailbox.publish_latest(RuntimeUpdate::Mute {
        generation: 7,
        reason: RuntimeMuteReason::MissingModel,
    });
    assert_eq!(
        chain.poll_runtime(&mailbox),
        Some(RuntimeApplyStatus::SafeMuted { generation: 7 })
    );

    let input = [1.0; CROSSFADE_SAMPLES];
    let mut output = [0.0; CROSSFADE_SAMPLES];
    chain.process_block(&input, &mut output);
    assert!(output[0] < 1.0 && output[0] > 0.0);
    assert_eq!(output[CROSSFADE_SAMPLES - 1], 0.0);

    let mut settled = [f32::NAN; TARGET_BLOCK_SIZE];
    chain.process_block(&[1.0; TARGET_BLOCK_SIZE], &mut settled);
    assert_eq!(settled, [0.0; TARGET_BLOCK_SIZE]);
    assert_eq!(chain.latency_samples(), 0);
}

#[test]
fn prepared_runtime_swap_is_same_sample_and_reports_zero_latency() {
    let runtime = build_maximum_runtime();
    let mailbox = RuntimeMailbox::new(2);
    mailbox.publish_latest(RuntimeUpdate::Ready {
        generation: 11,
        runtime: Box::new(runtime),
    });
    let mut chain = GuitarSignalChain::default();
    chain.reset(&AudioConfig::new(
        f64::from(REQUIRED_SAMPLE_RATE_HZ),
        TARGET_BLOCK_SIZE,
    ));
    assert_eq!(
        chain.poll_runtime(&mailbox),
        Some(RuntimeApplyStatus::Ready { generation: 11 })
    );
    assert_eq!(chain.latency_samples(), 0);

    let mut impulse = [0.0; TARGET_BLOCK_SIZE];
    impulse[0] = 1.0;
    let mut output = [0.0; TARGET_BLOCK_SIZE];
    chain.process_block(&impulse, &mut output);
    assert!(
        output[0].abs() > 1.0e-4,
        "runtime swap moved the impulse onset after sample zero"
    );
}

#[cfg(feature = "rt-paranoid")]
#[test]
fn maximum_runtime_processing_and_swap_allocate_nothing() {
    let runtime = build_maximum_runtime();
    let mailbox = RuntimeMailbox::new(2);
    mailbox.publish_latest(RuntimeUpdate::Ready {
        generation: 13,
        runtime: Box::new(runtime),
    });
    let mut chain = GuitarSignalChain::default();
    chain.reset(&AudioConfig::new(
        f64::from(REQUIRED_SAMPLE_RATE_HZ),
        TARGET_BLOCK_SIZE,
    ));
    let input = [0.125; TARGET_BLOCK_SIZE];
    let mut output = [0.0; TARGET_BLOCK_SIZE];

    let (status, allocations) = truce::rt::audit(|| {
        let _section = truce::rt::RtSection::enter();
        let status = chain.poll_runtime(&mailbox);
        chain.process_block(&input, &mut output);
        status
    });
    assert_eq!(status, Some(RuntimeApplyStatus::Ready { generation: 13 }));
    assert_eq!(allocations, 0);
}

#[test]
#[ignore = "target-machine release soak; simulates 30 minutes of 48k/32 callbacks without sleeping"]
fn m3_pro_48k_32_sample_30_minute_soak() {
    assert!(
        !std::hint::black_box(cfg!(debug_assertions)),
        "the RT acceptance soak must be run with --release"
    );
    assert_eq!(
        SOAK_CALLBACKS, 2_700_000,
        "30 minutes at 48 kHz / 32 samples must contain exactly 2.7M callbacks"
    );

    let mut runtime = build_maximum_runtime();
    let config = AudioConfig::new(f64::from(REQUIRED_SAMPLE_RATE_HZ), TARGET_BLOCK_SIZE);
    runtime.reset(&config);
    let input: [f32; TARGET_BLOCK_SIZE] = std::array::from_fn(|index| {
        let phase = index as f32;
        0.28 * (phase * 0.071).sin() + 0.11 * (phase * 0.193).cos()
    });
    let mut scratch = [0.0; TARGET_BLOCK_SIZE];
    let mut output = [0.0; TARGET_BLOCK_SIZE];

    // Populate every convolution partition before starting the clock.
    for _ in 0..MAX_IR_SAMPLES.div_ceil(TARGET_BLOCK_SIZE) {
        runtime.process_block(black_box(&input), &mut scratch, black_box(&mut output));
    }

    let cpu_clock_available = thread_cpu_time().is_some();
    let wall_started = Instant::now();
    let mut wall_deadline_outliers = 0_u64;
    let mut gate_deadline_misses = 0_u64;
    let mut scheduler_preemption_candidates = 0_u64;
    let mut largest_wall_minus_cpu = Duration::ZERO;
    let mut wall_callback_times = Vec::with_capacity(SOAK_CALLBACKS);
    let mut cpu_callback_times = cpu_clock_available.then(|| Vec::with_capacity(SOAK_CALLBACKS));
    for _ in 0..SOAK_CALLBACKS {
        let callback_started = Instant::now();
        let cpu_started = cpu_clock_available
            .then(|| thread_cpu_time().expect("thread CPU clock disappeared during release soak"));
        runtime.process_block(black_box(&input), &mut scratch, black_box(&mut output));
        let wall_elapsed = callback_started.elapsed();
        let cpu_elapsed = if let Some(cpu_started) = cpu_started {
            thread_cpu_time()
                .and_then(|finished| finished.checked_sub(cpu_started))
                .expect("thread CPU clock moved backwards during release soak")
        } else {
            wall_elapsed
        };

        wall_deadline_outliers += u64::from(wall_elapsed > CALLBACK_DEADLINE);
        gate_deadline_misses += u64::from(cpu_elapsed > CALLBACK_DEADLINE);
        if wall_elapsed > CALLBACK_DEADLINE && cpu_elapsed <= CALLBACK_DEADLINE {
            scheduler_preemption_candidates += 1;
            largest_wall_minus_cpu =
                largest_wall_minus_cpu.max(wall_elapsed.saturating_sub(cpu_elapsed));
        }
        wall_callback_times.push(wall_elapsed);
        if let Some(cpu_times) = &mut cpu_callback_times {
            cpu_times.push(cpu_elapsed);
        }
    }
    let wall_elapsed = wall_started.elapsed();

    wall_callback_times.sort_unstable();
    let wall_p99 = percentile(&wall_callback_times, 99.0);
    let wall_p999 = percentile(&wall_callback_times, 99.9);
    let wall_max = *wall_callback_times.last().expect("wall callback timings");

    let (gate_source, gate_p99, gate_p999, gate_max) =
        if let Some(cpu_times) = &mut cpu_callback_times {
            cpu_times.sort_unstable();
            (
                "thread CPU time",
                percentile(cpu_times, 99.0),
                percentile(cpu_times, 99.9),
                *cpu_times.last().expect("CPU callback timings"),
            )
        } else {
            (
                "wall time fallback (thread CPU clock unavailable)",
                wall_p99,
                wall_p999,
                wall_max,
            )
        };
    eprintln!(
        "MOT RT 48k/32 simulated 30-minute soak — callbacks={SOAK_CALLBACKS}, \
         execution={wall_elapsed:?}; \
         wall p99={wall_p99:?}, p99.9={wall_p999:?}, max={wall_max:?}, \
         >667us outliers={wall_deadline_outliers}; \
         {gate_source} p99={gate_p99:?}, p99.9={gate_p999:?}, max={gate_max:?}, \
         >667us gate misses={gate_deadline_misses}; \
         scheduler-preemption candidates={scheduler_preemption_candidates}, \
         largest wall-minus-CPU={largest_wall_minus_cpu:?}"
    );
    eprintln!(
        "Wall outliers in this normal-priority, unpaced process are scheduler \
         diagnostics only and are not claimed to be actual DAW audio xruns."
    );

    assert!(
        output.iter().all(|sample| sample.is_finite()),
        "maximum runtime became non-finite during the soak"
    );
    assert!(
        wall_p99 <= Duration::from_micros(167),
        "soak wall p99 {wall_p99:?} exceeds 0.167 ms"
    );
    assert!(
        wall_p999 <= Duration::from_micros(333),
        "soak wall p99.9 {wall_p999:?} exceeds 0.333 ms"
    );
    assert!(
        gate_p99 <= Duration::from_micros(167),
        "soak {gate_source} p99 {gate_p99:?} exceeds 0.167 ms"
    );
    assert!(
        gate_p999 <= Duration::from_micros(333),
        "soak {gate_source} p99.9 {gate_p999:?} exceeds 0.333 ms"
    );
    assert_eq!(
        gate_deadline_misses, 0,
        "callback work exceeded the 0.667 ms {gate_source} deadline; max={gate_max:?}"
    );
}

#[test]
#[ignore = "deterministic native-48k spectral regression gate; run in release mode"]
fn native_48k_aliasing_proxy_release_gate() {
    assert!(
        !std::hint::black_box(cfg!(debug_assertions)),
        "the aliasing regression gate must be run with --release"
    );

    const SWEEP_SAMPLES: usize = REQUIRED_SAMPLE_RATE_HZ as usize * 4;
    const SETTLE_SAMPLES: usize = 48_000;
    const ANALYSIS_SAMPLES: usize = 48_000;
    const LOW_TONE_HZ: f64 = 13_000.0;
    const HIGH_TONE_HZ: f64 = 17_000.0;
    // Each bin is a low-order product whose unfolded frequency is above the
    // 24 kHz Nyquist limit. 9 kHz is intentionally excluded: it is both the
    // folded third harmonic of 13 kHz and the legitimate 2f1-f2 product.
    const FOLD_CANDIDATES_HZ: [f64; 6] = [1_000.0, 3_000.0, 5_000.0, 14_000.0, 18_000.0, 22_000.0];
    const PALM_ANALYSIS_SAMPLES: usize = 48_000;
    const PALM_NYQUIST_PRESSURE_CUTOFF_HZ: f64 = 16_000.0;

    let runtime = build_maximum_runtime();
    let linear_reference = build_linear_reference_runtime();

    // Sweep metric: above an 8 kHz instantaneous fundamental, low-order
    // nonlinear harmonics increasingly cross Nyquist. Subtracting an
    // otherwise identical linear runtime removes the IR and dry path, leaving
    // a deliberately conservative nonlinear residual (wanted distortion and
    // aliases together).
    let sweep = logarithmic_sine_sweep(SWEEP_SAMPLES);
    let sweep_output = render_partitioned(&runtime, &sweep, &[TARGET_BLOCK_SIZE]);
    let sweep_linear = render_partitioned(&linear_reference, &sweep, &[TARGET_BLOCK_SIZE]);
    let sweep_residual = difference(&sweep_output, &sweep_linear);
    let sweep_analysis_start = logarithmic_sweep_frequency_sample(SWEEP_SAMPLES, 8_000.0);
    let sweep_residual_db = power_ratio_db(
        mean_square(&sweep_residual[sweep_analysis_start..]),
        mean_square(&sweep_output[sweep_analysis_start..]),
    );

    // Multitone metric: direct projection at analytically known folded
    // low-order products, relative to the two carriers. Integer-Hz tones and a
    // one-second window make the result independent of FFT binning.
    let sample_rate = f64::from(REQUIRED_SAMPLE_RATE_HZ);
    let input: Vec<f32> = (0..SETTLE_SAMPLES + ANALYSIS_SAMPLES)
        .map(|sample| {
            let time = sample as f64 / sample_rate;
            (0.26
                * ((std::f64::consts::TAU * LOW_TONE_HZ * time).sin()
                    + (std::f64::consts::TAU * HIGH_TONE_HZ * time).sin())) as f32
        })
        .collect();
    let output = render_partitioned(&runtime, &input, &[TARGET_BLOCK_SIZE]);
    let linear_output = render_partitioned(&linear_reference, &input, &[TARGET_BLOCK_SIZE]);
    let analysis = &output[SETTLE_SAMPLES..];
    let linear_analysis = &linear_output[SETTLE_SAMPLES..];
    let nonlinear_analysis = difference(analysis, linear_analysis);
    assert!(analysis.iter().all(|sample| sample.is_finite()));

    let low_fundamental = tone_projection_amplitude(analysis, LOW_TONE_HZ);
    let high_fundamental = tone_projection_amplitude(analysis, HIGH_TONE_HZ);
    let fundamental_rss = low_fundamental.hypot(high_fundamental);
    let fold_amplitudes: Vec<(f64, f64)> = FOLD_CANDIDATES_HZ
        .iter()
        .map(|frequency| {
            (
                *frequency,
                tone_projection_amplitude(&nonlinear_analysis, *frequency),
            )
        })
        .collect();
    let fold_rss = fold_amplitudes
        .iter()
        .map(|(_, amplitude)| amplitude * amplitude)
        .sum::<f64>()
        .sqrt();
    let fold_ratio_db = 20.0 * (fold_rss.max(1.0e-20) / fundamental_rss.max(1.0e-20)).log10();

    // Palm-mute metric: compare the nonlinear residual with the full output,
    // then pass the residual through a deterministic offline high-pass
    // analysis FIR. The latter is "Nyquist pressure", not isolated alias
    // energy: it deliberately catches nonlinear energy approaching 24 kHz
    // before a runtime change can make native-rate folding materially worse.
    let palm = synthetic_palm_mute_di(SETTLE_SAMPLES + PALM_ANALYSIS_SAMPLES);
    let palm_output = render_partitioned(&runtime, &palm, &[TARGET_BLOCK_SIZE]);
    let palm_linear = render_partitioned(&linear_reference, &palm, &[TARGET_BLOCK_SIZE]);
    let palm_analysis = &palm_output[SETTLE_SAMPLES..];
    let palm_residual = difference(palm_analysis, &palm_linear[SETTLE_SAMPLES..]);
    let palm_output_power = mean_square(palm_analysis);
    let palm_residual_db = power_ratio_db(mean_square(&palm_residual), palm_output_power);
    let palm_nyquist_pressure_db = power_ratio_db(
        high_pass_mean_square(&palm_residual, PALM_NYQUIST_PRESSURE_CUTOFF_HZ, 255),
        palm_output_power,
    );

    eprintln!(
        "MOT native-48k alias-risk regression gate — \
         logarithmic sweep nonlinear residual={sweep_residual_db:.2} dBr \
         (limit {ALIAS_SWEEP_MAX_RESIDUAL_DB:.2} dB); \
         13/17 kHz multitone fold candidates={fold_ratio_db:.2} dBc \
         (limit {ALIAS_MULTITONE_MAX_FOLD_DB:.2} dB), bins={fold_amplitudes:?}; \
         synthetic palm-mute nonlinear residual={palm_residual_db:.2} dBr \
         (limit {ALIAS_PALM_MAX_RESIDUAL_DB:.2} dB), \
         >16 kHz Nyquist pressure={palm_nyquist_pressure_db:.2} dBr \
         (limit {ALIAS_PALM_MAX_NYQUIST_PRESSURE_DB:.2} dB)"
    );
    eprintln!(
        "This is a fixed-model regression proxy, not an absolute aliasing \
         measurement: native-rate bins and nonlinear residuals include some \
         intended products, and the result does not establish subjective \
         transparency for every trained model."
    );

    assert!(
        sweep_residual_db <= ALIAS_SWEEP_MAX_RESIDUAL_DB,
        "sweep nonlinear residual {sweep_residual_db:.2} dB exceeds regression limit \
         {ALIAS_SWEEP_MAX_RESIDUAL_DB:.2} dB"
    );
    assert!(
        fold_ratio_db <= ALIAS_MULTITONE_MAX_FOLD_DB,
        "multitone fold proxy {fold_ratio_db:.2} dBc exceeds regression limit \
         {ALIAS_MULTITONE_MAX_FOLD_DB:.2} dB"
    );
    assert!(
        palm_residual_db <= ALIAS_PALM_MAX_RESIDUAL_DB,
        "palm-mute nonlinear residual {palm_residual_db:.2} dB exceeds regression limit \
         {ALIAS_PALM_MAX_RESIDUAL_DB:.2} dB"
    );
    assert!(
        palm_nyquist_pressure_db <= ALIAS_PALM_MAX_NYQUIST_PRESSURE_DB,
        "palm-mute Nyquist pressure {palm_nyquist_pressure_db:.2} dB exceeds regression \
         limit {ALIAS_PALM_MAX_NYQUIST_PRESSURE_DB:.2} dB"
    );
}

#[test]
#[ignore = "target-machine release microbenchmark; see module-level command"]
fn m3_pro_48k_32_sample_runtime_budget() {
    assert!(
        !std::hint::black_box(cfg!(debug_assertions)),
        "the RT acceptance benchmark must be run with --release"
    );
    let base = build_maximum_runtime();
    let config = AudioConfig::new(f64::from(REQUIRED_SAMPLE_RATE_HZ), TARGET_BLOCK_SIZE);
    let input: [f32; TARGET_BLOCK_SIZE] =
        std::array::from_fn(|index| (index as f32 * 0.37).sin() * 0.35);
    let mut output = [0.0; TARGET_BLOCK_SIZE];
    let mut scratch = [0.0; TARGET_BLOCK_SIZE];
    let mut runtime = base.clone();
    runtime.reset(&config);

    for _ in 0..1_000 {
        runtime.process_block(&input, &mut scratch, &mut output);
    }

    let mut callback_times = Vec::with_capacity(STEADY_CALLBACKS);
    for _ in 0..STEADY_CALLBACKS {
        let started = Instant::now();
        runtime.process_block(black_box(&input), &mut scratch, black_box(&mut output));
        callback_times.push(started.elapsed());
    }
    callback_times.sort_unstable();
    let steady_p99 = percentile(&callback_times, 99.0);
    let steady_p999 = percentile(&callback_times, 99.9);

    let mailbox = RuntimeMailbox::new(2);
    let mut chain = GuitarSignalChain::default();
    chain.reset(&config);
    let blocks_per_crossfade = CROSSFADE_SAMPLES / TARGET_BLOCK_SIZE;
    let warm_blocks = MAX_IR_SAMPLES.div_ceil(TARGET_BLOCK_SIZE);

    // Start every measured swap from a fully populated convolution history.
    // Measuring only a freshly installed IR materially understates the cost:
    // its frequency-domain delay line has not yet seen all 127 partitions.
    mailbox.publish_latest(RuntimeUpdate::Ready {
        generation: 0,
        runtime: Box::new(base.clone()),
    });
    assert_eq!(
        chain.poll_runtime(&mailbox),
        Some(RuntimeApplyStatus::Ready { generation: 0 })
    );
    for _ in 0..blocks_per_crossfade {
        chain.process_block(black_box(&input), black_box(&mut output));
    }
    assert_eq!(chain.poll_runtime(&mailbox), None);
    assert_eq!(mailbox.drain_retired(), 1);
    for _ in 0..warm_blocks {
        chain.process_block(black_box(&input), black_box(&mut output));
    }

    let mut swap_times = Vec::with_capacity(SWAP_SAMPLES * blocks_per_crossfade);
    for generation in 1..=SWAP_SAMPLES as u64 {
        mailbox.publish_latest(RuntimeUpdate::Ready {
            generation,
            runtime: Box::new(base.clone()),
        });
        for crossfade_block in 0..blocks_per_crossfade {
            let started = Instant::now();
            if crossfade_block == 0 {
                assert_eq!(
                    chain.poll_runtime(&mailbox),
                    Some(RuntimeApplyStatus::Ready { generation })
                );
            }
            chain.process_block(black_box(&input), black_box(&mut output));
            swap_times.push(started.elapsed());
        }
        assert_eq!(chain.poll_runtime(&mailbox), None);
        assert_eq!(mailbox.drain_retired(), 1);
        for _ in 0..warm_blocks {
            chain.process_block(black_box(&input), black_box(&mut output));
        }
    }
    swap_times.sort_unstable();
    let swap_p99 = percentile(&swap_times, 99.0);
    let swap_p999 = percentile(&swap_times, 99.9);
    let swap_max = *swap_times.last().expect("swap timings");

    eprintln!(
        "MOT RT 48k/32 — steady p99={steady_p99:?}, p99.9={steady_p999:?}; \
         swap p99={swap_p99:?}, p99.9={swap_p999:?}, max={swap_max:?}"
    );
    assert!(
        steady_p99 <= Duration::from_micros(167),
        "steady p99 {steady_p99:?} exceeds 0.167 ms"
    );
    assert!(
        steady_p999 <= Duration::from_micros(333),
        "steady p99.9 {steady_p999:?} exceeds 0.333 ms"
    );
    assert!(
        swap_p99 <= Duration::from_micros(333),
        "swap p99 {swap_p99:?} exceeds 0.333 ms"
    );
    assert!(
        swap_p999 <= Duration::from_micros(333),
        "swap p99.9 {swap_p999:?} exceeds 0.333 ms"
    );
    assert!(
        swap_max <= Duration::from_micros(667),
        "swap callback {swap_max:?} missed the 0.667 ms deadline"
    );
}

fn percentile(sorted: &[Duration], percentile: f64) -> Duration {
    assert!(!sorted.is_empty());
    let rank = (percentile / 100.0 * (sorted.len() - 1) as f64).ceil() as usize;
    sorted[rank.min(sorted.len() - 1)]
}

fn tone_projection_amplitude(samples: &[f32], frequency_hz: f64) -> f64 {
    assert!(!samples.is_empty());
    let angular_frequency =
        std::f64::consts::TAU * frequency_hz / f64::from(REQUIRED_SAMPLE_RATE_HZ);
    let (real, imaginary) = samples.iter().enumerate().fold(
        (0.0_f64, 0.0_f64),
        |(real, imaginary), (index, sample)| {
            let phase = angular_frequency * index as f64;
            (
                real + f64::from(*sample) * phase.cos(),
                imaginary - f64::from(*sample) * phase.sin(),
            )
        },
    );
    2.0 * real.hypot(imaginary) / samples.len() as f64
}

fn difference(left: &[f32], right: &[f32]) -> Vec<f32> {
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .map(|(left, right)| left - right)
        .collect()
}

fn mean_square(samples: &[f32]) -> f64 {
    assert!(!samples.is_empty());
    samples
        .iter()
        .map(|sample| f64::from(*sample).powi(2))
        .sum::<f64>()
        / samples.len() as f64
}

fn power_ratio_db(numerator: f64, denominator: f64) -> f64 {
    10.0 * (numerator.max(1.0e-30) / denominator.max(1.0e-30)).log10()
}

fn logarithmic_sweep_frequency_sample(sample_count: usize, frequency_hz: f64) -> usize {
    const START_HZ: f64 = 35.0;
    const END_HZ: f64 = 20_000.0;

    let progress = (frequency_hz / START_HZ).ln() / (END_HZ / START_HZ).ln();
    (progress.clamp(0.0, 1.0) * sample_count as f64).floor() as usize
}

/// Mean-square energy above `cutoff_hz`, measured by an offline, centered
/// Blackman-windowed sinc FIR.
///
/// This is analysis code only. Its non-causal centering and allocation are
/// intentionally unrelated to the zero-latency audio path.
fn high_pass_mean_square(samples: &[f32], cutoff_hz: f64, tap_count: usize) -> f64 {
    assert!(tap_count >= 3 && tap_count % 2 == 1);
    assert!(cutoff_hz > 0.0 && cutoff_hz < f64::from(REQUIRED_SAMPLE_RATE_HZ) * 0.5);
    assert!(samples.len() > tap_count);

    let radius = tap_count / 2;
    let normalized_cutoff = cutoff_hz / f64::from(REQUIRED_SAMPLE_RATE_HZ);
    let mut low_pass = Vec::with_capacity(tap_count);
    for tap in 0..tap_count {
        let offset = tap as f64 - radius as f64;
        let sinc = if offset == 0.0 {
            2.0 * normalized_cutoff
        } else {
            (std::f64::consts::TAU * normalized_cutoff * offset).sin()
                / (std::f64::consts::PI * offset)
        };
        let phase = std::f64::consts::TAU * tap as f64 / (tap_count - 1) as f64;
        let blackman = 0.42 - 0.5 * phase.cos() + 0.08 * (2.0 * phase).cos();
        low_pass.push(sinc * blackman);
    }
    let dc_gain = low_pass.iter().sum::<f64>();
    for coefficient in &mut low_pass {
        *coefficient /= dc_gain;
    }

    let mut energy = 0.0_f64;
    let measured_samples = samples.len() - radius * 2;
    for center in radius..samples.len() - radius {
        let low = low_pass
            .iter()
            .enumerate()
            .map(|(tap, coefficient)| f64::from(samples[center + tap - radius]) * coefficient)
            .sum::<f64>();
        let high = f64::from(samples[center]) - low;
        energy += high * high;
    }
    energy / measured_samples as f64
}
