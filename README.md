# MOT Guitar Plugin

Private mono guitar amp/cabinet laboratory for macOS ARM64.

The plug-in combines:

- a causal neural amp runtime;
- `INPUT GAIN`, `TIGHT`, and `BITE` controls;
- a zero-reported-latency cabinet IR loader;
- the MOT chromatic strobe tuner;
- a two-instance Source/Return capture lab;
- a native Rust trainer and immutable `.motmodel` library.

The live path is fixed to mono, 48 kHz, and reports exactly `0 samples` of
plug-in latency. There is no lookahead, hidden processing quantum, GPU
inference, STFT path, or runtime sample-rate conversion.

## Storage

User assets and generated models live outside the VST3 bundle:

```text
~/Library/Application Support/Plut&Mot/MOT Guitar Plugin/
├── Capture Assets/
│   └── input.wav
├── Capture Records/
├── IRs/
│   ├── <full-sha256>.wav
│   └── <full-sha256>.motir.json
├── Model Settings/
└── Models/
```

Imported IR WAV files are immutable, content-addressed RAW archives. The WAV
bytes are never rewritten; the sidecar records the original filename, exact
SHA-256, format, and the measured default leading-silence trim. Minimum-phase
conversion is prepared on a worker when the IR is loaded. Project and library
state retain the exact IR ID and digest, so replacing bytes at the same path
fails closed instead of silently changing the tone.

The required capture asset is Neural Amp Modeler's canonical mono, 48 kHz,
24-bit, 9,120,000-sample `input.wav`:

```text
SHA-256 70f8ec7f25686a1bd77f25973de8e51a6721e957e81eec121822e5e53366bc41
```

See [docs/capture-lab.md](docs/capture-lab.md) for software and hardware
capture routing, level checks, alignment, safety, and trainer behavior.

## Zero-latency architecture

- Amp model: diagonal causal recurrent `tanh` network, maximum 32 units.
- Amp controls: causal/minimum-phase filters with per-sample smoothing.
- Cabinet head: taps `0..63` evaluated directly.
- Cabinet tail: non-uniform `64 / 256 / 1024` overlap-add stages.
- IR length: maximum 8192 samples.
- Default IR import: leading-silence trim and minimum-phase transform.
- Runtime changes: complete prepared runtimes swapped on a host-block boundary
  and crossfaded at the same sample positions.
- Missing or corrupt selected assets: safe-mute amp/cab output; tuner remains
  available.

RAW IR mode preserves the file's phase and any delay contained in the IR
itself. That content delay is not reported as plug-in/PDC latency.

## Build

Requirements:

- Apple Silicon Mac;
- current stable Rust;
- `cargo-truce` 6.3;
- 48 kHz host session for amp/cab playback and capture.

Release VST3:

```bash
cargo truce build --vst3
cargo truce install --vst3 --user
```

The release profile uses full LTO and one codegen unit.

## Validation

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --lib --all-features
cargo test --lib --features rt-paranoid
cargo test --release --all-features \
  acceptance::m3_pro_48k_32_sample_runtime_budget \
  -- --ignored --nocapture
cargo test --release --all-features \
  acceptance::m3_pro_48k_32_sample_30_minute_soak \
  -- --ignored --nocapture
cargo test --release --all-features \
  acceptance::native_48k_aliasing_proxy_release_gate \
  -- --ignored --nocapture
/Applications/pluginval.app/Contents/MacOS/pluginval \
  --validate "$HOME/Library/Audio/Plug-Ins/VST3/MOT Guitar Plugin.vst3" \
  --strictness-level 5 \
  --sample-rates 48000 \
  --block-sizes 1,7,16,32,64,257,512
```

The machine-local performance gate targets one maximum model + 8192-sample IR
instance at 48 kHz / 32 samples on a MacBook Pro M3 Pro:

- p99 callback time: at most `0.167 ms`;
- p99.9 callback time: at most `0.333 ms`;
- callback deadline: `0.667 ms`.

Timing tests are deliberately ignored in ordinary debug test runs.
The soak reports wall-clock scheduler outliers separately from the current
thread's CPU time; neither synthetic measurement is presented as a DAW xrun
test.

The native-48 kHz alias-risk command is a deterministic release regression
gate over three fixed stimuli:

- a logarithmic sine sweep measures the nonlinear residual once the
  instantaneous fundamental exceeds 8 kHz;
- a 13/17 kHz multitone measures six analytically known folded-product bins
  relative to the carriers;
- a synthetic palm-mute DI measures both total nonlinear residual and
  nonlinear energy above 16 kHz ("Nyquist pressure").

The residual and Nyquist-pressure values are power ratios in dB relative to
the processed output (`dBr`); the multitone result is relative to its carriers
(`dBc`). The 0.3.0 ARM64 release baseline is respectively `-21.07 dBr`,
`-61.94 dBc`, `-6.26 dBr`, and `-66.19 dBr`. The fixed limits include explicit
regression headroom at `-18 dBr`, `-55 dBc`, `-4 dBr`, and `-58 dBr`.
They protect the current maximum-cost native-48 kHz runtime against spectral
regressions. They are deliberately conservative proxies: nonlinear residual
also contains wanted distortion, a native-rate spectrum cannot identify every
folded component without a trusted oversampled reference, and the fixed test
model cannot guarantee the behavior of every future trained model. Passing
the gate therefore does not replace the listening comparison below.

## Live acceptance

The final host check is deliberately manual:

1. use a native Apple Silicon host at 48 kHz and a 32-sample device buffer;
2. load one maximum-cost model and an 8192-sample IR;
3. confirm the host still reports zero samples of plug-in latency;
4. track or monitor for 30 minutes with no host xrun/dropout indication;
5. repeat the load test in both Fender Studio and REAPER;
6. perform a loudness-matched blind comparison against Pasadena at identical
   32- and 64-sample buffers.

The VST3 integration exposes transport state and sample position but no
dedicated host xrun/discontinuity flag. Capture therefore rejects stops, seeks,
loops, sample-rate changes, clipping, pair loss, and discontinuities inferred
from timeline position; a host dropout that leaves that reported timeline
continuous cannot be identified reliably by the plug-in alone.

## Scope

Version 0.3 is a private laboratory build. It does not provide physical dBu
calibration, 44.1/96 kHz model playback, cross-process Capture IPC, stereo
processing, or a high-latency/HQ mode. Commercial reference plug-ins, presets,
third-party IRs, rendered captures, and trained model files are local assets
and are intentionally excluded from Git.
