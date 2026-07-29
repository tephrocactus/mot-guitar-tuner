# MOT TUNER

Native Apple-Silicon chromatic strobe tuner for mono guitar input.

MOT TUNER is the only plug-in produced by this repository. Amplifier capture,
training, model playback, and cabinet processing are intentionally out of
scope; NAM model workflows are handled by NAM Gateway.

## Features

- Fast chromatic pitch detection with stable note latching.
- Seven configurable reference notes for a seven-string guitar.
- Independent per-string offsets in 0.1-cent steps.
- One switch to compare configured offsets against equal temperament.
- Large smooth strobe display with note and cents readout.
- Optional output mute.
- Bit-exact mono passthrough while unmuted.
- Exactly `0 samples` of reported plug-in latency.
- Native operation at 44.1, 48, 88.2, 96, and 192 kHz.
- VST3 for macOS ARM64.

The default tuning is `B1–E2–A2–D3–G3–B3–E4`. Plug-in identity and parameter
IDs are kept stable so existing MOT TUNER instances remain compatible.

## Repository

```text
mot-core
└── chromatic pitch detector

plugins/
├── mot-tuner
└── mot-ui
```

`mot-ui` is an internal UI support crate, not a separate plug-in.

## Build and install

Requirements: Apple Silicon macOS, stable Rust, and `cargo-truce` 6.3.

```bash
cargo fmt --all -- --check
cargo test --workspace --all-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo truce build --vst3 -p mot-tuner
cargo truce install --vst3 --user -p mot-tuner
```

Release builds use optimization level 3, full LTO, and one codegen unit.

## Validation

```bash
cargo test --workspace --all-features
cargo truce validate --pluginval -p mot-tuner
```

The test suite covers common sample rates, every semitone in the configured
range, harmonic-rich decays, octave-error rejection, note acquisition and
latching, cents polarity, offsets, bit-exact passthrough, mute/bypass behavior,
zero latency, and headless editor rendering.
