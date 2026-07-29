# MOT Guitar Suite

Private Apple-Silicon guitar tools built as three independent mono VST3
plug-ins in one Rust workspace:

- **MOT TRAINER** — records a DAW-aligned render of the canonical excitation,
  trains the causal A2 model, applies the validation gate, and publishes an
  immutable model.
- **MOT PLAYER** — browses and plays `.motmodel` files, adds the
  `INPUT GAIN`, `TIGHT`, and `BITE` controls, and loads cabinet IRs.
- **MOT TUNER** — the standalone chromatic strobe tuner for the Fender Studio
  input channel.

The wrappers have independent VST3 class IDs, parameter/state schemas, DSP
states, and single-page editors. Shared format-agnostic DSP and persistence
live in the root `mot-core` crate.

## Workspace

```text
mot-core
├── causal A2-C3 runtime
├── model format and library
├── zero-latency amp/cabinet path
├── reference generation, recording, and alignment validation
├── chromatic tuner
└── optional Candle/Metal offline trainer

plugins/
├── mot-trainer
├── mot-player
└── mot-tuner
```

Candle/Metal is enabled only for `mot-trainer`. Player inference remains a
fixed native CPU implementation with no allocation, lock, I/O, lookahead, or
hidden internal block in the audio callback.

## Shared storage

```text
~/Library/Application Support/Plut&Mot/MOT Guitar Plugin/
├── Capture Assets/
│   ├── input.wav
│   └── reference.wav
├── Capture Records/
├── IRs/
├── Model Settings/
└── Models/
```

The required excitation is NAM's canonical mono 48 kHz,
9,120,000-sample `input.wav`:

```text
SHA-256 70f8ec7f25686a1bd77f25973de8e51a6721e957e81eec121822e5e53366bc41
```

Trainer verifies this exact asset and generates `reference.wav`, a
DAW-ready file containing the pre-roll, synchronization header, excitation,
tail, and alignment margin expected by its recorder. Use `SHOW IN FINDER` in
Trainer, place that file on a mono DAW track, and render it through the
reference chain with the DAW's plug-in delay compensation. Then play the
aligned render through Trainer from its exact start. See
[docs/capture-lab.md](docs/capture-lab.md).

## Zero-latency Player

- Mono, native 48 kHz.
- Official causal NAM WaveNet A2-C3 shape: 1,870 trainable parameters.
- Receptive field: 6,347 current/past samples; it is history, not latency.
- VST3-reported latency: exactly `0 samples`.
- IR maximum: 8192 samples.
- IR head: direct 64-tap convolution.
- IR tail: non-uniform partitioned convolution.
- Default import: auto-trim + minimum phase.
- RAW mode preserves phase and any delay embedded in the IR itself.
- Missing/corrupt selected assets fail closed to safe mute.

## Trainer

Training is real full-dataset optimization, not the removed one-second
prototype. The control is `MAX PASSES`, 1–400, default 400. One pass is one
complete traversal of the available training windows. A contiguous
validation region selects the best checkpoint. A model is published only when
held-out ESR is below `0.035`; failed captures and their WAVs remain available
for diagnosis/retraining. The training input remains the verified canonical
`input.wav`; the target is the mono 48 kHz WAV rendered or recorded by the
DAW. Before publication, the exported native Player runtime is rendered over
the held-out region and its ESR must agree with the training graph.

The current loss is time-domain MSE. MRSTFT is not implemented. Training runs
off the audio callback. Player inference is CPU-only and unaffected by trainer
work.

## Build and install

Requirements: Apple Silicon macOS, stable Rust, and `cargo-truce` 6.3.

```bash
cargo fmt --all -- --check
cargo test --workspace --all-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo truce build --vst3
cargo truce install --vst3 --user
```

Release builds use optimization level 3, full LTO, and one codegen unit.

## Validation

```bash
cargo test --workspace --all-features
cargo test --workspace --all-features --features rt-paranoid
cargo truce validate --pluginval -p mot-trainer
cargo truce validate --pluginval -p mot-player
cargo truce validate --pluginval -p mot-tuner
```

The core acceptance suite also covers sample-zero onset, arbitrary host block
sizes, model/IR integrity, bit-exact transparent paths, RT allocation guards,
capture alignment, and Tuner operation at common sample rates.

## Scope

Version 0.6 is a private laboratory build. Model playback and capture are
fixed to mono/48 kHz. MOT TUNER alone intentionally supports the host's native
44.1/48/88.2/96/192 kHz rates. Physical dBu calibration, stereo processing,
AU wrappers, and a public installer are deferred.
