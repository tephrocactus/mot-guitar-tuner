# MOT amplifier reference capture

The amplifier core uses a fixed 5153-inspired sound as an artistic reference.
It is not intended to reproduce the controls or circuit behaviour of any
particular amplifier.

## Standard input

Use Neural Amp Modeler's canonical `input.wav`:

- Official NAM trainer page:
  <https://neural-amp-modeler.readthedocs.io/en/latest/tutorials/gui.html>
- Official download:
  <https://drive.google.com/file/d/1KbaS4oXXNEuh2aCPLwKrPdf5KFOjda8G/view>
- Verified format: mono WAV, 48 kHz, 24-bit, exactly 190 seconds /
  9,120,000 samples.
- MD5: `36cd1af62985c2fac3e654333e36431e`.
- SHA-256:
  `70f8ec7f25686a1bd77f25973de8e51a6721e957e81eec121822e5e53366bc41`.

Keep reference audio under `captures/`. Audio captures are intentionally
ignored by Git; only source code and capture instructions belong in the public
repository.

## Software reamp session

1. Create a 48 kHz session and import `input.wav` without sample-rate
   conversion, stretching, fades, normalization, clip gain, or level changes.
2. Route the file at unity gain through exactly one instance of the selected
   reference amp plugin.
3. Disable the input gate, drive pedals, cabinet, room, modulation, delay,
   reverb, post EQ, limiter, and every other effect. The target for this pass is
   the amp/power-amp section only.
4. Keep the chosen amp type, controls, input calibration, quality mode, and
   plugin version fixed. Save the preset alongside the rendered file.
5. Render mono, 48 kHz, 24-bit WAV from the exact start of `input.wav`, with no
   normalization, added tail, or loudness processing. Preserve the complete
   190-second length, including the alignment impulses and trailing silence.
6. Name the result `pasadena_amp_only.wav`.

The output must not contain clipped samples (`abs(sample) >= 1.0`). If level
reduction is required, use only the final output control of the reference
plugin and record its exact value.

For listening comparisons, a second render may include the preferred cabinet
and be named `pasadena_full_reference.wav`. It is not used as the input to the
separate MOT cabinet stage.

Before fitting or training, validate matching sample rate, channel count,
duration, absence of clipping, and sample alignment.

## Asset policy

Commercial plugin binaries, presets, captured model weights, rendered reference
audio, and third-party cabinet IRs are local development assets and must not be
committed or distributed without a licence that explicitly permits it.
