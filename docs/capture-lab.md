# MOT Guitar Plugin capture lab

This document describes the private two-instance capture workflow implemented
for MOT Guitar Plugin 0.3. Capture is intentionally isolated from the normal
amp/cab playback path: it may buffer minutes of audio and train on a worker,
while NORMAL playback continues to report zero samples of plug-in latency.

## Fixed format and storage

Capture is currently fixed to mono, 48 kHz. Do not enable project resampling,
item stretching, loop playback, normalization, fades, or any DAW effect that
changes the timing of the test signal.

The canonical NAM excitation must be installed at:

```text
~/Library/Application Support/Plut&Mot/MOT Guitar Plugin/
└── Capture Assets/
    └── input.wav
```

The loader requires the official mono 48 kHz, 24-bit, 9,120,000-sample file
with SHA-256:

```text
70f8ec7f25686a1bd77f25973de8e51a6721e957e81eec121822e5e53366bc41
```

Files produced by the lab use these locations:

```text
~/Library/Application Support/Plut&Mot/MOT Guitar Plugin/
├── Models/
│   └── <model_id>.motmodel
├── Model Settings/
│   └── <model_id>.json
└── Capture Records/
    └── <model_id>/
        ├── raw-return.wav
        ├── emitted.wav
        ├── aligned-return.wav
        └── capture.json
```

`.motmodel` files are immutable. Publication uses create-new semantics and
never overwrites an existing model. Editable INPUT GAIN, TIGHT, BITE, and IR
choices live separately under `Model Settings`.

## Two-instance session

Create two mono tracks and insert one MOT Guitar Plugin instance on each:

- `CAPTURE SOURCE` emits the test stream.
- `CAPTURE RETURN` records its own track input into a preallocated internal
  buffer and outputs silence.

Enter exactly the same session name in both instances. The name is converted
to a shared session ID inside the DAW process. A session permits one Source and
one Return; up to 16 concurrent session IDs are reserved by the current lab
build. The tracks need not be adjacent and may use completely different
hardware or software routing.

The default `AUTO` session is convenient when the project contains only one
Source/Return pair. Give every pair an explicit unique name when more than one
pair exists.

At present, press `ARM` on the **Source** instance. The Source atomically arms
both roles once the Return is present and that pair has passed a fresh
`CHECK LEVEL`. The level check is mandatory for both software and hardware
captures: start it on the Return, keep Source active through the final route,
and wait for `PASS`. If the DAW was already playing when armed, stop it first.
The capture begins on the next stopped-to-playing transport edge. DAW
recording is optional.

Do not seek, stop, change sample rate, enable looping, change the pair, or
change routing before the capture reaches `READY`. Any of those operations
invalidates the result. One failed side propagates the failure to its peer.

## Exact capture timeline

The internal stream is independent of DAW block size:

```text
1.000 s silence
→ 4096-sample deterministic sync header
→ 9,120,000-sample excitation
→ 2.000 s recorded tail
```

At 48 kHz the complete Return buffer is 9,268,096 samples, approximately
193.085 seconds. Source and Return process arbitrary host blocks directly; no
internal audio quantum is accumulated.

`SEND TRIM` is applied once to the Source output. The exact value is shared
with the Return through the session coordinator, saved in capture metadata,
and applied to the trainer input. Training input is not normalized.

The worker locates the sync header by normalized correlation within ±24,000
samples, refines the result to a fractional sample, and extracts an aligned
Return excitation. Correlation below 0.35 rejects the capture. Before
correlation or training starts, the exact complete preallocated Return is
written to `raw-return.wav`. The exact trimmed Source training input and
aligned target are retained as `emitted.wav` and `aligned-return.wav`.

The trainer supports 1–400 maximum passes, uses validation auto-stop, and
exports the best checkpoint rather than the final checkpoint.

## Software capture

A typical software setup is:

```text
MOT CAPTURE SOURCE
→ reference amp/profile plug-in
→ bus/send
→ MOT CAPTURE RETURN
```

Use only the chain intended to become the amp model. For an amp-only capture,
disable cabinet, room, gate, pedals, modulation, delay, reverb, post EQ,
limiter, normalization, and look-ahead processing. Keep every third-party
setting and version fixed and record those details in the Capture form.

The Return plug-in should be the first processor that receives the routed
result. Its output is deliberately silent to reduce feedback risk.
Run `CHECK LEVEL` on Return through this complete software route and require
`PASS` before pressing `ARM` on Source.

## Real amplifier capture

The supported first target is a complete amplifier into an unfiltered reactive
load, with the cabinet added later by MOT's IR loader:

```text
MOT CAPTURE SOURCE
→ interface LINE OUT
→ reamp box
→ amplifier input
→ amplifier SPEAKER OUT
→ correctly rated reactive load
→ load RAW / UNFILTERED LINE OUT
→ interface input
→ MOT CAPTURE RETURN
```

Never connect an amplifier speaker output directly to an audio interface.
Use a speaker cable between the amplifier and load, match the load impedance,
and verify that the load can dissipate the amplifier's output power. Keep the
Return track out of the Source hardware output to prevent a feedback loop.

Current captures are deliberately marked `uncalibrated`:

```text
calibration.status = uncalibrated
input_level_dbu = null
output_level_dbu = null
```

The exact Source send trim is still preserved. INPUT GAIN is adjusted by ear
after training until physical dBu calibration is implemented.

Before a hardware run, perform a Return level check with the final interface
gain and amplifier settings. The capture threshold is strict: any Return sample
above -1 dBFS invalidates the run.

Press `CHECK LEVEL` on the Return instance. While the session reports
`MEASURING`, Source automatically loops a precomputed one-second,
maximum-energy fragment of the excitation through the current SEND TRIM and
the real software/hardware route. Return measures exactly 48,000 incoming
samples without entering capture and shows `MEASURING`, `PASS`, or `FAIL` plus
the measured peak. The check rejects a peak above -1 dBFS and any NaN/infinite
input. `ARM` on Source remains locked until the Return in that same session has
published `PASS`. Starting another check, changing either member of the pair,
or changing Source SEND TRIM immediately clears the previous pass. Keep Source
active and unmuted during the check.

## Current implementation limits

The capture lab is functional infrastructure, not yet a public one-button
profiler. These gaps are intentional and should not be hidden:

- Host stop, timeline jumps, loop mode, sample-rate changes, non-finite level
  checks, pair loss, and clipping are detected. A dedicated host dropout/xrun
  notification is not currently available to this VST3 integration, so a
  dropout that does not also disturb the reported timeline may escape
  detection. Saving `raw-return.wav` makes later diagnosis possible but does
  not remove or relax this limitation.
- Only the Source instance initiates pair arming in the present UI/runtime.
  `ARM` on Return alone does not start a session.
- Pairing is process-local. Source and Return can be on different tracks in one
  DAW, but not in different applications or sandboxed plug-in processes.
- Capture is fixed to 48 kHz. There is no 44.1/96 kHz capture or training
  conversion.
- Physical dBu calibration is not implemented.
- Changing the Return role/session while its trainer still owns the completed
  buffer is not a supported workflow. Wait for `MODEL SAVED` (or cancellation
  completion) before reconfiguring the pair.

## Acceptance checks before relying on a capture

1. Confirm both instances show the same non-zero session ID and complementary
   Source/Return roles.
2. Confirm 48 kHz, loop off, and transport stopped before arming.
3. Run `CHECK LEVEL` on Return with the final routing and require `PASS`.
4. Let the complete pre-roll, sync, excitation, and tail finish without
   touching transport or routing.
5. Require successful sync correlation and alignment.
6. Confirm `raw-return.wav`, `emitted.wav`, `aligned-return.wav`, and
   `capture.json` exist.
7. Confirm the `.motmodel` appeared under the fixed `Models` path and validates
   as causal, 48 kHz, zero lookahead, and zero runtime latency.
8. Reload the model in NORMAL mode and compare it at matched loudness against
   the captured target before deleting any source recording.
