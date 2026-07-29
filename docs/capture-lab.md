# MOT Trainer capture lab

MOT TRAINER learns from a pair of mono 48 kHz signals:

```text
canonical excitation → reference chain → DAW-aligned target render
```

The DAW owns playback, plug-in delay compensation, recording, and rendering.
Trainer records the aligned result from its input, validates the pair, and runs
offline A2-C3 training. There is no separate Generator plug-in or
cross-plug-in transport protocol.

## Canonical source

Install the immutable NAM excitation here:

```text
~/Library/Application Support/Plut&Mot/MOT Guitar Plugin/Capture Assets/input.wav
```

Required contract:

```text
samples  9,120,000
rate     48,000 Hz
channels mono
SHA-256  70f8ec7f25686a1bd77f25973de8e51a6721e957e81eec121822e5e53366bc41
```

Trainer verifies this immutable source and generates a second file for use in
the DAW:

```text
~/Library/Application Support/Plut&Mot/MOT Guitar Plugin/Capture Assets/reference.wav
```

`reference.wav` contains the exact pre-roll, synchronization header,
canonical excitation, tail, and alignment margin expected by Trainer. Use
`SHOW IN FINDER` beside `REFERENCE WAV` to reveal it. VST3 provides no
portable host-independent command for inserting an audio file on the current
DAW track, so drag the revealed file into the project.

Do not normalize, fade, trim, time-stretch, transpose, or change the clip gain
of this file. Keep the DAW project and render at mono/48 kHz.

## Software capture

1. Drag `reference.wav` to a new mono track at a clearly defined project
   position.
2. Leave the source item and track gain at unity.
3. Insert the amp/profile chain that will become the target.
4. Disable every stage that should not become part of the learned amp:
   cabinet and room, gate and pedals, modulation, delay, reverb, post EQ,
   limiter, normalization, and lookahead mastering processing.
5. Render or freeze the processed track over the exact source time range with
   plug-in delay compensation enabled.
6. Keep the mono 48 kHz render in the project without normalization or fades.
7. Put MOT TRAINER on the rendered track, place the playhead at the exact
   start of the rendered item, stop transport, click `ARM`, then click Play.

Using a DAW render is preferable to capturing a live chain inside Trainer. The
DAW compensates latency explicitly reported by the reference plug-ins and
places source and target on one project timeline. Trainer therefore applies
zero timing shift to a software render. Correlation may diagnose a suspicious
capture, but its peak is not removed: filters and nonlinear plug-ins can have
legitimate causal group delay that belongs to the tone.

Input gain inside the reference amp/profile is part of the captured setting.
There is no independent Generator send trim to synchronize or record.

Enter a new name in `MODEL`, or select an existing model as a metadata
template. Retraining still creates a new immutable model and does not continue
from the old neural weights.

## Hardware amplifier

Use the same DAW-ready reference on a DAW track:

```text
DAW source track
→ interface LINE OUT
→ reamp box
→ amplifier input
→ amplifier SPEAKER OUT
→ correctly rated reactive load
→ load RAW / UNFILTERED LINE OUT
→ interface input
→ DAW Return track
```

Record the Return to its own mono track, align it with the DAW's calibrated
external-hardware insert or recording-offset facility, then play the aligned
Return through MOT TRAINER from the reference start. If the interface/driver
does not report the complete round trip accurately, measure a loopback offset
rather than treating the strongest correlation peak as pure transport
latency.

Never connect a speaker output directly to an audio interface. Use a speaker
cable, match impedance, and ensure the load can dissipate the amplifier's
power. Prevent the monitored Return from feeding the amplifier input.

Hardware captures remain explicitly uncalibrated in level. Record the
interface, reamp/load, amp/channel/control positions, impedance, and Return
gain in Trainer's metadata panel. Set amplifier drive with the interface
output and reamp box. If the Return clips, reduce the load-box line output or
interface Return gain rather than changing the amplifier drive and therefore
the captured tone.

## Validation, training, and files

Trainer verifies:

- canonical source identity and format;
- target sample rate and channel layout;
- target length and timeline consistency;
- peak below `−1 dBFS`;
- finite samples and usable signal energy.

The recorded response and its training artifacts are preserved under:

```text
~/Library/Application Support/Plut&Mot/MOT Guitar Plugin/Capture Records/<id>/
├── raw-return.wav
├── emitted.wav
├── aligned-return.wav
└── capture.json
```

For a DAW-rendered software target, `aligned-return.wav` preserves the
rendered causal timing; the applied training shift is zero. Capture metadata
records any measured correlation peak separately from an intentionally
applied hardware calibration.

Trainer optimizes the causal A2-C3 model for 1–400 full passes, retains the
best validation checkpoint, and publishes only when validation ESR is below
`0.035`. The exported native Player runtime is rendered over the same held-out
region and must reproduce the trainer's ESR before publication:

```text
~/Library/Application Support/Plut&Mot/MOT Guitar Plugin/Models/<id>.motmodel
```

Trainer work is offline and may take many minutes. It does not run in MOT
PLAYER's audio callback.

## First-version limitations

- VST3 cannot portably add `reference.wav` to a host track or initiate a
  host render; the user performs the drag, render/record, and aligned playback.
- DAW PDC compensates only latency declared by plug-ins. It does not and
  should not erase a processor's causal phase/group-delay response.
- Hardware round-trip compensation depends on DAW/interface calibration.
- Capture/model playback are mono/48 kHz only.
- Physical dBu calibration is deferred.
