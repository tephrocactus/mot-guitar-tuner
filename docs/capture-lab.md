# MOT Generator + MOT Trainer capture lab

Capture now uses two different VST3 plug-ins rather than two roles inside one
large plug-in:

```text
MOT GENERATOR → reference chain or hardware → MOT TRAINER
```

Both instances independently load the same verified excitation and observe
the DAW transport. Keep both tracks active/monitored, arm both while transport
is stopped, wait until both report that they are waiting for transport, then
press Play or Record once. The same stopped-to-playing edge is their shared
clock; there is no cross-bundle static, IPC, or hidden audio transfer.

## Fixed asset and format

The project and both tracks must run mono at 48 kHz. Install:

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

The emitted/recorded window is:

```text
1 second silence
→ 4096-sample deterministic sync header
→ canonical excitation
→ 2 seconds tail
→ shared 0.5-second silent alignment margin
```

Do not stop, seek, loop, change sample rate, or change routing until capture
finishes. Either plug-in invalidates its local run when its observed transport
timeline is discontinuous.

## Software capture

Use one mono track when the DAW supports the full serial route:

```text
MOT GENERATOR
→ reference amp/profile plug-in
→ MOT TRAINER
```

MOT GENERATOR ignores track input. MOT TRAINER records its mono input and
outputs silence by default to reduce feedback risk. Enable `MONITOR` in the
Trainer header to pass the Return to its output for auditioning; this does not
change the signal recorded for capture or used for training. DAW recording
itself is optional.

Disable every stage that should not become part of the learned amp:

- cabinet and room;
- gate and pedals;
- modulation, delay, and reverb;
- post EQ and limiter;
- normalization/lookahead processing.

MOT GENERATOR emits the immutable excitation at unity. MOT TRAINER loads the
same verified asset and uses it directly as the training input; there is no
manual gain value to synchronize between the plug-ins and the input is not
normalized.

Enter a new name in `MODEL`, or use its compact picker to reuse the name and
available capture metadata from an existing model. The picker is a retraining
template: training still creates a new immutable model and does not continue
from the old neural weights.

## Hardware amplifier

```text
MOT GENERATOR
→ interface LINE OUT
→ reamp box
→ amplifier input
→ amplifier SPEAKER OUT
→ correctly rated reactive load
→ load RAW / UNFILTERED LINE OUT
→ interface input
→ MOT TRAINER
```

Never connect a speaker output directly to an audio interface. Use a speaker
cable, match impedance, and ensure the load can dissipate the amplifier's
power. Keep the Return/input-monitor route out of the Generator hardware
output to prevent feedback. Before enabling Trainer `MONITOR`, verify that the
Return cannot feed the amplifier input.

Hardware captures remain explicitly uncalibrated. Record the interface,
reamp/load, amp/channel/control positions, impedance, and Return gain in
Trainer's metadata panel. Set amplifier drive with the interface output and
reamp box. If the Return clips, reduce the load-box line output or interface
Return gain rather than changing the amplifier drive and therefore the
captured tone. Never place an attenuator between Speaker Out and the required
reactive load unless it is explicitly designed and rated for that use.

## Alignment, training, and files

Trainer first writes the exact preallocated Return to `raw-return.wav`. It
finds the sync header by normalized correlation within ±24,000 samples,
performs fractional-sample refinement, then writes:

```text
~/Library/Application Support/Plut&Mot/MOT Guitar Plugin/Capture Records/<id>/
├── raw-return.wav
├── emitted.wav
├── aligned-return.wav
└── capture.json
```

Any Return peak above `−1 dBFS` rejects the run before training. Low sync
correlation also rejects it. The WAVs are preserved.

The trainer optimizes the causal A2-C3 model for 1–400 full passes (one pass
over every available training window), retains
the best validation checkpoint, and publishes only when validation ESR is
below `0.035`. The exported native Player runtime is then rendered over the
same held-out region and must reproduce the trainer's ESR before publication:

```text
~/Library/Application Support/Plut&Mot/MOT Guitar Plugin/Models/<id>.motmodel
```

Trainer work is offline and may take many minutes. It does not run in MOT
PLAYER's audio callback.

## First-version limitations

- Use exactly one armed Generator in the reachable routing graph. Version 0.4
  has no cross-bundle Session ID and cannot distinguish a summed second source.
- There is no cross-bundle automatic level-probe handshake yet. Set a safe
  Return level before the full run; the hard `−1 dBFS` gate still rejects a
  clipped capture.
- A host dropout that leaves the reported transport timeline continuous cannot
  be identified reliably.
- Capture/model playback are mono/48 kHz only.
- Physical dBu calibration is deferred.
