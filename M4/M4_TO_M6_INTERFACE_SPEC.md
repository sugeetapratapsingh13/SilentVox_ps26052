# M4 → M6 Audio Interface Specification

## Current software boundary

```text
Virtual REF / ERR / SPCH
      ↓
Virtual interface model
      ↓
16 kHz block processing
      ↓
SilentVox processing boundary
      ↓
OUT
```

## Future physical boundary

```text
USB audio interface
      ↓
Linux audio stack (ALSA/PipeWire as configured)
      ↓
REF / ERR / SPCH logical channels
      ↓
16 kHz block processing
      ↓
SilentVox
      ↓
OUT
```

## Input format

- 3 logical mono streams: REF / ERR / SPCH.
- Current software standard: 16 kHz.
- Current virtual WAV boundary: signed 16-bit PCM.
- Internal processing: float32.
- Physical bit depth: configurable/TBD until interface is selected and measured.
- Default block: 512 samples.
- Tested blocks: 256 / 512 / 1024.

## Output format

- 1 usable logical output channel minimum.
- Current software standard: 16 kHz.
- Internal processing: float32.
- Current software materialization: signed 16-bit PCM WAV.
- Future destination: physical interface output -> speaker/headset.

## Runtime requirements

- at least **3 simultaneous independent input channels**;
- at least **1 usable output channel**;
- 16 kHz capture/playback;
- explicit physical channel ordering;
- stable block operation;
- sustained capture/playback without unacceptable underruns/overruns;
- measurable processing and physical audio-interface latency.

## Latency contract

At 16 kHz:

- 256 samples = 16 ms frame duration;
- 512 samples = 32 ms frame duration;
- 1024 samples = 64 ms frame duration.

Frame duration is not total latency. M4 reports software processing time separately. Expected software processing time is whatever is recorded in `results/block_latency_benchmark.csv`; physical interface round-trip latency remains pending.

## Backend distinction

M6 should document the actual implementation as one of:

- direct ALSA;
- PipeWire;
- another application-level backend running over the Linux audio stack.

Do not describe Linux/ALSA/PipeWire availability as Raspberry Pi 5 validation.

## Physical channel-identification procedure

1. Connect candidate interface to Raspberry Pi 5.
2. Confirm USB enumeration.
3. List capture/playback devices.
4. Confirm input/output counts.
5. Inject a known signal into one physical input.
6. Capture all channels simultaneously.
7. Identify the corresponding captured channel.
8. Repeat for REF, ERR, and SPCH.
9. Record the physical-to-logical mapping.

## Sustained-operation acceptance test

The future physical test should record:

- test duration: **at least 30 minutes** for the initial acceptance run;
- acceptable underruns: **0** for an accepted run;
- acceptable overruns: **0** for an accepted run;
- dropped frames: **0** for an accepted run;
- CPU usage: record mean and peak;
- RAM usage: record mean and peak;
- sample-rate stability: record actual device setting and observed errors;
- output playback continuity: pass/fail.

These are acceptance criteria for the future physical test, not measurements already achieved.

## Current vs future

| Current software validation | Future physical validation |
|---|---|
| Virtual WAV REF/ERR/SPCH | Physical microphone/interface channels |
| 16 kHz software standard | Confirm 16 kHz on target hardware |
| Software block benchmark | Physical interface/round-trip latency |
| Simulated synchronization/delay | Physical channel synchronization |
| Virtual ADC/DAC-style simulation | Physical ADC/DAC/interface behaviour |
| Browser AudioWorklet demo | Raspberry Pi runtime |

## Current status

No physical audio interface is finally selected. The selection must follow the documented hardware procedure rather than a datasheet-only decision.
