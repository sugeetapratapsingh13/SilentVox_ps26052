# M4 Channel Map

## Logical map

| Logical channel | Role | Current source | Physical channel |
|---|---|---|---|
| REF | Environmental/reference audio | `virtual_mics/reference.wav` | TBD after physical test |
| ERR | Residual/error audio | `virtual_mics/error.wav` | TBD after physical test |
| SPCH | Speech/communication audio | `virtual_mics/speech.wav` | TBD after physical test |
| OUT | Processed output | `results/output_simulation.wav` in software validation | TBD after physical test |

## Required physical channel capability

The eventual physical interface must provide **at least 3 simultaneous independent input channels** and at least **1 usable output channel**. Physical channel numbers are intentionally not hard-coded before hardware validation.

## Current software format

- Input sample rate: 16 kHz
- Output sample rate: 16 kHz
- Logical input channels: REF / ERR / SPCH
- Current virtual/generated WAV interchange format: signed 16-bit PCM
- Internal processing: float32
- Default block: 512 samples
- Tested blocks: 256 / 512 / 1024
- Frame durations: 16 / 32 / 64 ms at 16 kHz
- Physical interface input/output bit depth: configurable/TBD

16-bit PCM is a **current virtual WAV interchange format**, not a permanent requirement for the final physical ADC/interface.

## Synchronization requirements

All logical streams should use:

1. the same sample rate;
2. the same frame size;
3. a common frame timeline;
4. known start/timestamp alignment;
5. explicit recording of intentional delays;
6. an explicit synchronization tolerance (current software parameter: 0.25 ms);
7. defined behavior if a stream is missing, late, or dropped.

The 0.25 ms value is a software acceptance parameter, not a physical measurement.

## Future physical channel-identification procedure

1. Enumerate the interface on Raspberry Pi 5.
2. Identify all physical inputs.
3. Inject a known signal into one physical input.
4. Capture all channels simultaneously.
5. Identify which captured channel corresponds to the injected signal.
6. Repeat for REF, ERR, and SPCH.
7. Document the resulting physical-to-logical mapping.

## Output specification

- Logical output: OUT
- Output sample rate: 16 kHz current software standard
- Output channel count: 1 usable logical playback channel
- Current software sample type: float32 internally; 16-bit PCM WAV when materialized
- Playback destination: virtual/PC speaker in software demo; future USB audio-interface output/headset boundary
