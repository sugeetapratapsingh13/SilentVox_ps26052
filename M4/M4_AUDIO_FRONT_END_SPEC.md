# M4 — Audio Front-End Specification

## Scope

M4 defines the audio acquisition/output boundary between microphone/audio sources, the audio interface, SilentVox processing, and playback. The current module is a reproducible software/digital implementation; physical microphones, a final audio interface, and Raspberry Pi 5 hardware are not physically validated.

## Intended architecture

### Current digital architecture

```text
Reference.wav -> Virtual Reference Mic / REF -+
Error.wav     -> Virtual Error Mic / ERR ----+
Speech.wav    -> Virtual Speech Mic / SPCH -+
                                             v
                              Virtual Audio Interface Model
                                             v
                                  Raspberry Pi input boundary
                                             v
                                    SilentVox processing
                                             v
                                  Virtual/PC playback OUT
```

### Future physical architecture — PENDING

```text
Physical microphones
      -> USB audio interface / ADC
      -> USB
      -> Raspberry Pi 5
      -> SilentVox
      -> USB audio interface / DAC
      -> speaker/headset
```

M4 is the audio acquisition/output boundary. It is not headset acoustic design, acoustic enclosure design, Raspberry Pi performance emulation, M5 secondary-path modelling, or physical headset validation.

## Current software contract

| Parameter | Current value | Status |
|---|---|---|
| Input sample rate | 16,000 Hz | SOFTWARE VALIDATED |
| Output sample rate | 16,000 Hz | SOFTWARE VALIDATED |
| Logical inputs | REF / ERR / SPCH, mono | SOFTWARE VALIDATED |
| Internal sample type | float32 | SOFTWARE VALIDATED |
| Virtual WAV boundary | signed 16-bit PCM | SOFTWARE VALIDATED |
| Physical input bit depth | Configurable/TBD | PHYSICAL VALIDATION PENDING |
| Physical output bit depth | Configurable/TBD | PHYSICAL VALIDATION PENDING |
| Default block | 512 samples | SOFTWARE VALIDATED |
| Tested blocks | 256 / 512 / 1024 | SOFTWARE VALIDATED |
| Frame durations | 16 / 32 / 64 ms | Calculated |
| Physical interface | Not selected | PHYSICAL VALIDATION PENDING |
| Pi 5 USB/audio validation | Not performed | PHYSICAL VALIDATION PENDING |

## Input specification

- REF: one logical mono reference/environmental stream.
- ERR: one logical mono residual/error stream.
- SPCH: one logical mono speech/communication stream.
- Current software standard: 16 kHz.
- Internal representation: float32.
- Current virtual WAV interchange: signed 16-bit PCM.
- Physical interface bit depth remains configurable/TBD.

## Output specification

- OUT: one logical playback stream in the current software demonstration.
- Output sample rate: 16 kHz current software standard.
- Internal sample type: float32.
- Current materialized software output: signed 16-bit PCM WAV.
- Future physical destination: audio-interface output -> speaker/headset.

## USB/Linux audio architecture

Future physical path:

```text
USB audio interface -> Raspberry Pi USB -> Linux audio stack -> SilentVox
SilentVox -> Linux audio stack -> Raspberry Pi USB -> USB audio interface output -> speaker
```

The Linux audio stack may expose devices through ALSA directly or through a higher-level framework such as PipeWire. These layers must not be conflated:

- **ALSA**: Linux kernel/user-space audio API/device layer used by applications and frameworks.
- **PipeWire**: higher-level Linux multimedia/audio session layer that can manage and route audio devices.
- **Application backend**: the specific library/API used by SilentVox to open the stream.

Raspberry Pi documentation notes that Raspberry Pi OS uses PulseAudio or PipeWire by default and that applications can communicate directly with ALSA when required. This documentation does not constitute physical validation of any candidate interface on Raspberry Pi 5.

## Latency categories

M4 keeps these quantities separate:

1. **Frame duration** — samples per block / sample rate.
2. **Buffering delay** — queued audio introduced by buffers.
3. **Software processing time** — measured on the development machine.
4. **Simulated digital pipeline delay** — deliberate software test delay.
5. **Future physical audio-interface round-trip latency** — must be measured on target hardware.
6. **Future acoustic/end-to-end headset latency** — must be measured with the physical system.

Therefore, a 512-sample block at 16 kHz means **32 ms frame duration**, not 32 ms system latency.

## ADC/DAC boundary

Current software simulation:

```text
digital WAV -> virtual ADC/interface behaviour -> processing -> virtual DAC/interface behaviour
```

The simulation models sampling/quantization/clipping/noise and a digital reconstruction-side operation. It does not reproduce exact electrical ADC/DAC behaviour.

Future physical architecture is pending hardware.

## Synchronization acceptance criteria

- same nominal sample rate;
- same frame size;
- common frame timeline;
- known start alignment;
- intentional delay recorded explicitly;
- current software tolerance parameter: 0.25 ms;
- missing/delayed streams detected rather than silently remapped.

## Physical channel requirement

The final interface must provide at least **3 simultaneous independent input channels** and at least **1 usable output channel**.

## Virtual-to-physical replacement

The virtual `REF`/`ERR`/`SPCH` WAV streams are designed to be replaced later by live physical interface channels without changing the downstream logical channel contract.

## Traceability

- M1 receives REF/ERR/SPCH constraints through `M4_TO_M1_AUDIO_CONSTRAINTS.md`.
- M6 receives the physical interface/runtime boundary through `M4_TO_M6_INTERFACE_SPEC.md`.
- M5 remains responsible for acoustic/secondary-path modelling; M4 does not claim acoustic validation.

## Acceptance status

- **IMPLEMENTED**: software/digital front-end, virtual channels, impairments, analysis, browser demo, documents.
- **SOFTWARE VALIDATED**: reproducible tests and generated evidence listed in `VALIDATION_STATUS.md`.
- **PHYSICAL VALIDATION PENDING**: microphones, final interface, Pi 5 USB/audio path, physical synchronization, physical latency, and headset acoustics.
