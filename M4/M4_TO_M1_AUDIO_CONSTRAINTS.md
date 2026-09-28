# M4 → M1 Audio Constraints

M4 supplies the audio-front-end contract and measured software evidence used by the existing ANC/adaptive-filter work. M4 does not establish physical headset ANC performance.

## Current contract

| Constraint | Value/status | Evidence |
|---|---|---|
| Sample rate | 16 kHz | `results/audio_metrics.csv` |
| REF | Logical mono reference stream | `CHANNEL_MAP.md` + virtual WAV |
| ERR | Logical mono residual/error stream | `CHANNEL_MAP.md` + virtual WAV |
| SPCH | Logical mono speech stream | `CHANNEL_MAP.md` + virtual WAV |
| Default block | 512 samples / 32 ms frame duration | `results/block_latency_benchmark.csv` |
| Tested blocks | 256 / 512 / 1024 | `results/block_latency_benchmark.csv` |
| Internal type | float32 | `config.py` / pipeline |
| Virtual WAV boundary | 16-bit PCM | `config.py` / generated WAVs |
| Physical interface depth | Configurable/TBD | Physical validation pending |
| Synchronization | common nominal frame timeline in simulation | `CHANNEL_MAP.md` |
| Software synchronization tolerance | 0.25 ms parameter | `config.py` |
| Physical synchronization | Pending | hardware test |
| Reference delays | 1 / 5 / 10 ms software simulation | `results/robustness_tests.csv` |
| Gain mismatch | 0.85 / 1.0 / 1.1 software simulation | `results/robustness_tests.csv` |
| Clipping | software saturation simulation | `results/robustness_tests.csv` |

## Dynamic range and normalization

- Internal normalized audio range: approximately `[-1, +1]`.
- Current virtual WAV boundary: signed 16-bit PCM.
- `soundfile` loads test streams into float32 for processing.
- No additional amplitude normalization is silently applied by the virtual interface; gains are explicit configuration parameters.
- Peak level can be reported as dBFS through `analysis/metrics.py`.

The current synthetic streams are normalized digital test signals, not calibrated acoustic pressure measurements.

## Stream formats

### REF
Mono float32 internally; 16 kHz software standard; source `virtual_mics/reference.wav`.

### ERR
Mono float32 internally; 16 kHz software standard; source `virtual_mics/error.wav`.

### SPCH
Mono float32 internally; 16 kHz software standard; source `virtual_mics/speech.wav`.

## Software benchmark evidence

The block benchmark reports frame duration separately from software processing time. It uses multiple trials and records mean/min/max/standard deviation plus real-time factor. Results are development-machine measurements and must not be called microphone/interface latency.

## Delay experiment

The software test applies explicit 1 ms, 5 ms, and 10 ms delays to a digital stream. These experiments demonstrate timing sensitivity at the digital boundary; they are **simulated delays**, not microphone, USB-interface, or acoustic delays.

## Gain mismatch and clipping

The robustness test exercises gain factors 0.85, 1.0, and 1.1 and a clipping/saturation condition. Results are digital impairment experiments.

## Digital SNR/noise results

`results/audio_metrics.csv` contains the reproducible speech-plus-noise condition where:

- signal = `speech_clean.wav`;
- noise = `speech_plus_noise.wav - speech_clean.wav`.

This produces a **digital test-stream SNR**, not microphone SNR.

## Important separation

Do not substitute any of the following for physical measurements:

- simulated delay -> physical microphone/interface/acoustic delay;
- digital test-stream SNR -> microphone SNR;
- software processing time -> Raspberry Pi audio-interface latency;
- block/frame duration -> total system latency.

Physical microphone, interface, acoustic-path, and end-to-end latency remain pending.

## M1 traceability

The existing ANC runtime configuration remains authoritative for the current ANC algorithm. M4 supplies compatible audio constraints rather than replacing M1's adaptive-filter implementation.
