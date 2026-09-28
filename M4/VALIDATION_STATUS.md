# M4 Validation Status

## Status summary

**M4 SOFTWARE/DIGITAL VALIDATION: COMPLETE when `python run_all.py` passes.**

**PHYSICAL AUDIO-FRONT-END VALIDATION: PENDING.**

## Implemented and software-validated

| Item | Status | Evidence |
|---|---|---|
| Logical REF/ERR/SPCH model | SOFTWARE VALIDATED | `CHANNEL_MAP.md`, virtual WAVs, tests |
| 16 kHz standardization | SOFTWARE VALIDATED | `results/audio_metrics.csv` |
| 256/512/1024 blocks | SOFTWARE VALIDATED | `results/block_latency_benchmark.csv` |
| 16/32/64 ms frame durations | SOFTWARE VALIDATED | benchmark CSV |
| Processing time separate from frame duration | SOFTWARE VALIDATED | benchmark CSV |
| Waveform | SOFTWARE VALIDATED | `plots/*_waveform.png` |
| FFT | SOFTWARE VALIDATED | `plots/*_fft.png` |
| Spectrogram | SOFTWARE VALIDATED | `plots/*_spectrogram.png` |
| RMS | SOFTWARE VALIDATED | `results/audio_metrics.csv` |
| Peak / dBFS | SOFTWARE VALIDATED | `results/audio_metrics.csv` |
| Crest factor | SOFTWARE VALIDATED | `results/audio_metrics.csv` |
| Digital test-stream SNR/noise result | SOFTWARE VALIDATED | `results/audio_metrics.csv`, `NOISE_FLOOR_ANALYSIS.md` |
| 1/5/10 ms simulated timing delay | SOFTWARE VALIDATED | `results/robustness_tests.csv` |
| Gain mismatch | SOFTWARE VALIDATED | `results/robustness_tests.csv` |
| Additive noise | SOFTWARE VALIDATED | `results/robustness_tests.csv` |
| Quantization | SOFTWARE VALIDATED | `results/robustness_tests.csv` |
| Clipping | SOFTWARE VALIDATED | `results/robustness_tests.csv` |
| Frequency limitation | SOFTWARE VALIDATED | `results/robustness_tests.csv` |
| ADC simulation | SOFTWARE VALIDATED | `audio_pipeline/hardware_sim.py` + robustness CSV |
| DAC-style digital simulation | SOFTWARE VALIDATED | `audio_pipeline/hardware_sim.py` + robustness CSV |
| Output-path simulation | SOFTWARE VALIDATED | `results/output_simulation.wav` |
| Software block benchmark | SOFTWARE VALIDATED | `results/block_latency_benchmark.csv` |
| Browser AudioWorklet | IMPLEMENTED / DEMO | `browser_demo/` |
| Automated tests | SOFTWARE VALIDATED | `tests/`, validation runner |
| Validation runner | SOFTWARE VALIDATED | `run_all.py`, `results/validation_summary.csv` |
| Microphone research | RESEARCH COMPLETE | `MICROPHONE_COMPARISON.csv`, `MIC_SELECTION.md` |
| Interface research | RESEARCH COMPLETE | `AUDIO_INTERFACE_COMPARISON.csv`, `AUDIO_INTERFACE_SELECTION.md` |

## Physical validation pending

- Physical microphone measurement.
- Final physical microphone selection.
- Final physical audio-interface selection.
- Raspberry Pi 5 USB enumeration.
- ALSA/PipeWire device/channel mapping.
- Physical channel synchronization.
- Physical audio-interface/round-trip latency.
- Long-duration hardware stability.
- End-to-end acoustic/headset latency.
- Final headset acoustic validation.

## Claim-control rules

- A datasheet SNR is a **DATASHEET SPECIFICATION**, not a physical SilentVox measurement.
- A generated WAV SNR is a **DIGITAL TEST-STREAM RESULT**, not microphone SNR.
- A software benchmark is a **SOFTWARE/VIRTUAL AUDIO BENCHMARK**, not microphone/interface/Pi latency.
- A 512-sample block at 16 kHz is **32 ms frame duration**, not 32 ms total system latency.
- Simulated ADC/DAC behavior is not a physical ADC/DAC measurement.
- Virtual microphone WAVs are not physical microphone measurements.
- No M4 result establishes complete physical headset ANC reduction.

## Final status wording

> M4 software/digital validation is complete and the module is ready for GitHub push when the validation runner and unit tests pass. Physical microphone, audio-interface, Raspberry Pi 5, and acoustic validation remain future hardware-stage work.
