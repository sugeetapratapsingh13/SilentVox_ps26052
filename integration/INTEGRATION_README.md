# SilentVox Stage-2 PC Software Integration

## Status

M1 + M2 + M3 are integrated into a PC-based software pipeline.

## Architecture

Input WAV -> M2 Noise Classification -> M1 FxLMS ANC -> M3 CNN Speech Enhancement -> Final Audio WAV

## Input

16 kHz mono WAV. Audio is converted to float32 at the integration boundary.

## M2

Provides noise class, confidence, majority label, probability label, and inference timing.

## M1

Uses the existing FxLMS adapter and retains ownership of ANC processing and parameters.

## M3

Uses the existing SpeechEnhancementCNN and existing STFT preprocessing for enhancement and reconstruction.

## End-to-End Tests

stationary: RAIN, confidence 0.3756, total 0.884193 s
rotor: ENGINE, confidence 0.4020, total 0.854554 s
impulsive: WIND, confidence 0.4712, total 0.825381 s
speech_plus_noise: ENGINE, confidence 0.2883, total 0.827716 s

All four tests produced 64000-sample output WAV files.

## PC Block Benchmark

256 samples: 0.020117 s processing for 0.016000 s audio, RTF 1.257x, deadline not met.
512 samples: 0.011848 s processing for 0.032000 s audio, RTF 0.370x, deadline met.
1024 samples: 0.025072 s processing for 0.064000 s audio, RTF 0.392x, deadline met.

These are PC benchmarks, not Raspberry Pi validation.

## Dependencies

numpy
librosa
soundfile
torch

## Limitations

No Raspberry Pi validation yet.
No physical headset validation yet.
No physical acoustic secondary-path validation yet.
The sequential M2 -> M1 -> M3 pipeline is a software integration test and is not the final physical acoustic architecture.

## Run

python .\\integration\\stage2_pipeline.py .\\M4\\virtual_mics\\speech_plus_noise.wav --output .\\integration\\stage2_speech_plus_noise.wav
