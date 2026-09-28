# SilentVox M2-P6 — Live Inference Benchmark

## Purpose

This test verifies continuous real-microphone inference through the P6 deployment pipeline.

This is a runtime verification test, not a classification-accuracy evaluation.

## Configuration

- Model: P5 augmented CNN
- Sample rate: 16000 Hz
- Channels: 1
- Window duration: 2.0 seconds
- Window samples: 32000
- Feature: 64-band Log-Mel
- Feature shape: 64 x 197
- Normalization: P3 training-set statistics
- Windows processed: 20

## Runtime Results

- Minimum processing time: 4.280 ms
- Median processing time: 10.134 ms
- Average processing time: 77.670 ms
- Maximum processing time: 1347.296 ms
- Real-time factor: 25.75

## Audio Integrity

- Overflow windows: 0
- Non-finite windows: 0
- Average RMS: 0.038930

## Confidence

- Mean predicted confidence: 0.567064

Confidence is reported descriptively only.
No validated UNKNOWN threshold has been applied.

## Prediction Distribution

- ENGINE: 0
- RAIN: 4
- SIREN: 14
- WIND: 2

These predictions are outputs for live microphone recordings and do not have ground-truth labels.

## Verified Pipeline

Microphone
|
v
16 kHz mono capture
|
v
2-second window
|
v
P3 Log-Mel extraction
|
v
P3 training-set normalization
|
v
P5 CNN inference
|
v
Class probability output

## Limitations

This benchmark does not establish:

- classification accuracy
- classification correctness
- confidence calibration
- final UNKNOWN threshold
- controlled noise robustness
- end-to-end headset latency
- Raspberry Pi hardware performance

Physical Raspberry Pi validation remains future work because the physical Pi is not currently available.

## Output Files

- deployment/results/P6_live_inference_benchmark.csv
- deployment/reports/P6_LIVE_INFERENCE_BENCHMARK.md

## Status

P6 LIVE INFERENCE BENCHMARK COMPLETE
