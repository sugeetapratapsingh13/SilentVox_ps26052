# P6 Temporal Smoothing Benchmark

## Purpose

This benchmark verifies runtime operation of the P6 temporal smoothing mechanisms using live microphone input. It does not measure classification accuracy because live microphone windows have no ground-truth labels.

## Configuration

- Sample rate: 16000 Hz
- Channels: 1
- Window duration: 2.0 s
- Window samples: 32000
- Smoothing history: 5
- Benchmark windows: 30
- Model: P5 augmented CNN
- UNKNOWN threshold: not configured

## Runtime Results

- Windows processed: 30
- Overflow windows: 0
- Non-finite windows: 0
- Minimum processing time: 2135.304 ms
- Median processing time: 2164.274 ms
- Average processing time: 2222.236 ms
- Maximum processing time: 4006.126 ms

## Prediction Transitions

- Raw prediction transitions: 10
- Majority-vote transitions: 4
- Probability-average transitions: 2

## Raw Prediction Distribution

- ENGINE: 12
- RAIN: 0
- SIREN: 0
- WIND: 18

## Majority-Vote Distribution

- ENGINE: 9
- RAIN: 0
- SIREN: 0
- WIND: 21

## Probability-Average Distribution

- ENGINE: 2
- RAIN: 0
- SIREN: 0
- WIND: 28

## Interpretation

The benchmark compares raw predictions with majority-vote and probability-average outputs. Prediction-transition counts are used as a runtime stability indicator. A lower transition count may indicate greater temporal stability, but it does not establish classification accuracy or generalization performance.

## Limitations

- Live microphone windows have no ground-truth labels.
- Classification accuracy is not measured here.
- Confidence values are not calibrated probabilities.
- UNKNOWN threshold selection remains pending validation.
- Physical Raspberry Pi validation has not been performed.
- Results are from the current Windows development environment.

## Status

P6 temporal smoothing runtime verification completed.
