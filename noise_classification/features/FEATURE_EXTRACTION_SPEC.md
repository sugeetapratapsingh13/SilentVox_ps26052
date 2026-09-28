# SilentVox — Feature Extraction Specification

## P3 Version

P3-v0.1

## Purpose

P3 converts the verified P2 processed audio dataset into model-ready features for SilentVox noise classification.

The P3 pipeline inherits the train/validation/test assignment from P2 and performs windowing only after the source-level split has been fixed.

## Input Dataset

- Sample rate: 16,000 Hz
- Channels: Mono
- Storage: WAV PCM16
- Preprocessing version: P2-v0.1

## Current Verified Classes

- ENGINE
- RAIN
- SIREN
- WIND

This is the current verified implementation subset and does not freeze the final SilentVox taxonomy.

## Windowing

- Window duration: 2 seconds
- Window length: 32,000 samples
- Window hop: 1 second
- Window hop length: 16,000 samples
- Partial windows: Not retained

Each generated window remains in the source recording's P2 split.

## Spectral Parameters

- Sample rate: 16,000 Hz
- FFT size: 512
- Window length: 400 samples
- Hop length: 160 samples
- Mel bands: 64
- Power: 2.0
- Center: False

## MFCC

- Number of coefficients: 13
- DCT type: 2
- DCT normalization: ortho

MFCC temporal dimensions are preserved.

## Log-Mel Spectrogram

- 64 Mel bands
- Power spectrogram
- Converted to decibels using the maximum value of each window as reference
- Temporal dimensions are preserved

## Normalization

Normalization statistics are calculated using the training split only.

Statistics are calculated independently for each feature coefficient/band across training windows and time frames.

Validation, test, and future inference features use the saved training statistics.

## Leakage Prevention

P2 source-level split assignments are inherited directly.

All windows from a single source recording remain in exactly one split.

## Outputs

```text
features/
├── mfcc/
│   ├── train/
│   ├── validation/
│   └── test/
│
└── log_mel/
    ├── train/
    ├── validation/
    └── test/
```

```text
metadata/
├── feature_manifest.csv
├── feature_normalization_stats.npz
└── feature_config.json
```

```text
statistics/
└── feature_statistics.csv
```

## P4/P6 Compatibility

P4 training and P6 live inference must use the same feature-extraction parameters and training-derived normalization statistics.

Any change requires a new P3 version and regeneration of affected feature artifacts.

## Reproducibility

Random seed: 42

Feature extraction itself is deterministic.

## Status

P3-v0.1 finalized for the current verified four-class P2 implementation dataset.
