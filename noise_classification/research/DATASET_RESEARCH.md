# SilentVox — Preprocessing Specification

## Version

```text
P2-v0.1
```

## Purpose

This document defines the preprocessing contract used by the SilentVox noise-classification dataset.

The purpose is to ensure that the audio used for model training and later inference follows the same fundamental preprocessing assumptions.

---

## 1. Input Audio

Current source:

```text
ESC-50
```

Raw files are stored under:

```text
raw/ESC-50/audio/
```

The original files are preserved without modification.

---

## 2. Target Audio Configuration

| Parameter                 | Configuration             |
| ------------------------- | ------------------------- |
| Sample rate               | 16,000 Hz                 |
| Channels                  | Mono                      |
| Internal processing dtype | float32                   |
| Stored WAV format         | PCM 16-bit                |
| Duration                  | Source duration preserved |
| Random seed               | 42                        |

---

## 3. Processing Operations

### Step 1 — Read Audio

Audio is loaded using `soundfile`.

Audio is initially represented as float32 during processing.

### Step 2 — Convert to Mono

If the source contains multiple channels, the channels are averaged:

```text
mono = mean(all channels)
```

### Step 3 — Resampling

Audio is resampled to:

```text
16,000 Hz
```

The actual source sample rate is read from the file rather than assumed.

### Step 4 — Numerical Validation

The processed signal is checked for:

* NaN values
* infinite values

### Step 5 — Storage

The processed audio is stored as:

```text
WAV
16,000 Hz
Mono
PCM 16-bit
```

---

## 4. Amplitude Normalization

No automatic amplitude normalization is applied in P2.

This is intentional.

The preprocessing pipeline should not silently modify recording-level amplitude characteristics before the downstream experiments.

If normalization is introduced later, it must be explicitly documented and applied identically during training and inference.

---

## 5. Duration Handling

The source duration is preserved.

The preprocessing script checks whether the source approximately matches the expected ESC-50 duration of five seconds.

Unexpected durations generate warnings but are not automatically cropped or padded.

Feature extraction may later divide recordings into shorter analysis windows.

---

## 6. Dataset Split

Target split:

```text
Train:       70%
Validation:  15%
Test:        15%
```

The split is performed at source-recording level.

All samples derived from one original recording must remain in one split.

Random seed:

```text
42
```

---

## 7. Leakage Prevention

The following rule is mandatory:

```text
One source recording → one dataset split
```

No source recording may appear simultaneously in:

```text
train + validation
train + test
validation + test
```

The split-generation script performs explicit overlap checks.

---

## 8. Current Class Subset

The current P2 implementation contains:

```text
ENGINE
MACHINERY
RAIN
SIREN
ALARM
WIND
```

These are the current implementation classes.

They are not declared to be the final SilentVox taxonomy.

The class mapping is controlled by:

```text
metadata/esc50_class_mapping.csv
```

---

## 9. Training/Inference Consistency

The following assumptions must remain consistent between training and inference:

```text
Sample rate = 16 kHz
Channels = mono
Audio preprocessing = same
Feature extraction parameters = same
Class ordering = saved with the trained model
UNKNOWN/rejection logic = defined separately by evaluation/deployment
```

Any change must result in a new preprocessing version.

---

## 10. Versioning

Current version:

```text
P2-v0.1
```

A preprocessing version must be incremented if any of the following change:

* target sample rate
* channel configuration
* amplitude normalization
* segmentation assumptions
* audio storage format
* preprocessing algorithm
* dataset mapping
* leakage/split policy
* class definition used by the dataset
