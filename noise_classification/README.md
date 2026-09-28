# SilentVox - M2 Noise Classification

## 1. Overview

SilentVox M2 is the AI/ML noise classification module of the SilentVox adaptive noise cancellation (ANC) defence hearing-protection headset.

The purpose of M2 is to identify the acoustic environment from incoming audio and provide a reliable noise-classification output that can later be used by the complete SilentVox system to select or adapt appropriate processing modes.

The current implementation focuses on a validated four-class operational model:

- ENGINE
- RAIN
- SIREN
- WIND

The broader research taxonomy contains additional defence-relevant categories, but those categories are not currently part of the trained operational CNN model.

---

## 2. Current Operational Model

The currently validated operational CNN contains four classes:

| Index | Class |
|---:|---|
| 0 | ENGINE |
| 1 | RAIN |
| 2 | SIREN |
| 3 | WIND |

The class mapping is defined in:

`models/architecture/cnn_logmel_labels.json`

Current model input:

- Sample rate: 16 kHz
- Audio: mono
- Window duration: 2 seconds
- Window size: 32,000 samples
- Feature representation: Log-Mel spectrogram
- Mel bands: 64
- Input shape: `1 x 64 x 197`

The current CNN has:

- Base channels: 8
- Training epochs: 40
- Batch size: 32
- Learning rate: 0.001
- Weight decay: 0.0001
- Random seed: 42
- Trainable parameters: approximately 6,132

---

## 3. Research Taxonomy vs Operational Taxonomy

The research stage defines a broader taxonomy for future SilentVox development.

Research taxonomy:

- ENGINE
- ROTOR
- VEHICLE
- MACHINERY
- WIND
- RAIN
- SIREN
- ALARM
- CROWD
- IMPACT/IMPULSIVE
- SPEECH
- MIXED
- UNKNOWN

This broader taxonomy is used for research, dataset investigation, class-definition work, and future expansion.

It must not be confused with the current operational CNN taxonomy.

### Current Operational Taxonomy

The currently trained and validated CNN uses only:

- ENGINE
- RAIN
- SIREN
- WIND

The remaining research classes require additional dataset validation, class balancing, source diversity analysis, and model evaluation before they can be introduced into the operational model.

### UNKNOWN

`UNKNOWN` is not a fifth CNN class.

It is intended as a future rejection/OOD mechanism that can identify predictions where the model does not have sufficient confidence that the input belongs to one of the supported operational classes.

The current confidence threshold has not been treated as a calibrated final deployment threshold.

---

## 4. M2 Processing Pipeline

The complete M2 pipeline is:

```text
Audio Input
    |
    v
16 kHz Mono Audio
    |
    v
2-second Audio Window
    |
    v
Preprocessing
    |
    v
Log-Mel Spectrogram
    |
    v
Normalization
    |
    v
CNN Model
    |
    v
Class Probabilities
    |
    v
Predicted Class + Confidence
    |
    v
Temporal Smoothing
    |
    v
Final Classification Output