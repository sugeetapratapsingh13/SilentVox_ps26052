# SilentVox — M2-P5 Robustness Analysis

## 1. Purpose

M2-P5 evaluates whether controlled feature-space augmentation improves the robustness of the SilentVox environmental-noise classifier.

The P4 CNN model is used as the baseline. A P5 augmented model is trained using the same architecture, dataset split, feature representation, optimization configuration, and evaluation procedure, with augmentation applied only to training features.

The purpose of this stage is to determine whether augmentation provides measurable improvement without changing the held-out validation or test data.

---

## 2. Experimental Configuration

### P4 Baseline

The P4 baseline uses the P3 Log-Mel feature representation:

```text
Sample rate:       16 kHz
Feature:           Log-Mel
Mel bands:         64
Window duration:   2 seconds
Window hop:        1 second
Feature shape:     64 × 197
Input tensor:      1 × 64 × 197
Classes:           ENGINE, RAIN, SIREN, WIND
```

The P4 CNN contains:

```text
Trainable parameters: 6,132
Estimated FP32 size:  23.95 KB
```

The P3 source-level train/validation/test split is retained.

---

## 3. P5 Augmentation

Augmentation is applied only to the training feature tensors.

The configuration is:

```text
Gaussian feature noise standard deviation: 0.02
Time-mask width:                           10
Frequency-mask width:                     6
```

Validation and test features are not augmented.

This prevents artificial modification of the evaluation data.

The P5 augmentation experiment uses:

```text
P5 version: P5-v0.1
Random seed: 42
```

---

## 4. Dataset and Split

The P5 experiment uses the authoritative P3 feature dataset.

Feature distribution:

```text
TRAIN:
ENGINE       116
RAIN         112
SIREN        108
WIND         120

VALIDATION:
ENGINE        24
RAIN          24
SIREN         20
WIND          24

TEST:
ENGINE        20
RAIN          24
SIREN         32
WIND          16
```

The test distribution is inherited from the source-level P2 split and is not artificially balanced.

There is no source-level leakage between training, validation, and test sets.

---

## 5. Validation Performance

The best P5 model was selected using validation macro-F1.

```text
Best epoch:                23
Best validation macro-F1:  0.5556
```

P4 baseline:

```text
Best validation macro-F1:  0.5538
```

The validation macro-F1 therefore changed from:

```text
0.5538 → 0.5556
```

This is a small increase and should not be interpreted as evidence of a large generalization improvement.

---

## 6. Window-Level Test Results

| Metric          | P4 Baseline | P5 Augmented |
| --------------- | ----------: | -----------: |
| Accuracy        |      0.3804 |       0.4130 |
| Macro Precision |           — |       0.4169 |
| Macro Recall    |           — |       0.4141 |
| Macro F1        |      0.3576 |       0.4047 |

The P5 augmented model produced higher window-level test accuracy and macro-F1 than the P4 baseline.

Absolute changes:

```text
Window accuracy:
0.3804 → 0.4130
Change: +0.0326

Window macro-F1:
0.3576 → 0.4047
Change: +0.0472
```

These results indicate improved classification performance at the individual feature-window level for this experiment.

---

## 7. Recording-Level Test Results

Recording-level aggregation provides a more relevant measure for a complete audio recording.

| Metric          | P4 Baseline | P5 Augmented |
| --------------- | ----------: | -----------: |
| Accuracy        |      0.4348 |       0.3913 |
| Macro Precision |      0.4196 |       0.3821 |
| Macro Recall    |      0.4313 |       0.4000 |
| Macro F1        |      0.4196 |       0.3825 |

The P5 augmented model therefore produced lower recording-level performance on the held-out test set.

Absolute changes:

```text
Recording accuracy:
0.4348 → 0.3913
Change: -0.0435

Recording macro-F1:
0.4196 → 0.3825
Change: -0.0372
```

Therefore, the augmentation experiment does not demonstrate an overall recording-level improvement.

---

## 8. Recording-Level Confusion Matrix

P5 recording-level confusion matrix:

```text
predicted_class  ENGINE  RAIN  SIREN  WIND
true_class
ENGINE                3     1      1     0
RAIN                  0     3      2     1
SIREN                 1     2      2     3
WIND                  3     0      0     1
```

Class-level observations:

```text
ENGINE: 3/5 correctly classified
RAIN:   3/6 correctly classified
SIREN:  2/8 correctly classified
WIND:   1/4 correctly classified
```

The major observed confusion patterns are:

```text
WIND → ENGINE
SIREN → WIND
SIREN → RAIN
RAIN  → SIREN
```

WIND remains strongly confused with ENGINE.

SIREN remains difficult to separate from the other environmental classes.

---

## 9. Comparison With P4 Confusion

P4 recording-level confusion matrix:

```text
predicted_class  ENGINE  RAIN  SIREN  WIND
true_class
ENGINE                3     1      1     0
RAIN                  0     3      2     1
SIREN                 1     2      3     2
WIND                  3     0      0     1
```

The main difference is the SIREN row.

P4:

```text
SIREN: 3 correct
```

P5:

```text
SIREN: 2 correct
```

The remaining class-level correct counts are unchanged.

Thus the recording-level decrease is primarily associated with the changed SIREN predictions.

---

## 10. Confidence Analysis

P4 confidence analysis showed that confidence cannot currently be interpreted as a calibrated probability of correctness.

P4 incorrect predictions included:

```text
ENGINE → SIREN    confidence 0.8792
RAIN   → SIREN    confidence 0.5646
SIREN  → WIND     confidence 0.5184
```

P5 also contains a high-confidence incorrect prediction:

```text
ENGINE → SIREN    confidence 0.8199
```

Therefore, high softmax confidence does not guarantee correct classification.

This is important for the deployment stage.

An UNKNOWN mechanism must not be implemented using an arbitrary confidence threshold without a validation-based calibration or threshold-selection procedure.

---

## 11. Recording-Level P5 Prediction Findings

Examples of P5 predictions include:

```text
ENGINE → SIREN    0.8199
RAIN   → SIREN    0.5574
SIREN  → WIND     0.5120
WIND   → ENGINE   0.5638
```

These examples demonstrate that the model continues to produce uncertain and occasionally overconfident classifications.

The deployment system should therefore retain confidence information but should not treat the raw softmax maximum as a calibrated confidence probability.

---

## 12. Interpretation

The P5 experiment demonstrates different behavior at window and recording levels.

At the window level:

```text
Performance increased.
```

At the recording level:

```text
Performance decreased.
```

This difference is important because a recording may contain multiple feature windows whose predictions are aggregated into one recording-level decision.

Improved individual-window predictions do not necessarily result in improved recording-level classification.

Therefore, model evaluation for SilentVox should retain both:

```text
Window-level metrics
Recording-level metrics
```

Recording-level evaluation is particularly important for deployment because live inference will generate a sequence of windows over time.

---

## 13. Robustness Findings

The current experiment provides the following findings:

### Finding 1 — Training-only augmentation is technically valid

Augmentation was applied only to the training features.

Validation and test data remained unchanged.

### Finding 2 — Window-level performance improved

The P5 augmented model achieved:

```text
Window accuracy:  41.30%
Window macro-F1:  0.4047
```

compared with:

```text
P4 accuracy:      38.04%
P4 macro-F1:      0.3576
```

### Finding 3 — Recording-level performance decreased

The P5 augmented model achieved:

```text
Recording accuracy: 39.13%
Recording macro-F1: 0.3825
```

compared with:

```text
P4 accuracy:         43.48%
P4 macro-F1:         0.4196
```

### Finding 4 — Class confusion remains

WIND remains frequently confused with ENGINE.

SIREN remains confused with RAIN and WIND.

### Finding 5 — Confidence remains uncalibrated

High-confidence incorrect predictions remain present.

Therefore, raw confidence should not yet be used as a validated UNKNOWN threshold.

---

## 14. Limitations

The current experiment has several limitations.

### Dataset size

The implementation dataset contains only:

```text
160 recordings
127 unique sources
4 classes
```

The recording-level test set contains only:

```text
23 recordings
```

Therefore, individual recording-level changes can have a substantial effect on the reported metrics.

### Class imbalance

The test set contains different numbers of recordings per class because the source-level split is inherited from P2.

The distribution is:

```text
ENGINE: 5
RAIN:   6
SIREN:  8
WIND:   4
```

The test set is not artificially balanced.

### Limited augmentation experiment

Only one augmentation configuration was evaluated in this P5 experiment:

```text
Noise std:        0.02
Time mask width:  10
Frequency mask:   6
```

Therefore, no conclusion should be made about all possible augmentation strategies.

### Confidence calibration

No dedicated confidence calibration method has yet been validated.

Therefore, the P5 results do not establish a final UNKNOWN threshold.

---

## 15. P5 Decision

The P5 augmentation model is retained as an experimental robustness model.

However, the P5 experiment does not justify replacing the P4 baseline solely on the basis of the current held-out test results.

The evidence is:

```text
Window-level performance:
P5 improved.

Recording-level performance:
P5 decreased.

Validation macro-F1:
P5 showed only a small increase.
```

The final deployment stage should therefore keep the P4 and P5 models/results traceable rather than claiming that augmentation universally improves the classifier.

---

## 16. Handoff to M2-P6

M2-P6 receives:

```text
P4 baseline model
P5 augmented model
P4 evaluation results
P5 evaluation results
P5 recording predictions
P5 augmentation metadata
P5 baseline-vs-augmented comparison
```

P6 must preserve the preprocessing and feature-generation assumptions used during training.

The live pipeline must use:

```text
Microphone
    ↓
Audio window
    ↓
16 kHz mono preprocessing
    ↓
2-second window
    ↓
64-band Log-Mel extraction
    ↓
197-frame feature representation
    ↓
Same training normalization
    ↓
CNN inference
    ↓
Class probabilities
    ↓
Temporal smoothing
    ↓
Final class decision
```

The exact preprocessing and normalization used during P3 must be reused during inference.

P6 should evaluate temporal decision methods including:

```text
Majority voting
Probability moving average
Confidence smoothing
Hysteresis
Minimum consecutive predictions
```

An UNKNOWN decision should only be introduced after a validation-based threshold or calibration procedure is established.

---

## 17. Generated P5 Artifacts

The completed P5 experiment produced:

```text
ROBUSTNESS_ANALYSIS/
├── models/
│   └── cnn_logmel_augmented_best.pt
│
├── results/
│   ├── P4_CNN_RESULTS_baseline.csv
│   ├── P4_window_classification_report.csv
│   ├── P4_recording_classification_report.csv
│   ├── P4_window_confusion_matrix.csv
│   ├── P4_recording_confusion_matrix.csv
│   ├── P4_recording_predictions.csv
│   └── P5_augmented_results.csv
│
├── reports/
│   ├── P5_augmented_training_history.csv
│   ├── P5_augmented_recording_predictions.csv
│   └── P5_baseline_vs_augmented.csv
│
├── metadata/
│   └── P5_AUGMENTATION_METADATA.json
│
└── scripts/
    └── train_p5_augmented.py
```

The exact locations of generated artifacts should be verified against the current repository structure before final GitHub documentation.

---

## 18. P5 Version and Status

```text
P5 version: P5-v0.1
Status: COMPLETE
```

Final status:

```text
Training augmentation experiment completed.
Window-level improvement observed.
Recording-level improvement not observed.
Confidence remains uncalibrated.
Ready for M2-P6 live inference and temporal decision analysis.
```
