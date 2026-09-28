# SilentVox — Final AI/ML Dataset & Classification Direction

## 1. Final Objective

SilentVox will **not** be built as a simple environmental-sound classifier.

The AI/ML system must classify acoustically meaningful noise conditions relevant to the intended defence hearing-protection application while preserving speech and supporting adaptive ANC mode selection.

The final system must therefore prioritize:

* acoustic-source validity
* defence-domain relevance
* dataset diversity
* source-level leakage prevention
* cross-dataset validation
* held-out-source testing
* unknown/OOD rejection
* speech-preservation testing
* real-time inference feasibility
* reproducibility
* deployment-domain validation

The project must not claim defence-domain performance solely from public environmental datasets.

---

# 2. Dataset Strategy

Use a **multi-dataset architecture**, not ESC-50 alone.

### Primary research datasets

1. **ESC-50**

   * Controlled environmental benchmark
   * Useful for initial preprocessing and baseline experiments
   * Do not force its labels into SilentVox classes

2. **AudioSet**

   * Large-scale sound-event coverage
   * Candidate source discovery
   * Additional training/evaluation material where licensing/access permits

3. **FSD50K**

   * Large-scale sound-event dataset
   * Useful for mechanical, alarm, vehicle, environmental and other candidate classes
   * Multi-label information is valuable

4. **UrbanSound8K**

   * Supplementary environmental/vehicle/mechanical recordings
   * Use for generalization rather than assuming direct taxonomy equivalence

5. **DEMAND**

   * Environmental/background noise
   * Use primarily for robustness, augmentation and mixed-noise experiments

6. **MUSAN**

   * Speech/noise/music corpus
   * Use primarily for speech-preservation and controlled noise mixing experiments

---

# 3. Do NOT Force a Fixed 6-Class ESC-50 Model

The previous provisional subset:

```text
ENGINE
MACHINERY
RAIN
SIREN
ALARM
WIND
```

must remain a **baseline/experimental subset**, not the final SilentVox taxonomy.

ESC-50 does not contain a clean one-to-one mapping for every desired SilentVox class.

Do not invent a mapping merely to obtain a balanced dataset.

---

# 4. Final Candidate Taxonomy

Use the following as the research taxonomy until experimental validation freezes the final training taxonomy:

```text
ENGINE
ROTOR
VEHICLE
MACHINERY
WIND
RAIN
SIREN
ALARM
CROWD
IMPACT/IMPULSIVE
SPEECH
```

Additionally:

```text
MIXED
UNKNOWN
```

must be implemented as mechanisms rather than blindly treated as ordinary single-label classes.

### MIXED

Represents simultaneous acoustic sources such as:

```text
speech + engine
speech + rotor
vehicle + wind
engine + rain
speech + machinery
```

Mixed conditions should primarily be generated and evaluated through controlled multi-source mixtures.

### UNKNOWN

UNKNOWN must be implemented through confidence/rejection/OOD logic.

It should **not** simply be another miscellaneous training class.

---

# 5. Dataset Mapping Rule

Every training/evaluation recording must have metadata containing at least:

```text
dataset
original_filename
original_label
SilentVox_class
mapping_type
source_recording_id
fold
split
```

Allowed mapping types:

```text
DIRECT
PARTIAL
INDIRECT
NONE
```

Only verified mappings should enter the corresponding training class.

Do not silently convert:

```text
dataset label → SilentVox label
```

without recording the mapping decision.

---

# 6. Important ESC-50 Rule

ESC-50 should be used according to its **actual original labels**.

Examples:

```text
ESC-50 engine
→ ENGINE
→ potentially DIRECT/PARTIAL after verification

ESC-50 helicopter
→ ROTOR
→ PARTIAL

ESC-50 car_horn
→ VEHICLE/ALARM candidate
→ requires verification

ESC-50 church_bells
→ ALARM candidate only if the project definition permits it
→ otherwise exclude

ESC-50 clock_alarm
→ ALARM
→ strong candidate

ESC-50 chainsaw
→ MACHINERY candidate
→ requires verification

ESC-50 vacuum_cleaner
→ MACHINERY candidate
→ requires verification

ESC-50 washing_machine
→ MACHINERY candidate
→ requires verification

ESC-50 train
→ VEHICLE candidate
→ requires verification
```

Do not call these final mappings until the mapping table explicitly records and validates them.

---

# 7. Machinery Problem

Do **not** continue trying to obtain every required class from ESC-50.

For MACHINERY, obtain additional evidence from datasets such as:

```text
FSD50K
AudioSet
UrbanSound8K
project-specific recordings
```

Candidate source labels should be filtered using the SilentVox MACHINERY definition.

Potential examples include:

```text
chainsaw
vacuum cleaner
washing machine
mechanical tools
industrial machinery
machine operation
power tools
```

but each source label must be individually mapped and verified.

---

# 8. Rotor Problem

Do not rely only on the ESC-50 `helicopter` class.

Use additional rotor-related recordings where available:

```text
helicopter rotor
drone rotor
propeller
fan/blade-dominant recordings
rotating aerodynamic sources
```

Rotor recordings should be distinguished from engine-dominant recordings whenever the source data permits.

---

# 9. Defence-Domain Data

This is the most important part for making the project scientifically credible.

Public datasets are **pretraining/research datasets**, not proof of defence deployment performance.

Create a separate:

```text
SILENTVOX_DEFENCE_DOMAIN_DATA
```

containing legally obtained/project-generated recordings representing relevant acoustic conditions.

Possible categories include:

```text
engine
rotor
vehicle
machinery
wind
rain
siren
alarm
impact/impulsive
speech
mixed
```

Record metadata such as:

```text
source type
recording environment
microphone/device
approximate source distance
recording condition
noise condition
sample rate
duration
session ID
source recording ID
```

Do not fabricate defence recordings.

If genuine defence recordings cannot legally be obtained, explicitly label project recordings as:

```text
deployment-domain proxy data
```

rather than claiming that they are military recordings.

---

# 10. Data Leakage Prevention

This is mandatory.

Never allow recordings derived from the same original source/session to appear across:

```text
TRAIN
VALIDATION
TEST
```

The split must occur at the **source/session level**, not randomly at individual audio-window level.

For augmented recordings:

```text
original source
        ↓
split assignment
        ↓
augmentation
        ↓
windows
```

NOT:

```text
audio
 ↓
windows
 ↓
random train/test split
```

This prevents artificially inflated results.

---

# 11. Recommended Evaluation Structure

Use:

```text
TRAIN
VALIDATION
TEST
```

with source-level separation.

Additionally create a stronger evaluation:

```text
IN-DATASET TEST
```

and:

```text
CROSS-DATASET TEST
```

and, when available:

```text
DEPLOYMENT-DOMAIN TEST
```

The model must therefore answer three different questions:

### Test A — Can the model learn the dataset?

Standard held-out evaluation.

### Test B — Can the model generalize to a different dataset?

Train on selected sources and evaluate on a different source/domain.

### Test C — Can the model handle deployment-domain audio?

Evaluate on separate project/deployment-domain recordings.

This is substantially stronger than reporting one random train/test accuracy.

---

# 12. Feature Pipeline

Keep the preprocessing contract:

```text
Input
↓
16 kHz
↓
Mono
↓
float32
↓
numerical validation
↓
feature extraction
```

Candidate features:

```text
MFCC
log-Mel spectrogram
spectral centroid
spectral bandwidth
spectral rolloff
zero-crossing rate
RMS energy
```

The final feature representation must be frozen before final benchmarking.

---

# 13. Model Strategy

Do NOT stop at one classical classifier.

Implement a progression:

```text
Baseline 1:
MFCC + classical classifier

Baseline 2:
MFCC + stronger classical classifier

Model 3:
log-Mel + CNN

Final candidate:
small CNN / lightweight neural network suitable for Raspberry Pi 5
```

The purpose is to demonstrate that the final architecture was selected through experimental comparison rather than arbitrarily chosen.

---

# 14. Required Metrics

Do not report accuracy alone.

Report:

```text
Accuracy
Macro F1
Weighted F1
Per-class precision
Per-class recall
Per-class F1
Confusion matrix
Balanced accuracy
Inference latency
Model size
Memory usage
```

For deployment:

```text
real-time factor
CPU usage
RAM usage
prediction stability
```

For UNKNOWN/OOD:

```text
AUROC where applicable
AUPRC where applicable
false acceptance rate
false rejection rate
known-vs-unknown separation
```

---

# 15. Speech Preservation

Speech must be explicitly evaluated.

Create mixtures such as:

```text
speech + engine
speech + rotor
speech + machinery
speech + wind
speech + rain
speech + siren
speech + alarm
speech + impulsive noise
```

at multiple SNR conditions.

Measure:

```text
speech intelligibility-related performance
SNR improvement
speech distortion
classification stability
```

The classifier must not be evaluated independently from the speech-preservation requirement.

---

# 16. Noise Mixing

For controlled experiments:

```text
clean speech
+
controlled noise
=
mixed signal
```

Generate multiple SNR conditions, for example:

```text
-10 dB
-5 dB
0 dB
5 dB
10 dB
15 dB
20 dB
```

Keep the mixing procedure reproducible.

Use fixed random seeds where appropriate.

---

# 17. Temporal Stability

Live classification must not change class every frame.

Implement and compare:

```text
raw prediction
majority voting
probability moving average
confidence smoothing
hysteresis
minimum consecutive predictions
```

The final deployment configuration should be selected experimentally.

---

# 18. UNKNOWN/OOD Strategy

The live system must not assume:

```text
highest probability = correct class
```

Implement rejection logic such as:

```text
if confidence < threshold:
    UNKNOWN
else:
    predicted class
```

Then calibrate the threshold using validation data.

Do not choose the threshold solely because it looks good on the test set.

Evaluate UNKNOWN behavior using genuinely unseen sound sources.

---

# 19. Final Dataset Manifest

Create one canonical manifest:

```text
dataset_manifest.csv
```

Minimum columns:

```text
sample_id
dataset
original_filename
original_label
SilentVox_class
mapping_type
source_recording_id
session_id
fold
split
sample_rate
duration
augmentation
snr_db
is_mixed
is_project_recording
```

This becomes the single source of truth for M2.

---

# 20. Required Dataset Documentation

Create:

```text
DATASET_RESEARCH.md
CLASS_TAXONOMY.md
CLASS_DEFINITION_TABLE.csv
DATASET_MAPPING.csv
DATASET_MANIFEST.csv
SPLIT_POLICY.md
LEAKAGE_CHECK.md
OOD_POLICY.md
AUGMENTATION_POLICY.md
```

---

# 21. Required Experimental Outputs

M2 must eventually produce:

```text
baseline_results.csv
model_comparison.csv
per_class_metrics.csv
confusion_matrix.png
cross_dataset_results.csv
ood_results.csv
speech_preservation_results.csv
latency_benchmark.csv
final_model_metrics.csv
```

---

# 22. Reproducibility

Every experiment must record:

```text
dataset version
taxonomy version
preprocessing version
feature version
model configuration
random seed
training configuration
evaluation configuration
software environment
```

A result should be reproducible from the repository.

---

# 23. Final Model Selection Rule

The final model must **not** be selected using accuracy alone.

Selection must consider:

```text
classification performance
cross-dataset generalization
OOD/rejection performance
speech preservation
temporal stability
inference latency
memory usage
Raspberry Pi 5 feasibility
```

The final model must therefore be justified experimentally.

---

# 24. What Makes This a Serious Defence-Oriented ML Pipeline

The project should demonstrate:

```text
Research taxonomy
        ↓
Multi-dataset construction
        ↓
Verified label mapping
        ↓
Source-level leakage prevention
        ↓
Controlled augmentation
        ↓
Classical baseline
        ↓
CNN comparison
        ↓
Cross-dataset validation
        ↓
OOD/UNKNOWN rejection
        ↓
Speech-preservation evaluation
        ↓
Temporal smoothing
        ↓
Real-time benchmarking
        ↓
Deployment-domain validation
        ↓
Raspberry Pi 5 deployment
```

This is the required direction.

Do not optimize the project around obtaining a convenient dataset with exactly the desired class names.

Optimize it around **scientifically valid data, reproducible experiments, generalization, rejection of unknown conditions, speech preservation, and real-time deployment**.

---

# 25. Current Status

Current ESC-50 preprocessing:

```text
Metadata: 2000
Audio: 1999
```

The missing ESC-50 recording should be treated as a dataset-integrity issue and documented rather than silently fabricated or replaced.

The existing renamed/duplicated audio filenames must also be resolved before the ESC-50 manifest is considered valid.

The current ESC-50 dataset should therefore remain:

```text
PREPROCESSING / BASELINE DATA
```

until its integrity and mapping are verified.

The final SilentVox training dataset should be constructed only after the multi-dataset mapping and leakage policy are implemented.

## Final Principle

```text
Do not force the data to fit the taxonomy.

Validate the taxonomy against the data,
validate the model against unseen data,
and validate the final system against the deployment domain.
```

This is the standard the SilentVox AI/ML pipeline should follow.
