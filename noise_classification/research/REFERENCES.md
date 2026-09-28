# SilentVox — M2-P1 References

## Dataset References

### 1. AudioSet

Google Research AudioSet.

https://research.google.com/audioset/

Purpose in SilentVox research:

* Large-scale sound-event research
* Candidate class discovery
* Broad acoustic coverage
* Multi-label sound-event analysis

---

### 2. FSD50K

FSD50K — Freesound Dataset 50K.

Zenodo:

https://zenodo.org/records/4060432

Purpose in SilentVox research:

* Large-scale waveform data
* Sound-event classification
* Multi-label research
* Candidate training data

---

### 3. ESC-50

ESC-50 Environmental Sound Classification dataset.

GitHub:

https://github.com/karolpiczak/ESC-50

Purpose in SilentVox research:

* Controlled environmental sound benchmark
* Preprocessing validation
* Initial classifier benchmarking
* Candidate class mapping

---

### 4. UrbanSound8K

UrbanSound8K dataset.

https://urbansounddataset.weebly.com/urbansound8k.html

Purpose in SilentVox research:

* Supplementary environmental sound data
* Generalization testing
* Urban acoustic conditions

---

### 5. DEMAND

DEMAND — Diverse Environments Multichannel Acoustic Noise Database.

Zenodo:

https://zenodo.org/records/1227121

Purpose in SilentVox research:

* Environmental background noise
* Robustness testing
* Noise augmentation
* Multichannel acoustic environment research

---

### 6. MUSAN

MUSAN corpus.

OpenSLR:

https://www.openslr.org/17/

Purpose in SilentVox research:

* Speech/noise augmentation
* Speech preservation experiments
* Background noise generation

---

# Research Interpretation Notes

Dataset coverage does not prove acoustic separability.

A dataset label must not automatically be treated as equivalent to a SilentVox class.

For every dataset used in the final training pipeline:

1. Identify the exact original dataset label.
2. Record the source dataset.
3. Map the original label to the SilentVox taxonomy.
4. Assign a mapping category.
5. Verify inclusion/exclusion rules.
6. Record known ambiguity.
7. Track the source recording for leakage prevention.

---

# Mapping Categories

| Mapping  | Meaning                                                                   |
| -------- | ------------------------------------------------------------------------- |
| DIRECT   | Original dataset label closely matches the SilentVox class                |
| PARTIAL  | Original label covers only part of the SilentVox class definition         |
| INDIRECT | Related acoustic content exists but does not directly represent the class |
| NONE     | No appropriate evidence for the class                                     |

Example:

```text
ESC-50 "Helicopter"
→ SilentVox "ROTOR"
→ PARTIAL
```

This is a project-specific mapping and does not change the original ESC-50 taxonomy.

---

# Research Limitations

The datasets above were selected for research relevance, but they do not by themselves provide complete defence-domain coverage.

Important limitations include:

* Limited defence-specific recordings
* Different recording environments
* Different microphones
* Different source distances
* Different background conditions
* Different class definitions
* Weak labels in some datasets
* Multi-label overlap
* Dataset imbalance
* Potential domain shift

Therefore project-specific recordings remain important for deployment-domain validation.

---

# P1 Research Conclusion

The reviewed datasets provide complementary resources for SilentVox:

* AudioSet and FSD50K provide large-scale candidate sound-event coverage.
* ESC-50 provides a controlled environmental benchmark.
* UrbanSound8K provides supplementary environmental recordings.
* DEMAND provides realistic environmental background noise.
* MUSAN provides speech/noise augmentation material.
* Project-specific recordings provide deployment-domain validation.

No dataset is assumed to be a perfect representation of the final SilentVox taxonomy.

The final dataset selection and class mapping must be frozen only after verification and experimental validation.

**Current taxonomy version: `P1-v0.1-investigate`**
