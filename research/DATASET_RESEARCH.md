# Dataset Research — Speech Enhancement (M3-P1)

Per-dataset details below are drawn from each dataset's own paper/documentation
(cited in `REFERENCES.md`), not re-measured by SilentVox. This feeds M3-P2's
choice of source material for the mixture-generation pipeline.

| Dataset | Speech content | Noise content | Sample rate | SNR convention used in the original work | License | Relevance to SilentVox | Limitations |
|---|---|---|---|---|---|---|---|
| **DNS Challenge** (Interspeech 2020/2021, Reddy et al.) | ~500 h clean speech, 2,150 speakers, sourced from Librivox | ~180 h / 60,000+ clips across 150 classes, from AudioSet + Freesound | 16 kHz (wideband track); fullband track extends higher | Official challenge: SNR sampled uniformly from 0–40 dB, mixed to a target RMS of -15 to -35 dBFS. Most follow-on training papers instead dynamically mix at SNRs sampled uniformly between -5 dB and +20 dB | Open, released for the challenge (see DNS-Challenge GitHub for exact terms) | Largest, most standardized public speech+noise resource; the de facto benchmark for comparing any new enhancement model | Noise classes are general-purpose (AudioSet/Freesound categories) — no engine/rotor/machinery/gunfire classes specific to SilentVox's military/tactical environment; a documented gap, not a SilentVox-specific one (see `SPEECH_ENHANCEMENT_RESEARCH.md` Section 4) |
| **WHAM!** (Wichern et al., 2019) | WSJ0-2mix speech (~30 h train / ~8 h val / ~5 h test, 119 speakers) | ~80 h of real urban recordings (coffee shops, restaurants, bars, parks, offices) | 8 kHz and 16 kHz variants (min/max modes) | Noise mixed so the louder speaker is at an SNR of -6 dB to +3 dB relative to the noise | Research use, distributed via the WHAM! site | Standard noisy-speech-separation benchmark; useful as a *methodology* reference (how to structure train/val/test with disjoint speakers) even though its noise classes (cafe/office) don't match SilentVox's | SNR range (-6 to +3 dB) is much narrower than SilentVox's proposed -10 to +20 dB operating range (Section "SNR range justification" in `DATASET_STRUCTURE.md`) — not usable as-is for SilentVox's SNR sweep |
| **WHAMR!** (Maciejewski et al., 2020) | Same WSJ0-2mix speech as WHAM! | Same WHAM! noise, plus synthetic room impulse responses (reverberant) | 8/16 kHz | Same -6 to +3 dB convention as WHAM!, applied before reverberation | Research use, distributed via the WHAM! site | Its reverberation-simulation *methodology* (synthetic RIRs approximating room T60 in the 0.1–1.0 s range) is directly reusable for the reverberation augmentation already implemented in the noise-classifier track (Person 2's `augmentation.py`), and is relevant if SilentVox later wants to model vehicle-cabin acoustics for speech enhancement | Same narrow SNR range as WHAM!; adds significant complexity (RIR simulation) that may not be justified before a simpler noise-only enhancement pipeline is working |
| **MUSAN** (Snyder, Chen & Povey, 2015) | ~60 h speech (LibriVox read speech + US government hearings, 12 languages) | ~6–7 h of assorted technical and ambient noise (car idling, wind, footsteps, etc.); also ~42 h music | 16 kHz WAV | N/A — a raw source corpus, not a pre-mixed dataset; users choose their own SNR when mixing | Creative Commons / US Public Domain — explicitly redistributable, including for commercial use | Good source of clean, license-unencumbered noise and speech clips to mix ourselves at SilentVox's own SNR range and with SilentVox's own noise-class taxonomy, since MUSAN provides raw source audio rather than a fixed mixing recipe | Noise portion is comparatively small (~6–7 h) and general-purpose (not military/vehicle-specific); speech portion is read/formal speech, not the conversational range SilentVox eventually needs |
| **Other suitable sources (not full datasets)** | | | | | | | |
| VoiceBank+DEMAND (Valentini-Botinhao et al., 2016) | 28 speakers train / 2 speakers test (small) | 10 DEMAND noise types (train) + 5 disjoint types (test) | 48 kHz (commonly downsampled to 16 kHz) | 0/5/15/20 dB (train), 2.5/7.5/12.5/17.5 dB (test) | Research use | Small but very widely used as a quick sanity-check benchmark; useful for early pipeline debugging before scaling to DNS-Challenge-sized data | Too small and too narrow in noise diversity to be a primary training set for SilentVox |
| Recorded / self-collected engine, rotor, machinery, wind and impulsive-noise clips (shared with Member 2's noise-classifier taxonomy) | N/A | SilentVox-specific | To be fixed at 16 kHz to match the rest of the project | N/A | Project-owned | The *only* realistic source of the actual noise classes SilentVox needs (engine/rotor/machinery/wind/impulsive) — none of the public speech-enhancement datasets above contain these | Requires actual collection/curation; this is exactly what M3-P2's `dataset/noise/` folder and `noise_metadata` schema are built to receive |

## Why no single existing dataset is sufficient on its own

None of the public datasets above combine (a) SilentVox's actual noise
taxonomy (engine, rotor, machinery, wind, broadband, impulsive/transient) with
(b) a full -10 dB to +20 dB SNR sweep. This mirrors the dataset-relevance
caveat already documented for the noise classifier (Person 2's
`CLASS_TAXONOMY.md` process) and is the reason M3-P2 builds its own
`mixture_generator.py` rather than only pointing at DNS Challenge or WHAM! —
those remain useful as (i) large sources of clean, diverse *speech*, (ii)
methodology references for leakage-free splitting, and (iii) an external
benchmark to sanity-check that a SilentVox-trained model isn't wildly out of
line with published results, but the actual noise side of the mixtures has to
come from SilentVox's own noise recordings (or Member 2's already-being-built
noise corpus, shared across the classifier and speech-enhancement tracks).
