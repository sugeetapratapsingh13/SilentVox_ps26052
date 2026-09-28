# Speech Enhancement — Research & Literature Review

**SilentVox Project — M3-P1 (Speech Enhancement Research + Literature)**
**Owner:** Person M3-P1 — theoretical/research foundation only; this document does not
train models (that is M3-P2/M3-P3's scope).

---

## 1. Scope

This review covers the areas specified for M3-P1: classical speech enhancement
(spectral subtraction, Wiener filtering, spectral masking), neural speech
enhancement (CNN/CRNN, lightweight DNNs), speech separation, and the
SilentVox-specific concerns of speech preservation during ANC and communication
enhancement in hearing-protection systems. It is written as a companion to the
M3-P2 dataset/mixture pipeline and to Person 2's `Related Work` document, which
already covers the ANC literature (Widrow, Morgan, Kuo & Morgan) and the
commercial PELTOR-/INVISIO-type hearing-protection landscape — this document
does not repeat that ground, only cross-references it where relevant.

---

## 2. Classical Speech Enhancement

### 2.1 Spectral subtraction

Spectral subtraction, introduced by Boll (1979), estimates the noise magnitude
spectrum during non-speech segments and subtracts it from the noisy speech
magnitude spectrum, leaving phase untouched (the ear is far less sensitive to
phase distortion than magnitude distortion). It is the historical starting
point for almost all single-channel speech enhancement that followed. Its
well-documented weakness is "musical noise" — isolated, randomly-spaced
spectral peaks left behind by imperfect subtraction, later addressed by
over-subtraction with a spectral floor (Berouti, Schwartz & Makhoul, 1979).

### 2.2 Wiener filtering and statistical estimators

The Wiener filter (Wiener, 1949) is the minimum-mean-square-error (MMSE)
linear estimator for a signal corrupted by additive noise, and is the
theoretical ancestor of most statistical speech-enhancement gain functions.
Ephraim & Malah (1984) derived the MMSE short-time spectral amplitude (STSA)
estimator, and its 1985 log-spectral-amplitude variant, which remain the
reference statistical baseline against which most later work — classical or
neural — is still compared, because they largely eliminate the musical-noise
artefact that plain spectral subtraction produces.

### 2.3 Spectral masking

Time-frequency masking reframes enhancement as estimating, per
time-frequency bin, how much of the observed energy is speech versus noise.
The Ideal Binary Mask (IBM) and Ideal Ratio Mask (IRM) — formalized in the
computational-auditory-scene-analysis literature (Wang & Brown, 2006) — are
the target representations essentially all neural speech-enhancement and
speech-separation models are trained to approximate, whether they predict a
mask explicitly (T-F domain models) or implicitly (time-domain models like
Conv-TasNet, Section 3.3).

---

## 3. Neural Speech Enhancement

### 3.1 CNN / CRNN architectures

Convolutional and convolutional-recurrent networks (CRNNs) extend masking
approaches by learning the mask (or a spectral mapping) directly from
log-power or log-mel spectrograms, using convolution to exploit local
time-frequency structure (analogous to the reasoning behind M3's CNN choice
for noise classification, Person 2) and recurrence to model longer temporal
context than a single frame. This family is the dominant approach in the
DNS Challenge leaderboards (Section 5).

### 3.2 Lightweight DNNs for real-time / edge enhancement

This is the subcategory most directly relevant to SilentVox, since any
speech-enhancement stage has to share the Pi 5 with the ANC and
noise-classification pipelines (Members 1 and 2):

- **RNNoise** (Valin, 2018) is a hybrid DSP + GRU system: classical DSP
  handles the FFT, Bark-scale (22-band) analysis, and pitch/comb filtering,
  while a small recurrent network (215 units across 5 layers) predicts a
  per-band gain rather than a full spectrum. It processes 10 ms frames with
  10 ms look-ahead and is explicitly reported to run in real time on a
  Raspberry Pi — the closest existing system, architecturally, to what M3
  needs to design for.
- **DTLN** (Westhausen & Meyer, 2020) combines an STFT-domain stage with a
  learned-basis time-domain stage in two stacked LSTMs, with under one
  million parameters total, processes audio one frame in/one frame out (true
  streaming, no batch look-ahead), and was reported to outperform the DNS
  Challenge baseline by 0.24 MOS points. Stamenovic et al. (2021) later
  applied weight pruning to a DTLN-style model and compressed it from 3.7 MB
  to 87 KB with only a 0.1 dB SDR loss — a concrete existence proof that
  speech-enhancement models this small are viable, which directly supports
  scoping M3's own model size target aggressively.

### 3.3 Speech separation

Speech separation (isolating individual speakers, as opposed to
speech-vs-noise enhancement) is a related but distinct problem; Conv-TasNet
(Luo & Mesgarani, 2019) is the standard reference: a fully time-domain,
end-to-end model (learned encoder → temporal convolutional mask estimator →
learned decoder) that avoids STFT/ISTFT latency entirely and was reported to
have a substantially smaller model size and shorter minimum latency than
prior time-frequency-masking separators, making it a reasonable reference
architecture if SilentVox ever needs to separate a specific speaker (e.g. the
radio operator) from other talkers rather than just speech-from-noise.

---

## 4. Speech Preservation During ANC and Communication Enhancement

This is the area Person 2's related-work document identifies as the
project's core research gap ("Gap A — explicit speech preservation inside
the adaptive loop"): classical ANC treats residual speech as residual noise,
and commercial PELTOR-/INVISIO-type systems measure speech intelligibility
only *after* the fact rather than optimizing for it directly. The literature
on explicit speech-preserving ANC is genuinely thin — the relevant threads
are:

- End-to-end deep controllers that add a speech-retention loss term
  alongside the noise-cancellation objective, so the network is
  discouraged from cancelling energy in the speech band.
- Linearly-constrained minimum-variance (LCMV) formulations, which enforce
  a hard spatial/spectral constraint that protects a desired signal (here,
  speech/radio audio) while the adaptive filter is otherwise free to
  cancel noise.
- Classification-guided selective cancellation — i.e. exactly the
  Member-2-classifier → Member-1-ANC-controller interface SilentVox already
  uses, where a "speech present" label can trigger a less aggressive
  cancellation mode.

M3's practical contribution to this gap is not a novel ANC algorithm (that
is Member 1's scope) but a speech-enhancement stage that runs on the ANC
residual to recover intelligibility that classical cancellation degrades —
i.e. treating speech preservation as a post-ANC cleanup problem as well as
an in-loop constraint.

---

## 5. Required Paper Comparison Table

| Paper | Year | Problem | Model | Dataset | Metrics | Latency | Result | Limitation | SilentVox relevance |
|---|---|---|---|---|---|---|---|---|---|
| Boll, *Suppression of Acoustic Noise in Speech Using Spectral Subtraction* | 1979 | Remove additive noise from speech magnitude spectrum | Spectral subtraction (non-parametric) | N/A (algorithm paper) | Informal/listening | Per-frame, negligible compute | Established the field; audible "musical noise" artefact | Musical noise; no explicit speech-preservation objective | Historical baseline; too crude alone for SilentVox, but computationally free enough to combine with a lightweight NN front-end |
| Ephraim & Malah, *Speech Enhancement Using a MMSE Short-Time Spectral Amplitude Estimator* | 1984 | Statistically optimal gain function, reduce musical noise vs. spectral subtraction | MMSE-STSA estimator (statistical, model-based) | N/A (algorithm paper) | Informal/listening, later widely used as PESQ/STOI baseline in follow-on work | Per-frame, negligible compute | Standard reference baseline for decades of later work | Assumes a noise PSD estimate; stationarity assumptions degrade under fast-changing or impulsive noise | Strong, near-zero-cost baseline to benchmark any learned SilentVox model against |
| Valin, *A Hybrid DSP/Deep Learning Approach to Real-Time Fullband Speech Enhancement* (RNNoise) | 2018 | Real-time noise suppression on constrained hardware | Hybrid DSP (Bark-band FFT, pitch filter) + small GRU (215 units, 5 layers) predicting per-band gains | Synthetic speech+noise with SNR/gain/bandwidth/resampling augmentation | Listening-based (no PESQ/STOI reported in the cited materials) | 10 ms frame, 10 ms look-ahead | Reported real-time operation on a Raspberry Pi | Full-band gain only, no explicit multi-speaker or spatial handling | Closest architectural precedent for an M3 model sharing the Pi 5 with ANC + classifier |
| Luo & Mesgarani, *Conv-TasNet: Surpassing Ideal T-F Magnitude Masking for Speech Separation* | 2019 | Speaker separation without STFT/ISTFT latency | Fully time-domain: learned encoder → temporal conv-net mask estimator → learned decoder | WSJ0-2mix / WSJ0-3mix | SI-SDR, PESQ (reported to surpass ideal T-F masks) | Reported smaller minimum latency than prior T-F masking separators (exact ms not specified in the cited abstract) | Outperformed prior separation methods on 2- and 3-speaker mixtures | Designed for speaker separation, not speech-vs-noise enhancement; would need retargeting for SilentVox's noise classes | Reference architecture if SilentVox later needs target-speaker isolation (e.g. radio operator vs. bystanders) rather than just denoising |
| Westhausen & Meyer, *Dual-Signal Transformation LSTM Network for Real-Time Noise Suppression* (DTLN) | 2020 | Real-time, low-parameter speech enhancement for the DNS Challenge | Two stacked LSTM stages: STFT-domain + learned-basis time-domain | DNS Challenge (500 h noisy speech) | MOS (+0.24 absolute over DNS baseline) | True streaming: one frame in, one frame out | State-of-the-art-competitive with <1M parameters | Still LSTM-based (sequential, less parallelizable than a CNN at inference) | Directly informs M3's parameter-budget target; strong evidence a sub-1M-parameter model is sufficient |
| Stamenovic, Westhausen, Yang, Jensen & Pawlicki, *Weight, Block or Unit? Exploring Sparsity Tradeoffs for Speech Enhancement on Tiny Neural Accelerators* | 2021 | Compress a DTLN-class model for extremely constrained hardware | Weight-pruned DTLN variant | (DTLN-derived training setup) | SDR (0.1 dB loss after pruning) | Not separately re-benchmarked in the cited abstract | 3.7 MB → 87 KB (42x compression) at 0.1 dB SDR cost | Aggressive pruning requires careful retraining/fine-tuning | Existence proof that SilentVox's speech-enhancement model can target a model-size budget in the same league as the M3 CNN classifier (Person 2's P4), not just the ANC filter |

---

## 6. What This Document Does Not Cover

Per the M3-P1 boundary, no models were trained or benchmarked here — Section
5's Latency/Result columns are the figures reported in each cited paper's own
experiments, not numbers produced by SilentVox. Actual training, mixture
generation, and evaluation on SilentVox's own data are M3-P2 and later
M3 sub-tasks respectively.

See `DATASET_RESEARCH.md` for the dataset survey and `REFERENCES.md` for the
full reference list.
