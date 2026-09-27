# SilentVox Literature Review
## Adaptive Noise Control for Speech-Preserving Military Communication Hearing Protection

**Person 2 - Literature + Research**  
**Date:** 2026-09-24

---

## 1. Foundational Adaptive Filtering and Noise Cancelling

### 1.1 Widrow & Hoff (1960) - Adaptive Switching Circuits / LMS Algorithm

Widrow and Hoff introduced the Least Mean Squares (LMS) algorithm in the context of the ADALINE (Adaptive Linear Neuron). The work demonstrated that a simple stochastic-gradient update could iteratively adjust linear combiner weights to minimize mean-square error between a desired response and the filter output. The update rule

    w(n+1) = w(n) + mu * e(n) * x(n)

requires only instantaneous estimates of the gradient and is computationally lightweight. This algorithm became the workhorse of adaptive signal processing and the direct ancestor of virtually all adaptive noise-cancelling and active-noise-control (ANC) schemes.

**Relevance to SilentVox:** LMS (and its filtered-x variants) remains the core real-time adaptation engine for any edge-deployable ANC system that must track non-stationary noise while remaining computationally tractable on embedded hardware.

### 1.2 Widrow et al. (1975) - Adaptive Noise Cancelling: Principles and Applications

This landmark paper formalized the two-input adaptive noise canceller: a primary channel containing signal-plus-noise and a reference channel containing noise correlated with the primary interference. The adaptive filter (typically LMS) produces an estimate of the primary noise that is subtracted, yielding a cleaned signal. Key theoretical results include:

- Wiener-solution analysis for stationary stochastic inputs;
- output SNR equal to the reciprocal of the reference-input SNR (under ideal conditions);
- the adaptive notch-filter interpretation for periodic interference (infinite null, frequency tracking).

Experimental demonstrations covered ECG artifact removal, speech enhancement, and antenna sidelobe cancellation.

**Relevance to SilentVox:** Establishes the classic primary/reference architecture that underpins both feed-forward ANC and speech-preserving "noise-cancelling" modes in communication headsets. Any SilentVox design that retains a reference microphone inherits this framework.

### 1.3 Morgan (1980) - Analysis of Multiple Correlation Cancellation Loops with a Filter in the Auxiliary Path

Morgan extended the LMS analysis to the practical situation in which a linear filter (the secondary path) appears between the adaptive filter output and the error sensor. The resulting filtered-x LMS (FxLMS) algorithm filters the reference signal by an estimate of the secondary path before the weight update. This paper supplied the rigorous convergence analysis that later became standard in ANC literature.

**Relevance to SilentVox:** FxLMS is the default algorithm for real-time acoustic ANC. Accurate secondary-path modeling (or robust online identification) is a prerequisite for stable, low-latency cancellation inside a headset or helmet.

### 1.4 Kuo & Morgan (1996) - Active Noise Control Systems: Algorithms and DSP Implementations

This monograph is the definitive engineering reference for digital ANC. It systematically treats:

- broadband and narrowband feed-forward ANC;
- adaptive feedback ANC;
- multi-channel extensions;
- online secondary-path modeling;
- leaky, normalized, and frequency-domain LMS variants;
- practical DSP implementations (primarily TMS320 family).

The book bridges theory and real-time realization, emphasizing computational cost, causality constraints, and stability under plant uncertainty.

**Relevance to SilentVox:** Provides the algorithmic and implementation baseline against which any modern edge/embedded ANC solution must be compared. SilentVox's real-time, power-constrained targets map directly onto the DSP-centric design philosophy of this text.

---

## 2. Contemporary Research Themes

### 2.1 Real-Time and Adaptive ANC

Modern work focuses on reducing latency to the sub-millisecond regime required for causal cancellation of broadband noise at the ear. Techniques include:

- optimized FxLMS on embedded GPGPUs or dual-core DSPs (e.g., TMS320F28379D for helmet ANC);
- selective fixed-filter ANC driven by CNN classifiers that switch among pre-trained filters without continuous adaptation;
- hybrid adaptive + deep-learning controllers that combine the tracking ability of LMS with the modeling power of recurrent networks.

### 2.2 Speech-Preserving ANC

Conventional ANC treats speech as residual noise and can degrade intelligibility. Recent approaches explicitly protect the speech component:

- deep end-to-end controllers (CRN + complex spectral mapping) with speech-retention loss functions;
- linearly-constrained minimum-variance (LCMV) ANC that enforces spatial or spectral constraints on desired signals;
- selective cancellation guided by real-time sound-event classification (sirens, machinery, gunfire vs. speech).

These methods are directly relevant to military communication headsets, where radio traffic and face-to-face speech must remain intelligible while continuous or impulsive noise is suppressed.

### 2.3 Military Communication Hearing Protection & Impulsive-Noise Suppression

Military literature emphasizes:

- level-dependent (non-linear) attenuation that passes low-level speech/ambient cues while compressing high-level impulses (weapon fire, blasts);
- dual-protection configurations (in-ear + over-ear) that raise continuous-noise attenuation without complete occlusion;
- integration of radio audio into the same signal path as active attenuation so that communication remains available when ANR is engaged or fails.

Documented performance metrics are typically SNR/NRR/APV attenuation tables, impulse peak insertion loss (IPIL), and speech intelligibility scores (Modified Rhyme Test, MRT) under continuous and impulsive noise.

### 2.4 Edge/Embedded ANC and Machine-Learning Classification

Edge implementations on Cortex-M, dual-core C2000, or specialized audio NPUs demonstrate that both classical LMS and lightweight neural networks can run at sampling rates >= 32-48 kHz with power budgets compatible with battery-powered headsets. Classification of noise type (stationary, non-stationary, impulsive) enables mode switching or selective filter banks, reducing the need for continuous high-order adaptation.

---

## 3. Synthesis for SilentVox

The foundational papers supply the mathematical and algorithmic core (LMS → FxLMS → multi-channel ANC). Contemporary research shows that:

1. pure adaptive cancellation is mature but can harm speech;
2. speech-preserving and selective-cancellation strategies are emerging but still largely laboratory or high-end consumer prototypes;
3. military systems prioritize impulse protection, situational awareness, and radio integration over aggressive continuous-noise cancellation;
4. edge hardware is now capable of hosting both classical and hybrid ML-enhanced controllers.

SilentVox's investigation therefore sits at the intersection of real-time adaptive cancellation, explicit speech preservation, impulsive-noise handling, and low-power embedded realization-precisely the region where commercial military headsets still rely predominantly on level-dependent compression rather than sophisticated adaptive or learning-based controllers.

---

## References

See `CITATIONS.bib` for full bibliographic entries.