# Competitor Research: Electronic & Adaptive Hearing Protection Systems

**Focus:** Documented technical capabilities (not marketing claims) of Peltor-type and INVISIO-type systems versus the research questions SilentVox is investigating.

---

## 1. Peltor-Type Systems (3M PELTOR family)

### What is actually documented

| Feature | Documented evidence | Source type |
|---------|---------------------|-------------|
| Passive attenuation | SNR / HML / frequency-dependent mean attenuation + SD tables (EN 352) | Product data sheets, EU type-examination |
| Level-dependent (talk-through) | Omni-directional external microphones + compression circuitry that amplifies low-level sounds and limits high-level continuous or impulsive noise | Technical datasheets (ComTac V, Tactical XP, Sport Tactical 500, etc.) |
| Impulse protection | Impulse Peak Insertion Loss (IPIL) measured at 160-170 dBP; e.g., ComTac V ~25-28 dB IPIL | Independent military lab reports (DTIC) |
| Communication integration | Boom or in-ear microphones, radio push-to-talk, Bluetooth in some models | Product literature + independent SI testing |
| Active Noise Reduction (ANR) | Present in selected models; primarily low-frequency continuous-noise reduction (classic feedback/feed-forward ANR) | Limited quantitative public data; mostly "ANR available" statements |
| Speech intelligibility | Modified Rhyme Test (MRT) scores under continuous noise (up to ~105 dBA) show improvement over passive foam when level-dependent mode is active | Peer-reviewed / DTIC military studies |
| Situational awareness | Stereo external microphones; user-adjustable volume | Product descriptions |

**What is not documented in public technical literature**

- Real-time adaptive filter coefficients or online secondary-path modeling.
- Explicit speech-preserving loss functions or selective cancellation of specific noise classes while protecting speech.
- Machine-learning classification of noise type (stationary / impulsive / speech).
- Detailed latency budgets or edge-DSP architecture beyond "electronic circuitry."
- Performance under rapidly changing non-stationary noise beyond level-dependent compression.

### Representative products examined

- 3M PELTOR ComTac V (level-dependent, IPIL data available).
- PELTOR Tactical XP / ProTac / LiteCom series (SNR tables, radio integration).
- Electronic earplugs (EEP-100) with level-dependent function.

---

## 2. INVISIO-Type Systems

### What is actually documented

| Feature | Documented evidence | Source type |
|---------|---------------------|-------------|
| Hearing protection | SNR 28-39 dB (single layer); dual-layer options up to ~43 dB SNR with hear-through retained | Product pages + EN 352 / ANSI data |
| Active Noise Reduction | Explicit ANR on T30 / RA-series; protects low-frequency continuous noise while preserving situational awareness | Product technical descriptions |
| Bone-conduction / Voice Pick-up Sensor (VPU) | Jawbone or in-ear vibration sensing for speech; external noise largely rejected at the microphone itself | X5 / X7 product literature |
| AI-powered audio | "INVISIO Audio" with AI enhancements for speech quality, automatic RX volume adjustment, noise filtering on transmit | Product announcements (X7, T30) |
| Situational awareness | Multiple hear-through modes; claimed localization accuracy (e.g., 9° on T30) | Product claims backed by acoustic design descriptions |
| Impulse / high-noise | Level-dependent + dual-protection modes; submersible designs | MIL-STD-810G certification statements |
| Radio integration | Multi-net control units (V60 etc.), VOX, whisper mode | System architecture documents |

**What is not documented in public technical literature**

- Detailed adaptive-filter algorithms (FxLMS order, step-size, secondary-path identification method).
- Open description of the AI model architecture, training data, or latency.
- Quantitative comparison of speech intelligibility with versus without the adaptive/AI stage under controlled impulsive + continuous noise.
- Power consumption or computational load of the onboard processing.
- Peer-reviewed independent measurements of residual speech distortion after ANR/AI processing.

### Representative products examined

- INVISIO T30 (over-ear with ANR + dual-protection option).
- INVISIO X5 / X7 (in-ear bone-conduction / VPU).
- INVISIO T7 and associated control units.

---

## 3. Broader Adaptive / Electronic Hearing Protection Landscape

Military and industrial literature (DTIC reports, NATO studies, peer-reviewed HPD evaluations) consistently describes two dominant technical approaches:

1. **Level-dependent (non-linear) electronic HPDs**  
   - Pass low-level ambient sound and speech.  
   - Compress or clip high-level continuous and impulsive noise.  
   - No continuous adaptive filtering of the residual noise spectrum.

2. **Classic ANR (feedback or feed-forward)**  
   - Effective mainly below ~500-1000 Hz.  
   - Usually fixed or slowly adapting controllers.  
   - Communication audio is mixed into the same path; ANR may be switched off if instability occurs.

Hybrid systems that combine (1) and (2) exist, but public documentation rarely discloses adaptive-filter details or speech-preservation metrics beyond overall SNR and MRT scores.

---

## 4. Gap Analysis: Competitor Documentation vs. SilentVox Research Questions

| SilentVox investigation focus | What competitors document | Gap |
|------------------------------|---------------------------|-----|
| Real-time adaptive (FxLMS-class) cancellation of continuous + non-stationary noise | Classic ANR or level-dependent compression | Adaptive tracking algorithms and secondary-path handling are not publicly detailed |
| Explicit speech-preserving cancellation (protect radio / face-to-face speech while attenuating noise) | Speech intelligibility measured post-hoc; no explicit speech-retention objective | No published speech-preservation loss or constrained ANC formulation |
| Impulsive-noise suppression that does not destroy subsequent speech | IPIL / peak clipping data | Limited evidence of adaptive recovery after impulse or selective impulse classification |
| Edge / embedded implementation constraints (latency, power, form-factor) | Battery life and weight figures; little algorithmic complexity data | No open benchmarks of algorithmic load on the actual headset DSP |
| ML-based noise classification to switch cancellation modes | "AI-powered audio" claims (INVISIO) | Architecture, training regime, and real-time classification accuracy not disclosed |
| Quantitative residual speech distortion after cancellation | MRT / PESQ occasionally reported for whole system | Rarely isolated to the adaptive stage itself |

---

## 5. Summary Statement

Commercial military-grade systems (Peltor-type and INVISIO-type) deliver robust passive attenuation, level-dependent impulse protection, radio integration, and, in selected models, classic low-frequency ANR plus proprietary AI post-processing. Their public technical documentation emphasizes certified attenuation numbers, speech-intelligibility scores under standardized tests, and operational features (hear-through modes, dual protection, bone conduction).

They do **not** publicly document:

- continuous real-time adaptive filtering of the type analyzed by Widrow/Morgan/Kuo,
- explicit speech-preserving objective functions,
- open ML classification pipelines for noise-type-dependent cancellation,
- detailed edge-implementation metrics for adaptive algorithms.

SilentVox is therefore investigating a region-speech-preserving, adaptive, potentially ML-augmented, edge-realizable ANC for communication hearing protection-that sits beyond the currently published technical baselines of these product families.