# ANC Theory — SilentVox
## Person 1: ANC Theory + Mathematics

### 1. Purpose

This document provides the theoretical foundation for the SilentVox active noise control (ANC) work. It defines the main ANC architectures, signals, acoustic paths, adaptive-filter concepts, and the concepts required for LMS, NLMS, and FxLMS implementation.

The purpose of this document is to provide a consistent specification for the implementation and experimental members of the team. It does not prescribe which algorithm is ultimately preferred; that decision must be based on measured experiments.

---

## 2. What is Active Noise Control?

Active Noise Control (ANC) is a signal-processing technique that reduces unwanted acoustic noise by generating a control signal, commonly called anti-noise, that interferes destructively with the unwanted sound at a target location.

A simplified conceptual relationship is:

`Residual = Unwanted noise - effective anti-noise`

The practical system must account for amplitude, timing, phase, delays, the acoustic environment, and the secondary path between the controller and the error microphone.

The ANC objective is to minimize the residual error rather than simply generate a signal that is the mathematical negative of the noise.

---

## 3. Fundamental ANC Signals

### 3.1 Reference signal x(n)

The reference signal is a measured signal correlated with the unwanted disturbance. In feedforward ANC, it is normally obtained from a reference sensor/microphone.

### 3.2 Desired/disturbance signal d(n)

`d(n)` denotes the unwanted disturbance at the error/cancellation location in the simplified mathematical model.

### 3.3 Adaptive-filter output y(n)

For an L-tap FIR adaptive filter:

`y(n) = w^T(n) x(n)`

where `w(n)` is the adaptive coefficient vector and `x(n)` is the reference-signal vector.

### 3.4 Error/residual signal e(n)

Under the simplified sign convention used for the LMS/NLMS baseline:

`e(n) = d(n) - y(n)`

The error is the quantity minimized by the adaptive algorithm.

In a physical ANC system, the anti-noise is modified by the secondary path before it reaches the error microphone. FxLMS explicitly accounts for this effect.

---

## 4. Feedforward ANC

Feedforward ANC uses a reference sensor to obtain information about the disturbance and generates a control signal before/while the disturbance reaches the cancellation location.

Conceptually:

`Reference -> Adaptive controller -> Anti-noise -> Secondary path -> Cancellation location`

The measured residual/error is fed back to the adaptation algorithm.

A suitable reference should contain useful information correlated with the disturbance.

---

## 5. Feedback ANC

Feedback ANC uses information derived from the residual/error signal to control the anti-noise generation. A separate external reference signal is not used in the same manner as in feedforward ANC.

Feedback ANC can be useful when a suitable external reference is unavailable, but controller design and stability become important considerations.

---

## 6. Hybrid ANC

Hybrid ANC combines feedforward and feedback structures. The feedforward branch can use an external reference while the feedback branch uses residual/error information.

The exact architecture should be defined by the eventual system design; this document does not assume that hybrid ANC is universally preferable.

---

## 7. Primary Acoustic Path P(z)

The primary path represents the acoustic path through which the unwanted disturbance travels from its source toward the error/cancellation location.

Conceptually:

`Noise source -> Primary path P(z) -> Error/cancellation location`

The primary path can include propagation, reflections, frequency-dependent attenuation, and delays.

---

## 8. Secondary Acoustic Path S(z)

The secondary path represents the path taken by the generated anti-noise from the controller output to the error sensor.

Depending on the system boundary, it may include:

- digital-to-analog conversion and associated processing;
- amplifier/electronic response;
- loudspeaker response;
- acoustic propagation;
- microphone response;
- analog-to-digital conversion;
- system delays.

The secondary path is central to physical ANC because it changes the anti-noise before it contributes to the measured error.

---

## 9. Adaptive Filters

An adaptive filter changes its coefficients during operation according to an adaptation rule.

For an L-tap FIR adaptive filter:

`x(n) = [x(n), x(n-1), ..., x(n-L+1)]^T`

`w(n) = [w_0(n), w_1(n), ..., w_{L-1}(n)]^T`

The output is:

`y(n) = w^T(n)x(n)`

The filter coefficients are updated to reduce the selected error/cost function.

---

## 10. LMS

Least Mean Squares (LMS) is a gradient-based adaptive-filter algorithm.

Using the instantaneous squared-error cost:

`J(n) = 1/2 e^2(n)`

and the simplified error:

`e(n) = d(n) - w^T(n)x(n)`

the stochastic-gradient update is:

`w(n+1) = w(n) + mu e(n)x(n)`

where `mu` is the adaptation step size.

The LMS implementation is the baseline algorithm assigned to Person 3.

---

## 11. NLMS

Normalized Least Mean Squares (NLMS) normalizes the adaptation update using the energy of the current input vector.

The baseline update is:

`w(n+1) = w(n) + [mu / (epsilon + ||x(n)||^2)] e(n)x(n)`

where:

- `mu` is the adaptation parameter;
- `epsilon > 0` prevents a zero or excessively small denominator;
- `||x(n)||^2 = x^T(n)x(n)` is the input-vector energy.

NLMS is the second baseline implementation assigned to Person 3.

---

## 12. LMS and NLMS Parameters

### Step size mu

Controls the magnitude of coefficient adaptation.

A smaller value generally produces smaller updates and slower adaptation. A larger value produces larger updates but can increase excess error and, depending on conditions, cause instability.

The final value must be established experimentally.

### Filter length L

The number of adaptive coefficients. A larger filter can represent longer relationships but increases computation and memory requirements.

### Epsilon

Used by NLMS to regularize the denominator when input energy is very small.

### Sampling rate

The number of samples processed per second. It affects representable frequency range, computational workload, latency, and the number of samples corresponding to a physical delay.

---

## 13. Convergence

Convergence describes the process by which the adaptive coefficients approach a useful solution and the error decreases toward a steady-state region.

Convergence speed depends on factors including:

- input signal statistics;
- step size;
- filter length;
- signal-to-noise conditions;
- secondary-path effects in physical ANC;
- implementation details.

Convergence must be measured experimentally rather than assumed from theory alone.

---

## 14. Stability

Stability refers to whether the adaptive process remains bounded rather than diverging.

For standard LMS, a commonly used mean-convergence condition is:

`0 < mu < 2 / lambda_max(R)`

where `R` is the input autocorrelation matrix and `lambda_max(R)` is its largest eigenvalue.

This is a theoretical condition for the corresponding model; it should not be treated as a universal fixed value for all practical signals and implementations.

---

## 15. Misadjustment

Misadjustment describes the excess steady-state mean-square error associated with adaptive operation relative to the minimum achievable error under the given conditions.

It represents a trade-off associated with continued adaptation and finite/noisy observations.

---

## 16. FxLMS Concept

In physical feedforward ANC, the controller-generated anti-noise passes through the secondary path before affecting the error microphone.

Therefore, the reference used for adaptation must account for an estimate of the secondary path.

Let `S_hat(z)` denote the estimated secondary path. The filtered reference is conceptually:

`x_f(n) = S_hat(z) * x(n)`

where `*` denotes filtering/convolution.

The corresponding update, using the sign convention adopted for the project, is:

`w(n+1) = w(n) + mu e(n)x_f(n)`

FxLMS implementation and secondary-path modelling are assigned to Person 5.

---

## 17. FxNLMS

FxNLMS combines the filtered-reference concept of FxLMS with normalized adaptation. Its exact implementation should use the same project sign convention and a clearly defined normalization term.

Person 5 is responsible for the implementation details after the secondary-path model is established.

---

## 18. SilentVox Relevance

For SilentVox, the theory establishes:

1. a common signal and sign convention;
2. LMS and NLMS as independently testable baseline adaptive algorithms;
3. the role of the reference, disturbance, control signal, and residual error;
4. the importance of the primary and secondary acoustic paths;
5. the mathematical motivation for FxLMS;
6. the parameters that later experiments must vary.

The theory does not determine the final preferred algorithm or parameter configuration. Those conclusions must come from the experimental and benchmarking work of Persons 3, 4, 5, and 6.

---

## 19. Scope Boundary

Person 1 owns the theoretical and mathematical specification.

Person 3 owns LMS/NLMS coding and baseline experiments.

Person 4 owns controlled simulation and noise-condition experiments.

Person 5 owns FxLMS and secondary-path modelling.

Person 6 owns parameter sweeps, real-time benchmarking, and final configuration based on measured evidence.
