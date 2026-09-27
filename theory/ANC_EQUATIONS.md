# ANC Equations — SilentVox
## Person 1: Mathematical Specification

> **Sign convention:** This document uses `e(n) = d(n) - y(n)` for the simplified LMS/NLMS baseline. All implementations should use one consistent convention. Physical ANC with a secondary path requires the FxLMS formulation described below.

---

## 1. Reference vector

For an L-tap adaptive FIR filter:

`x(n) = [x(n), x(n-1), ..., x(n-L+1)]^T`

The coefficient vector is:

`w(n) = [w_0(n), w_1(n), ..., w_{L-1}(n)]^T`

---

## 2. Adaptive-filter output

The FIR output is:

`y(n) = w^T(n)x(n)`

Expanded:

`y(n) = sum_{k=0}^{L-1} w_k(n)x(n-k)`

---

## 3. Error

For the simplified baseline model:

`e(n) = d(n) - y(n)`

Substituting the filter output:

`e(n) = d(n) - w^T(n)x(n)`

---

## 4. Instantaneous cost function

Define:

`J(n) = 1/2 e^2(n)`

The factor `1/2` is convenient because it cancels the factor of 2 introduced by differentiation.

The optimization objective is to reduce the cost:

`min J(n)`

---

## 5. LMS gradient derivation

Starting with:

`J(n) = 1/2 e^2(n)`

Differentiate with respect to `w`:

`grad_w J(n) = e(n) grad_w e(n)`

From:

`e(n) = d(n) - w^T(n)x(n)`

we obtain:

`grad_w e(n) = -x(n)`

Therefore:

`grad_w J(n) = -e(n)x(n)`

---

## 6. Gradient descent

The generic gradient-descent update is:

`w(n+1) = w(n) - mu grad_w J(n)`

Substitute the LMS gradient:

`w(n+1) = w(n) - mu[-e(n)x(n)]`

Therefore:

`w(n+1) = w(n) + mu e(n)x(n)`

### Final LMS equation

`w(n+1) = w(n) + mu e(n)x(n)`

---

## 7. Scalar LMS coefficient update

For each coefficient `k`:

`w_k(n+1) = w_k(n) + mu e(n)x(n-k)`

for:

`k = 0, 1, ..., L-1`

---

## 8. NLMS normalization

Define the input-vector energy:

`E_x(n) = ||x(n)||^2`

and:

`||x(n)||^2 = x^T(n)x(n)`

NLMS normalizes the LMS update using the current input energy.

### Final NLMS equation

`w(n+1) = w(n) + [mu / (epsilon + ||x(n)||^2)] e(n)x(n)`

where:

`epsilon > 0`

---

## 9. Scalar NLMS update

For each coefficient:

`w_k(n+1) = w_k(n) + [mu e(n)x(n-k)] / [epsilon + sum_{j=0}^{L-1} x^2(n-j)]`

---

## 10. Meaning of parameters

### mu

Adaptation step size.

Controls update magnitude.

### epsilon

Small positive regularization constant in NLMS.

Prevents division by zero or an excessively small denominator.

### L

Adaptive filter length.

Number of coefficients.

### Sampling rate

Number of samples per second.

### e(n)

Residual/error signal used by the adaptation algorithm.

---

## 11. LMS stability/convergence reference

A commonly used mean-convergence condition for standard LMS is:

`0 < mu < 2 / lambda_max(R)`

where:

`R = E[x(n)x^T(n)]`

is the input autocorrelation matrix and `lambda_max(R)` is its largest eigenvalue.

This condition is theoretical and depends on the input statistics. Experimental parameter selection is therefore still required.

---

## 12. FxLMS

Let the true secondary path be:

`S(z)`

and its model/estimate be:

`S_hat(z)`

The filtered reference is:

`x_f(n) = S_hat(z) * x(n)`

where `*` denotes filtering/convolution.

The adaptive update under the project's adopted sign convention is:

`w(n+1) = w(n) + mu e(n)x_f(n)`

The secondary-path estimate is essential because the physical anti-noise is altered before reaching the error microphone.

---

## 13. FxNLMS conceptual extension

FxNLMS uses the filtered reference together with normalization. A project implementation should explicitly define:

1. the filtered-reference vector;
2. the normalization energy;
3. epsilon;
4. the update sign convention;
5. the secondary-path model.

These details must be documented alongside the implementation so that results remain reproducible.

---

## 14. Convergence metrics for experiments

Person 3/4/6 may use:

- instantaneous squared error: `e^2(n)`;
- moving-average error power;
- mean-square error (MSE);
- convergence time;
- residual noise level;
- processing time.

A convergence curve should state clearly what quantity is plotted and over what time/sample axis.

---

## 15. Parameter relationships

Increasing `mu`:
- increases adaptation step size;
- can speed initial adaptation;
- can increase excess error;
- can cause instability if too large.

Increasing `L`:
- increases model capacity;
- increases computation and memory;
- changes convergence behavior.

Changing sampling rate:
- changes time represented by each sample;
- changes available frequency range;
- changes computational workload and latency requirements.

Changing `epsilon`:
- changes the regularization of NLMS normalization;
- should be tested experimentally rather than selected solely by intuition.

---

## 16. Implementation contract for Person 3

### LMS

Inputs:
- audio/reference signal;
- desired/disturbance signal or equivalent baseline input representation;
- sampling rate;
- filter length `L`;
- step size `mu`.

State:
- coefficient vector `w`.

Per sample:
1. build reference vector;
2. calculate `y(n)`;
3. calculate `e(n)`;
4. update `w`.

Output:
- filtered/control output;
- residual/error audio;
- adaptive coefficients;
- convergence metrics.

### NLMS

Same baseline contract, with:
- epsilon `epsilon`;
- input-energy calculation;
- normalized coefficient update.

---

## 17. Implementation contract for Person 5

For FxLMS:
- define/estimate secondary path;
- filter reference through the secondary-path estimate;
- construct filtered-reference vector;
- use filtered reference in adaptation;
- document delay/filter assumptions;
- compare simulated and measured secondary-path models when hardware becomes available.
