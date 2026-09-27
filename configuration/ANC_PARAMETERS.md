# ANC Parameters

## Primary configuration

| Parameter | Value |
|---|---:|
| Sampling rate | 16000 Hz |
| Primary algorithm | NLMS |
| Filter length | 32 |
| NLMS step size | 0.001 |
| NLMS epsilon | 1e-08 |

## LMS

| Parameter | Value |
|---|---:|
| Filter length | 32 |
| Step size | 0.001 |
| Epsilon | 1e-08 |

## FxLMS

| Parameter | Value |
|---|---:|
| Sampling rate | 16000 Hz |
| Filter length | 64 |
| Learning rate | 1e-05 |
| Secondary-path delay | 8 samples |
| Secondary-path delay | 0.5 ms |
| Secondary-path attenuation | 0.7 |
| Secondary-path FIR | [0.20, 0.30, 0.30, 0.15, 0.05] |

## Validation status

LMS: executed successfully.

NLMS: executed successfully.

FxLMS: executed successfully using a simulated secondary path.

Secondary path: SIMULATED.

Measured secondary-path identification: not claimed.

Physical closed-loop ANC validation: not claimed.
