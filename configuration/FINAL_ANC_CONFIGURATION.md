\# FINAL ANC CONFIGURATION



\## 1. Project Configuration



Project: SilentVox Adaptive Noise Cancellation (ANC)



Target Platform: Raspberry Pi 5



Audio Processing: Real-Time, Block-Based



Sampling Rate: 16 kHz



Audio Channels: Mono



Audio Format: 16-bit PCM



\---



\## 2. Tested Algorithms



Three ANC algorithms were evaluated:



\- LMS

\- NLMS

\- FxLMS



The comparison is based on measured experimental results.



\---



\## 3. LMS Configuration



\- Algorithm: LMS

\- Sampling Rate: 16 kHz

\- Filter Length: 128 taps

\- Step Size (μ): 0.0001

\- Processing Time: 0.1901 s

\- Measured Noise Reduction: 1.3038 dB

\- Real-Time Testing: PASS



\---



\## 4. NLMS Configuration



\- Algorithm: NLMS

\- Sampling Rate: 16 kHz

\- Filter Length: 128 taps

\- Step Size (μ): 0.001

\- Epsilon: 1e-8

\- Processing Time: 0.2696 s

\- Measured Noise Reduction: 1.2992 dB

\- Real-Time Testing: PASS



\---



\## 5. FxLMS Configuration



\- Algorithm: FxLMS

\- Sampling Rate: 16 kHz

\- Filter Length: 64 taps

\- Step Size (μ): 1e-5

\- Secondary-path model: Tested

\- Secondary-path delay: 8 samples

\- Processing Time: 0.2083 s

\- Measured Noise Reduction: 1.9492 dB

\- Stability: Stable

\- Real-Time Factor: 14.40

\- Benchmark Status: PASS



Note: FxLMS was evaluated using its existing sample-by-sample simulation.

Therefore, its latency measurement is not directly comparable with the

block-based LMS/NLMS latency measurements.



\---



\## 6. LMS Real-Time Measurements



| Block Size | Average Latency | Maximum Latency | RTF |

|---:|---:|---:|---:|

| 256 | 0.2925 ms | 0.8491 ms | 54.69 |

| 512 | 0.8288 ms | 1.5654 ms | 38.61 |

| 1024 | 2.1545 ms | 3.7247 ms | 29.71 |



\---



\## 7. NLMS Real-Time Measurements



| Block Size | Average Latency | Maximum Latency | RTF |

|---:|---:|---:|---:|

| 256 | 0.4102 ms | 0.9044 ms | 39.01 |

| 512 | 1.2167 ms | 2.2675 ms | 26.30 |

| 1024 | 3.0429 ms | 3.8340 ms | 21.03 |



All tested LMS and NLMS block configurations passed the real-time benchmark.



\---



\## 8. Parameter Optimization



The parameter experiments included:



\- Filter length

\- Step size (μ)

\- Epsilon

\- Block size

\- LMS

\- NLMS

\- FxLMS



LMS and NLMS parameter sweeps were performed using the project audio

benchmark dataset.



NLMS epsilon values tested:



\- 1e-10

\- 1e-8

\- 1e-6

\- 1e-4



All tested epsilon values were stable.



\---



\## 9. Resource Measurements



Measured development-system process usage:



| Algorithm | CPU | RAM |

|---|---:|---:|

| LMS | 84.5–102.5% | 68.05–68.33 MB |

| NLMS | 97.5–105.3% | 68.34 MB |

| FxLMS | Not measured | Not measured |



CPU and RAM values were measured on the Windows development computer.



They are not Raspberry Pi 5 system-wide measurements.



\---



\## 10. Current Runtime Requirements



Target:



\- Sampling Rate: 16 kHz

\- Channels: Mono

\- Processing: Block-based

\- Tested Block Sizes: 256, 512, 1024 samples

\- Filter Length: 128 taps for LMS/NLMS

\- LMS μ: 0.0001

\- NLMS μ: 0.001

\- NLMS ε: 1e-8



For Raspberry Pi 5 integration, the final implementation should be benchmarked

directly on the target hardware.



\---



\## 11. Algorithm Comparison Status



| Algorithm | Implementation | Parameter Testing | Real-Time Testing |

|---|---|---|---|

| LMS | Complete | Complete | Complete |

| NLMS | Complete | Complete | Complete |

| FxLMS | Complete | Tested baseline | Tested baseline |



The measurements are documented in:



\- ALGORITHM\_DECISION\_MATRIX.csv

\- REALTIME\_BENCHMARK.csv

\- EPSILON\_SWEEP.csv

\- FXLMS\_BENCHMARK.csv

\- results/parameter\_sweep.csv

\- results/FINAL\_ANC\_SUMMARY.csv



\---



\## 12. Raspberry Pi 5 Validation



The following must be measured again on the actual Raspberry Pi 5:



\- CPU utilization

\- RAM usage

\- End-to-end latency

\- Block processing time

\- Real-time factor

\- Long-duration stability

\- Audio quality



This is required because current CPU and RAM measurements were obtained on the

development computer.



\---



\## 13. Final Person 6 Deliverable Status



Experiment matrix: COMPLETE



Parameter experimentation: COMPLETE



LMS testing: COMPLETE



NLMS testing: COMPLETE



FxLMS testing: COMPLETE



Processing-time measurement: COMPLETE



CPU measurement: COMPLETE on development system



RAM measurement: COMPLETE on development system



Latency measurement: COMPLETE for LMS/NLMS block benchmark



Stability measurement: COMPLETE



Noise-reduction measurement: COMPLETE



Algorithm decision matrix: COMPLETE



Runtime requirements: COMPLETE



Final configuration: COMPLETE



Raspberry Pi 5 hardware validation: PENDING



Overall Person 6 software benchmarking work: COMPLETE

