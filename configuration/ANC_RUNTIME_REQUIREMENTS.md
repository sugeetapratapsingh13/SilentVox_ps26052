\# ANC Runtime Requirements



\## 1. Target Platform



Target embedded platform: Raspberry Pi 5



The ANC engine is intended for real-time audio processing with low processing

latency and continuous block-based operation.



\---



\## 2. Audio Configuration



| Parameter | Requirement |

|---|---|

| Sampling Rate | 16 kHz |

| Audio Channels | Mono |

| Audio Format | 16-bit PCM |

| Processing Mode | Block-based real-time |

| Reference Signal | Environmental noise reference |

| Desired Signal | Noisy environmental audio |



\---



\## 3. Adaptive Filter Configuration



| Parameter | LMS | NLMS |

|---|---:|---:|

| Filter Length | 128 | 128 |

| Step Size (μ) | 0.0001 | 0.001 |

| Epsilon | Not used in standard LMS update | Tested: 1e-10 to 1e-4 |

| Tested Block Sizes | 256, 512, 1024 | 256, 512, 1024 |



\---



\## 4. Real-Time Performance Measurements



Measurements were obtained using 16 kHz, 80,000-sample audio.



\### LMS



| Block Size | Average Latency (ms) | Maximum Latency (ms) | RTF | Status |

|---:|---:|---:|---:|---|

| 256 | 0.2925 | 0.8491 | 54.69 | PASS |

| 512 | 0.8288 | 1.5654 | 38.61 | PASS |

| 1024 | 2.1545 | 3.7247 | 29.71 | PASS |



\### NLMS



| Block Size | Average Latency (ms) | Maximum Latency (ms) | RTF | Status |

|---:|---:|---:|---:|---|

| 256 | 0.4102 | 0.9044 | 39.01 | PASS |

| 512 | 1.2167 | 2.2675 | 26.30 | PASS |

| 1024 | 3.0429 | 3.8340 | 21.03 | PASS |



RTF (Real-Time Factor) is the ratio between audio duration and processing

time. Values greater than 1 indicate that processing completed faster than

the audio playback duration.



\---



\## 5. Resource Measurements



The measured Windows process resource usage was:



| Algorithm | CPU Range | RAM Range |

|---|---:|---:|

| LMS | 84.5–102.5% | 68.05–68.33 MB |

| NLMS | 97.5–105.3% | 68.34 MB |



CPU values are process CPU percentages reported by the Windows measurement

environment and should not be interpreted as Raspberry Pi 5 system-wide CPU

usage.



\---



\## 6. Epsilon Testing



NLMS was tested using:



\- 1e-10

\- 1e-8

\- 1e-6

\- 1e-4



All tested epsilon values produced:



\- Stable output

\- 0.1566 dB measured noise reduction under the epsilon test configuration



Processing times varied between approximately 0.283 s and 0.333 s.



The epsilon parameter therefore requires no further tuning based on the

current measured dataset.



\---



\## 7. Real-Time Requirements



The ANC implementation should satisfy the following engineering requirements:



1\. Sampling rate: 16 kHz

2\. Mono audio processing

3\. Block-based processing

4\. Processing time must remain below the duration of the processed audio block

5\. Real-time factor must remain greater than 1

6\. Output must remain numerically stable

7\. Processing latency should remain low enough for real-time audio operation

8\. CPU and RAM usage should be measured on the target Raspberry Pi 5 during

&#x20;  final deployment

9\. The final embedded implementation should preserve the tested filter length,

&#x20;  step size, and sampling rate unless further hardware testing requires changes



\---



\## 8. Current Measured Configuration



Current measured configurations:



\### LMS



\- Sampling rate: 16 kHz

\- Filter length: 128

\- Step size: 0.0001

\- Block sizes tested: 256, 512, 1024

\- Maximum measured latency: 3.7247 ms

\- Real-time status: PASS



\### NLMS



\- Sampling rate: 16 kHz

\- Filter length: 128

\- Step size: 0.001

\- Epsilon: 1e-8 in the real-time benchmark

\- Block sizes tested: 256, 512, 1024

\- Maximum measured latency: 3.8340 ms

\- Real-time status: PASS



\---



\## 9. Deployment Validation



The current measurements were obtained on the development computer.



Before final deployment, the same benchmark should be repeated on the

Raspberry Pi 5 to measure:



\- Actual CPU utilization

\- Actual RAM utilization

\- End-to-end audio latency

\- Block processing time

\- Real-time factor

\- Long-duration stability



These measurements will provide the final hardware-specific runtime

validation.

