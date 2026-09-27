\# M3-P3 Classical Enhancement Baseline Report



\## 1. Purpose



This work establishes the classical, non-neural speech-enhancement baseline required before evaluating a neural enhancement system.



The baseline compares:



1\. No enhancement

2\. Wiener filtering

3\. Spectral suppression / spectral subtraction



The purpose is to establish reproducible reference performance for classical enhancement methods.



This work does \*\*not\*\* implement or evaluate the neural enhancement model.



\---



\## 2. Experimental Boundary



M3-P3 is restricted to classical speech-enhancement baselines.



\### Included



\- No-enhancement reference

\- Wiener filtering

\- Spectral suppression

\- Objective speech-enhancement metrics

\- Baseline comparison plots

\- Reproducible audio outputs



\### Not included



\- Neural speech enhancement

\- Neural network training

\- Neural model inference

\- Neural architecture design

\- Neural model optimization



\---



\## 3. Dataset



The experiments use the audio files available in the project `input\_audio` directory.



All four noisy speech conditions and their corresponding noise-only recordings are sampled at:



\- Sampling rate: 16 kHz

\- Duration: 5 seconds

\- Samples: 80,000



The clean reference is:



```text

clean\_signal.wav

