# SilentVox Stage 2 — Final PC Software Integration Results

## Status

Stage 2 PC-based software integration of M1, M2, and M3 is complete.

Pipeline:

Audio Input
→ M2 Noise Classification
→ M1 FxLMS ANC
→ M3 CNN Speech Enhancement
→ Final Audio

This validates software interoperability on a PC. It is not physical headset, Raspberry Pi, or physical acoustic-loop validation.

## 1. End-to-End Integration

Four WAV test cases were executed through the complete M2 → M1 → M3 pipeline.

| Test | M2 label | Confidence | M1 residual RMS | M3 output RMS | Total time |
|---|---|---:|---:|---:|---:|
| Stationary | RAIN | 0.3756 | -14.872 dBFS | -28.610 dBFS | 0.519267 s |
| Rotor | ENGINE | 0.4020 | -17.299 dBFS | -23.790 dBFS | 0.438840 s |
| Impulsive | WIND | 0.4712 | -28.388 dBFS | -29.821 dBFS | 0.442756 s |
| Speech + noise | ENGINE | 0.2883 | -18.341 dBFS | -22.841 dBFS | 0.433960 s |

All inputs were 16 kHz mono, 64,000 samples, and 4 seconds long.

All four end-to-end runs completed successfully and produced 64,000-sample M3 output.

## 2. M1 Validation

M1 quantitative validation was completed independently using the Stage-2 validation inputs.

The existing M1 validation log is:

integration/m1_validation_results.txt

Measured PC processing times were below the 4-second duration of the test inputs.

M1 remains responsible for ANC processing and adaptive-filter parameters.

## 3. M2 Validation

M2 classification was executed at the integration boundary for all four test inputs.

Observed outputs:

- stationary: RAIN, confidence 0.3756
- rotor: ENGINE, confidence 0.4020
- impulsive: WIND, confidence 0.4712
- speech + noise: ENGINE, confidence 0.2883

M2 inference timing was measured for every end-to-end run.

These results demonstrate that M2 produces a usable classification result and confidence value for the integration pipeline. They do not constitute a separate classifier accuracy evaluation.

## 4. M3 Validation

M3 uses the existing SpeechEnhancementCNN model and existing STFT preprocessing.

Model:

SpeechEnhancementCNN

Parameter count:

18,817

The independent M3 validation log is:

integration/m3_validation_results.txt

The integrated speech-path evaluation compares the final M3 output against the available clean-speech reference.

## 5. Integrated Speech Metrics

Reference:

M4/virtual_mics/speech_clean.wav

Integrated path:

clean speech reference
→ speech + noise input
→ M2
→ M1
→ M3
→ integrated output

Measured metrics:

- STOI: 0.437609
- SI-SDR: 12.301356 dB

Metrics log:

integration/integrated_speech_metrics.txt

These metrics evaluate the available software integration path against the available clean-speech reference. They do not constitute physical headset or Raspberry Pi validation.

## 6. PC Continuous Block Benchmark

Block processing was tested at the required block sizes.

| Block size | Audio duration | Processing time | Real-time factor | Deadline |
|---:|---:|---:|---:|---|
| 256 | 0.016000 s | 0.009477 s | 0.592x | PASS |
| 512 | 0.032000 s | 0.006469 s | 0.202x | PASS |
| 1024 | 0.064000 s | 0.010507 s | 0.164x | PASS |

Processing time was below the available audio time for all three tested block sizes on the PC benchmark.

These are PC measurements only and must not be interpreted as Raspberry Pi real-time performance.

Benchmark log:

integration/block_benchmark_results.txt

## 7. End-to-End Log

Complete four-test end-to-end execution log:

integration/stage2_final_end_to_end_log.txt

Final speech-plus-noise output:

integration/stage2_output.wav

The pipeline currently reuses the same output filename for each invocation; therefore the final file represents the most recently executed test case.

## 8. Software Interfaces

### M2

Input:
- 16 kHz mono audio

Output:
- noise class
- confidence
- temporal/majority label where provided
- probability label where provided
- inference timing

### M1

Input:
- audio samples

Output:
- ANC residual audio

Current integrated implementation:
- FxLMS
- filter length: 64
- learning rate: 1e-5
- existing M1 native simulation secondary path

### M3

Input:
- M1 residual audio

Output:
- enhanced audio

M3 retains ownership of:
- STFT preprocessing
- CNN inference
- mask application
- reconstruction

## 9. Reproducibility

Example:

python .\integration\stage2_pipeline.py .\M4\virtual_mics\speech_plus_noise.wav

M1 validation:

python .\integration\m1_validation.py

M3 validation:

python .\integration\m3_validation.py

Block benchmark:

python .\integration\block_benchmark.py

Integrated speech metrics:

python .\integration\integrated_speech_metrics.py

## 10. Stage-2 Artifacts

- integration/stage2_pipeline.py
- integration/m1_validation.py
- integration/m1_validation_results.txt
- integration/m3_validation.py
- integration/m3_validation_results.txt
- integration/integrated_speech_metrics.py
- integration/integrated_speech_metrics.txt
- integration/block_benchmark.py
- integration/block_benchmark_results.txt
- integration/stage2_final_end_to_end_log.txt
- integration/stage2_output.wav
- integration/INTEGRATION_README.md

## 11. Limitations

Stage 2 does not include:

- Raspberry Pi validation
- physical headset validation
- physical microphone validation
- physical acoustic secondary-path measurement
- final hardware ANC-loop validation

The sequential M2 → M1 → M3 pipeline is a PC software integration test and is not claimed to represent the final physical acoustic signal-routing architecture.

## 12. Stage-2 Conclusion

M1, M2, and M3 have been integrated and tested as a PC-based software pipeline.

The complete software path executes successfully:

M2 Classification
→ M1 ANC
→ M3 Speech Enhancement
→ Final Audio

End-to-end execution, quantitative module validation, continuous block processing at 256/512/1024 samples, and integrated speech-path STOI/SI-SDR evaluation have been completed.

The resulting interfaces and measurements provide the software integration baseline for subsequent M4/M5/M6 hardware integration.
