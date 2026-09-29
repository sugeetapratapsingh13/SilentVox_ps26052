# M3-P6 → M6 Handoff Report

This is the measured M3-P6 deployment handoff to M6.
All completed deployment values below are taken from the accepted
M3-P6 ONNX export, ONNX Runtime validation, and deployment
optimization measurements performed on 29-09-2026.

## 1. Model Identity

| Field | Value |
|---|---|
| Model name | `SpeechEnhancementCNN` |
| PyTorch checkpoint | `M3-P4/model/trained_model.pth` |
| Deployment format | ONNX |
| Primary deployment artifact | `deployment/m3_speech_enhancement.onnx` |
| Parameter count | `18,817` |
| ONNX IR version | `10` |
| ONNX opset | `18` |
| Graph nodes | `10` |

## 2. Audio / STFT Contract

| Field | Value |
|---|---|
| Sample rate | `16,000 Hz` |
| Audio channels | Mono |
| Audio type at integration boundary | `float32` |
| Input integration shape | 1-D waveform |
| FFT size | `512` |
| Window length | `512` |
| Hop length | `128` |
| Frequency bins | `257` |

## 3. ONNX Tensor Contract

### Input

| Field | Value |
|---|---|
| Tensor name | `input` |
| Data type | `float32` |
| Shape | `[batch, 1, 257, time]` |
| Representation | Log-magnitude STFT |

### Output

| Field | Value |
|---|---|
| Tensor name | `mask` |
| Data type | `float32` |
| Shape | `[batch, 1, 257, time]` |
| Representation | Predicted enhancement mask |

The predicted mask is applied element-wise to the mixture magnitude.
The enhanced magnitude is reconstructed with the original mixture phase
using ISTFT.

## 4. Export Validation

### PyTorch → ONNX

| Measurement | Result |
|---|---:|
| ONNX checker | `PASS` |
| Maximum absolute output difference | `1.7881393433e-07` |
| Mean absolute output difference | `3.1553355484e-08` |

The exported ONNX model reproduced the PyTorch model within the
measured numerical tolerance.

### ONNX Runtime

Execution provider tested:

`CPUExecutionProvider`

Validation input:

`(1, 1, 257, 100)`

Validation output:

`(1, 1, 257, 100)`

Results:

- Output finite: `True`
- ONNX Runtime validation: `PASS`

## 5. Deployment Artifacts

### FP32 ONNX

`deployment/m3_speech_enhancement.onnx`

Measured size:

`16,577 bytes`

### External ONNX Data

`deployment/m3_speech_enhancement.onnx.data`

Measured size:

`74,880 bytes`

The `.onnx` artifact and associated external data file were generated
during the accepted export.

### INT8 ONNX

`deployment/m3_speech_enhancement_int8.onnx`

Measured size:

`33,817 bytes`

## 6. INT8 Optimization Experiment

Dynamic INT8 quantization was successfully performed and the resulting
model was validated with ONNX Runtime.

### Numerical comparison

| Measurement | Result |
|---|---:|
| Maximum FP32/INT8 output difference | `0.00570488` |
| Mean FP32/INT8 output difference | `0.00113890` |
| Output shape agreement | `PASS` |
| FP32 output finite | `True` |
| INT8 output finite | `True` |

### CPU latency

| Model | Mean latency |
|---|---:|
| FP32 | `9.3289 ms` |
| INT8 | `42.4774 ms` |

Measured INT8/FP32 speed ratio:

`0.220x`

The INT8 result is recorded as an optimization experiment.
It is NOT claimed to be a performance improvement over FP32 on the
tested CPU.

The INT8 artifact was also larger than the FP32 ONNX artifact in this
experiment:

- FP32: `16,577 bytes`
- INT8: `33,817 bytes`

Therefore, the INT8 experiment is retained as measured evidence and
is not designated as the primary deployment artifact.

## 7. Primary M3-P6 Deployment Decision

The primary deployment artifact is:

`deployment/m3_speech_enhancement.onnx`

Reason:

- Actual ONNX export completed successfully.
- ONNX checker passed.
- ONNX Runtime validation passed.
- PyTorch/ONNX numerical difference was very small.
- INT8 was experimentally validated but did not improve measured CPU
  latency or model size on the tested platform.

No claim of Raspberry Pi performance is made from these measurements.

## 8. M3 Integration Boundary

M3 owns:

1. STFT preprocessing
2. CNN inference
3. Enhancement-mask generation
4. Mask application
5. ISTFT reconstruction

The integration layer supplies:

- Mono audio
- `float32`
- `16,000 Hz`

The integration layer receives:

- Enhanced mono audio
- `float32`
- `16,000 Hz`

The ONNX model itself operates on the log-magnitude STFT tensor,
not directly on the waveform.

## 9. Inference / Runtime Status

### Completed

- [x] Actual ONNX export
- [x] ONNX checker validation
- [x] ONNX Runtime validation
- [x] FP32 numerical equivalence validation
- [x] INT8 quantization experiment
- [x] INT8 numerical validation
- [x] Model I/O contract documented

### Not yet completed

- [ ] Raspberry Pi 5 deployment
- [ ] Raspberry Pi 5 real-time benchmark
- [ ] Sustained continuous Pi 5 run
- [ ] Pi 5 CPU measurement
- [ ] Pi 5 RAM measurement
- [ ] Pi 5 thermal/throttling observation

## 10. Real-Time Capability Sign-Off

The following requirements MUST NOT be signed off from the current
Windows/CPU measurements:

- [ ] Physical Raspberry Pi 5 run exists
- [ ] Platform is confirmed as `raspberry_pi_5`
- [ ] Every tested audio block meets the real-time margin
- [ ] Continuous sustained run completed
- [ ] Warm-up blocks excluded from steady-state measurement
- [ ] Pi CPU utilization recorded
- [ ] Pi memory/RSS recorded
- [ ] Pi thermal behavior checked

The current laptop/Windows measurements are software validation only.

## 11. Physical Hardware Status

The following work remains open:

- [ ] Physical microphone selection
- [ ] Physical audio interface selection
- [ ] Microphone measurements
- [ ] Audio-interface measurements
- [ ] Channel synchronization measurement
- [ ] End-to-end acoustic validation
- [ ] Final physical ANC/headset validation

No physical microphone, audio interface, acoustic path, or closed-loop
ANC performance is claimed by this handoff.

## 12. Open Items for M6

M6 should be aware that:

1. The primary deployment artifact is the FP32 ONNX model.
2. The INT8 model is an experimental artifact, not a demonstrated
   performance improvement.
3. The ONNX model expects log-magnitude STFT input with shape
   `[batch, 1, 257, time]`.
4. Raspberry Pi 5 performance has not yet been measured.
5. Physical audio hardware has not yet been selected or characterized.
6. End-to-end acoustic ANC validation has not yet been performed.
7. No physical closed-loop ANC performance claim is made.

## 13. M3-P6 Completion Status

### Software deployment portion

**COMPLETE**

### Hardware deployment portion

**NOT YET COMPLETED**

### Overall M3-P6 → M6 handoff

**PARTIALLY COMPLETE — software deployment evidence is complete;
Raspberry Pi and physical hardware evidence remain pending.**

