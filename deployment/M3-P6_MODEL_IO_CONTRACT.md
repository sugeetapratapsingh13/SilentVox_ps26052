# M3-P6 Model I/O Contract

## Model Identity

- Model name: `SpeechEnhancementCNN`
- Deployment format: ONNX
- Primary deployment artifact: `deployment/m3_speech_enhancement.onnx`
- PyTorch checkpoint: `M3-P4/model/trained_model.pth`
- Parameter count: `18,817`

## Audio Contract

- Sample rate: `16,000 Hz`
- Audio channels: mono
- Input audio type: `float32`
- Input audio shape at integration boundary: 1-D waveform
- STFT FFT size: `512`
- STFT window length: `512`
- STFT hop length: `128`
- Frequency bins: `257`

## ONNX Tensor Contract

### Input

- Tensor name: `input`
- Data type: `float32`
- Shape: `[batch, 1, 257, time]`
- Representation: log-magnitude STFT

### Output

- Tensor name: `mask`
- Data type: `float32`
- Shape: `[batch, 1, 257, time]`
- Representation: predicted enhancement mask

The output mask is applied element-wise to the mixture magnitude before ISTFT reconstruction.

## ONNX Metadata

- ONNX IR version: `10`
- ONNX opset: `18`
- Graph nodes: `10`

## Model Validation

### PyTorch vs ONNX

- Maximum absolute output difference: `1.7881393433e-07`
- Mean absolute output difference: `3.1553355484e-08`
- ONNX checker: `PASS`

### ONNX Runtime

- Execution provider tested: `CPUExecutionProvider`
- Input shape: `(1, 1, 257, 100)`
- Output shape: `(1, 1, 257, 100)`
- Output finite: `True`
- Runtime validation: `PASS`

## Deployment Artifacts

### FP32

`deployment/m3_speech_enhancement.onnx`

Size measured during export:

`16,577 bytes`

### INT8

`deployment/m3_speech_enhancement_int8.onnx`

Size measured during optimization:

`33,817 bytes`

## INT8 Optimization Measurements

The INT8 model was successfully generated and numerically validated.

- Maximum FP32/INT8 output difference: `0.00570488`
- Mean FP32/INT8 output difference: `0.00113890`

### CPU latency measurement

- FP32 mean latency: `9.3289 ms`
- INT8 mean latency: `42.4774 ms`
- Measured INT8/FP32 speed ratio: `0.220x`

The measured INT8 result is recorded as an optimization experiment. It is not claimed to be a performance improvement over FP32 on the tested CPU.

## Current Deployment Status

- Actual ONNX export: `PASS`
- ONNX checker: `PASS`
- ONNX Runtime validation: `PASS`
- INT8 quantization: `PASS`
- Raspberry Pi 5 deployment: `NOT YET COMPLETED`
- Raspberry Pi 5 real-time benchmark: `NOT YET COMPLETED`
- Physical audio hardware validation: `NOT YET COMPLETED`

## Contract Boundary

M3 owns:

1. STFT preprocessing
2. CNN inference
3. Enhancement-mask generation
4. Mask application
5. ISTFT reconstruction

The integration layer supplies mono `float32` audio at `16 kHz` and receives enhanced mono `float32` audio at `16 kHz`.

## Validation Status

**M3-P6 MODEL I/O CONTRACT: COMPLETE**

This document records the measured software deployment contract and validation results. Raspberry Pi and physical hardware measurements are intentionally not included until they are actually performed.
