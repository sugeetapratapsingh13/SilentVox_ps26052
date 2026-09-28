# Model Parameters — I/O Contract

This document is the precise technical contract for whoever loads the
exported model — filled in from what `export.py` was actually run with.
Keep this in sync with the `--input-shape`, `--sample-rate`,
`--frame-size`, `--hop-size` flags used for the accepted export.

## I/O contract

| Parameter | Value | Notes |
|---|---|---|
| Input format | float32 waveform, mono, range [-1, 1] | Matches `EnhancerRuntime.process_block` in `inference.py` |
| Input shape | `(1, 1, frame_size)` or `(1, frame_size)` | Set via `--input-shape-rank` (3 or 2) at export/inference time — *fill in which one the real model uses* |
| Output format | float32 waveform, mono, range [-1, 1] | Same shape convention as input |
| Sampling rate | *fill in, e.g. 16000 Hz* | Must match training data's sample rate — resample upstream if the Pi's audio device runs at a different rate |
| Frame size | *fill in, e.g. 512 samples* | Samples per inference call |
| Hop size | *fill in, e.g. 256 samples* | Samples of new audio per block in streaming mode; determines the real-time budget (see `HANDOFF_REPORT.md`) |
| Normalization | *fill in — e.g. none, or peak/RMS normalization applied before inference* | State explicitly: silent mismatch here is a common source of "works in testing, degrades on-device" bugs |

## Export shape note

Because the exported ONNX graph's sample-length dimension can be baked
in as static by the exporter (see `deployment/export.py` docstring), the
model **must be exported with `--input-shape` set to the exact
`frame_size` used at inference time** — not an arbitrary placeholder
length. If frame_size changes, re-export.

## Preprocessing / postprocessing expected outside the model

*Fill in anything the model assumes has already happened before its
input tensor is built* — e.g.:
- DC offset removal
- A specific windowing function applied to each frame
- Overlap-add reconstruction across hops (the reference `inference.py`
  file-mode harness does a **naive** concatenation for a quick
  listen-check only — a real deployment needs proper overlap-add if
  `hop_size < frame_size`)

## Dependencies (see also HANDOFF_REPORT.md)

- `onnxruntime` (or `tflite_runtime` if the TFLite path is used)
- `numpy`
- `sounddevice` (live audio capture/playback on the Pi)
- `soundfile` (only needed for file-mode testing, not required on-device
  for live streaming)
