# Edge Deployment Guide — Raspberry Pi 5

## Pipeline

```
Training (M3) → Trained Model → Export (deployment/export.py)
             → Optimized Runtime (ONNX Runtime, int8)
             → Raspberry Pi 5 → Live Audio → Inference (deployment/inference.py)
             → Enhanced Speech
```

## Runtime choice: ONNX Runtime over TFLite

Both were investigated per the source spec ("ONNX, TFLite, appropriate
lightweight runtime"). **ONNX Runtime is the recommended runtime for
this project's Pi 5 deployment**, for concrete reasons:

- `onnxruntime` ships a standard `pip`-installable ARM64 wheel for
  Raspberry Pi OS (64-bit) — no separate cross-compilation step.
- `onnxruntime.quantization.quantize_dynamic` gave a real, measured
  size/latency win directly on the ONNX graph produced by `export.py`
  (see `benchmarks/optimization_results.csv`, `onnx_quant` row) without
  requiring TensorFlow at all.
- The TFLite path requires `onnx-tf` + full TensorFlow to convert from
  the PyTorch-exported ONNX graph, then `tflite_runtime` (a separate,
  Pi-specific wheel that isn't always in lockstep with the latest
  TensorFlow release) to run it. That's two extra heavy dependencies and
  a less-portable conversion step, for a runtime this project did not
  end up needing.

`deployment/inference.py` supports both (`--runtime onnx` / `--runtime
tflite`) so this decision can be revisited if a future model architecture
converts more cleanly to TFLite or the team standardizes on it project-wide.

## Setup on the Raspberry Pi 5

1. Raspberry Pi OS (64-bit, Bookworm or later recommended) — a 64-bit OS
   is required for the standard `onnxruntime` ARM64 wheel.
2. `python3 -m venv venv && source venv/bin/activate`
3. `pip install onnxruntime numpy sounddevice soundfile`
   (add `tflite_runtime` only if using the TFLite path)
4. Copy the accepted exported model (from `deployment/export.py`'s
   `--output-dir`, the artifact with `status == ACCEPTED` in
   `optimization_results.csv`) onto the Pi.
5. Confirm the audio device: `python3 -c "import sounddevice as sd; print(sd.query_devices())"`
   and set the correct input/output device index if the default isn't right.

## Running live inference

```bash
python3 deployment/inference.py --mode live \
  --model-path model_onnx_int8.onnx --runtime onnx \
  --sample-rate 16000 --frame-size 512 --hop-size 256 \
  --platform raspberry_pi_5 --duration 30 \
  --results-csv benchmarks/realtime_results.csv
```

## Validating real-time performance — do this ON the Pi, not a laptop

The source spec is explicit: **do not claim real-time merely because it
runs on a laptop.** `inference.py` enforces this by recording whatever
`--platform` you pass into every result row and printing an explicit
warning banner whenever that value isn't `raspberry_pi_5`. Before
signing off on real-time capability in `HANDOFF_REPORT.md`:

1. Run the exact command above, on the Pi 5 itself, with
   `--platform raspberry_pi_5`.
2. Confirm every row in `realtime_results.csv` has `meets_margin = True`.
3. If some blocks miss the margin under real audio-driver load (jitter,
   thermal throttling, background processes), that's exactly the kind
   of thing lab-only laptop testing would have hidden — increase
   `--hop-size`, apply a lighter optimization technique, or both.

## Common Pi-specific failure modes to check for

- **Thermal throttling** under sustained inference — run
  `vcgencmd measure_temp` during a longer soak test, not just the
  30-second smoke test above.
- **USB audio device latency** — onboard 3.5mm audio and cheap USB
  audio interfaces can have very different buffer latencies; test with
  the actual hardware that will ship.
- **Cold-start latency** — the first few inference calls after process
  start are typically slower (runtime warm-up, page faults); the
  benchmark script's warmup runs account for this in `export.py`'s
  profiling, but `inference.py`'s live mode does not currently discard
  a warm-up period — discard the first ~10 blocks' timings when judging
  steady-state real-time margin.
