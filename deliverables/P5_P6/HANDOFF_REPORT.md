# M3-P6 → M6 Handoff Report

This is the required handoff package from the deployment stage (M3-P6)
to the whole-project integration stage (M6). Every field below must be
filled in from the actual accepted export before this is considered
complete — the structure and field names are final, the values are
templated pending a real optimization + Pi run.

| Field | Value | Source |
|---|---|---|
| **Model file** | *e.g. `model_onnx_int8.onnx`* | `deployment/export.py --output-dir`, the artifact with `status == ACCEPTED` in `optimization_results.csv` |
| **Runtime format** | ONNX (int8 quantized) — see `EDGE_DEPLOYMENT.md` for why ONNX Runtime over TFLite | — |
| **Input format** | float32 mono waveform, shape `(1, 1, frame_size)`, range [-1, 1] | `final/MODEL_PARAMETERS.md` |
| **Output format** | float32 mono waveform, same shape as input | `final/MODEL_PARAMETERS.md` |
| **Sampling rate** | *fill in, e.g. 16000 Hz* | `MODEL_PARAMETERS.md` |
| **Frame size** | *fill in, e.g. 512 samples* | `MODEL_PARAMETERS.md` — must match the shape used at export time |
| **Hop size** | *fill in, e.g. 256 samples* | `MODEL_PARAMETERS.md` |
| **Latency** | *fill in mean + p95, ms/block* | `benchmarks/optimization_results.csv` (single-block) and `benchmarks/realtime_results.csv` (streaming, on-Pi) |
| **CPU requirement** | *fill in observed CPU% during on-Pi run* | `benchmarks/realtime_results.csv`, platform=raspberry_pi_5 rows only |
| **RAM requirement** | *fill in observed RSS, MB* | `benchmarks/realtime_results.csv`, `rss_bytes` column |
| **Model size** | *fill in bytes* | `benchmarks/optimization_results.csv`, `model_size_bytes` for the accepted row |
| **Dependencies** | `onnxruntime`, `numpy`, `sounddevice` (live audio), `soundfile` (file-mode testing only) | `MODEL_PARAMETERS.md` |
| **Inference API** | `EnhancerRuntime(model_path, runtime="onnx", input_shape_rank=3).process_block(block: np.ndarray[frame_size]) -> np.ndarray[frame_size]` | `deployment/inference.py` |

## Real-time capability sign-off

- [ ] `benchmarks/realtime_results.csv` contains a run with
      `platform == raspberry_pi_5` (not a laptop — see the platform-honesty
      rule in `deployment/inference.py` and `EDGE_DEPLOYMENT.md`)
- [ ] Every block in that run has `meets_margin == True`
- [ ] The run covers at least [duration TBD by team] of continuous audio,
      not just a short smoke test, to catch thermal throttling
- [ ] First ~10 blocks (warm-up) excluded from the steady-state RTF claim

**Do not check these boxes from a laptop run.** Re-run on the physical
Raspberry Pi 5 before M6 treats this as validated.

## Quality gate summary (Part 1 boundary)

Per the do-not-accept-degradation rule, only techniques that passed the
quality gate in `deployment/export.py` (`status == ACCEPTED` in
`optimization_results.csv`) are eligible to be the handed-off model file
above. Rejected techniques remain visible in that CSV for the record but
are not this handoff's deliverable.

*Fill in: accepted technique(s), and the STOI/SI-SDR delta vs. the
unoptimized baseline model, for the record.*

## Open items for M6

*List anything M6 needs to know that isn't captured in the table above —
e.g. known edge cases, audio devices tested, or follow-up optimization
work that was investigated but not adopted.*
