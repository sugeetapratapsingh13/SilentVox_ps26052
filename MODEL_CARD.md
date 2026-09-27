# Model Card — Edge-Optimized Speech Enhancement Model

**Prepared by:** M3-P6 (Optimization + Edge Deployment)
**Status:** Template wired to `deployment/export.py`. Fill in the
italicized fields once run against the actual trained model from M3's
training work; the field names match the columns `export.py` produces.

## Overview

| Field | Value |
|---|---|
| Task | Speech enhancement (noise suppression / speech preservation) for real-time communication |
| Upstream source | *Trained model artifact from M3 training stage* |
| Intended deployment target | Raspberry Pi 5, real-time streaming inference |
| Not intended for | Offline batch processing at scale (a server-side unoptimized model is more appropriate there — this optimization work specifically targets edge constraints) |

## Architecture

*Fill in from the actual model:* layer types, parameter count, receptive
field / context length, whether it operates on raw waveform or a
time-frequency representation (STFT/mel). `deployment/export.py` assumes
a raw-waveform-in/raw-waveform-out contract by default — see
`MODEL_PARAMETERS.md` for the exact I/O contract used.

## Optimization applied

Selected from `benchmarks/optimization_results.csv` — report only the
techniques with `status == ACCEPTED` here; `REJECTED` techniques are
recorded in the benchmarks file for transparency but are explicitly
**not** part of the deployed model, per the do-not-accept-degradation
rule in Part 1 of this module's spec.

| Field | Value |
|---|---|
| Technique(s) accepted | *e.g. onnx_quant* |
| Size before → after | *e.g. 38,285 B → 15,299 B* |
| Latency before → after | *e.g. 3.3 ms → 5.5 ms mean, per-block* |
| STOI before → after (Δ) | *fill in* |
| SI-SDR before → after (Δ, dB) | *fill in* |

## Known limitations

- Unstructured pruning does not reduce on-disk size unless exported to a
  sparse-aware format/runtime; it was evaluated but is not treated as a
  size-reduction technique on its own in this project (see
  `optimization_results.csv` notes column).
- TFLite conversion requires optional heavy dependencies (`onnx-tf`,
  TensorFlow) not present in the environment this project was authored
  in; the code path exists in `export.py` but was not exercised here. See
  `EDGE_DEPLOYMENT.md` for why ONNX Runtime is the recommended runtime
  regardless.
- Quality metrics (STOI/SI-SDR) were computed on whatever `--quality-clean`
  / `--quality-noisy` pair was supplied at export time — a single pair (or
  small set) is not a substitute for the full evaluation corpus already
  used in M3-P5; treat the optimization gate as a fast sanity check, not
  a replacement for the M3-P5 evaluation.

## Where the numbers come from

Every number in this card should trace to a row in
`benchmarks/optimization_results.csv` or `benchmarks/realtime_results.csv`.
Do not hand-enter numbers that don't appear in those files.
