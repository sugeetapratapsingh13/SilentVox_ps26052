# M3-P2 — Speech Dataset + Mixture Generation

Builds Clean Speech + Environmental Noise → Noisy Speech mixtures across the
required SNR sweep, with leakage prevented on both the speaker axis and the
noise-source axis. See `DATASET_STRUCTURE.md` for the full folder layout,
manifest schemas, and the SNR-range justification.

## Status

| Deliverable | Status |
|---|---|
| `mixture_generator.py` | ✅ written, self-contained (numpy/scipy/pandas only) |
| Dual leakage-free split (speaker + noise source) | ✅ implemented and **verified**: 392 synthetic mixtures generated, zero overlap on either axis, confirmed programmatically |
| SNR sweep (-10/-5/0/+5/+10/+15/+20 dB) | ✅ verified — all 7 values present, achieved SNR matched target to within rounding error on spot-checks |
| `dataset_manifest.csv` / `snr_manifest.csv` | ✅ generated correctly on synthetic data; **empty until you add real audio** |
| `DATASET_STRUCTURE.md` | ✅ complete |
| Real speech/noise audio | ⏳ **not provided** — `dataset/speech/`, `dataset/noise/` are empty; templates for the metadata CSVs are in `dataset/metadata/` |

This was verified end-to-end here using synthetic tone-plus-noise WAV files
(6 speakers, 18 noise recordings across the 6 classes) — the pipeline
mechanics are confirmed correct; the content still needs real recordings.

## How to run it

```bash
pip install numpy scipy pandas

# 1. Add real audio:
#    dataset/speech/*.wav   (16 kHz mono, or any rate -- auto-resampled)
#    dataset/noise/*.wav

# 2. Fill in metadata (copy the _TEMPLATE files, drop the _TEMPLATE suffix):
#    dataset/metadata/speech_metadata.csv
#    dataset/metadata/noise_metadata.csv

# 3. Generate mixtures
python mixture_generator.py \
    --speech_metadata dataset/metadata/speech_metadata.csv \
    --noise_metadata dataset/metadata/noise_metadata.csv \
    --speech_dir dataset/speech \
    --noise_dir dataset/noise \
    --mixtures_dir dataset/mixtures \
    --manifest_dir dataset/metadata
```

This writes every mixture .wav under `dataset/mixtures/<split>/<noise_class>/`,
plus `dataset_manifest.csv` and `snr_manifest.csv` in `dataset/metadata/`.

`--noise_per_speech_per_class` (default 1) controls how many different noise
recordings of each class get paired with each speech clip — raise it if you
have plenty of noise recordings and want more mixture diversity per speaker.

## What's honestly NOT done yet

- No real speech or noise recordings — collection/curation is still needed.
- No reverberation modelling (see `DATASET_STRUCTURE.md` limitations).
- This script does not train or evaluate any enhancement model — per the
  M3-P2 boundary, that's a separate sub-task.
