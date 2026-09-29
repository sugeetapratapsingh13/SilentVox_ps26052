from pathlib import Path
import numpy as np
import soundfile as sf
from pystoi import stoi

ROOT = Path(__file__).resolve().parents[1]

clean_path = ROOT / "M4" / "virtual_mics" / "speech_clean.wav"
output_path = ROOT / "integration" / "stage2_output.wav"
result_path = ROOT / "integration" / "integrated_speech_metrics.txt"

clean, sr_clean = sf.read(clean_path, dtype="float32")
enhanced, sr_out = sf.read(output_path, dtype="float32")

if clean.ndim > 1:
    clean = clean.mean(axis=1)
if enhanced.ndim > 1:
    enhanced = enhanced.mean(axis=1)

if sr_clean != sr_out:
    raise ValueError(f"Sample-rate mismatch: clean={sr_clean}, output={sr_out}")

n = min(len(clean), len(enhanced))
clean = np.asarray(clean[:n], dtype=np.float64)
enhanced = np.asarray(enhanced[:n], dtype=np.float64)

if not np.isfinite(clean).all() or not np.isfinite(enhanced).all():
    raise ValueError("NaN/Inf detected")

# SI-SDR with scale-invariant projection.
target_energy = np.sum(clean ** 2) + 1e-12
scale = np.sum(enhanced * clean) / target_energy
target = scale * clean
error = enhanced - target

si_sdr_db = 10.0 * np.log10(
    (np.sum(target ** 2) + 1e-12) /
    (np.sum(error ** 2) + 1e-12)
)

# STOI expects signals at the same sampling rate.
stoi_score = stoi(clean, enhanced, sr_out, extended=False)

duration = n / sr_out

text = f"""SILENTVOX STAGE-2 INTEGRATED SPEECH METRICS
================================================

Reference:
  {clean_path}

Integrated output:
  {output_path}

Sample rate:        {sr_out} Hz
Samples compared:   {n}
Duration:            {duration:.6f} sec

STOI:               {stoi_score:.6f}
SI-SDR:             {si_sdr_db:.6f} dB

Metrics path:
  Clean speech -> speech+noise -> M2 -> M1 -> M3 -> integrated output

These metrics evaluate the integrated speech path against the
available clean-speech reference. They do not constitute physical
headset or Raspberry Pi validation.
"""

print(text)
result_path.write_text(text, encoding="utf-8")
print(f"Results saved: {result_path}")
