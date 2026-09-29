from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import time
import numpy as np
import librosa

from M6.adapters.m1_adapter import create_adapter


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "M4" / "virtual_mics"

TESTS = [
    "stationary_noise.wav",
    "rotor_noise.wav",
    "impulsive_noise.wav",
    "speech_plus_noise.wav",
]


def rms(audio):
    audio = np.asarray(audio, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(audio))))


def dbfs(audio):
    return 20.0 * np.log10(max(rms(audio), 1e-12))


m1 = create_adapter()

print("=" * 78)
print("SILENTVOX STAGE-2 M1 QUANTITATIVE VALIDATION")
print("=" * 78)

for filename in TESTS:
    path = INPUT_DIR / filename

    audio, sr = librosa.load(
        path,
        sr=16000,
        mono=True,
    )

    audio = np.asarray(audio, dtype=np.float32)

    start = time.perf_counter()
    result = m1.process(audio)
    elapsed = time.perf_counter() - start

    residual = np.asarray(
        result.residual,
        dtype=np.float32,
    )

    input_rms = rms(audio)
    output_rms = rms(residual)

    input_db = dbfs(audio)
    output_db = dbfs(residual)

    noise_reduction_db = (
        20.0 * np.log10(
            max(input_rms, 1e-12)
            / max(output_rms, 1e-12)
        )
    )

    duration = len(audio) / sr
    realtime_factor = elapsed / duration

    print()
    print(f"TEST:              {filename}")
    print(f"Sample rate:       {sr} Hz")
    print(f"Samples:           {len(audio)}")
    print(f"Duration:          {duration:.6f} sec")
    print(f"Input RMS:         {input_db:.3f} dBFS")
    print(f"Output RMS:        {output_db:.3f} dBFS")
    print(f"Noise reduction:   {noise_reduction_db:.3f} dB")
    print(f"M1 processing:     {elapsed:.6f} sec")
    print(f"PC real-time factor: {realtime_factor:.3f}x")
    print(f"Finite output:     {np.isfinite(residual).all()}")

print()
print("=" * 78)
print("M1 QUANTITATIVE VALIDATION COMPLETE")
print("=" * 78)
