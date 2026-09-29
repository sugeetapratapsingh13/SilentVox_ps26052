import numpy as np
import soundfile as sf
import time
from pathlib import Path

from m3_adapter import create_adapter

ROOT = Path(".")
INPUTS = [
    ROOT / "M4" / "virtual_mics" / "stationary_noise.wav",
    ROOT / "M4" / "virtual_mics" / "rotor_noise.wav",
    ROOT / "M4" / "virtual_mics" / "impulsive_noise.wav",
    ROOT / "M4" / "virtual_mics" / "speech_plus_noise.wav",
]

def rms_dbfs(x):
    rms = np.sqrt(np.mean(np.square(x)) + 1e-12)
    return 20.0 * np.log10(rms + 1e-12)

m3 = create_adapter()

print("=" * 78)
print("SILENTVOX STAGE-2 M3 QUANTITATIVE VALIDATION")
print("=" * 78)

for path in INPUTS:
    audio, sr = sf.read(str(path), dtype="float32")

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    audio = np.asarray(audio, dtype=np.float32)

    start = time.perf_counter()
    result = m3.process(audio, sr)
    wall_time = time.perf_counter() - start

    duration = len(audio) / sr
    rtf = wall_time / duration

    print()
    print(f"TEST:              {path.name}")
    print(f"Sample rate:       {sr} Hz")
    print(f"Samples:           {len(audio)}")
    print(f"Duration:          {duration:.6f} sec")
    print(f"Input RMS:         {rms_dbfs(audio):.3f} dBFS")
    print(f"Output RMS:        {rms_dbfs(result.enhanced_audio):.3f} dBFS")
    print(f"M3 inference:      {result.inference_time_sec:.6f} sec")
    print(f"M3 wall time:      {wall_time:.6f} sec")
    print(f"PC real-time factor: {rtf:.3f}x")
    print(f"Model:             {result.model_name}")
    print(f"Parameters:        {result.parameter_count}")
    print(f"Output samples:    {len(result.enhanced_audio)}")
    print(f"Finite output:     {np.isfinite(result.enhanced_audio).all()}")

print()
print("=" * 78)
print("M3 QUANTITATIVE VALIDATION COMPLETE")
print("=" * 78)
