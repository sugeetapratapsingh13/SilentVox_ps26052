import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import time
import numpy as np
import librosa

from M6.adapters.m1_adapter import create_adapter as create_m1_adapter
from integration.m3_adapter import create_adapter as create_m3_adapter

audio, _ = librosa.load(
    r".\M4\virtual_mics\speech_plus_noise.wav",
    sr=16000,
    mono=True,
)

audio = np.asarray(audio, dtype=np.float32)

m1 = create_m1_adapter()
m3 = create_m3_adapter()

print("=" * 70)
print("SILENTVOX PC BLOCK PROCESSING BENCHMARK")
print("=" * 70)

for block_size in (256, 512, 1024):
    block = audio[:block_size]

    start = time.perf_counter()
    m1_result = m1.process(block)
    m1_time = time.perf_counter() - start

    residual = np.asarray(
        m1_result.residual,
        dtype=np.float32,
    )

    # M3's existing STFT uses a 512-sample window.
    # Preserve the benchmark block size while padding only
    # the M3 boundary input when the block is shorter than 512.
    m3_input = residual
    if len(m3_input) < 512:
        m3_input = np.pad(
            m3_input,
            (0, 512 - len(m3_input)),
        )

    start = time.perf_counter()
    m3_result = m3.process(
        m3_input,
        16000,
    )
    m3_time = time.perf_counter() - start

    total = m1_time + m3_time
    audio_duration = block_size / 16000.0
    realtime_factor = total / audio_duration

    print()
    print(f"BLOCK SIZE:       {block_size}")
    print(f"Audio duration:   {audio_duration:.6f} sec")
    print(f"M1 time:          {m1_time:.6f} sec")
    print(f"M3 time:          {m3_time:.6f} sec")
    print(f"Total time:       {total:.6f} sec")
    print(f"Real-time factor: {realtime_factor:.3f}x")
    print(f"Within deadline:  {total < audio_duration}")
