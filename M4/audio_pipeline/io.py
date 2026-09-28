from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def load_audio(path, target_sr=None, mono=False):
    data, sr = sf.read(str(path), always_2d=True, dtype="float32")
    if mono:
        data = np.mean(data, axis=1, keepdims=True)
    if target_sr and sr != target_sr:
        g = np.gcd(int(sr), int(target_sr))
        up, down = int(target_sr // g), int(sr // g)
        data = np.stack([resample_poly(data[:, c], up, down) for c in range(data.shape[1])], axis=1).astype(np.float32)
        sr = target_sr
    return data, sr


def save_audio(path, audio, sr):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(audio, dtype=np.float32)
    sf.write(str(path), np.clip(audio, -1.0, 1.0), sr, subtype="PCM_16")


def validate_audio(audio, sr, expected_sr=16000, expected_channels=None):
    audio = np.asarray(audio)
    assert audio.ndim == 2, f"Expected [samples, channels], got {audio.shape}"
    assert sr == expected_sr, f"Expected {expected_sr} Hz, got {sr} Hz"
    if expected_channels is not None:
        assert audio.shape[1] == expected_channels, f"Expected {expected_channels} channels, got {audio.shape[1]}"
    assert np.isfinite(audio).all(), "Audio contains non-finite values"


def block_iter(audio, block_size):
    audio = np.asarray(audio)
    for start in range(0, len(audio), block_size):
        block = audio[start:start + block_size]
        if len(block) == block_size:
            yield start, block
