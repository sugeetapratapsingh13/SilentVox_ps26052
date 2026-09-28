import numpy as np


def quantize(x, bits=16):
    levels = 2 ** (bits - 1) - 1
    return np.round(np.clip(x, -1, 1) * levels) / levels


def clip(x, limit=0.95):
    return np.clip(x, -limit, limit)


def apply_gain(x, gain=1.0):
    return np.asarray(x) * gain


def add_noise(x, rms=0.01, seed=42):
    rng = np.random.default_rng(seed)
    return np.asarray(x) + rng.normal(0, rms, size=len(x))


def delay_samples(x, samples):
    samples = int(samples)
    x = np.asarray(x, dtype=np.float32)
    if samples <= 0:
        return x.copy()
    if samples >= len(x):
        return np.zeros_like(x)
    return np.concatenate([np.zeros(samples, dtype=np.float32), x[:-samples]])


def lowpass(x, sr=16000, cutoff=6000):
    from scipy.signal import butter, sosfilt
    sos = butter(6, cutoff, btype="lowpass", fs=sr, output="sos")
    return sosfilt(sos, np.asarray(x)).astype(np.float32)


def adc_simulate(x, bits=16, gain=1.0, noise_rms=0.0, clip_limit=1.0, seed=42):
    y = apply_gain(x, gain)
    if noise_rms:
        y = add_noise(y, noise_rms, seed)
    y = np.clip(y, -clip_limit, clip_limit)
    return quantize(y, bits).astype(np.float32)


def dac_simulate(x, bits=16, clip_limit=1.0):
    """Digital reconstruction-side model; not an electrical DAC measurement."""
    return quantize(np.clip(np.asarray(x), -clip_limit, clip_limit), bits).astype(np.float32)
