import numpy as np
from pathlib import Path
from .io import save_audio


def time_axis(n, sr):
    return np.arange(n, dtype=np.float32) / sr


def sine(freq, duration, sr, amplitude=0.3, phase=0.0):
    t = time_axis(int(duration * sr), sr)
    return amplitude * np.sin(2 * np.pi * freq * t + phase)


def chirp(f0, f1, duration, sr, amplitude=0.25):
    from scipy.signal import chirp as scipy_chirp
    t = time_axis(int(duration * sr), sr)
    return amplitude * scipy_chirp(t, f0=f0, f1=f1, t1=duration, method="linear")


def add_impulses(x, sr, times=(0.7, 1.8), amplitude=0.85):
    y = x.copy()
    width = max(1, int(0.004 * sr))
    for sec in times:
        center = int(sec * sr)
        lo, hi = max(0, center - width), min(len(y), center + width)
        window = np.hanning(max(2, hi - lo))
        y[lo:hi] += amplitude * window
    return np.clip(y, -1, 1)


def write_demo_signals(out_dir, sr=16000, duration=4.0, seed=42):
    rng = np.random.default_rng(seed)
    n = int(sr * duration)
    t = time_axis(n, sr)
    speech_like = 0.18 * np.sin(2*np.pi*180*t) + 0.10*np.sin(2*np.pi*320*t) + 0.05*np.sin(2*np.pi*650*t)
    speech_like *= (0.55 + 0.45*(np.sin(2*np.pi*1.2*t)**2))
    stationary = 0.18 * rng.normal(size=n)
    rotor = 0.16*np.sin(2*np.pi*120*t) + 0.10*np.sin(2*np.pi*240*t) + 0.06*np.sin(2*np.pi*360*t)
    rotor += 0.03*rng.normal(size=n)
    nonstationary = (0.08 + 0.14*(0.5+0.5*np.sin(2*np.pi*0.35*t))) * rng.normal(size=n)
    impulsive = add_impulses(0.02*rng.normal(size=n), sr)
    speech_plus_noise = np.clip(speech_like + 0.22*stationary, -1, 1)
    reference = np.clip(0.65*stationary + 0.35*rotor + 0.03*rng.normal(size=n), -1, 1)
    error = np.clip(0.38*stationary + 0.15*rotor + 0.10*speech_like + 0.02*rng.normal(size=n), -1, 1)
    speech = np.clip(speech_like + 0.015*rng.normal(size=n), -1, 1)
    files = {
        "speech_clean.wav": speech_like,
        "stationary_noise.wav": stationary,
        "rotor_noise.wav": rotor,
        "impulsive_noise.wav": impulsive,
        "speech_plus_noise.wav": speech_plus_noise,
        "reference.wav": reference,
        "error.wav": error,
        "speech.wav": speech,
    }
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for name, sig in files.items(): save_audio(out_dir/name, sig, sr)
    return files
