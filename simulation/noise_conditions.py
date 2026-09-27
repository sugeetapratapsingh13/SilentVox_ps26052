"""
noise_conditions.py

Generates the 5 required test conditions for the ANC simulation:

    1. Stationary          - constant-statistics noise (white)
    2. Non-stationary       - noise whose amplitude/energy drifts over time
    3. Periodic/mechanical  - harmonic "motor hum" style noise
    4. Impulsive/transient  - sparse loud clicks/bangs over a quiet floor
    5. Speech + noise       - synthetic speech-like signal + background noise

Every generator returns (reference_noise, desired_signal) at the requested
sample_rate/length, where:
    reference_noise  -> what the reference microphone picks up
    desired_signal   -> the "wanted" signal that must be preserved
                         (zero for pure noise-cancellation conditions 1-4)
"""

import numpy as np


# ------------------------------------------------------------------
# 1. STATIONARY NOISE
# ------------------------------------------------------------------
def stationary_noise(n_samples, sample_rate, seed=1):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.5, n_samples)


# ------------------------------------------------------------------
# 2. NON-STATIONARY NOISE
# ------------------------------------------------------------------
def non_stationary_noise(n_samples, sample_rate, seed=2):
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 0.5, n_samples)
    t = np.arange(n_samples) / sample_rate
    # slow envelope so the noise energy visibly drifts over the clip
    envelope = 0.3 + 0.7 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.2 * t))
    return base * envelope


# ------------------------------------------------------------------
# 3. PERIODIC / MECHANICAL NOISE
# ------------------------------------------------------------------
def periodic_mechanical_noise(n_samples, sample_rate, fundamental=60.0,
                               n_harmonics=5, seed=3):
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / sample_rate
    noise = np.zeros(n_samples)
    for k in range(1, n_harmonics + 1):
        amp = 1.0 / k
        phase = rng.uniform(0, 2 * np.pi)
        noise += amp * np.sin(2 * np.pi * fundamental * k * t + phase)
    # small amplitude wobble to mimic a real motor/fan
    mod = 1.0 + 0.05 * np.sin(2 * np.pi * 1.3 * t)
    noise *= mod
    noise = noise / (np.max(np.abs(noise)) + 1e-12)
    return 0.5 * noise


# ------------------------------------------------------------------
# 4. IMPULSIVE / TRANSIENT NOISE
# ------------------------------------------------------------------
def impulsive_noise(n_samples, sample_rate, background_level=0.05,
                     impulse_rate_hz=2.0, seed=4):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, background_level, n_samples)

    duration = n_samples / sample_rate
    n_impulses = max(int(rng.poisson(impulse_rate_hz * duration)), 1)
    click_len = max(int(0.003 * sample_rate), 4)  # ~3 ms clicks

    for _ in range(n_impulses):
        start = rng.integers(0, max(n_samples - click_len, 1))
        amp = rng.uniform(0.6, 1.0) * rng.choice([-1, 1])
        decay = np.exp(-np.linspace(0, 8, click_len))
        noise[start:start + click_len] += amp * decay

    return noise


# ------------------------------------------------------------------
# 5. SPEECH + NOISE
# ------------------------------------------------------------------
def synthetic_speech(n_samples, sample_rate, seed=5):
    """
    Lightweight formant-style synthetic speech stand-in (no external
    speech file needed). Replace with a real speech recording later
    if/when one is available -- just load it and resample instead of
    calling this function.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / sample_rate

    f0 = 120 + 20 * np.sin(2 * np.pi * 0.5 * t)          # pitch contour
    phase = 2 * np.pi * np.cumsum(f0) / sample_rate

    formants = [700, 1200, 2500]
    formant_amps = [1.0, 0.6, 0.3]

    signal = np.zeros(n_samples)
    for f, a in zip(formants, formant_amps):
        signal += a * np.sin(phase * (f / np.mean(f0)))

    # syllable-like bursts instead of one continuous tone
    env = np.zeros(n_samples)
    syllable_len = int(0.25 * sample_rate)
    gap_len = int(0.10 * sample_rate)
    pos = 0
    while pos < n_samples:
        seg = min(syllable_len, n_samples - pos)
        if seg > 1:
            env[pos:pos + seg] = np.hanning(seg)
        pos += syllable_len + gap_len

    signal *= env
    signal = signal / (np.max(np.abs(signal)) + 1e-12)
    return 0.6 * signal


def speech_plus_noise(n_samples, sample_rate, noise_level=0.3, seed=6):
    rng = np.random.default_rng(seed)
    speech = synthetic_speech(n_samples, sample_rate, seed=seed)
    background = rng.normal(0, noise_level, n_samples)
    return speech, background


# ------------------------------------------------------------------
# UNIFIED ACCESS POINT
# ------------------------------------------------------------------
CONDITIONS = [
    "stationary",
    "non_stationary",
    "periodic_mechanical",
    "impulsive_transient",
    "speech_plus_noise",
]


def get_noise_condition(name, n_samples, sample_rate):
    """
    Returns (reference_noise, desired_signal) for the named condition.
    """
    if name == "stationary":
        reference = stationary_noise(n_samples, sample_rate)
        desired = np.zeros(n_samples)

    elif name == "non_stationary":
        reference = non_stationary_noise(n_samples, sample_rate)
        desired = np.zeros(n_samples)

    elif name == "periodic_mechanical":
        reference = periodic_mechanical_noise(n_samples, sample_rate)
        desired = np.zeros(n_samples)

    elif name == "impulsive_transient":
        reference = impulsive_noise(n_samples, sample_rate)
        desired = np.zeros(n_samples)

    elif name == "speech_plus_noise":
        speech, background = speech_plus_noise(n_samples, sample_rate)
        reference = background
        desired = speech

    else:
        raise ValueError(f"Unknown condition: {name}")

    return reference, desired