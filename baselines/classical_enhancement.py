"""
M3-P3 Classical Speech Enhancement Baselines

Methods:
1. Wiener filtering
2. Spectral suppression / spectral subtraction

Noise estimation is performed using the corresponding
noise-only recording rather than assuming that a segment
of the noisy speech contains only noise.
"""

from pathlib import Path
import argparse

import numpy as np
import soundfile as sf
from scipy import signal


EPS = 1e-10


def normalize_audio(x):
    """Prevent clipping when writing the enhanced signal."""
    x = np.asarray(x, dtype=np.float64)

    peak = np.max(np.abs(x))

    if peak > 0.99:
        x = x / peak * 0.99

    return x


def load_mono(path):
    """Load an audio file as mono float64."""
    audio, sr = sf.read(path)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    return audio.astype(np.float64), sr


def estimate_noise_power(
    noise,
    n_fft=512,
    hop_length=128
):
    """Estimate average noise power spectrum."""

    _, _, noise_stft = signal.stft(
        noise,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        boundary="zeros"
    )

    noise_power = np.mean(
        np.abs(noise_stft) ** 2,
        axis=1,
        keepdims=True
    )

    return noise_power


def estimate_noise_magnitude(
    noise,
    n_fft=512,
    hop_length=128
):
    """Estimate average noise magnitude spectrum."""

    _, _, noise_stft = signal.stft(
        noise,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        boundary="zeros"
    )

    noise_magnitude = np.mean(
        np.abs(noise_stft),
        axis=1,
        keepdims=True
    )

    return noise_magnitude


def wiener_filter(
    noisy,
    noise,
    sr,
    n_fft=512,
    hop_length=128,
    gain_floor=0.10
):
    """
    Classical Wiener filtering using an external
    noise-only recording.
    """

    noisy = np.asarray(noisy, dtype=np.float64)
    noise = np.asarray(noise, dtype=np.float64)

    _, _, Z = signal.stft(
        noisy,
        fs=sr,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        boundary="zeros"
    )

    noise_power = estimate_noise_power(
        noise,
        n_fft=n_fft,
        hop_length=hop_length
    )

    noisy_power = np.abs(Z) ** 2

    # Estimate clean speech power
    speech_power = np.maximum(
        noisy_power - noise_power,
        0.0
    )

    # Wiener gain
    gain = speech_power / (
        noisy_power + EPS
    )

    # Avoid complete spectral cancellation
    gain = np.maximum(
        gain,
        gain_floor
    )

    Z_enhanced = gain * Z

    _, enhanced = signal.istft(
        Z_enhanced,
        fs=sr,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        input_onesided=True
    )

    enhanced = enhanced[:len(noisy)]

    return normalize_audio(enhanced)


def spectral_suppression(
    noisy,
    noise,
    sr,
    n_fft=512,
    hop_length=128,
    oversubtraction=1.0,
    spectral_floor=0.10
):
    """
    Classical spectral subtraction using an external
    noise-only recording.
    """

    noisy = np.asarray(noisy, dtype=np.float64)
    noise = np.asarray(noise, dtype=np.float64)

    _, _, Z = signal.stft(
        noisy,
        fs=sr,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        boundary="zeros"
    )

    noise_magnitude = estimate_noise_magnitude(
        noise,
        n_fft=n_fft,
        hop_length=hop_length
    )

    magnitude = np.abs(Z)
    phase = np.angle(Z)

    # Spectral subtraction
    enhanced_magnitude = (
        magnitude
        - oversubtraction * noise_magnitude
    )

    # Keep a small spectral floor to reduce musical-noise artifacts
    floor = spectral_floor * magnitude

    enhanced_magnitude = np.maximum(
        enhanced_magnitude,
        floor
    )

    Z_enhanced = (
        enhanced_magnitude
        * np.exp(1j * phase)
    )

    _, enhanced = signal.istft(
        Z_enhanced,
        fs=sr,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        input_onesided=True
    )

    enhanced = enhanced[:len(noisy)]

    return normalize_audio(enhanced)


def process_file(
    input_path,
    noise_path,
    output_dir
):
    """Run both classical methods."""

    noisy, sr = load_mono(input_path)
    noise, noise_sr = load_mono(noise_path)

    if sr != noise_sr:
        raise ValueError(
            f"Sample-rate mismatch: "
            f"noisy={sr}, noise={noise_sr}"
        )

    # Use common length for noise estimation
    noise = noise[:len(noisy)]

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    wiener_output = wiener_filter(
        noisy,
        noise,
        sr
    )

    spectral_output = spectral_suppression(
        noisy,
        noise,
        sr
    )

    stem = input_path.stem

    wiener_path = (
        output_dir /
        f"{stem}_wiener.wav"
    )

    spectral_path = (
        output_dir /
        f"{stem}_spectral.wav"
    )

    sf.write(
        wiener_path,
        wiener_output,
        sr
    )

    sf.write(
        spectral_path,
        spectral_output,
        sr
    )

    print("Classical enhancement completed.")
    print(f"Noisy input:        {input_path}")
    print(f"Noise reference:    {noise_path}")
    print(f"Wiener output:      {wiener_path}")
    print(f"Spectral output:    {spectral_path}")
    print(f"Sample rate:        {sr} Hz")
    print(f"Samples:            {len(noisy)}")


def main():

    parser = argparse.ArgumentParser(
        description="M3-P3 classical enhancement baselines"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input noisy WAV file"
    )

    parser.add_argument(
        "--noise",
        required=True,
        help="Corresponding noise-only WAV file"
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for enhanced WAV files"
    )

    args = parser.parse_args()

    process_file(
        Path(args.input),
        Path(args.noise),
        Path(args.output_dir)
    )


if __name__ == "__main__":
    main()