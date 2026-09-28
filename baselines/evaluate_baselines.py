"""
M3-P3 Classical Enhancement Evaluation

Evaluates:
1. No enhancement
2. Wiener filtering
3. Spectral suppression

Metrics:
- STOI
- SI-SDR
- PESQ (when available)
- Input SNR
- Output SNR
- SNR improvement
- Speech distortion
- Residual noise
- Processing time
"""

from pathlib import Path
import time

import numpy as np
import pandas as pd
import soundfile as sf
from pystoi import stoi

try:
    from pesq import pesq
    PESQ_AVAILABLE = True
except ImportError:
    pesq = None
    PESQ_AVAILABLE = False


INPUT_DIR = Path(
    r"C:\Users\Tanisha\Desktop\Desktop\Person6_ANC_Benchmark\input_audio"
)

OUTPUT_DIR = Path("./baseline_outputs")
RESULTS_FILE = Path("./baseline_results.csv")

CLEAN_FILE = INPUT_DIR / "clean_signal.wav"

NOISE_PAIRS = {
    "environmental": (
        "noisy_environmental.wav",
        "environmental_noise.wav",
    ),
    "pink": (
        "noisy_pink.wav",
        "pink_noise.wav",
    ),
    "sine": (
        "noisy_sine.wav",
        "sine_noise.wav",
    ),
    "white": (
        "noisy_white.wav",
        "white_noise.wav",
    ),
}

EPS = 1e-12


def load_audio(path):
    """Load audio as mono floating-point data."""

    audio, sr = sf.read(path)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    return audio.astype(np.float64), sr


def align_signals(reference, estimate):
    """Trim signals to the same length."""

    length = min(
        len(reference),
        len(estimate)
    )

    return (
        reference[:length],
        estimate[:length]
    )


def rms(x):
    """Root-mean-square amplitude."""

    return np.sqrt(
        np.mean(x ** 2) + EPS
    )


def calculate_snr(clean, estimate):
    """Calculate SNR relative to the clean reference."""

    clean, estimate = align_signals(
        clean,
        estimate
    )

    noise = estimate - clean

    signal_power = np.mean(
        clean ** 2
    )

    noise_power = np.mean(
        noise ** 2
    )

    return 10 * np.log10(
        (signal_power + EPS)
        /
        (noise_power + EPS)
    )


def calculate_speech_distortion(
    clean,
    enhanced
):
    """
    Speech distortion relative to clean speech.

    More negative values indicate a smaller
    error relative to the clean signal.
    """

    clean, enhanced = align_signals(
        clean,
        enhanced
    )

    distortion = enhanced - clean

    return 20 * np.log10(
        (rms(distortion) + EPS)
        /
        (rms(clean) + EPS)
    )


def calculate_residual_noise(
    clean,
    enhanced
):
    """
    Residual error/noise level relative to
    the clean reference.
    """

    clean, enhanced = align_signals(
        clean,
        enhanced
    )

    residual = enhanced - clean

    return 20 * np.log10(
        rms(residual) + EPS
    )


def calculate_si_sdr(
    reference,
    estimate
):
    """Calculate Scale-Invariant SDR."""

    reference, estimate = align_signals(
        reference,
        estimate
    )

    reference = (
        reference
        - np.mean(reference)
    )

    estimate = (
        estimate
        - np.mean(estimate)
    )

    reference_energy = np.sum(
        reference ** 2
    )

    if reference_energy < EPS:
        return np.nan

    scale = (
        np.dot(
            estimate,
            reference
        )
        /
        reference_energy
    )

    target = scale * reference

    error = (
        estimate - target
    )

    target_energy = np.sum(
        target ** 2
    )

    error_energy = np.sum(
        error ** 2
    )

    return 10 * np.log10(
        (target_energy + EPS)
        /
        (error_energy + EPS)
    )


def calculate_stoi(
    clean,
    enhanced,
    sr
):
    """Calculate STOI."""

    clean, enhanced = align_signals(
        clean,
        enhanced
    )

    try:

        return stoi(
            clean,
            enhanced,
            sr,
            extended=False
        )

    except Exception as exc:

        print(
            f"STOI warning: {exc}"
        )

        return np.nan


def calculate_pesq(
    clean,
    enhanced,
    sr
):
    """
    Calculate wideband PESQ.

    PESQ is optional because the current
    Python environment may not provide
    a compatible PESQ installation.
    """

    if not PESQ_AVAILABLE:
        return np.nan

    clean, enhanced = align_signals(
        clean,
        enhanced
    )

    if sr != 16000:
        return np.nan

    try:

        return pesq(
            sr,
            clean,
            enhanced,
            "wb"
        )

    except Exception as exc:

        print(
            f"PESQ warning: {exc}"
        )

        return np.nan


def evaluate_method(
    clean,
    noisy,
    enhanced,
    sr,
    method,
    noise_type,
    processing_time
):
    """Calculate all evaluation metrics."""

    input_snr = calculate_snr(
        clean,
        noisy
    )

    output_snr = calculate_snr(
        clean,
        enhanced
    )

    return {
        "noise_type": noise_type,

        "method": method,

        "sample_rate": sr,

        "stoi": calculate_stoi(
            clean,
            enhanced,
            sr
        ),

        "si_sdr_db": calculate_si_sdr(
            clean,
            enhanced
        ),

        "pesq": calculate_pesq(
            clean,
            enhanced,
            sr
        ),

        "input_snr_db": input_snr,

        "output_snr_db": output_snr,

        "snr_improvement_db": (
            output_snr
            - input_snr
        ),

        "speech_distortion_db": (
            calculate_speech_distortion(
                clean,
                enhanced
            )
        ),

        "residual_noise_db": (
            calculate_residual_noise(
                clean,
                enhanced
            )
        ),

        "processing_time_sec": (
            processing_time
        ),
    }


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    clean, clean_sr = load_audio(
        CLEAN_FILE
    )

    rows = []

    # Import our classical algorithms.
    from classical_enhancement import (
        wiener_filter,
        spectral_suppression
    )

    for noise_type, (
        noisy_filename,
        noise_filename
    ) in NOISE_PAIRS.items():

        noisy_path = (
            INPUT_DIR
            / noisy_filename
        )

        noise_path = (
            INPUT_DIR
            / noise_filename
        )

        noisy, sr = load_audio(
            noisy_path
        )

        noise, noise_sr = load_audio(
            noise_path
        )

        if sr != clean_sr:
            raise ValueError(
                f"Sample-rate mismatch: "
                f"{noisy_filename}={sr}, "
                f"clean={clean_sr}"
            )

        if noise_sr != clean_sr:
            raise ValueError(
                f"Sample-rate mismatch: "
                f"{noise_filename}={noise_sr}, "
                f"clean={clean_sr}"
            )

        print()
        print("=" * 60)
        print(
            f"Noise condition: {noise_type}"
        )
        print(
            f"Noisy file:      {noisy_filename}"
        )
        print(
            f"Noise reference: {noise_filename}"
        )
        print("=" * 60)

        # --------------------------------------------------
        # 1. NO ENHANCEMENT
        # --------------------------------------------------

        start = time.perf_counter()

        no_enhancement = noisy.copy()

        processing_time = (
            time.perf_counter()
            - start
        )

        rows.append(
            evaluate_method(
                clean,
                noisy,
                no_enhancement,
                sr,
                "no_enhancement",
                noise_type,
                processing_time
            )
        )

        # --------------------------------------------------
        # 2. WIENER FILTER
        # --------------------------------------------------

        start = time.perf_counter()

        wiener_output = wiener_filter(
            noisy,
            noise,
            sr
        )

        processing_time = (
            time.perf_counter()
            - start
        )

        wiener_path = (
            OUTPUT_DIR
            / f"{noise_type}_wiener.wav"
        )

        sf.write(
            wiener_path,
            wiener_output,
            sr
        )

        rows.append(
            evaluate_method(
                clean,
                noisy,
                wiener_output,
                sr,
                "wiener",
                noise_type,
                processing_time
            )
        )

        # --------------------------------------------------
        # 3. SPECTRAL SUPPRESSION
        # --------------------------------------------------

        start = time.perf_counter()

        spectral_output = (
            spectral_suppression(
                noisy,
                noise,
                sr
            )
        )

        processing_time = (
            time.perf_counter()
            - start
        )

        spectral_path = (
            OUTPUT_DIR
            / f"{noise_type}_spectral.wav"
        )

        sf.write(
            spectral_path,
            spectral_output,
            sr
        )

        rows.append(
            evaluate_method(
                clean,
                noisy,
                spectral_output,
                sr,
                "spectral_suppression",
                noise_type,
                processing_time
            )
        )

    # ------------------------------------------------------
    # SAVE RESULTS
    # ------------------------------------------------------

    results = pd.DataFrame(
        rows
    )

    results.to_csv(
        RESULTS_FILE,
        index=False
    )

    print()
    print("=" * 60)
    print(
        "M3-P3 BASELINE EVALUATION COMPLETE"
    )
    print("=" * 60)

    print(
        f"Results: {RESULTS_FILE}"
    )

    print(
        f"Rows:    {len(results)}"
    )

    print()

    print(
        results[
            [
                "noise_type",
                "method",
                "stoi",
                "si_sdr_db",
                "pesq",
                "input_snr_db",
                "output_snr_db",
                "snr_improvement_db",
                "speech_distortion_db",
                "residual_noise_db",
            ]
        ].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
