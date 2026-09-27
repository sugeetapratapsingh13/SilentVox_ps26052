import sys
import os
import numpy as np
import soundfile as sf
import csv

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.append(PROJECT_ROOT)

from LMS import lms_filter
from NLMS import nlms_filter
from benchmark import (
    calculate_noise_reduction_db,
    check_stability
)

INPUT_DIR = os.path.join(PROJECT_ROOT, "input_audio")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

SAMPLING_RATE = 16000
FILTER_LENGTH = 32
STEP_SIZE = 0.001
EPSILON = 1e-8

tests = [
    ("sine", "sine_noise.wav", "noisy_sine.wav"),
    ("white", "white_noise.wav", "noisy_white.wav"),
    ("pink", "pink_noise.wav", "noisy_pink.wav"),
    ("environmental", "environmental_noise.wav", "noisy_environmental.wav")
]

results = []

print("\n========================================")
print("       PERSON 6 ANC BENCHMARK")
print("========================================")

for name, reference_name, desired_name in tests:

    print(f"\n========== {name.upper()} NOISE ==========")

    reference_file = os.path.join(
        INPUT_DIR, reference_name
    )

    desired_file = os.path.join(
        INPUT_DIR, desired_name
    )

    reference_signal, reference_sr = sf.read(
        reference_file
    )

    desired_signal, desired_sr = sf.read(
        desired_file
    )

    if reference_signal.ndim > 1:
        reference_signal = np.mean(
            reference_signal, axis=1
        )

    if desired_signal.ndim > 1:
        desired_signal = np.mean(
            desired_signal, axis=1
        )

    if reference_sr != desired_sr:
        raise ValueError(
            f"Sampling rates do not match for {name}"
        )

    length = min(
        len(reference_signal),
        len(desired_signal)
    )

    reference_signal = reference_signal[:length]
    desired_signal = desired_signal[:length]

    # ---------------- LMS ----------------

    (
        lms_output,
        lms_estimated,
        lms_coefficients,
        lms_convergence,
        lms_time
    ) = lms_filter(
        reference_signal,
        desired_signal,
        SAMPLING_RATE,
        FILTER_LENGTH,
        STEP_SIZE,
        EPSILON
    )

    lms_reduction = calculate_noise_reduction_db(
        desired_signal,
        lms_output
    )

    lms_stable, lms_status = check_stability(
        lms_output
    )

    # ---------------- NLMS ----------------

    (
        nlms_output,
        nlms_estimated,
        nlms_coefficients,
        nlms_convergence,
        nlms_time
    ) = nlms_filter(
        reference_signal,
        desired_signal,
        SAMPLING_RATE,
        FILTER_LENGTH,
        STEP_SIZE,
        EPSILON
    )

    nlms_reduction = calculate_noise_reduction_db(
        desired_signal,
        nlms_output
    )

    nlms_stable, nlms_status = check_stability(
        nlms_output
    )

    print(
        f"LMS  -> Time: {lms_time:.6f}s | "
        f"Reduction: {lms_reduction:.4f} dB | "
        f"{lms_status}"
    )

    print(
        f"NLMS -> Time: {nlms_time:.6f}s | "
        f"Reduction: {nlms_reduction:.4f} dB | "
        f"{nlms_status}"
    )

    results.append([
        name,
        "LMS",
        lms_time,
        lms_reduction,
        lms_status
    ])

    results.append([
        name,
        "NLMS",
        nlms_time,
        nlms_reduction,
        nlms_status
    ])

# ---------------- SAVE CSV ----------------

csv_file = os.path.join(
    RESULTS_DIR,
    "ANC_MULTI_NOISE_RESULTS.csv"
)

with open(
    csv_file,
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "Noise Type",
        "Algorithm",
        "Processing Time (s)",
        "Noise Reduction (dB)",
        "Stability"
    ])

    writer.writerows(results)

print("\n========================================")
print("ALL BENCHMARKS COMPLETED")
print("========================================")

print(
    "Results saved to:",
    csv_file
)