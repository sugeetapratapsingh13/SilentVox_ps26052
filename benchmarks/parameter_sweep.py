import os
import sys
import csv
import numpy as np
import soundfile as sf
import time

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.append(PROJECT_ROOT)

from LMS import lms_filter
from NLMS import nlms_filter
from benchmark import calculate_noise_reduction_db

INPUT_DIR = os.path.join(PROJECT_ROOT, "input_audio")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

# Input audio
reference, sr = sf.read(
    os.path.join(INPUT_DIR, "sine_noise.wav")
)

desired, desired_sr = sf.read(
    os.path.join(INPUT_DIR, "noisy_sine.wav")
)

if reference.ndim > 1:
    reference = np.mean(reference, axis=1)

if desired.ndim > 1:
    desired = np.mean(desired, axis=1)

length = min(len(reference), len(desired))

reference = reference[:length]
desired = desired[:length]

# Parameters to test
filter_lengths = [16, 32, 64, 128]
step_sizes = [0.0001, 0.0005, 0.001, 0.005, 0.01]

epsilon = 1e-8

results = []

print("\n========================================")
print("       ANC PARAMETER SWEEP")
print("========================================")

for filter_length in filter_lengths:

    for step_size in step_sizes:

        print(
            f"\nTesting Filter={filter_length}, "
            f"mu={step_size}"
        )

        # ---------------- LMS ----------------

        start = time.perf_counter()

        (
            lms_output,
            _,
            _,
            _,
            _
        ) = lms_filter(
            reference,
            desired,
            sr,
            filter_length,
            step_size,
            epsilon
        )

        lms_time = time.perf_counter() - start

        lms_reduction = calculate_noise_reduction_db(
            desired,
            lms_output
        )

        # ---------------- NLMS ----------------

        start = time.perf_counter()

        (
            nlms_output,
            _,
            _,
            _,
            _
        ) = nlms_filter(
            reference,
            desired,
            sr,
            filter_length,
            step_size,
            epsilon
        )

        nlms_time = time.perf_counter() - start

        nlms_reduction = calculate_noise_reduction_db(
            desired,
            nlms_output
        )

        results.append([
            filter_length,
            step_size,
            "LMS",
            lms_time,
            lms_reduction
        ])

        results.append([
            filter_length,
            step_size,
            "NLMS",
            nlms_time,
            nlms_reduction
        ])

        print(
            f"LMS  : {lms_reduction:.4f} dB | "
            f"{lms_time:.4f} s"
        )

        print(
            f"NLMS : {nlms_reduction:.4f} dB | "
            f"{nlms_time:.4f} s"
        )

# Save results
csv_file = os.path.join(
    RESULTS_DIR,
    "parameter_sweep.csv"
)

with open(
    csv_file,
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "Filter Length",
        "Step Size",
        "Algorithm",
        "Processing Time (s)",
        "Noise Reduction (dB)"
    ])

    writer.writerows(results)

print("\n========================================")
print("PARAMETER SWEEP COMPLETED")
print("========================================")

print(
    "Results saved to:",
    csv_file
)