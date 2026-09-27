"""
run_simulation.py

MAIN ENTRY POINT for Person 4's work.

Pipeline:
    Reference Noise -> Primary Path -> Noisy Signal -> Adaptive ANC (LMS/NLMS) -> Residual Error

Runs all 5 required noise conditions through both LMS and NLMS
(imported directly from Person 3's LMS.py / NLMS.py -- no modification
of their algorithms), measures the required metrics, saves all
visualizations, and writes a consolidated results CSV for Person 6.

USAGE:
    Place this file, primary_path.py, noise_conditions.py, metrics.py,
    visualize.py, LMS.py and NLMS.py all in the SAME folder, then run:

        python run_simulation.py
"""

import os
import numpy as np
import pandas as pd

from LMS import lms_filter
from NLMS import nlms_filter

from primary_path import make_primary_path, apply_primary_path
from noise_conditions import get_noise_condition, CONDITIONS
from metrics import compute_metrics
from visualize import (
    plot_waveform,
    plot_spectrogram,
    plot_convergence,
    plot_noise_reduction_summary,
)

# ============================================================
# SETTINGS -- change these to experiment
# ============================================================

SAMPLE_RATE = 16000
DURATION = 5  # seconds
N_SAMPLES = SAMPLE_RATE * DURATION

FILTER_LENGTH = 32
STEP_SIZE_LMS = 0.01
STEP_SIZE_NLMS = 0.5
EPSILON = 1e-8

OUTPUT_DIR = "simulation_outputs"

# ============================================================
# OUTPUT FOLDERS
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "waveforms"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "spectrograms"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "convergence"), exist_ok=True)

# ============================================================
# PRIMARY PATH (shared across all conditions)
# ============================================================

primary_path_ir = make_primary_path(
    delay_samples=5,
    tap_length=10,
    decay=0.6,
    attenuation=0.8,
)

# ============================================================
# MAIN LOOP
# ============================================================

results = []

for condition in CONDITIONS:

    print()
    print("=" * 60)
    print(f"CONDITION: {condition}")
    print("=" * 60)

    reference_noise, desired_clean = get_noise_condition(
        condition, N_SAMPLES, SAMPLE_RATE
    )

    # Push the reference noise through the simulated primary (acoustic) path
    primary_noise = apply_primary_path(reference_noise, primary_path_ir)

    # What the error microphone actually measures
    noisy_signal = desired_clean + primary_noise

    # Normalize to avoid clipping, keep reference on the same scale
    peak = np.max(np.abs(noisy_signal))
    if peak > 0:
        noisy_signal = noisy_signal / peak
        reference_noise = reference_noise / peak

    for algo_name, filter_func, step_size in [
        ("LMS", lms_filter, STEP_SIZE_LMS),
        ("NLMS", nlms_filter, STEP_SIZE_NLMS),
    ]:
        print(f"  Running {algo_name}...")

        (
            error_signal,
            estimated_noise,
            coefficients,
            convergence_curve,
            processing_time,
        ) = filter_func(
            reference_noise,
            noisy_signal,
            SAMPLE_RATE,
            FILTER_LENGTH,
            step_size,
            EPSILON,
        )

        m = compute_metrics(
            noisy_signal, error_signal, convergence_curve,
            FILTER_LENGTH, SAMPLE_RATE, processing_time,
        )
        m["Condition"] = condition
        m["Algorithm"] = algo_name
        m["Filter_Length"] = FILTER_LENGTH
        m["Step_Size"] = step_size
        m["Epsilon"] = EPSILON
        results.append(m)

        tag = f"{condition}_{algo_name}"

        plot_waveform(
            noisy_signal, error_signal, SAMPLE_RATE,
            f"{condition} - {algo_name}",
            os.path.join(OUTPUT_DIR, "waveforms", f"{tag}_waveform.png"),
        )

        plot_spectrogram(
            noisy_signal, error_signal, SAMPLE_RATE,
            f"{condition} - {algo_name}",
            os.path.join(OUTPUT_DIR, "spectrograms", f"{tag}_spectrogram.png"),
        )

        plot_convergence(
            convergence_curve,
            f"Convergence - {condition} - {algo_name}",
            os.path.join(OUTPUT_DIR, "convergence", f"{tag}_convergence.png"),
        )

        print(
            f"    Noise Reduction: {m['Noise_Reduction_dB']:.2f} dB | "
            f"Convergence: {m['Convergence_Time_s']:.3f}s | "
            f"Stable: {m['Stable']} | "
            f"Proc time: {m['Processing_Time_s']:.4f}s"
        )

# ============================================================
# SAVE CONSOLIDATED RESULTS
# ============================================================

results_df = pd.DataFrame(results)
csv_path = os.path.join(OUTPUT_DIR, "simulation_results.csv")
results_df.to_csv(csv_path, index=False)

plot_noise_reduction_summary(
    results_df,
    os.path.join(OUTPUT_DIR, "noise_reduction_summary.png"),
)

print()
print("=" * 60)
print("SIMULATION COMPLETE")
print("=" * 60)
print(f"Results CSV : {csv_path}")
print(f"Plots       : {OUTPUT_DIR}/waveforms, /spectrograms, /convergence")
print(f"Summary     : {OUTPUT_DIR}/noise_reduction_summary.png")
print()
print(results_df[[
    "Condition", "Algorithm", "Noise_Reduction_dB",
    "Convergence_Time_s", "Stable", "Processing_Time_s"
]].to_string(index=False))