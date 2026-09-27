import os
import sys
import time
import numpy as np
import pandas as pd
import soundfile as sf

# Allow import from project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from NLMS import nlms_filter


# ==========================================================
# CONFIGURATION
# ==========================================================

SAMPLE_RATE = 16000
FILTER_LENGTH = 128
STEP_SIZE = 0.001

EPSILON_VALUES = [
    1e-10,
    1e-8,
    1e-6,
    1e-4
]

REFERENCE_FILE = os.path.join(
    PROJECT_ROOT,
    "input_audio",
    "environmental_noise.wav"
)

DESIRED_FILE = os.path.join(
    PROJECT_ROOT,
    "input_audio",
    "noisy_environmental.wav"
)

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "EPSILON_SWEEP.csv"
)


# ==========================================================
# LOAD AUDIO
# ==========================================================

reference, ref_sr = sf.read(REFERENCE_FILE)
desired, desired_sr = sf.read(DESIRED_FILE)

reference = np.asarray(reference, dtype=np.float64).flatten()
desired = np.asarray(desired, dtype=np.float64).flatten()

n = min(len(reference), len(desired))

reference = reference[:n]
desired = desired[:n]

print("=" * 65)
print("              NLMS EPSILON SWEEP")
print("=" * 65)

print(f"Reference sample rate : {ref_sr} Hz")
print(f"Desired sample rate   : {desired_sr} Hz")
print(f"Samples tested        : {n}")
print(f"Filter length         : {FILTER_LENGTH}")
print(f"Step size (mu)        : {STEP_SIZE}")
print()


# ==========================================================
# RUN EXPERIMENTS
# ==========================================================

results = []

for epsilon in EPSILON_VALUES:

    start = time.perf_counter()

    error_signal, estimated_noise, coefficient_history, \
        convergence_curve, processing_time = nlms_filter(
            reference,
            desired,
            SAMPLE_RATE,
            FILTER_LENGTH,
            STEP_SIZE,
            epsilon
        )

    total_time = time.perf_counter() - start

    # Input/output RMS
    input_rms = np.sqrt(np.mean(desired ** 2))
    output_rms = np.sqrt(np.mean(error_signal ** 2))

    # Noise reduction in dB
    if output_rms > 0:
        noise_reduction_db = 20 * np.log10(
            input_rms / output_rms
        )
    else:
        noise_reduction_db = 0

    # Stability check
    if np.all(np.isfinite(error_signal)):
        stability = "STABLE"
    else:
        stability = "UNSTABLE"

    results.append({
        "Algorithm": "NLMS",
        "Filter_Length": FILTER_LENGTH,
        "Step_Size": STEP_SIZE,
        "Epsilon": epsilon,
        "Processing_Time_s": round(processing_time, 6),
        "Total_Runtime_s": round(total_time, 6),
        "Noise_Reduction_dB": round(noise_reduction_db, 6),
        "Input_RMS": round(input_rms, 8),
        "Output_RMS": round(output_rms, 8),
        "Stability": stability
    })

    print("-" * 65)
    print(f"Epsilon          : {epsilon}")
    print(f"Processing time   : {processing_time:.6f} s")
    print(f"Noise reduction   : {noise_reduction_db:.4f} dB")
    print(f"Stability         : {stability}")


# ==========================================================
# SAVE RESULTS
# ==========================================================

df = pd.DataFrame(results)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("=" * 65)
print("EPSILON SWEEP COMPLETE")
print("=" * 65)
print()
print(f"Created:")
print(OUTPUT_FILE)