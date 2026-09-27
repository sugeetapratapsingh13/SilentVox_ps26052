import time
import numpy as np
from scipy.signal import lfilter
import pandas as pd
import os

# ==========================================================
# FxLMS BENCHMARK
# ==========================================================

FS = 16000
DURATION = 3.0
FILTER_LENGTH = 64
MU = 1e-5

DELAY_SAMPLES = 8
ATTENUATION = 0.7

SECONDARY_FILTER = np.array([
    0.20, 0.30, 0.30, 0.15, 0.05
])

# Create reference noise
t = np.arange(0, DURATION, 1 / FS)

x = (
    0.5 * np.sin(2 * np.pi * 300 * t)
    + 0.3 * np.sin(2 * np.pi * 1000 * t)
    + 0.2 * np.sin(2 * np.pi * 2500 * t)
)

d = x.copy()

# Filtered reference
x_filtered = lfilter(
    SECONDARY_FILTER,
    [1.0],
    x
)

w = np.zeros(FILTER_LENGTH)

x_buffer = np.zeros(FILTER_LENGTH)
x_filtered_buffer = np.zeros(FILTER_LENGTH)

secondary_buffer = np.zeros(DELAY_SAMPLES + 1)

error = np.zeros(len(x))

# ==========================================================
# RUN FxLMS
# ==========================================================

start_time = time.perf_counter()

for n in range(len(x)):

    x_buffer[1:] = x_buffer[:-1]
    x_buffer[0] = x[n]

    control_output = np.dot(w, x_buffer)

    secondary_buffer[1:] = secondary_buffer[:-1]
    secondary_buffer[0] = control_output

    delayed_output = secondary_buffer[DELAY_SAMPLES]

    secondary_output = ATTENUATION * (
        SECONDARY_FILTER[0] * delayed_output
    )

    for k in range(1, len(SECONDARY_FILTER)):
        if k <= DELAY_SAMPLES:
            secondary_output += ATTENUATION * (
                SECONDARY_FILTER[k]
                * secondary_buffer[DELAY_SAMPLES - k]
            )

    error[n] = d[n] - secondary_output

    x_filtered_buffer[1:] = x_filtered_buffer[:-1]
    x_filtered_buffer[0] = x_filtered[n]

    w += MU * error[n] * x_filtered_buffer

processing_time = time.perf_counter() - start_time

# ==========================================================
# PERFORMANCE
# ==========================================================

initial_power = np.mean(d ** 2)

steady_start = int(0.5 * FS)

final_power = np.mean(
    error[steady_start:] ** 2
)

noise_reduction_db = 10 * np.log10(
    initial_power / final_power
)

stable = np.all(np.isfinite(error)) and np.all(np.isfinite(w))

audio_duration = len(x) / FS
rtf = audio_duration / processing_time

avg_latency_ms = (
    processing_time / len(x)
) * 1000 * 16000

results = [{
    "Algorithm": "FxLMS",
    "Sampling_Rate": FS,
    "Filter_Length": FILTER_LENGTH,
    "Step_Size": MU,
    "Processing_Time_s": processing_time,
    "Noise_Reduction_dB": noise_reduction_db,
    "Average_Latency_ms": avg_latency_ms,
    "RTF": rtf,
    "Stability": "Stable" if stable else "Unstable",
    "Status": "PASS" if stable and rtf > 1 else "CHECK"
}]

df = pd.DataFrame(results)

output = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "FXLMS_BENCHMARK.csv"
)

df.to_csv(output, index=False)

print("=" * 65)
print("                 FxLMS BENCHMARK")
print("=" * 65)

print(f"Sampling rate    : {FS} Hz")
print(f"Duration         : {DURATION} s")
print(f"Filter length    : {FILTER_LENGTH}")
print(f"Step size        : {MU}")
print(f"Processing time  : {processing_time:.6f} s")
print(f"Noise reduction  : {noise_reduction_db:.4f} dB")
print(f"Average latency  : {avg_latency_ms:.4f} ms")
print(f"RTF              : {rtf:.4f}")
print(f"Stability        : {'STABLE' if stable else 'UNSTABLE'}")
print(f"Status           : {'PASS' if stable and rtf > 1 else 'CHECK'}")

print()
print("Created:")
print(output)