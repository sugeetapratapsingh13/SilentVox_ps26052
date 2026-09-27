import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter

# ============================================================
# SILENTVOX - FxLMS SIMULATION
# ============================================================

FS = 16000
DURATION = 3.0

FILTER_LENGTH = 64
MU = 1e-5

DELAY_SAMPLES = 8
ATTENUATION = 0.7

SECONDARY_FILTER = np.array([
    0.20, 0.30, 0.30, 0.15, 0.05
])

# ============================================================
# 1. CREATE PRIMARY NOISE
# ============================================================

t = np.arange(0, DURATION, 1 / FS)

x = (
    0.5 * np.sin(2 * np.pi * 300 * t)
    + 0.3 * np.sin(2 * np.pi * 1000 * t)
    + 0.2 * np.sin(2 * np.pi * 2500 * t)
)

# Desired disturbance
d = x.copy()

# ============================================================
# 2. FILTERED-X
# ============================================================

x_filtered = lfilter(
    SECONDARY_FILTER,
    [1.0],
    x
)

# ============================================================
# 3. INITIALIZE CONTROLLER
# ============================================================

w = np.zeros(FILTER_LENGTH)

x_buffer = np.zeros(FILTER_LENGTH)
x_filtered_buffer = np.zeros(FILTER_LENGTH)

secondary_buffer = np.zeros(DELAY_SAMPLES + 1)

error = np.zeros(len(x))

# ============================================================
# 4. FxLMS ADAPTATION
# ============================================================

for n in range(len(x)):

    # Update reference buffer
    x_buffer[1:] = x_buffer[:-1]
    x_buffer[0] = x[n]

    # Controller output
    control_output = np.dot(w, x_buffer)

    # Apply secondary-path delay
    secondary_buffer[1:] = secondary_buffer[:-1]
    secondary_buffer[0] = control_output

    delayed_output = secondary_buffer[DELAY_SAMPLES]

    # Apply secondary-path filter
    secondary_output = ATTENUATION * (
        SECONDARY_FILTER[0] * delayed_output
    )

    for k in range(1, len(SECONDARY_FILTER)):
        if k <= DELAY_SAMPLES:
            secondary_output += ATTENUATION * (
                SECONDARY_FILTER[k]
                * secondary_buffer[DELAY_SAMPLES - k]
            )

    # Error signal
    error[n] = d[n] - secondary_output

    # Update filtered-X buffer
    x_filtered_buffer[1:] = x_filtered_buffer[:-1]
    x_filtered_buffer[0] = x_filtered[n]

    # FxLMS weight update
    w += MU * error[n] * x_filtered_buffer

# ============================================================
# 5. PERFORMANCE CALCULATION
# ============================================================

initial_power = np.mean(d ** 2)

# Ignore initial transient
steady_start = int(0.5 * FS)

final_power = np.mean(
    error[steady_start:] ** 2
)

noise_reduction_db = 10 * np.log10(
    initial_power / final_power
)

print("\n==========================================")
print(" SILENTVOX - FxLMS SIMULATION")
print("==========================================")

print(f"Sampling frequency : {FS} Hz")
print(f"Duration            : {DURATION} seconds")
print(f"Filter length       : {FILTER_LENGTH}")
print(f"Learning rate       : {MU}")

print("\nPerformance:")
print(f"Initial power       : {initial_power:.8f}")
print(f"Final power         : {final_power:.8f}")
print(f"Noise reduction     : {noise_reduction_db:.2f} dB")

# ============================================================
# 6. CONVERGENCE DATA
# ============================================================

WINDOW = 800

error_power = error ** 2

smoothed_power = np.convolve(
    error_power,
    np.ones(WINDOW) / WINDOW,
    mode="same"
)

power_db = 10 * np.log10(
    smoothed_power + 1e-12
)

# ============================================================
# 7. CREATE GRAPH WINDOW
# ============================================================

fig, axes = plt.subplots(
    3,
    1,
    figsize=(14, 16)
)

# Increase vertical space between graphs
fig.subplots_adjust(
    hspace=0.50,
    top=0.94,
    bottom=0.06,
    left=0.08,
    right=0.96
)

# ============================================================
# GRAPH 1 - ORIGINAL VS RESIDUAL
# ============================================================

plot_samples = int(0.05 * FS)

axes[0].plot(
    t[:plot_samples],
    d[:plot_samples],
    label="Original disturbance"
)

axes[0].plot(
    t[:plot_samples],
    error[:plot_samples],
    label="Residual error"
)

axes[0].set_title(
    "Original Disturbance vs Residual Error",
    fontsize=14,
    pad=12
)

axes[0].set_xlabel("Time (seconds)")
axes[0].set_ylabel("Amplitude")

axes[0].legend()
axes[0].grid(True)

# ============================================================
# GRAPH 2 - CONVERGENCE CURVE
# ============================================================

axes[1].plot(
    t,
    power_db
)

axes[1].set_title(
    "FxLMS Convergence Curve",
    fontsize=14,
    pad=12
)

axes[1].set_xlabel("Time (seconds)")
axes[1].set_ylabel("Error Power (dB)")

axes[1].grid(True)

# ============================================================
# GRAPH 3 - STEADY STATE
# ============================================================

steady_samples = int(0.1 * FS)

axes[2].plot(
    t[-steady_samples:],
    d[-steady_samples:],
    label="Original disturbance"
)

axes[2].plot(
    t[-steady_samples:],
    error[-steady_samples:],
    label="Residual error"
)

axes[2].set_title(
    "Steady-State Comparison",
    fontsize=14,
    pad=12
)

axes[2].set_xlabel("Time (seconds)")
axes[2].set_ylabel("Amplitude")

axes[2].legend()
axes[2].grid(True)

# ============================================================
# 8. SHOW ALL GRAPHS
# ============================================================

plt.show()

# ============================================================
# END
# ============================================================