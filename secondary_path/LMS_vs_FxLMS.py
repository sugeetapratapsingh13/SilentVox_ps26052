import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter

# ============================================================
# SILENTVOX - LMS vs FxLMS COMPARISON
# ============================================================

FS = 16000
DURATION = 3.0

FILTER_LENGTH = 64
MU_LMS = 1e-5
MU_FXLMS = 1e-5

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

d = x.copy()

# ============================================================
# 2. SECONDARY PATH
# ============================================================

def secondary_path(signal):

    filtered = lfilter(
        SECONDARY_FILTER,
        [1.0],
        signal
    )

    delayed = np.zeros_like(filtered)

    delayed[DELAY_SAMPLES:] = (
        filtered[:-DELAY_SAMPLES]
    )

    return ATTENUATION * delayed


# ============================================================
# 3. ORDINARY LMS
# ============================================================

def run_lms():

    w = np.zeros(FILTER_LENGTH)

    x_buffer = np.zeros(FILTER_LENGTH)

    error = np.zeros(len(x))

    control_signal = np.zeros(len(x))

    for n in range(len(x)):

        x_buffer[1:] = x_buffer[:-1]
        x_buffer[0] = x[n]

        y = np.dot(w, x_buffer)

        control_signal[n] = y

        anti_noise = secondary_path(control_signal)[n]

        error[n] = d[n] - anti_noise

        w += MU_LMS * error[n] * x_buffer

    return error


# ============================================================
# 4. FxLMS
# ============================================================

def run_fxlms():

    w = np.zeros(FILTER_LENGTH)

    x_buffer = np.zeros(FILTER_LENGTH)
    filtered_x_buffer = np.zeros(FILTER_LENGTH)

    error = np.zeros(len(x))

    control_signal = np.zeros(len(x))

    filtered_x = lfilter(
        SECONDARY_FILTER,
        [1.0],
        x
    )

    filtered_x_delayed = np.zeros_like(filtered_x)

    filtered_x_delayed[DELAY_SAMPLES:] = (
        filtered_x[:-DELAY_SAMPLES]
    )

    filtered_x_delayed *= ATTENUATION

    for n in range(len(x)):

        x_buffer[1:] = x_buffer[:-1]
        x_buffer[0] = x[n]

        y = np.dot(w, x_buffer)

        control_signal[n] = y

        anti_noise = secondary_path(control_signal)[n]

        error[n] = d[n] - anti_noise

        filtered_x_buffer[1:] = filtered_x_buffer[:-1]
        filtered_x_buffer[0] = filtered_x_delayed[n]

        w += MU_FXLMS * error[n] * filtered_x_buffer

    return error


# ============================================================
# 5. RUN BOTH
# ============================================================

print("\n==========================================")
print(" SILENTVOX - LMS vs FxLMS")
print("==========================================")

print("\nRunning ordinary LMS...")
error_lms = run_lms()

print("Running FxLMS...")
error_fxlms = run_fxlms()


# ============================================================
# 6. PERFORMANCE
# ============================================================

steady_start = int(0.5 * FS)

initial_power = np.mean(
    d[steady_start:] ** 2
)

lms_power = np.mean(
    error_lms[steady_start:] ** 2
)

fxlms_power = np.mean(
    error_fxlms[steady_start:] ** 2
)

lms_reduction = 10 * np.log10(
    initial_power / lms_power
)

fxlms_reduction = 10 * np.log10(
    initial_power / fxlms_power
)


print("\nPerformance:")
print("------------------------------------------")

print(f"Initial disturbance power : {initial_power:.8f}")

print("\nLMS:")
print(f"Final error power         : {lms_power:.8f}")
print(f"Noise reduction           : {lms_reduction:.2f} dB")

print("\nFxLMS:")
print(f"Final error power         : {fxlms_power:.8f}")
print(f"Noise reduction           : {fxlms_reduction:.2f} dB")

print("\n==========================================")


# ============================================================
# 7. CONVERGENCE DATA
# ============================================================

WINDOW = 800

lms_smoothed = np.convolve(
    error_lms ** 2,
    np.ones(WINDOW) / WINDOW,
    mode="same"
)

fxlms_smoothed = np.convolve(
    error_fxlms ** 2,
    np.ones(WINDOW) / WINDOW,
    mode="same"
)

lms_db = 10 * np.log10(
    lms_smoothed + 1e-12
)

fxlms_db = 10 * np.log10(
    fxlms_smoothed + 1e-12
)


# ============================================================
# GRAPH 1 - LMS vs FxLMS CONVERGENCE
# ============================================================

plt.figure(figsize=(12, 7))

plt.plot(
    t,
    lms_db,
    label="LMS"
)

plt.plot(
    t,
    fxlms_db,
    label="FxLMS"
)

plt.title(
    "LMS vs FxLMS - Convergence Comparison",
    fontsize=15
)

plt.xlabel("Time (seconds)")
plt.ylabel("Error Power (dB)")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# GRAPH 2 - LMS RESIDUAL
# ============================================================

plot_samples = int(0.05 * FS)

plt.figure(figsize=(12, 7))

plt.plot(
    t[:plot_samples],
    d[:plot_samples],
    label="Original disturbance"
)

plt.plot(
    t[:plot_samples],
    error_lms[:plot_samples],
    label="LMS residual"
)

plt.title(
    "Ordinary LMS - Original vs Residual",
    fontsize=15
)

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# GRAPH 3 - FxLMS RESIDUAL
# ============================================================

plt.figure(figsize=(12, 7))

plt.plot(
    t[:plot_samples],
    d[:plot_samples],
    label="Original disturbance"
)

plt.plot(
    t[:plot_samples],
    error_fxlms[:plot_samples],
    label="FxLMS residual"
)

plt.title(
    "FxLMS - Original vs Residual",
    fontsize=15
)

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# END
# ============================================================