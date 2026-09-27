import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz

# ============================================================
# SILENTVOX - SECONDARY PATH ANALYSIS
# ============================================================

FS = 16000

DELAY_SAMPLES = 8
ATTENUATION = 0.7

SECONDARY_FILTER = np.array([
    0.20, 0.30, 0.30, 0.15, 0.05
])

# ============================================================
# 1. FREQUENCY RESPONSE OF SECONDARY PATH
# ============================================================

frequencies, response = freqz(
    SECONDARY_FILTER,
    worN=2048,
    fs=FS
)

magnitude = np.abs(response)

magnitude_db = 20 * np.log10(
    magnitude + 1e-12
)

# Include attenuation
total_magnitude_db = (
    magnitude_db
    + 20 * np.log10(ATTENUATION)
)

# ============================================================
# 2. PHASE RESPONSE
# ============================================================

phase = np.unwrap(
    np.angle(response)
)

phase_degrees = np.degrees(phase)

# ============================================================
# 3. PRINT INFORMATION
# ============================================================

print("\n==========================================")
print(" SILENTVOX - SECONDARY PATH ANALYSIS")
print("==========================================")

print(f"Sampling frequency : {FS} Hz")
print(f"Delay              : {DELAY_SAMPLES} samples")
print(f"Delay time         : "
      f"{DELAY_SAMPLES / FS * 1000:.3f} ms")

print(f"Attenuation        : {ATTENUATION}")

print("\nSecondary-path filter coefficients:")

for i, coefficient in enumerate(SECONDARY_FILTER):
    print(f"h[{i}] = {coefficient:.3f}")

# ============================================================
# 4. MAXIMUM RESPONSE
# ============================================================

max_index = np.argmax(total_magnitude_db)

print("\nFrequency response:")
print(
    f"Maximum magnitude : "
    f"{total_magnitude_db[max_index]:.2f} dB"
)

print(
    f"At frequency      : "
    f"{frequencies[max_index]:.1f} Hz"
)

# ============================================================
# 5. PLOT 1 - MAGNITUDE RESPONSE
# ============================================================

plt.figure(figsize=(12, 7))

plt.plot(
    frequencies,
    total_magnitude_db
)

plt.title(
    "Secondary Path - Magnitude Frequency Response",
    fontsize=15
)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")

plt.xlim(0, FS / 2)

plt.grid(True)

plt.tight_layout()

plt.show()

# ============================================================
# 6. PLOT 2 - PHASE RESPONSE
# ============================================================

plt.figure(figsize=(12, 7))

plt.plot(
    frequencies,
    phase_degrees
)

plt.title(
    "Secondary Path - Phase Response",
    fontsize=15
)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase (degrees)")

plt.xlim(0, FS / 2)

plt.grid(True)

plt.tight_layout()

plt.show()

# ============================================================
# END
# ============================================================