"""
visualize.py

Plotting helpers: waveform before/after, spectrogram before/after,
convergence curve, and a cross-condition noise-reduction summary chart.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram


def plot_waveform(before, after, sample_rate, title, filename):
    t = np.arange(len(before)) / sample_rate

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(t, before, linewidth=0.5)
    axes[0].set_title(f"{title} - Before (Noisy)")
    axes[0].set_ylabel("Amplitude")

    axes[1].plot(t, after, linewidth=0.5, color="green")
    axes[1].set_title(f"{title} - After (Residual)")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_spectrogram(before, after, sample_rate, title, filename):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, sig, label in [(axes[0], before, "Before"), (axes[1], after, "After")]:
        f, tt, Sxx = spectrogram(sig, fs=sample_rate, nperseg=256, noverlap=128)
        ax.pcolormesh(tt, f, 10 * np.log10(Sxx + 1e-12), shading="gouraud")
        ax.set_title(f"{title} - {label}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_convergence(convergence_curve, title, filename):
    plt.figure(figsize=(10, 4))
    curve = convergence_curve + 1e-12
    plt.plot(10 * np.log10(curve), linewidth=0.7)
    plt.xlabel("Iteration")
    plt.ylabel("Squared Error (dB)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_noise_reduction_summary(results_df, filename):
    pivot = results_df.pivot_table(
        index="Condition", columns="Algorithm", values="Noise_Reduction_dB"
    )
    ax = pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_ylabel("Noise Reduction (dB)")
    ax.set_title("Noise Reduction by Condition and Algorithm")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()