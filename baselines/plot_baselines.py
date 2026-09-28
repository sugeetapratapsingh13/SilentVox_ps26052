"""
M3-P3 Classical Baseline Plotting

Reads baseline_results.csv and generates comparison plots
for the classical speech-enhancement baselines.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


RESULTS_FILE = Path("./baseline_results.csv")
PLOT_DIR = Path("./baseline_plots")

PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


df = pd.read_csv(RESULTS_FILE)

method_order = [
    "no_enhancement",
    "wiener",
    "spectral_suppression",
]

method_labels = {
    "no_enhancement": "No enhancement",
    "wiener": "Wiener",
    "spectral_suppression": "Spectral suppression",
}

noise_order = [
    "environmental",
    "pink",
    "sine",
    "white",
]


def plot_metric(
    column,
    ylabel,
    filename,
    title,
):
    """Generate grouped bar chart for one metric."""

    pivot = df.pivot(
        index="noise_type",
        columns="method",
        values=column
    )

    pivot = pivot.reindex(
        noise_order
    )

    pivot = pivot[
        [
            m for m in method_order
            if m in pivot.columns
        ]
    ]

    pivot = pivot.rename(
        columns=method_labels
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(10, 6)
    )

    ax.set_title(title)
    ax.set_xlabel("Noise condition")
    ax.set_ylabel(ylabel)

    ax.legend(
        title="Method"
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / filename,
        dpi=200
    )

    plt.close()


# ---------------------------------------------------------
# STOI
# ---------------------------------------------------------

plot_metric(
    "stoi",
    "STOI",
    "stoi_comparison.png",
    "STOI Comparison"
)


# ---------------------------------------------------------
# SI-SDR
# ---------------------------------------------------------

plot_metric(
    "si_sdr_db",
    "SI-SDR (dB)",
    "si_sdr_comparison.png",
    "SI-SDR Comparison"
)


# ---------------------------------------------------------
# SNR improvement
# ---------------------------------------------------------

plot_metric(
    "snr_improvement_db",
    "SNR improvement (dB)",
    "snr_improvement_comparison.png",
    "SNR Improvement Comparison"
)


# ---------------------------------------------------------
# Speech distortion
# ---------------------------------------------------------

plot_metric(
    "speech_distortion_db",
    "Speech distortion (dB)",
    "speech_distortion_comparison.png",
    "Speech Distortion Comparison"
)


# ---------------------------------------------------------
# Residual noise
# ---------------------------------------------------------

plot_metric(
    "residual_noise_db",
    "Residual noise (dB)",
    "residual_noise_comparison.png",
    "Residual Noise Comparison"
)


# ---------------------------------------------------------
# Processing time
# ---------------------------------------------------------

plot_metric(
    "processing_time_sec",
    "Processing time (seconds)",
    "processing_time_comparison.png",
    "Processing Time Comparison"
)


print()
print("=" * 60)
print("M3-P3 BASELINE PLOTS GENERATED")
print("=" * 60)

for path in sorted(
    PLOT_DIR.glob("*.png")
):
    print(path)