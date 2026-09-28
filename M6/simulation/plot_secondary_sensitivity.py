from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = (
    REPO_ROOT
    / "M6"
    / "evidence"
    / "simulation"
    / "secondary_path_sensitivity.csv"
)

PLOT_PATH = (
    REPO_ROOT
    / "M6"
    / "evidence"
    / "simulation"
    / "secondary_path_sensitivity.png"
)


rows = []

with CSV_PATH.open(
    "r",
    encoding="utf-8",
) as handle:

    reader = csv.DictReader(handle)

    for row in reader:
        rows.append(row)


delays = sorted(
    {
        float(row["delay_ms"])
        for row in rows
    }
)

attenuations = sorted(
    {
        float(
            row[
                "secondary_path_attenuation"
            ]
        )
        for row in rows
    }
)


matrix = np.zeros(
    (
        len(attenuations),
        len(delays),
    ),
    dtype=np.float64,
)


for row in rows:

    delay = float(
        row["delay_ms"]
    )

    attenuation = float(
        row[
            "secondary_path_attenuation"
        ]
    )

    value = float(
        row["attenuation_db"]
    )

    x = delays.index(delay)
    y = attenuations.index(attenuation)

    matrix[y, x] = value


plt.figure(
    figsize=(9, 5)
)

image = plt.imshow(
    matrix,
    aspect="auto",
    origin="lower",
    extent=[
        min(delays),
        max(delays),
        min(attenuations),
        max(attenuations),
    ],
)

plt.colorbar(
    image,
    label="Simulated attenuation (dB)",
)

plt.xlabel(
    "Secondary-path delay (ms)"
)

plt.ylabel(
    "Secondary-path attenuation"
)

plt.title(
    "SilentVox M6 Simulated Secondary-Path Sensitivity"
)

plt.tight_layout()

plt.savefig(
    PLOT_PATH,
    dpi=160,
)

plt.close()

print("Plot created:")
print(PLOT_PATH)
