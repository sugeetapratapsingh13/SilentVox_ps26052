from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = REPO_ROOT / "M6" / "evidence" / "simulation" / "block_size_benchmark.csv"
PLOT_PATH = REPO_ROOT / "M6" / "evidence" / "simulation" / "block_size_processing.png"


rows = []

with CSV_PATH.open("r", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)

    for row in reader:
        rows.append(row)

block_sizes = np.array(
    [int(row["block_size_samples"]) for row in rows]
)

frame_duration = np.array(
    [float(row["block_duration_ms"]) for row in rows]
)

mean_processing = np.array(
    [float(row["mean_processing_ms"]) for row in rows]
)

max_processing = np.array(
    [float(row["max_processing_ms"]) for row in rows]
)


plt.figure(figsize=(9, 5))

plt.plot(
    block_sizes,
    frame_duration,
    marker="o",
    label="Block duration",
)

plt.plot(
    block_sizes,
    mean_processing,
    marker="o",
    label="Mean processing time",
)

plt.plot(
    block_sizes,
    max_processing,
    marker="o",
    label="Maximum processing time",
)

plt.xlabel("Block size (samples)")
plt.ylabel("Time (ms)")
plt.title("SilentVox M6 Software Block-Processing Benchmark")
plt.xticks(block_sizes)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(PLOT_PATH, dpi=160)
plt.close()

print("Plot created:")
print(PLOT_PATH)
