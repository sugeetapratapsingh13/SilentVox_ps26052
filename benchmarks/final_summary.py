import os
import csv

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results"
)

input_file = os.path.join(
    RESULTS_DIR,
    "parameter_sweep.csv"
)

output_file = os.path.join(
    RESULTS_DIR,
    "FINAL_ANC_SUMMARY.csv"
)

with open(input_file, "r", newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Convert numbers
for row in rows:
    row["Filter Length"] = int(row["Filter Length"])
    row["Step Size"] = float(row["Step Size"])
    row["Processing Time (s)"] = float(
        row["Processing Time (s)"]
    )
    row["Noise Reduction (dB)"] = float(
        row["Noise Reduction (dB)"]
    )

# Find highest reduction for each algorithm
lms_rows = [
    r for r in rows if r["Algorithm"] == "LMS"
]

nlms_rows = [
    r for r in rows if r["Algorithm"] == "NLMS"
]

best_lms = max(
    lms_rows,
    key=lambda r: r["Noise Reduction (dB)"]
)

best_nlms = max(
    nlms_rows,
    key=lambda r: r["Noise Reduction (dB)"]
)

# Write summary
with open(output_file, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "Algorithm",
        "Best Filter Length",
        "Best Step Size",
        "Processing Time (s)",
        "Noise Reduction (dB)"
    ])

    writer.writerow([
        "LMS",
        best_lms["Filter Length"],
        best_lms["Step Size"],
        best_lms["Processing Time (s)"],
        best_lms["Noise Reduction (dB)"]
    ])

    writer.writerow([
        "NLMS",
        best_nlms["Filter Length"],
        best_nlms["Step Size"],
        best_nlms["Processing Time (s)"],
        best_nlms["Noise Reduction (dB)"]
    ])

print("\n========================================")
print("       FINAL ANC SUMMARY")
print("========================================")

print("\nLMS")
print(
    f"Filter Length : {best_lms['Filter Length']}"
)
print(
    f"Step Size     : {best_lms['Step Size']}"
)
print(
    f"Processing    : "
    f"{best_lms['Processing Time (s)']:.6f} s"
)
print(
    f"Reduction     : "
    f"{best_lms['Noise Reduction (dB)']:.4f} dB"
)

print("\nNLMS")
print(
    f"Filter Length : {best_nlms['Filter Length']}"
)
print(
    f"Step Size     : {best_nlms['Step Size']}"
)
print(
    f"Processing    : "
    f"{best_nlms['Processing Time (s)']:.6f} s"
)
print(
    f"Reduction     : "
    f"{best_nlms['Noise Reduction (dB)']:.4f} dB"
)

print("\n========================================")
print("Summary saved to:")
print(output_file)
print("========================================")