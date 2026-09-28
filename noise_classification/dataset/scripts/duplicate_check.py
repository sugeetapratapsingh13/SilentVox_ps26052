import hashlib
import pandas as pd
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MANIFEST_PATH = (
    PROJECT_ROOT
    / "manifests"
    / "split_manifest.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "statistics"
    / "duplicate_report.csv"
)


# ============================================================
# LOAD MANIFEST
# ============================================================

if not MANIFEST_PATH.exists():
    raise FileNotFoundError(
        f"Manifest not found: {MANIFEST_PATH}"
    )

df = pd.read_csv(MANIFEST_PATH)

if "processed_audio_path" not in df.columns:
    raise ValueError(
        "processed_audio_path not found. "
        "Run preprocess.py first."
    )


# ============================================================
# HASH FILES
# ============================================================

hash_records = []

missing_files = []

for _, row in df.iterrows():

    relative_path = row["processed_audio_path"]

    path = PROJECT_ROOT / relative_path

    if not path.exists():

        missing_files.append(
            str(path)
        )

        continue

    sha256 = hashlib.sha256()

    with open(path, "rb") as file:

        for block in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):
            sha256.update(block)

    hash_records.append({
        "sha256":
            sha256.hexdigest(),

        "file":
            relative_path,

        "split":
            row.get("split", ""),

        "target_class":
            row.get("target_class", ""),

        "source_recording_id":
            row.get(
                "source_recording_id",
                ""
            ),
    })


# ============================================================
# MISSING FILE CHECK
# ============================================================

if missing_files:

    print("\nMissing processed files:")

    for path in missing_files:
        print(f"  {path}")

    raise FileNotFoundError(
        f"{len(missing_files)} processed files are missing."
    )


# ============================================================
# BUILD HASH TABLE
# ============================================================

hash_df = pd.DataFrame(
    hash_records
)

if hash_df.empty:

    raise RuntimeError(
        "No processed files were available for duplicate checking."
    )


# ============================================================
# FIND DUPLICATES
# ============================================================

duplicate_counts = (
    hash_df["sha256"]
    .value_counts()
)

duplicate_hashes = duplicate_counts[
    duplicate_counts > 1
].index


duplicates = hash_df[
    hash_df["sha256"].isin(
        duplicate_hashes
    )
].copy()


# ============================================================
# ADD DUPLICATE INFORMATION
# ============================================================

if not duplicates.empty:

    duplicates["duplicate_count"] = (
        duplicates["sha256"]
        .map(duplicate_counts)
    )

    split_counts = (
        duplicates.groupby("sha256")[
            "split"
        ]
        .nunique()
    )

    duplicates["cross_split_duplicate"] = (
        duplicates["sha256"]
        .map(split_counts)
        > 1
    )

    duplicates = duplicates[
        [
            "sha256",
            "duplicate_count",
            "cross_split_duplicate",
            "split",
            "target_class",
            "source_recording_id",
            "file",
        ]
    ]

else:

    duplicates = pd.DataFrame(
        columns=[
            "sha256",
            "duplicate_count",
            "cross_split_duplicate",
            "split",
            "target_class",
            "source_recording_id",
            "file",
        ]
    )


# ============================================================
# SAVE REPORT
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

duplicates.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# REPORT
# ============================================================

print("\n========================================")
print("DUPLICATE CHECK")
print("========================================")

print(
    f"Processed files checked: "
    f"{len(hash_df)}"
)

print(
    f"Unique SHA-256 hashes:   "
    f"{hash_df['sha256'].nunique()}"
)

print(
    f"Duplicate files found:   "
    f"{len(duplicates)}"
)

if duplicates.empty:

    print("\nNO EXACT BINARY DUPLICATES FOUND.")

else:

    print("\nDuplicate groups:")

    print(duplicates)

    if duplicates[
        "cross_split_duplicate"
    ].any():

        print(
            "\nWARNING: Duplicate files occur "
            "across different dataset splits."
        )

print(
    f"\nSaved to: {OUTPUT_PATH}"
)

print(
    "\nDuplicate check completed."
)