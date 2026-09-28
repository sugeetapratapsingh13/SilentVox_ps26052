from pathlib import Path
import sys

import pandas as pd
import soundfile as sf


# ============================================================
# P2 INPUT CONFIGURATION
# ============================================================

P3_ROOT = Path(__file__).resolve().parents[1]

P2_ROOT = (
    P3_ROOT.parent
    / "DATASET_PREPROCESSING"
)

MANIFEST_PATH = (
    P2_ROOT
    / "manifests"
    / "split_manifest.csv"
)

EXPECTED_CLASSES = {
    "ENGINE",
    "RAIN",
    "SIREN",
    "WIND",
}

EXPECTED_SPLITS = {
    "train",
    "validation",
    "test",
}

EXPECTED_REQUIRED_COLUMNS = {
    "source_dataset",
    "source_recording_id",
    "original_filename",
    "audio_path",
    "original_category",
    "target_class",
    "original_fold",
    "take",
    "split",
    "processed_audio_path",
    "preprocessing_version",
}

EXPECTED_ROW_COUNT = 160
EXPECTED_P2_VERSION = "P2-v0.1"
EXPECTED_SAMPLE_RATE = 16000
EXPECTED_CHANNELS = 1


# ============================================================
# HELPERS
# ============================================================

def fail(message):
    print()
    print("=" * 70)
    print("VERIFICATION FAILED")
    print("=" * 70)
    print(message)
    print("=" * 70)
    sys.exit(1)


def resolve_processed_audio_path(relative_or_absolute_path):
    """
    Resolve processed_audio_path from the P2 manifest.

    P2 stores paths relative to the P2 root, for example:

        dataset\\test\\ENGINE\\file.wav

    Therefore the path must be resolved against P2_ROOT.
    """

    audio_path = Path(str(relative_or_absolute_path))

    if not audio_path.is_absolute():
        audio_path = P2_ROOT / audio_path

    return audio_path


# ============================================================
# MAIN VERIFICATION
# ============================================================

def main():

    print("=" * 70)
    print("SilentVox P3 — P2 Input Verification")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Verify P2 root
    # --------------------------------------------------------

    print()
    print("[1/9] Checking P2 root...")

    print(f"P2 root: {P2_ROOT}")

    if not P2_ROOT.exists():
        fail(
            f"P2 dataset root does not exist:\n"
            f"{P2_ROOT}"
        )

    print("P2 root: OK")

    # --------------------------------------------------------
    # 2. Verify manifest
    # --------------------------------------------------------

    print()
    print("[2/9] Checking split manifest...")

    print(f"Manifest: {MANIFEST_PATH}")

    if not MANIFEST_PATH.exists():
        fail(
            f"split_manifest.csv was not found:\n"
            f"{MANIFEST_PATH}"
        )

    try:
        df = pd.read_csv(MANIFEST_PATH)
    except Exception as exc:
        fail(
            f"Could not read split_manifest.csv:\n{exc}"
        )

    print("Manifest: OK")

    # --------------------------------------------------------
    # 3. Verify required columns
    # --------------------------------------------------------

    print()
    print("[3/9] Checking manifest columns...")

    actual_columns = set(df.columns)

    missing_columns = (
        EXPECTED_REQUIRED_COLUMNS - actual_columns
    )

    if missing_columns:
        fail(
            "Required manifest columns are missing:\n"
            + "\n".join(
                sorted(missing_columns)
            )
        )

    print("Required columns: OK")

    # --------------------------------------------------------
    # 4. Verify row count and classes
    # --------------------------------------------------------

    print()
    print("[4/9] Checking manifest records and classes...")

    print(f"Manifest rows: {len(df)}")

    if len(df) != EXPECTED_ROW_COUNT:
        fail(
            f"Expected {EXPECTED_ROW_COUNT} manifest rows, "
            f"but found {len(df)}."
        )

    actual_classes = set(
        df["target_class"]
        .dropna()
        .astype(str)
        .str.upper()
        .unique()
    )

    print(f"Classes: {sorted(actual_classes)}")

    if actual_classes != EXPECTED_CLASSES:
        fail(
            "Unexpected target classes.\n"
            f"Expected: {sorted(EXPECTED_CLASSES)}\n"
            f"Found:    {sorted(actual_classes)}"
        )

    class_counts = (
        df["target_class"]
        .astype(str)
        .str.upper()
        .value_counts()
        .sort_index()
    )

    print()
    print("Class distribution:")

    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count}")

    # --------------------------------------------------------
    # 5. Verify splits
    # --------------------------------------------------------

    print()
    print("[5/9] Checking train/validation/test splits...")

    actual_splits = set(
        df["split"]
        .dropna()
        .astype(str)
        .str.lower()
        .unique()
    )

    print(f"Splits: {sorted(actual_splits)}")

    if actual_splits != EXPECTED_SPLITS:
        fail(
            "Unexpected dataset splits.\n"
            f"Expected: {sorted(EXPECTED_SPLITS)}\n"
            f"Found:    {sorted(actual_splits)}"
        )

    split_counts = (
        df["split"]
        .astype(str)
        .str.lower()
        .value_counts()
        .reindex(
            ["train", "validation", "test"],
            fill_value=0
        )
    )

    print()
    print("Split distribution:")

    for split_name, count in split_counts.items():
        print(f"  {split_name}: {count}")

    # --------------------------------------------------------
    # 6. Verify processed audio files
    # --------------------------------------------------------

    print()
    print("[6/9] Checking processed audio files...")

    missing_files = []
    invalid_files = []

    valid_audio_count = 0

    for _, row in df.iterrows():

        processed_path = resolve_processed_audio_path(
            row["processed_audio_path"]
        )

        if not processed_path.exists():
            missing_files.append(
                str(row["processed_audio_path"])
            )
            continue

        try:
            info = sf.info(processed_path)

            if info.samplerate != EXPECTED_SAMPLE_RATE:
                invalid_files.append(
                    (
                        str(processed_path),
                        f"sample_rate={info.samplerate}"
                    )
                )
                continue

            if info.channels != EXPECTED_CHANNELS:
                invalid_files.append(
                    (
                        str(processed_path),
                        f"channels={info.channels}"
                    )
                )
                continue

            valid_audio_count += 1

        except Exception as exc:
            invalid_files.append(
                (
                    str(processed_path),
                    str(exc)
                )
            )

    print(f"Expected processed files: {len(df)}")
    print(f"Valid processed files: {valid_audio_count}")
    print(f"Missing files: {len(missing_files)}")
    print(f"Invalid files: {len(invalid_files)}")

    if missing_files:

        print()
        print("First missing files:")

        for file_path in missing_files[:5]:
            print(file_path)

    if invalid_files:

        print()
        print("First invalid files:")

        for file_path, reason in invalid_files[:5]:
            print(f"{file_path}")
            print(f"  Reason: {reason}")

    if missing_files:
        fail(
            "Processed audio files are missing."
        )

    if invalid_files:
        fail(
            "Processed audio files failed validation."
        )

    if valid_audio_count != len(df):
        fail(
            "Processed audio count does not match "
            "manifest row count."
        )

    print("Processed audio files: OK")

    # --------------------------------------------------------
    # 7. Verify class × split distribution
    # --------------------------------------------------------

    print()
    print("[7/9] Checking class × split distribution...")

    class_split = pd.crosstab(
        df["target_class"].astype(str).str.upper(),
        df["split"].astype(str).str.lower()
    )

    class_split = class_split.reindex(
        index=sorted(EXPECTED_CLASSES),
        columns=["train", "validation", "test"],
        fill_value=0
    )

    print()
    print(class_split.to_string())

    # --------------------------------------------------------
    # 8. Verify source-level split leakage
    # --------------------------------------------------------

    print()
    print("[8/9] Checking source-level split leakage...")

    source_split_counts = (
        df.groupby("source_recording_id")["split"]
        .nunique()
    )

    leaked_sources = (
        source_split_counts[
            source_split_counts > 1
        ]
    )

    print(
        f"Unique source recordings: "
        f"{df['source_recording_id'].nunique()}"
    )

    print(
        f"Sources appearing in multiple splits: "
        f"{len(leaked_sources)}"
    )

    if len(leaked_sources) > 0:

        print()
        print("Leaked source recordings:")

        for source_id in leaked_sources.index[:10]:
            print(source_id)

        fail(
            "Source-level split leakage detected."
        )

    print("Source-level split leakage: NONE")

    # --------------------------------------------------------
    # 9. Verify preprocessing version
    # --------------------------------------------------------

    print()
    print("[9/9] Checking P2 preprocessing version...")

    preprocessing_versions = (
        df["preprocessing_version"]
        .dropna()
        .astype(str)
        .unique()
    )

    print(
        f"Preprocessing versions: "
        f"{list(preprocessing_versions)}"
    )

    if len(preprocessing_versions) != 1:
        fail(
            "Multiple preprocessing versions were found."
        )

    actual_version = preprocessing_versions[0]

    if actual_version != EXPECTED_P2_VERSION:
        fail(
            f"Expected preprocessing version "
            f"{EXPECTED_P2_VERSION}, "
            f"but found {actual_version}."
        )

    print(
        f"Preprocessing version {EXPECTED_P2_VERSION}: OK"
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("P3 INPUT VERIFICATION PASSED")
    print("=" * 70)

    print()
    print("P2 root:")
    print(f"  {P2_ROOT}")

    print()
    print("Manifest:")
    print(f"  {MANIFEST_PATH}")

    print()
    print("Dataset:")
    print(f"  Recordings: {len(df)}")
    print(
        f"  Unique sources: "
        f"{df['source_recording_id'].nunique()}"
    )

    print()
    print("Classes:")
    for class_name in sorted(EXPECTED_CLASSES):
        count = class_counts.get(class_name, 0)
        print(f"  {class_name}: {count}")

    print()
    print("Splits:")
    print(f"  Train:      {split_counts['train']}")
    print(f"  Validation: {split_counts['validation']}")
    print(f"  Test:       {split_counts['test']}")

    print()
    print("Processed audio:")
    print(f"  Sample rate: {EXPECTED_SAMPLE_RATE} Hz")
    print(f"  Channels:    {EXPECTED_CHANNELS}")
    print("  All files valid: YES")

    print()
    print("Leakage:")
    print("  Source-level leakage: NONE")

    print()
    print("P2 preprocessing version:")
    print(f"  {actual_version}")

    print()
    print("=" * 70)
    print("READY FOR P3 FEATURE EXTRACTION")
    print("=" * 70)


if __name__ == "__main__":
    main()