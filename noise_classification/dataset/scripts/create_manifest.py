import pandas as pd
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ESC50_METADATA = (
    PROJECT_ROOT
    / "raw"
    / "ESC-50"
    / "meta"
    / "esc50.csv"
)

CLASS_MAPPING = (
    PROJECT_ROOT
    / "metadata"
    / "esc50_class_mapping.csv"
)

OUTPUT_MANIFEST = (
    PROJECT_ROOT
    / "manifests"
    / "dataset_manifest.csv"
)


# ============================================================
# LOAD FILES
# ============================================================

if not ESC50_METADATA.exists():
    raise FileNotFoundError(
        f"ESC-50 metadata not found: {ESC50_METADATA}"
    )

if not CLASS_MAPPING.exists():
    raise FileNotFoundError(
        f"Class mapping not found: {CLASS_MAPPING}"
    )

esc50 = pd.read_csv(ESC50_METADATA)
mapping = pd.read_csv(CLASS_MAPPING)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_esc50 = [
    "filename",
    "fold",
    "target",
    "category",
    "esc10",
    "src_file",
    "take",
]

missing_esc50 = [
    column for column in required_esc50
    if column not in esc50.columns
]

if missing_esc50:
    raise ValueError(
        f"Missing ESC-50 columns: {missing_esc50}"
    )


required_mapping = [
    "esc50_category",
    "target_class",
    "status",
]

missing_mapping = [
    column for column in required_mapping
    if column not in mapping.columns
]

if missing_mapping:
    raise ValueError(
        f"Missing mapping columns: {missing_mapping}"
    )


# ============================================================
# KEEP ONLY ACTIVE MAPPINGS
# ============================================================

keep_mapping = mapping[
    mapping["status"].astype(str).str.upper() == "KEEP"
].copy()

if keep_mapping.empty:
    raise ValueError(
        "No ESC-50 categories are marked KEEP."
    )

if keep_mapping["esc50_category"].isna().any():
    raise ValueError(
        "KEEP mappings contain missing ESC-50 categories."
    )

if keep_mapping["target_class"].isna().any():
    raise ValueError(
        "KEEP mappings contain missing target classes."
    )

if keep_mapping["esc50_category"].duplicated().any():
    duplicates = keep_mapping.loc[
        keep_mapping["esc50_category"].duplicated(),
        "esc50_category"
    ].tolist()

    raise ValueError(
        f"Duplicate ESC-50 categories in mapping: {duplicates}"
    )


# ============================================================
# MERGE DATASET METADATA WITH PROJECT MAPPING
# ============================================================

merged = esc50.merge(
    keep_mapping[
        ["esc50_category", "target_class"]
    ],
    left_on="category",
    right_on="esc50_category",
    how="inner"
)


# ============================================================
# CHECK MAPPING RESULT
# ============================================================

if merged.empty:
    raise ValueError(
        "Mapping produced zero dataset records."
    )

print("\nESC-50 categories mapped to SilentVox classes:")

for category, target_class in (
    merged[
        ["category", "target_class"]
    ]
    .drop_duplicates()
    .sort_values(["target_class", "category"])
    .itertuples(index=False, name=None)
):
    print(f"  {category} -> {target_class}")


# ============================================================
# BUILD MANIFEST
# ============================================================

manifest = pd.DataFrame({
    "source_dataset": "ESC-50",

    "source_recording_id":
        merged["src_file"].astype(str),

    "original_filename":
        merged["filename"].astype(str),

    "audio_path":
        "raw/ESC-50/audio/"
        + merged["filename"].astype(str),

    "original_category":
        merged["category"].astype(str),

    "target_class":
        merged["target_class"].astype(str),

    "original_fold":
        merged["fold"],

    "take":
        merged["take"],

    "split":
        "UNASSIGNED",
})


# ============================================================
# DUPLICATE CHECKS
# ============================================================

if manifest["original_filename"].duplicated().any():

    duplicates = manifest.loc[
        manifest["original_filename"].duplicated(keep=False),
        "original_filename"
    ].tolist()

    raise ValueError(
        "Duplicate original filenames found:\n"
        + "\n".join(duplicates)
    )


if manifest["source_recording_id"].isna().any():

    raise ValueError(
        "Missing source_recording_id values found."
    )


if manifest["target_class"].isna().any():

    raise ValueError(
        "Missing target_class values found."
    )


# ============================================================
# VERIFY SOURCE FILES
# ============================================================

missing_files = []

for audio_path in manifest["audio_path"]:

    path = PROJECT_ROOT / audio_path

    if not path.exists():
        missing_files.append(str(path))

if missing_files:

    print("\nMissing source files:")

    for path in missing_files:
        print(f"  {path}")

    raise FileNotFoundError(
        f"{len(missing_files)} source audio files are missing."
    )


# ============================================================
# SORT AND SAVE
# ============================================================

manifest = manifest.sort_values(
    [
        "target_class",
        "source_recording_id",
        "original_filename",
    ]
).reset_index(drop=True)


OUTPUT_MANIFEST.parent.mkdir(
    parents=True,
    exist_ok=True
)

manifest.to_csv(
    OUTPUT_MANIFEST,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n========================================")
print("DATASET MANIFEST CREATED")
print("========================================")

print(f"Total recordings: {len(manifest)}")

print(
    f"Unique sources:   "
    f"{manifest['source_recording_id'].nunique()}"
)

print("\nClass distribution:")

print(
    manifest["target_class"]
    .value_counts()
    .sort_index()
)

print("\nUnique sources by class:")

print(
    manifest.groupby("target_class")[
        "source_recording_id"
    ]
    .nunique()
    .sort_index()
)

print("\nSplit status:")

print(
    manifest["split"]
    .value_counts()
)

print(f"\nSaved to: {OUTPUT_MANIFEST}")