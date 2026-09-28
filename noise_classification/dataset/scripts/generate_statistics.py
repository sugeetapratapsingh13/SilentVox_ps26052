import pandas as pd
import soundfile as sf
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
    / "dataset_statistics.csv"
)


# ============================================================
# EXPECTED PROCESSING CONFIGURATION
# ============================================================

EXPECTED_SAMPLE_RATE = 16000
EXPECTED_CHANNELS = 1
EXPECTED_SUBTYPE = "PCM_16"


# ============================================================
# LOAD MANIFEST
# ============================================================

if not MANIFEST_PATH.exists():
    raise FileNotFoundError(
        f"Manifest not found: {MANIFEST_PATH}"
    )

df = pd.read_csv(MANIFEST_PATH)

required_columns = [
    "processed_audio_path",
    "target_class",
    "split",
    "source_recording_id",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# COLLECT PROCESSED AUDIO INFORMATION
# ============================================================

audio_stats = []

for _, row in df.iterrows():

    path = PROJECT_ROOT / row["processed_audio_path"]

    if not path.exists():

        raise FileNotFoundError(
            f"Processed file missing: {path}"
        )

    info = sf.info(path)

    audio_stats.append({
        "duration_seconds":
            info.frames / info.samplerate,

        "sample_rate":
            info.samplerate,

        "channels":
            info.channels,

        "subtype":
            info.subtype,

        "frames":
            info.frames,
    })


audio_df = pd.DataFrame(audio_stats)

full_df = pd.concat(
    [
        df.reset_index(drop=True),
        audio_df.reset_index(drop=True),
    ],
    axis=1
)


# ============================================================
# VERIFY PROCESSED FORMAT
# ============================================================

if not (
    full_df["sample_rate"]
    == EXPECTED_SAMPLE_RATE
).all():

    raise RuntimeError(
        "Not all processed files are 16 kHz."
    )

if not (
    full_df["channels"]
    == EXPECTED_CHANNELS
).all():

    raise RuntimeError(
        "Not all processed files are mono."
    )

if not (
    full_df["subtype"]
    == EXPECTED_SUBTYPE
).all():

    raise RuntimeError(
        "Not all processed files are PCM_16."
    )


# ============================================================
# STATISTICS FUNCTION
# ============================================================

def calculate_statistics(
    subset,
    scope,
    split="ALL",
    target_class="ALL",
):

    return {
        "scope": scope,
        "split": split,
        "target_class": target_class,

        "recording_count":
            len(subset),

        "unique_source_count":
            subset[
                "source_recording_id"
            ].nunique(),

        "total_duration_seconds":
            subset[
                "duration_seconds"
            ].sum(),

        "mean_duration_seconds":
            subset[
                "duration_seconds"
            ].mean(),

        "min_duration_seconds":
            subset[
                "duration_seconds"
            ].min(),

        "max_duration_seconds":
            subset[
                "duration_seconds"
            ].max(),

        "sample_rate":
            subset[
                "sample_rate"
            ].mode().iloc[0],

        "channels":
            subset[
                "channels"
            ].mode().iloc[0],

        "subtype":
            subset[
                "subtype"
            ].mode().iloc[0],
    }


# ============================================================
# BUILD STATISTICS
# ============================================================

statistics = []

# Overall
statistics.append(
    calculate_statistics(
        full_df,
        scope="overall"
    )
)


# Per split
for split in [
    "train",
    "validation",
    "test",
]:

    subset = full_df[
        full_df["split"] == split
    ]

    if subset.empty:
        continue

    statistics.append(
        calculate_statistics(
            subset,
            scope="split",
            split=split
        )
    )


# Per class
for target_class in sorted(
    full_df["target_class"].unique()
):

    subset = full_df[
        full_df["target_class"]
        == target_class
    ]

    statistics.append(
        calculate_statistics(
            subset,
            scope="class",
            target_class=target_class
        )
    )


# Class × split
for target_class in sorted(
    full_df["target_class"].unique()
):

    for split in [
        "train",
        "validation",
        "test",
    ]:

        subset = full_df[
            (full_df["target_class"] == target_class)
            &
            (full_df["split"] == split)
        ]

        if subset.empty:
            continue

        statistics.append(
            calculate_statistics(
                subset,
                scope="class_split",
                split=split,
                target_class=target_class
            )
        )


statistics_df = pd.DataFrame(
    statistics
)


# ============================================================
# ROUND NUMERICAL VALUES
# ============================================================

duration_columns = [
    "total_duration_seconds",
    "mean_duration_seconds",
    "min_duration_seconds",
    "max_duration_seconds",
]

for column in duration_columns:

    statistics_df[column] = (
        statistics_df[column]
        .round(3)
    )


# ============================================================
# SAVE
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

statistics_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# REPORT
# ============================================================

overall = full_df

print("\n========================================")
print("DATASET STATISTICS")
print("========================================")

print(
    f"Total recordings: "
    f"{len(overall)}"
)

print(
    f"Unique sources:   "
    f"{overall['source_recording_id'].nunique()}"
)

print(
    f"Total duration:   "
    f"{overall['duration_seconds'].sum():.2f} seconds"
)

print("\nClass distribution:")

print(
    overall["target_class"]
    .value_counts()
    .sort_index()
)

print("\nSplit distribution:")

print(
    overall["split"]
    .value_counts()
    .reindex(
        ["train", "validation", "test"]
    )
)

print("\nProcessed format:")

print(
    f"Sample rate: "
    f"{overall['sample_rate'].unique()}"
)

print(
    f"Channels: "
    f"{overall['channels'].unique()}"
)

print(
    f"Subtype: "
    f"{overall['subtype'].unique()}"
)

print(
    f"\nSaved to: {OUTPUT_PATH}"
)

print("\nDATASET STATISTICS GENERATED SUCCESSFULLY.")