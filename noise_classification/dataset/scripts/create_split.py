import pandas as pd
import numpy as np
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_MANIFEST = (
    PROJECT_ROOT
    / "manifests"
    / "dataset_manifest.csv"
)

OUTPUT_MANIFEST = (
    PROJECT_ROOT
    / "manifests"
    / "split_manifest.csv"
)


TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

SPLITS = [
    "train",
    "validation",
    "test",
]


ratio_sum = (
    TRAIN_RATIO
    + VALIDATION_RATIO
    + TEST_RATIO
)

if not np.isclose(ratio_sum, 1.0):
    raise ValueError(
        f"Split ratios must sum to 1.0. "
        f"Current sum = {ratio_sum}"
    )


if not INPUT_MANIFEST.exists():
    raise FileNotFoundError(
        f"Dataset manifest not found: {INPUT_MANIFEST}"
    )

df = pd.read_csv(INPUT_MANIFEST)

required_columns = [
    "source_recording_id",
    "target_class",
    "original_filename",
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

if df.empty:
    raise ValueError("Dataset manifest is empty.")


df["split"] = "UNASSIGNED"

rng = np.random.default_rng(RANDOM_SEED)


for target_class in sorted(
    df["target_class"].dropna().unique()
):

    class_mask = (
        df["target_class"] == target_class
    )

    class_sources = (
        df.loc[
            class_mask,
            "source_recording_id"
        ]
        .drop_duplicates()
        .to_numpy()
        .copy()
    )

    rng.shuffle(class_sources)

    total_sources = len(class_sources)

    if total_sources < 3:
        raise ValueError(
            f"Class '{target_class}' has only "
            f"{total_sources} source recordings. "
            "At least 3 are required."
        )

    test_count = max(
        1,
        int(round(
            total_sources * TEST_RATIO
        ))
    )

    validation_count = max(
        1,
        int(round(
            total_sources * VALIDATION_RATIO
        ))
    )

    if (
        test_count
        + validation_count
        >= total_sources
    ):
        test_count = 1
        validation_count = 1

    test_sources = class_sources[
        :test_count
    ]

    validation_sources = class_sources[
        test_count:
        test_count + validation_count
    ]

    train_sources = class_sources[
        test_count + validation_count:
    ]

    if len(train_sources) == 0:
        raise ValueError(
            f"No training sources remain for "
            f"class '{target_class}'."
        )

    df.loc[
        df["source_recording_id"].isin(
            train_sources
        )
        & class_mask,
        "split"
    ] = "train"

    df.loc[
        df["source_recording_id"].isin(
            validation_sources
        )
        & class_mask,
        "split"
    ] = "validation"

    df.loc[
        df["source_recording_id"].isin(
            test_sources
        )
        & class_mask,
        "split"
    ] = "test"


if (df["split"] == "UNASSIGNED").any():

    unassigned = df.loc[
        df["split"] == "UNASSIGNED"
    ]

    raise RuntimeError(
        "Some recordings remain UNASSIGNED:\n"
        + str(unassigned)
    )


split_sources = {}

for split in SPLITS:

    split_sources[split] = set(
        df.loc[
            df["split"] == split,
            "source_recording_id"
        ]
    )


for i, split_a in enumerate(SPLITS):

    for split_b in SPLITS[i + 1:]:

        overlap = (
            split_sources[split_a]
            & split_sources[split_b]
        )

        if overlap:

            raise RuntimeError(
                f"SOURCE LEAKAGE DETECTED between "
                f"{split_a} and {split_b}: "
                f"{sorted(overlap)}"
            )


classes = sorted(
    df["target_class"].dropna().unique()
)

for target_class in classes:

    for split in SPLITS:

        count = len(
            df[
                (df["target_class"] == target_class)
                &
                (df["split"] == split)
            ]
        )

        if count == 0:

            raise RuntimeError(
                f"Class '{target_class}' has "
                f"zero recordings in {split}."
            )


df = df.sort_values(
    [
        "split",
        "target_class",
        "source_recording_id",
        "original_filename",
    ]
).reset_index(drop=True)


OUTPUT_MANIFEST.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_MANIFEST,
    index=False
)


print("\n========================================")
print("SOURCE-LEVEL DATASET SPLIT CREATED")
print("========================================")

print(f"Random seed: {RANDOM_SEED}")

print("\nRecording distribution:")

print(
    df["split"]
    .value_counts()
    .reindex(SPLITS)
)

print("\nSource distribution:")

print(
    df.groupby("split")[
        "source_recording_id"
    ]
    .nunique()
    .reindex(SPLITS)
)

print("\nClass x split distribution:")

print(
    pd.crosstab(
        df["target_class"],
        df["split"]
    ).reindex(
        columns=SPLITS,
        fill_value=0
    )
)

print("\nActual split ratios:")

total = len(df)

for split in SPLITS:

    count = (
        df["split"] == split
    ).sum()

    print(
        f"{split:12s}: "
        f"{count:4d} "
        f"({count / total:.3f})"
    )

print("\nSource overlap check: PASSED")
print("Class presence check: PASSED")

print(
    f"\nSaved to: {OUTPUT_MANIFEST}"
)