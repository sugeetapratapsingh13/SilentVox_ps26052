from pathlib import Path
import json
import re
import sys

import numpy as np
import pandas as pd
import soundfile as sf
import librosa


# ============================================================
# SilentVox — M2-P3 Feature Extraction
# Version: P3-v0.1
# ============================================================

P3_VERSION = "P3-v0.1"
P2_VERSION = "P2-v0.1"
RANDOM_SEED = 42

P3_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = P3_ROOT.parent
P2_ROOT = PROJECT_ROOT / "DATASET_PREPROCESSING"

P2_MANIFEST = P2_ROOT / "manifests" / "split_manifest.csv"

FEATURE_ROOT = P3_ROOT / "features"
MFCC_ROOT = FEATURE_ROOT / "mfcc"
LOG_MEL_ROOT = FEATURE_ROOT / "log_mel"

METADATA_ROOT = P3_ROOT / "metadata"
STATISTICS_ROOT = P3_ROOT / "statistics"

FEATURE_MANIFEST = METADATA_ROOT / "feature_manifest.csv"
NORMALIZATION_FILE = METADATA_ROOT / "feature_normalization_stats.npz"
CONFIG_FILE = METADATA_ROOT / "feature_config.json"
STATISTICS_FILE = STATISTICS_ROOT / "feature_statistics.csv"
SPEC_FILE = P3_ROOT / "FEATURE_EXTRACTION_SPEC.md"


# ============================================================
# Dataset configuration
# ============================================================

EXPECTED_CLASSES = [
    "ENGINE",
    "RAIN",
    "SIREN",
    "WIND",
]

EXPECTED_SPLITS = [
    "train",
    "validation",
    "test",
]


# ============================================================
# Audio configuration
# ============================================================

SAMPLE_RATE = 16000
CHANNELS = 1

WINDOW_SECONDS = 2.0
HOP_SECONDS = 1.0

WINDOW_SAMPLES = int(
    SAMPLE_RATE * WINDOW_SECONDS
)

HOP_SAMPLES = int(
    SAMPLE_RATE * HOP_SECONDS
)


# ============================================================
# Feature configuration
# ============================================================

N_FFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 160

N_MELS = 64
N_MFCC = 13

POWER = 2.0
CENTER = False

NORMALIZATION_EPSILON = 1e-8


# ============================================================
# Reproducibility
# ============================================================

np.random.seed(RANDOM_SEED)


# ============================================================
# Utility functions
# ============================================================

def fail(message):
    print()
    print("=" * 70)
    print("P3 FEATURE EXTRACTION FAILED")
    print("=" * 70)
    print(message)
    print("=" * 70)
    sys.exit(1)


def sanitize_filename(value):
    return re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        str(value),
    )


def relative_path(path):
    return path.relative_to(
        P3_ROOT
    ).as_posix()


def ensure_directories():

    MFCC_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG_MEL_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    METADATA_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    STATISTICS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


def resolve_p2_audio_path(value):

    path = Path(str(value))

    if path.is_absolute():
        return path

    return P2_ROOT / path


# ============================================================
# P2 verification
# ============================================================

def load_and_verify_p2_manifest():

    print()
    print("=" * 70)
    print("SilentVox M2-P3 — Feature Extraction")
    print("=" * 70)

    print()
    print(
        "[1/8] Loading P2 split manifest..."
    )

    if not P2_ROOT.exists():
        fail(
            f"P2 root does not exist:\n{P2_ROOT}"
        )

    if not P2_MANIFEST.exists():
        fail(
            "P2 split manifest does not exist:\n"
            f"{P2_MANIFEST}"
        )

    df = pd.read_csv(
        P2_MANIFEST,
        dtype=str,
    )

    required_columns = [
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
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        fail(
            "Missing required P2 manifest columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing_columns
            )
        )

    if len(df) == 0:
        fail(
            "P2 split manifest contains zero records."
        )

    print(
        f"Manifest rows: {len(df)}"
    )

    print()
    print(
        "[2/8] Verifying P2 classes and splits..."
    )

    classes = sorted(
        df["target_class"]
        .dropna()
        .unique()
        .tolist()
    )

    splits = sorted(
        df["split"]
        .dropna()
        .unique()
        .tolist()
    )

    if set(classes) != set(
        EXPECTED_CLASSES
    ):

        fail(
            "Unexpected P2 classes.\n"
            f"Expected: {EXPECTED_CLASSES}\n"
            f"Found:    {classes}"
        )

    if set(splits) != set(
        EXPECTED_SPLITS
    ):

        fail(
            "Unexpected P2 splits.\n"
            f"Expected: {EXPECTED_SPLITS}\n"
            f"Found:    {splits}"
        )

    print("Classes: OK")
    print("Splits: OK")

    print()
    print("Class distribution:")

    class_counts = (
        df["target_class"]
        .value_counts()
        .sort_index()
    )

    for target_class, count in (
        class_counts.items()
    ):

        print(
            f"  {target_class}: {count}"
        )

    print()
    print("Split distribution:")

    split_counts = (
        df["split"]
        .value_counts()
        .reindex(
            EXPECTED_SPLITS,
            fill_value=0,
        )
    )

    for split, count in (
        split_counts.items()
    ):

        print(
            f"  {split}: {count}"
        )

    print()
    print(
        "[3/8] Verifying P2 preprocessing version..."
    )

    versions = sorted(
        df["preprocessing_version"]
        .dropna()
        .unique()
        .tolist()
    )

    if versions != [P2_VERSION]:

        fail(
            "Unexpected preprocessing version.\n"
            f"Expected: {[P2_VERSION]}\n"
            f"Found:    {versions}"
        )

    print(
        f"Preprocessing version: "
        f"{P2_VERSION}"
    )

    print("P2 version: OK")

    return df


# ============================================================
# Source-level leakage verification
# ============================================================

def verify_source_split_leakage(df):

    print()
    print(
        "[4/8] Checking source-level split leakage..."
    )

    source_split_counts = (
        df.groupby(
            "source_recording_id"
        )["split"]
        .nunique()
    )

    leaked_sources = (
        source_split_counts[
            source_split_counts > 1
        ]
    )

    unique_sources = (
        df["source_recording_id"]
        .nunique()
    )

    print(
        f"Unique source recordings: "
        f"{unique_sources}"
    )

    print(
        "Sources appearing in multiple splits: "
        f"{len(leaked_sources)}"
    )

    if len(leaked_sources) > 0:

        print()
        print("LEAKED SOURCES:")

        for source_id in (
            leaked_sources.index
        ):

            source_rows = df[
                df["source_recording_id"]
                == source_id
            ]

            print(
                f"  {source_id}: "
                f"{sorted(source_rows['split'].unique().tolist())}"
            )

        fail(
            "Source-level split leakage detected."
        )

    print(
        "Source-level split leakage: NONE"
    )


# ============================================================
# Audio validation
# ============================================================

def validate_audio_file(
    audio_path
):

    if not audio_path.exists():

        raise FileNotFoundError(
            str(audio_path)
        )

    info = sf.info(
        audio_path
    )

    if info.samplerate != SAMPLE_RATE:

        raise ValueError(
            f"Expected {SAMPLE_RATE} Hz, "
            f"found {info.samplerate} Hz"
        )

    if info.channels != CHANNELS:

        raise ValueError(
            f"Expected {CHANNELS} channel, "
            f"found {info.channels}"
        )

    if info.frames < WINDOW_SAMPLES:

        raise ValueError(
            f"Audio shorter than "
            f"{WINDOW_SECONDS} seconds"
        )

    if info.subtype != "PCM_16":

        raise ValueError(
            f"Expected PCM_16, "
            f"found {info.subtype}"
        )


# ============================================================
# Feature extraction
# ============================================================

def extract_features(audio):

    mel_power = (
        librosa.feature.melspectrogram(
            y=audio,
            sr=SAMPLE_RATE,
            n_fft=N_FFT,
            win_length=WIN_LENGTH,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            power=POWER,
            center=CENTER,
        )
    )

    log_mel = (
        librosa.power_to_db(
            mel_power,
            ref=np.max,
        )
    )

    mfcc = (
        librosa.feature.mfcc(
            y=None,
            sr=SAMPLE_RATE,
            S=log_mel,
            n_mfcc=N_MFCC,
            dct_type=2,
            norm="ortho",
        )
    )

    return (
        mfcc.astype(np.float32),
        log_mel.astype(np.float32),
    )


# ============================================================
# Process one recording
# ============================================================

def process_recording(row):

    audio_path = (
        resolve_p2_audio_path(
            row["processed_audio_path"]
        )
    )

    validate_audio_file(
        audio_path
    )

    audio, sr = sf.read(
        audio_path,
        dtype="float32",
        always_2d=False,
    )

    if sr != SAMPLE_RATE:

        raise ValueError(
            f"Unexpected sample rate: {sr}"
        )

    if audio.ndim != 1:

        raise ValueError(
            "Expected mono audio array, "
            f"got shape {audio.shape}"
        )

    total_samples = len(audio)

    windows = []

    start_sample = 0
    window_index = 0

    while (
        start_sample + WINDOW_SAMPLES
        <= total_samples
    ):

        end_sample = (
            start_sample
            + WINDOW_SAMPLES
        )

        segment = audio[
            start_sample:end_sample
        ]

        mfcc, log_mel = (
            extract_features(
                segment
            )
        )

        windows.append(
            {
                "window_index": (
                    window_index
                ),
                "start_sample": (
                    start_sample
                ),
                "end_sample": (
                    end_sample
                ),
                "start_sec": (
                    start_sample
                    / SAMPLE_RATE
                ),
                "end_sec": (
                    end_sample
                    / SAMPLE_RATE
                ),
                "mfcc": mfcc,
                "log_mel": log_mel,
            }
        )

        start_sample += HOP_SAMPLES
        window_index += 1

    if len(windows) == 0:

        raise ValueError(
            "No complete 2-second windows "
            "could be generated."
        )

    duration = (
        total_samples
        / SAMPLE_RATE
    )

    return windows, duration


# ============================================================
# Save configuration
# ============================================================

def save_feature_config():

    config = {
        "feature_version": P3_VERSION,
        "p2_version": P2_VERSION,
        "random_seed": RANDOM_SEED,
        "sample_rate_hz": SAMPLE_RATE,
        "channels": CHANNELS,
        "window_duration_seconds": (
            WINDOW_SECONDS
        ),
        "window_hop_seconds": (
            HOP_SECONDS
        ),
        "window_samples": (
            WINDOW_SAMPLES
        ),
        "hop_samples": (
            HOP_SAMPLES
        ),
        "n_fft": N_FFT,
        "win_length": WIN_LENGTH,
        "hop_length": HOP_LENGTH,
        "n_mels": N_MELS,
        "n_mfcc": N_MFCC,
        "power": POWER,
        "center": CENTER,
        "feature_types": [
            "MFCC",
            "Log-Mel Spectrogram",
        ],
        "normalization": {
            "method": (
                "training_split_global_"
                "per_feature_coefficient"
            ),
            "statistics_source": (
                "train_only"
            ),
            "epsilon": (
                NORMALIZATION_EPSILON
            ),
        },
        "split_policy": (
            "inherited_from_P2_"
            "source_level_split"
        ),
    }

    with open(
        CONFIG_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            config,
            file,
            indent=4,
        )


# ============================================================
# Save finalized specification
# ============================================================

def save_specification():

    lines = [
        "# SilentVox — Feature Extraction Specification",
        "",
        "## P3 Version",
        "",
        P3_VERSION,
        "",
        "## Purpose",
        "",
        "P3 converts the verified P2 processed "
        "audio dataset into model-ready features "
        "for SilentVox noise classification.",
        "",
        "The P3 pipeline inherits the "
        "train/validation/test assignment "
        "from P2 and performs windowing only "
        "after the source-level split has "
        "been fixed.",
        "",
        "## Input Dataset",
        "",
        "- Sample rate: 16,000 Hz",
        "- Channels: Mono",
        "- Storage: WAV PCM16",
        f"- Preprocessing version: {P2_VERSION}",
        "",
        "## Current Verified Classes",
        "",
        "- ENGINE",
        "- RAIN",
        "- SIREN",
        "- WIND",
        "",
        "This is the current verified "
        "implementation subset and does not "
        "freeze the final SilentVox taxonomy.",
        "",
        "## Windowing",
        "",
        "- Window duration: 2 seconds",
        "- Window length: 32,000 samples",
        "- Window hop: 1 second",
        "- Window hop length: 16,000 samples",
        "- Partial windows: Not retained",
        "",
        "Each generated window remains in the "
        "source recording's P2 split.",
        "",
        "## Spectral Parameters",
        "",
        "- Sample rate: 16,000 Hz",
        "- FFT size: 512",
        "- Window length: 400 samples",
        "- Hop length: 160 samples",
        "- Mel bands: 64",
        "- Power: 2.0",
        "- Center: False",
        "",
        "## MFCC",
        "",
        "- Number of coefficients: 13",
        "- DCT type: 2",
        "- DCT normalization: ortho",
        "",
        "MFCC temporal dimensions are preserved.",
        "",
        "## Log-Mel Spectrogram",
        "",
        "- 64 Mel bands",
        "- Power spectrogram",
        "- Converted to decibels using the "
        "maximum value of each window as reference",
        "- Temporal dimensions are preserved",
        "",
        "## Normalization",
        "",
        "Normalization statistics are calculated "
        "using the training split only.",
        "",
        "Statistics are calculated independently "
        "for each feature coefficient/band across "
        "training windows and time frames.",
        "",
        "Validation, test, and future inference "
        "features use the saved training statistics.",
        "",
        "## Leakage Prevention",
        "",
        "P2 source-level split assignments are "
        "inherited directly.",
        "",
        "All windows from a single source recording "
        "remain in exactly one split.",
        "",
        "## Outputs",
        "",
        "```text",
        "features/",
        "├── mfcc/",
        "│   ├── train/",
        "│   ├── validation/",
        "│   └── test/",
        "│",
        "└── log_mel/",
        "    ├── train/",
        "    ├── validation/",
        "    └── test/",
        "```",
        "",
        "```text",
        "metadata/",
        "├── feature_manifest.csv",
        "├── feature_normalization_stats.npz",
        "└── feature_config.json",
        "```",
        "",
        "```text",
        "statistics/",
        "└── feature_statistics.csv",
        "```",
        "",
        "## P4/P6 Compatibility",
        "",
        "P4 training and P6 live inference must "
        "use the same feature-extraction parameters "
        "and training-derived normalization statistics.",
        "",
        "Any change requires a new P3 version and "
        "regeneration of affected feature artifacts.",
        "",
        "## Reproducibility",
        "",
        f"Random seed: {RANDOM_SEED}",
        "",
        "Feature extraction itself is deterministic.",
        "",
        "## Status",
        "",
        "P3-v0.1 finalized for the current verified "
        "four-class P2 implementation dataset.",
        "",
    ]

    with open(
        SPEC_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "\n".join(lines)
        )


# ============================================================
# Extract all features
# ============================================================

def extract_all_features(df):

    print()
    print(
        "[5/8] Extracting MFCC and Log-Mel features..."
    )

    all_records = []
    failed_records = []

    total_recordings = len(df)

    for record_number, (_, row) in enumerate(
        df.iterrows(),
        start=1,
    ):

        source_id = str(
            row["source_recording_id"]
        )

        original_filename = str(
            row["original_filename"]
        )

        target_class = str(
            row["target_class"]
        )

        split = str(
            row["split"]
        ).lower()

        print(
            f"\rProcessing recording "
            f"{record_number}/{total_recordings}",
            end="",
            flush=True,
        )

        try:

            windows, duration = (
                process_recording(row)
            )

            # IMPORTANT:
            # source_recording_id can repeat.
            # Combine it with the original filename
            # to create a unique recording identifier.
            recording_key = (
                f"{source_id}_"
                f"{Path(original_filename).stem}"
            )

            safe_recording_key = (
                sanitize_filename(
                    recording_key
                )
            )

            for window in windows:

                window_index = (
                    window["window_index"]
                )

                filename = (
                    f"{safe_recording_key}"
                    f"_w{window_index:03d}.npy"
                )

                mfcc_dir = (
                    MFCC_ROOT
                    / split
                    / target_class
                )

                log_mel_dir = (
                    LOG_MEL_ROOT
                    / split
                    / target_class
                )

                mfcc_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                log_mel_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                mfcc_path = (
                    mfcc_dir
                    / filename
                )

                log_mel_path = (
                    log_mel_dir
                    / filename
                )

                np.save(
                    mfcc_path,
                    window["mfcc"].astype(
                        np.float32
                    ),
                )

                np.save(
                    log_mel_path,
                    window["log_mel"].astype(
                        np.float32
                    ),
                )

                feature_id = (
                    f"{safe_recording_key}"
                    f"_w{window_index:03d}"
                )

                all_records.append(
                    {
                        "feature_id": feature_id,
                        "source_dataset": row[
                            "source_dataset"
                        ],
                        "source_recording_id": source_id,
                        "original_filename": (
                            original_filename
                        ),
                        "original_category": row[
                            "original_category"
                        ],
                        "target_class": target_class,
                        "split": split,
                        "original_fold": row[
                            "original_fold"
                        ],
                        "take": row["take"],
                        "window_index": (
                            window_index
                        ),
                        "start_sample": window[
                            "start_sample"
                        ],
                        "end_sample": window[
                            "end_sample"
                        ],
                        "start_sec": window[
                            "start_sec"
                        ],
                        "end_sec": window[
                            "end_sec"
                        ],
                        "source_duration_sec": (
                            duration
                        ),
                        "mfcc_path": relative_path(
                            mfcc_path
                        ),
                        "log_mel_path": relative_path(
                            log_mel_path
                        ),
                        "mfcc_shape": str(
                            list(
                                window[
                                    "mfcc"
                                ].shape
                            )
                        ),
                        "log_mel_shape": str(
                            list(
                                window[
                                    "log_mel"
                                ].shape
                            )
                        ),
                        "sample_rate_hz": (
                            SAMPLE_RATE
                        ),
                        "window_duration_sec": (
                            WINDOW_SECONDS
                        ),
                        "window_hop_sec": (
                            HOP_SECONDS
                        ),
                        "feature_version": (
                            P3_VERSION
                        ),
                        "p2_version": row[
                            "preprocessing_version"
                        ],
                    }
                )

        except Exception as error:

            failed_records.append(
                {
                    "source_recording_id": (
                        source_id
                    ),
                    "original_filename": (
                        original_filename
                    ),
                    "target_class": (
                        target_class
                    ),
                    "split": split,
                    "error": str(error),
                }
            )

    print()

    if failed_records:

        print()
        print(
            "Feature extraction failures:"
        )

        for failure in failed_records:

            print(
                f"  {failure['source_recording_id']} "
                f"{failure['original_filename']}: "
                f"{failure['error']}"
            )

        fail(
            "Feature extraction failed for "
            f"{len(failed_records)} recording(s)."
        )

    if not all_records:

        fail(
            "No feature windows were generated."
        )

    feature_df = pd.DataFrame(
        all_records
    )

    print(
        f"Feature windows generated: "
        f"{len(feature_df)}"
    )

    return feature_df


# ============================================================
# Calculate training statistics
# ============================================================

def calculate_training_statistics(
    feature_df
):

    print()
    print(
        "[6/8] Calculating training-only "
        "normalization statistics..."
    )

    train_df = feature_df[
        feature_df["split"] == "train"
    ].copy()

    if len(train_df) == 0:

        fail(
            "No training features available "
            "for normalization."
        )

    mfcc_arrays = []
    log_mel_arrays = []

    for _, row in train_df.iterrows():

        mfcc_path = (
            P3_ROOT
            / row["mfcc_path"]
        )

        log_mel_path = (
            P3_ROOT
            / row["log_mel_path"]
        )

        mfcc = np.load(
            mfcc_path
        ).astype(np.float32)

        log_mel = np.load(
            log_mel_path
        ).astype(np.float32)

        mfcc_arrays.append(
            mfcc
        )

        log_mel_arrays.append(
            log_mel
        )

    mfcc_stack = np.stack(
        mfcc_arrays,
        axis=0,
    )

    log_mel_stack = np.stack(
        log_mel_arrays,
        axis=0,
    )

    mfcc_mean = np.mean(
        mfcc_stack,
        axis=(0, 2),
    ).astype(np.float32)

    mfcc_std = np.std(
        mfcc_stack,
        axis=(0, 2),
    ).astype(np.float32)

    log_mel_mean = np.mean(
        log_mel_stack,
        axis=(0, 2),
    ).astype(np.float32)

    log_mel_std = np.std(
        log_mel_stack,
        axis=(0, 2),
    ).astype(np.float32)

    mfcc_std = np.maximum(
        mfcc_std,
        NORMALIZATION_EPSILON,
    )

    log_mel_std = np.maximum(
        log_mel_std,
        NORMALIZATION_EPSILON,
    )

    np.savez(
        NORMALIZATION_FILE,
        mfcc_mean=mfcc_mean,
        mfcc_std=mfcc_std,
        log_mel_mean=log_mel_mean,
        log_mel_std=log_mel_std,
        sample_rate_hz=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_mels=N_MELS,
        feature_version=P3_VERSION,
        normalization_source="train_only",
    )

    print(
        f"Training windows used: "
        f"{len(train_df)}"
    )

    print(
        f"MFCC mean shape: "
        f"{mfcc_mean.shape}"
    )

    print(
        f"MFCC std shape: "
        f"{mfcc_std.shape}"
    )

    print(
        f"Log-Mel mean shape: "
        f"{log_mel_mean.shape}"
    )

    print(
        f"Log-Mel std shape: "
        f"{log_mel_std.shape}"
    )

    return (
        mfcc_mean,
        mfcc_std,
        log_mel_mean,
        log_mel_std,
    )


# ============================================================
# Apply normalization
# ============================================================

def normalize_features(
    feature_df,
    mfcc_mean,
    mfcc_std,
    log_mel_mean,
    log_mel_std,
):

    print()
    print(
        "[7/8] Applying training-only normalization..."
    )

    for _, row in feature_df.iterrows():

        mfcc_path = (
            P3_ROOT
            / row["mfcc_path"]
        )

        log_mel_path = (
            P3_ROOT
            / row["log_mel_path"]
        )

        mfcc = np.load(
            mfcc_path
        ).astype(np.float32)

        log_mel = np.load(
            log_mel_path
        ).astype(np.float32)

        normalized_mfcc = (
            mfcc
            - mfcc_mean[:, None]
        ) / mfcc_std[:, None]

        normalized_log_mel = (
            log_mel
            - log_mel_mean[:, None]
        ) / log_mel_std[:, None]

        np.save(
            mfcc_path,
            normalized_mfcc.astype(
                np.float32
            ),
        )

        np.save(
            log_mel_path,
            normalized_log_mel.astype(
                np.float32
            ),
        )

    print(
        f"Normalized feature windows: "
        f"{len(feature_df)}"
    )


# ============================================================
# Save feature manifest
# ============================================================

def save_feature_manifest(
    feature_df
):

    ordered_columns = [
        "feature_id",
        "source_dataset",
        "source_recording_id",
        "original_filename",
        "original_category",
        "target_class",
        "split",
        "original_fold",
        "take",
        "window_index",
        "start_sample",
        "end_sample",
        "start_sec",
        "end_sec",
        "source_duration_sec",
        "mfcc_path",
        "log_mel_path",
        "mfcc_shape",
        "log_mel_shape",
        "sample_rate_hz",
        "window_duration_sec",
        "window_hop_sec",
        "feature_version",
        "p2_version",
    ]

    feature_df = feature_df[
        ordered_columns
    ]

    feature_df.to_csv(
        FEATURE_MANIFEST,
        index=False,
    )


# ============================================================
# Verify generated features
# ============================================================

def verify_feature_manifest(
    feature_df
):

    print()
    print(
        "Verifying generated feature manifest..."
    )

    if feature_df[
        "feature_id"
    ].duplicated().any():

        duplicates = (
            feature_df[
                feature_df[
                    "feature_id"
                ].duplicated(
                    keep=False
                )
            ]["feature_id"]
            .tolist()
        )

        fail(
            "Duplicate feature IDs detected:\n"
            + "\n".join(
                f"  {item}"
                for item in duplicates[:20]
            )
        )

    source_split_counts = (
        feature_df.groupby(
            "source_recording_id"
        )["split"]
        .nunique()
    )

    leaked_sources = (
        source_split_counts[
            source_split_counts > 1
        ]
    )

    if len(leaked_sources) > 0:

        fail(
            "Feature-level source split "
            "leakage detected."
        )

    feature_paths_seen = set()

    for _, row in feature_df.iterrows():

        mfcc_path = (
            P3_ROOT
            / row["mfcc_path"]
        )

        log_mel_path = (
            P3_ROOT
            / row["log_mel_path"]
        )

        if not mfcc_path.exists():

            fail(
                f"Missing MFCC feature:\n"
                f"{mfcc_path}"
            )

        if not log_mel_path.exists():

            fail(
                f"Missing Log-Mel feature:\n"
                f"{log_mel_path}"
            )

        mfcc_key = str(
            mfcc_path.resolve()
        )

        log_mel_key = str(
            log_mel_path.resolve()
        )

        if mfcc_key in feature_paths_seen:

            fail(
                f"Duplicate MFCC path:\n"
                f"{mfcc_path}"
            )

        if log_mel_key in feature_paths_seen:

            fail(
                f"Duplicate Log-Mel path:\n"
                f"{log_mel_path}"
            )

        feature_paths_seen.add(
            mfcc_key
        )

        feature_paths_seen.add(
            log_mel_key
        )

        mfcc = np.load(
            mfcc_path
        )

        log_mel = np.load(
            log_mel_path
        )

        if mfcc.ndim != 2:

            fail(
                f"Invalid MFCC dimensions: "
                f"{mfcc.shape}"
            )

        if mfcc.shape[0] != N_MFCC:

            fail(
                f"Invalid MFCC shape: "
                f"{mfcc.shape}"
            )

        if log_mel.ndim != 2:

            fail(
                f"Invalid Log-Mel dimensions: "
                f"{log_mel.shape}"
            )

        if log_mel.shape[0] != N_MELS:

            fail(
                f"Invalid Log-Mel shape: "
                f"{log_mel.shape}"
            )

        if (
            mfcc.shape[1]
            != log_mel.shape[1]
        ):

            fail(
                "MFCC and Log-Mel temporal "
                "dimensions do not match."
            )

        if not np.all(
            np.isfinite(mfcc)
        ):

            fail(
                f"Non-finite values in MFCC:\n"
                f"{mfcc_path}"
            )

        if not np.all(
            np.isfinite(log_mel)
        ):

            fail(
                "Non-finite values in Log-Mel:\n"
                f"{log_mel_path}"
            )

    print(
        "Feature manifest: OK"
    )

    print(
        "Feature files: OK"
    )

    print(
        "Feature source-level split leakage: NONE"
    )

    print(
        "Feature path uniqueness: OK"
    )

    print(
        "Finite feature values: OK"
    )


# ============================================================
# Save feature statistics
# ============================================================

def save_statistics(
    feature_df
):

    rows = []

    total_windows = len(
        feature_df
    )

    unique_sources = (
        feature_df[
            "source_recording_id"
        ].nunique()
    )

    rows.append(
        {
            "metric": (
                "total_feature_windows"
            ),
            "split": "all",
            "target_class": "all",
            "value": total_windows,
        }
    )

    rows.append(
        {
            "metric": (
                "unique_source_recordings"
            ),
            "split": "all",
            "target_class": "all",
            "value": unique_sources,
        }
    )

    for split in EXPECTED_SPLITS:

        split_df = feature_df[
            feature_df["split"]
            == split
        ]

        rows.append(
            {
                "metric": (
                    "feature_windows"
                ),
                "split": split,
                "target_class": "all",
                "value": len(split_df),
            }
        )

        rows.append(
            {
                "metric": (
                    "unique_sources"
                ),
                "split": split,
                "target_class": "all",
                "value": split_df[
                    "source_recording_id"
                ].nunique(),
            }
        )

    for target_class in (
        EXPECTED_CLASSES
    ):

        class_df = feature_df[
            feature_df["target_class"]
            == target_class
        ]

        rows.append(
            {
                "metric": (
                    "feature_windows"
                ),
                "split": "all",
                "target_class": (
                    target_class
                ),
                "value": len(class_df),
            }
        )

    for split in EXPECTED_SPLITS:

        for target_class in (
            EXPECTED_CLASSES
        ):

            count = len(
                feature_df[
                    (
                        feature_df[
                            "split"
                        ]
                        == split
                    )
                    &
                    (
                        feature_df[
                            "target_class"
                        ]
                        == target_class
                    )
                ]
            )

            rows.append(
                {
                    "metric": (
                        "feature_windows"
                    ),
                    "split": split,
                    "target_class": (
                        target_class
                    ),
                    "value": count,
                }
            )

    first_mfcc = np.load(
        P3_ROOT
        / feature_df.iloc[0][
            "mfcc_path"
        ]
    )

    first_log_mel = np.load(
        P3_ROOT
        / feature_df.iloc[0][
            "log_mel_path"
        ]
    )

    rows.extend(
        [
            {
                "metric": (
                    "sample_rate_hz"
                ),
                "split": "all",
                "target_class": "all",
                "value": SAMPLE_RATE,
            },
            {
                "metric": (
                    "window_duration_sec"
                ),
                "split": "all",
                "target_class": "all",
                "value": WINDOW_SECONDS,
            },
            {
                "metric": (
                    "window_hop_sec"
                ),
                "split": "all",
                "target_class": "all",
                "value": HOP_SECONDS,
            },
            {
                "metric": "n_fft",
                "split": "all",
                "target_class": "all",
                "value": N_FFT,
            },
            {
                "metric": (
                    "win_length"
                ),
                "split": "all",
                "target_class": "all",
                "value": WIN_LENGTH,
            },
            {
                "metric": (
                    "hop_length"
                ),
                "split": "all",
                "target_class": "all",
                "value": HOP_LENGTH,
            },
            {
                "metric": "n_mels",
                "split": "all",
                "target_class": "all",
                "value": N_MELS,
            },
            {
                "metric": "n_mfcc",
                "split": "all",
                "target_class": "all",
                "value": N_MFCC,
            },
            {
                "metric": "mfcc_shape",
                "split": "all",
                "target_class": "all",
                "value": str(
                    first_mfcc.shape
                ),
            },
            {
                "metric": (
                    "log_mel_shape"
                ),
                "split": "all",
                "target_class": "all",
                "value": str(
                    first_log_mel.shape
                ),
            },
            {
                "metric": (
                    "feature_version"
                ),
                "split": "all",
                "target_class": "all",
                "value": P3_VERSION,
            },
        ]
    )

    statistics_df = pd.DataFrame(
        rows
    )

    statistics_df.to_csv(
        STATISTICS_FILE,
        index=False,
    )


# ============================================================
# Final summary
# ============================================================

def print_final_summary(
    feature_df
):

    print()
    print("=" * 70)
    print(
        "P3 FEATURE EXTRACTION COMPLETED"
    )
    print("=" * 70)

    print()
    print("P3 version:")
    print(
        f"  {P3_VERSION}"
    )

    print()
    print("Input:")
    print(
        f"  P2 manifest: {P2_MANIFEST}"
    )
    print(
        f"  P2 version:  {P2_VERSION}"
    )

    print()
    print("Feature windows:")
    print(
        f"  Total: {len(feature_df)}"
    )

    print()
    print("Unique sources:")
    print(
        f"  {feature_df['source_recording_id'].nunique()}"
    )

    print()
    print("Class distribution:")

    class_counts = (
        feature_df[
            "target_class"
        ]
        .value_counts()
        .reindex(
            EXPECTED_CLASSES,
            fill_value=0,
        )
    )

    for target_class, count in (
        class_counts.items()
    ):

        print(
            f"  {target_class}: {count}"
        )

    print()
    print("Split distribution:")

    split_counts = (
        feature_df[
            "split"
        ]
        .value_counts()
        .reindex(
            EXPECTED_SPLITS,
            fill_value=0,
        )
    )

    for split, count in (
        split_counts.items()
    ):

        print(
            f"  {split}: {count}"
        )

    print()
    print("Feature parameters:")

    print(
        f"  Sample rate:       "
        f"{SAMPLE_RATE} Hz"
    )

    print(
        f"  Window:            "
        f"{WINDOW_SECONDS} sec"
    )

    print(
        f"  Window hop:        "
        f"{HOP_SECONDS} sec"
    )

    print(
        f"  FFT:               "
        f"{N_FFT}"
    )

    print(
        f"  Win length:        "
        f"{WIN_LENGTH}"
    )

    print(
        f"  Hop length:        "
        f"{HOP_LENGTH}"
    )

    print(
        f"  Mel bands:         "
        f"{N_MELS}"
    )

    print(
        f"  MFCC coefficients: "
        f"{N_MFCC}"
    )

    print(
        f"  Center:            "
        f"{CENTER}"
    )

    print()
    print("Normalization:")
    print(
        "  Training split only: YES"
    )

    print()
    print("Leakage:")
    print(
        "  Source-level leakage: NONE"
    )

    print()
    print("Feature IDs:")
    print(
        "  Unique: YES"
    )

    print()
    print("Outputs:")

    print(
        f"  {FEATURE_MANIFEST}"
    )

    print(
        f"  {NORMALIZATION_FILE}"
    )

    print(
        f"  {CONFIG_FILE}"
    )

    print(
        f"  {STATISTICS_FILE}"
    )

    print(
        f"  {SPEC_FILE}"
    )

    print()
    print("=" * 70)
    print(
        "READY FOR M2-P4 MODEL TRAINING"
    )
    print("=" * 70)


# ============================================================
# Main
# ============================================================

def main():

    ensure_directories()

    df = (
        load_and_verify_p2_manifest()
    )

    verify_source_split_leakage(
        df
    )

    save_feature_config()

    save_specification()

    feature_df = (
        extract_all_features(
            df
        )
    )

    (
        mfcc_mean,
        mfcc_std,
        log_mel_mean,
        log_mel_std,
    ) = (
        calculate_training_statistics(
            feature_df
        )
    )

    normalize_features(
        feature_df,
        mfcc_mean,
        mfcc_std,
        log_mel_mean,
        log_mel_std,
    )

    save_feature_manifest(
        feature_df
    )

    verify_feature_manifest(
        feature_df
    )

    save_statistics(
        feature_df
    )

    print_final_summary(
        feature_df
    )


if __name__ == "__main__":
    main()