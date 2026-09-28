from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, TensorDataset

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

P3_ROOT = PROJECT_ROOT / "FEATURE_EXTRACTION"
P5_ROOT = PROJECT_ROOT / "ROBUSTNESS_ANALYSIS"

P3_MANIFEST = (
    P3_ROOT
    / "metadata"
    / "feature_manifest.csv"
)

P3_CONFIG = (
    P3_ROOT
    / "metadata"
    / "feature_config.json"
)

RESULTS_DIR = P5_ROOT / "results"
MODELS_DIR = P5_ROOT / "models"
REPORTS_DIR = P5_ROOT / "reports"
METADATA_DIR = P5_ROOT / "metadata"
PLOTS_DIR = P5_ROOT / "plots"

P4_BASELINE_RESULTS = (
    RESULTS_DIR
    / "P4_CNN_RESULTS_baseline.csv"
)

BEST_MODEL_PATH = (
    MODELS_DIR
    / "cnn_logmel_augmented_best.pt"
)

EXPECTED_CLASSES = [
    "ENGINE",
    "RAIN",
    "SIREN",
    "WIND",
]

EXPECTED_LOGMEL_SHAPE = (
    64,
    197,
)

EXPECTED_P3_VERSION = "P3-v0.1"
EXPECTED_P2_VERSION = "P2-v0.1"

SEED = 42


# ============================================================
# SEED
# ============================================================

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# IMPORT P4 MODEL
# ============================================================

MODEL_SCRIPT_DIR = (
    PROJECT_ROOT
    / "MODEL_TRAINING"
    / "scripts"
)

if str(MODEL_SCRIPT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(MODEL_SCRIPT_DIR),
    )

from cnn_model import SmallCNN


# ============================================================
# DIRECTORIES
# ============================================================

def create_directories() -> None:
    for directory in [
        RESULTS_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        METADATA_DIR,
        PLOTS_DIR,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


# ============================================================
# FEATURE PATH
# ============================================================

def resolve_feature_path(
    value: str,
) -> Path:

    raw = Path(value)

    if raw.is_absolute():
        return raw

    normalized = (
        str(value)
        .replace("\\", "/")
    )

    candidate_1 = (
        P3_ROOT
        / normalized
    )

    candidate_2 = (
        PROJECT_ROOT
        / normalized
    )

    if candidate_1.exists():
        return candidate_1

    if candidate_2.exists():
        return candidate_2

    return candidate_1


# ============================================================
# P3 MANIFEST
# ============================================================

def load_manifest() -> pd.DataFrame:

    if not P3_MANIFEST.exists():
        raise FileNotFoundError(
            f"P3 manifest not found:\n"
            f"{P3_MANIFEST}"
        )

    df = pd.read_csv(
        P3_MANIFEST
    )

    required_columns = [
        "target_class",
        "split",
        "log_mel_path",
        "source_recording_id",
        "original_filename",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required P3 columns: "
            f"{missing}"
        )

    df["_class"] = (
        df["target_class"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["_split"] = (
        df["split"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["_source_id"] = (
        df["source_recording_id"]
        .astype(str)
    )

    df["_recording_key"] = (
        df["source_recording_id"]
        .astype(str)
        + "::"
        + df["original_filename"]
        .astype(str)
    )

    if len(df) != 640:
        raise ValueError(
            f"Expected 640 P3 windows, "
            f"found {len(df)}."
        )

    if set(df["_class"]) != set(
        EXPECTED_CLASSES
    ):
        raise ValueError(
            "P3 classes do not match "
            "expected classes."
        )

    if set(df["_split"]) != {
        "train",
        "validation",
        "test",
    }:
        raise ValueError(
            "P3 splits do not match "
            "expected splits."
        )

    return df


# ============================================================
# SOURCE-LEVEL LEAKAGE CHECK
# ============================================================

def verify_source_leakage(
    df: pd.DataFrame,
) -> None:

    train_sources = set(
        df.loc[
            df["_split"] == "train",
            "_source_id",
        ]
    )

    validation_sources = set(
        df.loc[
            df["_split"] == "validation",
            "_source_id",
        ]
    )

    test_sources = set(
        df.loc[
            df["_split"] == "test",
            "_source_id",
        ]
    )

    if (
        train_sources
        & validation_sources
    ):
        raise ValueError(
            "Train/validation "
            "source leakage detected."
        )

    if (
        train_sources
        & test_sources
    ):
        raise ValueError(
            "Train/test source leakage "
            "detected."
        )

    if (
        validation_sources
        & test_sources
    ):
        raise ValueError(
            "Validation/test source leakage "
            "detected."
        )


# ============================================================
# LOAD FEATURES
# ============================================================

def load_features(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:

    class_to_index = {
        class_name: index
        for index, class_name
        in enumerate(EXPECTED_CLASSES)
    }

    features = []
    labels = []

    for _, row in df.iterrows():

        feature_path = resolve_feature_path(
            row["log_mel_path"]
        )

        if not feature_path.exists():
            raise FileNotFoundError(
                f"Feature file not found:\n"
                f"{feature_path}"
            )

        feature = np.load(
            feature_path
        )

        if feature.shape != (
            EXPECTED_LOGMEL_SHAPE
        ):
            raise ValueError(
                f"Unexpected feature shape "
                f"{feature.shape}.\n"
                f"Expected "
                f"{EXPECTED_LOGMEL_SHAPE}"
            )

        if not np.isfinite(
            feature
        ).all():
            raise ValueError(
                f"Non-finite feature values:\n"
                f"{feature_path}"
            )

        features.append(
            feature.astype(
                np.float32
            )
        )

        labels.append(
            class_to_index[
                row["_class"]
            ]
        )

    X = np.stack(
        features,
        axis=0,
    )

    y = np.asarray(
        labels,
        dtype=np.int64,
    )

    return X, y


# ============================================================
# TRAINING-ONLY FEATURE AUGMENTATION
# ============================================================

def add_feature_noise(
    X: np.ndarray,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:

    if noise_std <= 0:
        return X.copy()

    noise = rng.normal(
        loc=0.0,
        scale=noise_std,
        size=X.shape,
    ).astype(
        np.float32
    )

    return (
        X + noise
    ).astype(
        np.float32
    )


def time_mask(
    X: np.ndarray,
    max_width: int,
    rng: np.random.Generator,
) -> np.ndarray:

    X_aug = X.copy()

    if max_width <= 0:
        return X_aug

    n_samples, _, n_frames = (
        X_aug.shape
    )

    for index in range(n_samples):

        width = int(
            rng.integers(
                0,
                max_width + 1,
            )
        )

        if width <= 0:
            continue

        width = min(
            width,
            n_frames,
        )

        start_max = (
            n_frames - width
        )

        start = int(
            rng.integers(
                0,
                start_max + 1,
            )
        )

        fill_value = float(
            X_aug[index].mean()
        )

        X_aug[
            index,
            :,
            start:start + width,
        ] = fill_value

    return X_aug


def frequency_mask(
    X: np.ndarray,
    max_width: int,
    rng: np.random.Generator,
) -> np.ndarray:

    X_aug = X.copy()

    if max_width <= 0:
        return X_aug

    n_samples, n_mels, _ = (
        X_aug.shape
    )

    for index in range(n_samples):

        width = int(
            rng.integers(
                0,
                max_width + 1,
            )
        )

        if width <= 0:
            continue

        width = min(
            width,
            n_mels,
        )

        start_max = (
            n_mels - width
        )

        start = int(
            rng.integers(
                0,
                start_max + 1,
            )
        )

        fill_value = float(
            X_aug[index].mean()
        )

        X_aug[
            index,
            start:start + width,
            :,
        ] = fill_value

    return X_aug


def augment_training_features(
    X: np.ndarray,
    noise_std: float,
    time_mask_width: int,
    frequency_mask_width: int,
    seed: int,
) -> np.ndarray:

    rng = np.random.default_rng(
        seed
    )

    X_aug = X.copy()

    X_aug = add_feature_noise(
        X_aug,
        noise_std,
        rng,
    )

    X_aug = time_mask(
        X_aug,
        time_mask_width,
        rng,
    )

    X_aug = frequency_mask(
        X_aug,
        frequency_mask_width,
        rng,
    )

    return X_aug.astype(
        np.float32
    )


# ============================================================
# DATA LOADER
# ============================================================

def create_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:

    tensors = TensorDataset(
        torch.from_numpy(
            X[:, np.newaxis, :, :]
        ),
        torch.from_numpy(y),
    )

    return DataLoader(
        tensors,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "macro_precision": float(
            precision
        ),
        "macro_recall": float(
            recall
        ),
        "macro_f1": float(
            f1
        ),
    }


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:

    model.eval()

    targets = []
    probabilities = []

    for X_batch, y_batch in loader:

        X_batch = X_batch.to(
            device
        )

        logits = model(
            X_batch
        )

        probs = torch.softmax(
            logits,
            dim=1,
        )

        targets.append(
            y_batch.numpy()
        )

        probabilities.append(
            probs.cpu().numpy()
        )

    y_true = np.concatenate(
        targets,
        axis=0,
    )

    y_prob = np.concatenate(
        probabilities,
        axis=0,
    )

    return y_true, y_prob


# ============================================================
# RECORDING-LEVEL EVALUATION
# ============================================================

def recording_level_evaluation(
    df: pd.DataFrame,
    probabilities: np.ndarray,
) -> pd.DataFrame:

    test_df = (
        df[df["_split"] == "test"]
        .copy()
        .reset_index(drop=True)
    )

    if len(test_df) != len(
        probabilities
    ):
        raise ValueError(
            "Test probability count does "
            "not match test manifest."
        )

    test_df["_probabilities"] = list(
        probabilities
    )

    records = []

    for recording_key, group in (
        test_df.groupby(
            "_recording_key",
            sort=True,
        )
    ):

        matrix = np.stack(
            group[
                "_probabilities"
            ].to_list()
        )

        mean_probability = (
            matrix.mean(axis=0)
        )

        prediction = int(
            np.argmax(
                mean_probability
            )
        )

        true_classes = group[
            "_class"
        ].unique()

        if len(true_classes) != 1:
            raise ValueError(
                f"Multiple classes in "
                f"recording {recording_key}: "
                f"{true_classes}"
            )

        record = {
            "recording_key":
                recording_key,
            "true_class":
                true_classes[0],
            "predicted_class":
                EXPECTED_CLASSES[
                    prediction
                ],
            "confidence":
                float(
                    mean_probability[
                        prediction
                    ]
                ),
            "window_count":
                len(group),
        }

        for index, class_name in (
            enumerate(
                EXPECTED_CLASSES
            )
        ):
            record[
                f"prob_{class_name}"
            ] = float(
                mean_probability[
                    index
                ]
            )

        records.append(
            record
        )

    return pd.DataFrame(
        records
    )


# ============================================================
# TRAIN ONE MODEL
# ============================================================

def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[
    SmallCNN,
    pd.DataFrame,
]:

    train_loader = create_loader(
        X_train,
        y_train,
        args.batch_size,
        shuffle=True,
    )

    validation_loader = create_loader(
        X_validation,
        y_validation,
        args.batch_size,
        shuffle=False,
    )

    model = SmallCNN(
        num_classes=len(
            EXPECTED_CLASSES
        ),
        base_channels=args.base_channels,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_f1 = -np.inf
    best_epoch = 0

    history = []

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        start = time.perf_counter()

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in (
            train_loader
        ):

            X_batch = X_batch.to(
                device
            )

            y_batch = y_batch.to(
                device
            )

            optimizer.zero_grad()

            logits = model(
                X_batch
            )

            loss = criterion(
                logits,
                y_batch,
            )

            loss.backward()
            optimizer.step()

            running_loss += (
                loss.item()
                * X_batch.size(0)
            )

            predictions = (
                logits.argmax(
                    dim=1
                )
            )

            correct += (
                (
                    predictions
                    == y_batch
                )
                .sum()
                .item()
            )

            total += (
                X_batch.size(0)
            )

        train_loss = (
            running_loss
            / total
        )

        train_accuracy = (
            correct
            / total
        )

        val_true, val_prob = (
            evaluate(
                model,
                validation_loader,
                device,
            )
        )

        val_pred = (
            val_prob.argmax(
                axis=1
            )
        )

        metrics = calculate_metrics(
            val_true,
            val_pred,
        )

        epoch_time = (
            time.perf_counter()
            - start
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss":
                    train_loss,
                "train_accuracy":
                    train_accuracy,
                "validation_accuracy":
                    metrics[
                        "accuracy"
                    ],
                "validation_macro_precision":
                    metrics[
                        "macro_precision"
                    ],
                "validation_macro_recall":
                    metrics[
                        "macro_recall"
                    ],
                "validation_macro_f1":
                    metrics[
                        "macro_f1"
                    ],
                "epoch_time_seconds":
                    epoch_time,
            }
        )

        if (
            metrics["macro_f1"]
            > best_f1
        ):
            best_f1 = (
                metrics["macro_f1"]
            )

            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),
                    "classes":
                        EXPECTED_CLASSES,
                    "input_shape":
                        [1, 64, 197],
                    "base_channels":
                        args.base_channels,
                    "epoch":
                        epoch,
                    "validation_macro_f1":
                        float(best_f1),
                    "seed":
                        args.seed,
                    "p3_version":
                        EXPECTED_P3_VERSION,
                    "augmentation":
                        {
                            "noise_std":
                                args.noise_std,
                            "time_mask_width":
                                args.time_mask_width,
                            "frequency_mask_width":
                                args.frequency_mask_width,
                        },
                },
                BEST_MODEL_PATH,
            )

    checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    history_df = pd.DataFrame(
        history
    )

    return (
        model,
        history_df,
    )


# ============================================================
# MAIN
# ============================================================

def run(
    args: argparse.Namespace,
) -> None:

    set_seed(args.seed)
    create_directories()

    print("=" * 70)
    print(
        "SILENTVOX M2-P5 — "
        "TRAINING ROBUSTNESS AUGMENTATION"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # P3 CONFIG
    # --------------------------------------------------------

    if not P3_CONFIG.exists():
        raise FileNotFoundError(
            f"P3 config not found:\n"
            f"{P3_CONFIG}"
        )

    with open(
        P3_CONFIG,
        "r",
        encoding="utf-8",
    ) as file:
        config = json.load(file)

    if config.get(
        "feature_version"
    ) != EXPECTED_P3_VERSION:
        raise ValueError(
            "Unexpected P3 version."
        )

    if config.get(
        "p2_version"
    ) != EXPECTED_P2_VERSION:
        raise ValueError(
            "Unexpected P2 version."
        )

    # --------------------------------------------------------
    # MANIFEST
    # --------------------------------------------------------

    df = load_manifest()

    verify_source_leakage(
        df
    )

    train_df = (
        df[df["_split"] == "train"]
        .reset_index(drop=True)
    )

    validation_df = (
        df[df["_split"] == "validation"]
        .reset_index(drop=True)
    )

    test_df = (
        df[df["_split"] == "test"]
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # LOAD P3 FEATURES
    # --------------------------------------------------------

    print()
    print(
        "Loading P3 Log-Mel features..."
    )

    X_train, y_train = load_features(
        train_df
    )

    X_validation, y_validation = (
        load_features(
            validation_df
        )
    )

    X_test, y_test = load_features(
        test_df
    )

    print(
        f"Train:       {X_train.shape}"
    )

    print(
        f"Validation:  {X_validation.shape}"
    )

    print(
        f"Test:        {X_test.shape}"
    )

    # --------------------------------------------------------
    # AUGMENT TRAINING ONLY
    # --------------------------------------------------------

    print()
    print(
        "Applying training-only "
        "feature augmentation..."
    )

    X_train_augmented = (
        augment_training_features(
            X_train,
            noise_std=args.noise_std,
            time_mask_width=(
                args.time_mask_width
            ),
            frequency_mask_width=(
                args.frequency_mask_width
            ),
            seed=args.seed,
        )
    )

    # Validation and test remain untouched.
    X_validation_augmented = (
        X_validation.copy()
    )

    X_test_augmented = (
        X_test.copy()
    )

    print(
        "Validation augmentation: NONE"
    )

    print(
        "Test augmentation: NONE"
    )

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    print()
    print(
        f"Device: {device}"
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model, history_df = (
        train_model(
            X_train_augmented,
            y_train,
            X_validation_augmented,
            y_validation,
            args,
            device,
        )
    )

    history_df.to_csv(
        REPORTS_DIR
        / "P5_augmented_training_history.csv",
        index=False,
    )

    best_row = (
        history_df.loc[
            history_df[
                "validation_macro_f1"
            ].idxmax()
        ]
    )

    best_epoch = int(
        best_row["epoch"]
    )

    best_validation_f1 = float(
        best_row[
            "validation_macro_f1"
        ]
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    test_loader = create_loader(
        X_test_augmented,
        y_test,
        args.batch_size,
        shuffle=False,
    )

    test_true, test_probabilities = (
        evaluate(
            model,
            test_loader,
            device,
        )
    )

    test_predictions = (
        test_probabilities.argmax(
            axis=1
        )
    )

    window_metrics = (
        calculate_metrics(
            test_true,
            test_predictions,
        )
    )

    # --------------------------------------------------------
    # RECORDING TEST
    # --------------------------------------------------------

    recording_predictions = (
        recording_level_evaluation(
            df,
            test_probabilities,
        )
    )

    class_to_index = {
        class_name: index
        for index, class_name
        in enumerate(
            EXPECTED_CLASSES
        )
    }

    recording_true = (
        recording_predictions[
            "true_class"
        ]
        .map(class_to_index)
        .to_numpy()
    )

    recording_pred = (
        recording_predictions[
            "predicted_class"
        ]
        .map(class_to_index)
        .to_numpy()
    )

    recording_metrics = (
        calculate_metrics(
            recording_true,
            recording_pred,
        )
    )

    recording_predictions.to_csv(
        REPORTS_DIR
        / "P5_augmented_recording_predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # SAVE METRICS
    # --------------------------------------------------------

    metrics = {
        "p5_version": "P5-v0.1",
        "p4_baseline":
            "P4-v0.1",
        "p3_version":
            EXPECTED_P3_VERSION,
        "p2_version":
            EXPECTED_P2_VERSION,
        "model":
            "SmallCNN",
        "feature":
            "Log-Mel Spectrogram",
        "input_shape":
            "(64, 197)",
        "classes":
            len(EXPECTED_CLASSES),
        "train_windows":
            len(train_df),
        "validation_windows":
            len(validation_df),
        "test_windows":
            len(test_df),
        "test_recordings":
            len(recording_predictions),
        "best_epoch":
            best_epoch,
        "best_validation_macro_f1":
            best_validation_f1,
        "window_accuracy":
            window_metrics[
                "accuracy"
            ],
        "window_macro_precision":
            window_metrics[
                "macro_precision"
            ],
        "window_macro_recall":
            window_metrics[
                "macro_recall"
            ],
        "window_macro_f1":
            window_metrics[
                "macro_f1"
            ],
        "recording_accuracy":
            recording_metrics[
                "accuracy"
            ],
        "recording_macro_precision":
            recording_metrics[
                "macro_precision"
            ],
        "recording_macro_recall":
            recording_metrics[
                "macro_recall"
            ],
        "recording_macro_f1":
            recording_metrics[
                "macro_f1"
            ],
        "augmentation_noise_std":
            args.noise_std,
        "augmentation_time_mask_width":
            args.time_mask_width,
        "augmentation_frequency_mask_width":
            args.frequency_mask_width,
        "seed":
            args.seed,
        "device":
            str(device),
    }

    metrics_path = (
        RESULTS_DIR
        / "P5_augmented_results.csv"
    )

    pd.DataFrame(
        [metrics]
    ).to_csv(
        metrics_path,
        index=False,
    )

    # --------------------------------------------------------
    # BASELINE COMPARISON
    # --------------------------------------------------------

    comparison_rows = []

    if P4_BASELINE_RESULTS.exists():

        baseline_df = pd.read_csv(
            P4_BASELINE_RESULTS
        )

        if not baseline_df.empty:

            baseline = (
                baseline_df.iloc[0]
            )

            comparison_rows.append(
                {
                    "model":
                        "P4_Baseline",
                    "window_accuracy":
                        baseline.get(
                            "window_accuracy",
                            np.nan,
                        ),
                    "window_macro_f1":
                        baseline.get(
                            "window_macro_f1",
                            np.nan,
                        ),
                    "recording_accuracy":
                        baseline.get(
                            "recording_accuracy",
                            np.nan,
                        ),
                    "recording_macro_f1":
                        baseline.get(
                            "recording_macro_f1",
                            np.nan,
                        ),
                }
            )

    comparison_rows.append(
        {
            "model":
                "P5_Augmented",
            "window_accuracy":
                window_metrics[
                    "accuracy"
                ],
            "window_macro_f1":
                window_metrics[
                    "macro_f1"
                ],
            "recording_accuracy":
                recording_metrics[
                    "accuracy"
                ],
            "recording_macro_f1":
                recording_metrics[
                    "macro_f1"
                ],
        }
    )

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    comparison_df.to_csv(
        REPORTS_DIR
        / "P5_baseline_vs_augmented.csv",
        index=False,
    )

    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    metadata = {
        "p5_version":
            "P5-v0.1",
        "p4_baseline_version":
            "P4-v0.1",
        "p3_version":
            EXPECTED_P3_VERSION,
        "p2_version":
            EXPECTED_P2_VERSION,
        "classes":
            EXPECTED_CLASSES,
        "input_shape":
            [1, 64, 197],
        "augmentation":
            {
                "noise_std":
                    args.noise_std,
                "time_mask_width":
                    args.time_mask_width,
                "frequency_mask_width":
                    args.frequency_mask_width,
                "training_only":
                    True,
                "validation_untouched":
                    True,
                "test_untouched":
                    True,
            },
        "training":
            {
                "epochs":
                    args.epochs,
                "batch_size":
                    args.batch_size,
                "learning_rate":
                    args.learning_rate,
                "weight_decay":
                    args.weight_decay,
                "base_channels":
                    args.base_channels,
                "seed":
                    args.seed,
            },
        "best_epoch":
            best_epoch,
        "best_validation_macro_f1":
            best_validation_f1,
        "device":
            str(device),
    }

    with open(
        METADATA_DIR
        / "P5_AUGMENTATION_METADATA.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "P5 AUGMENTED TRAINING COMPLETED"
    )
    print("=" * 70)

    print()
    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Best validation macro-F1: "
        f"{best_validation_f1:.4f}"
    )

    print()
    print("Window-level test:")

    print(
        f"  Accuracy:        "
        f"{window_metrics['accuracy']:.4f}"
    )

    print(
        f"  Macro Precision: "
        f"{window_metrics['macro_precision']:.4f}"
    )

    print(
        f"  Macro Recall:    "
        f"{window_metrics['macro_recall']:.4f}"
    )

    print(
        f"  Macro F1:        "
        f"{window_metrics['macro_f1']:.4f}"
    )

    print()
    print("Recording-level test:")

    print(
        f"  Accuracy:        "
        f"{recording_metrics['accuracy']:.4f}"
    )

    print(
        f"  Macro Precision: "
        f"{recording_metrics['macro_precision']:.4f}"
    )

    print(
        f"  Macro Recall:    "
        f"{recording_metrics['macro_recall']:.4f}"
    )

    print(
        f"  Macro F1:        "
        f"{recording_metrics['macro_f1']:.4f}"
    )

    print()
    print("Augmentation:")
    print(
        f"  Noise std: "
        f"{args.noise_std}"
    )
    print(
        f"  Time mask width: "
        f"{args.time_mask_width}"
    )
    print(
        f"  Frequency mask width: "
        f"{args.frequency_mask_width}"
    )

    print()
    print("Outputs:")
    print(
        f"  {BEST_MODEL_PATH}"
    )
    print(
        f"  {metrics_path}"
    )
    print(
        f"  {REPORTS_DIR / 'P5_augmented_training_history.csv'}"
    )
    print(
        f"  {REPORTS_DIR / 'P5_augmented_recording_predictions.csv'}"
    )
    print(
        f"  {REPORTS_DIR / 'P5_baseline_vs_augmented.csv'}"
    )
    print(
        f"  {METADATA_DIR / 'P5_AUGMENTATION_METADATA.json'}"
    )

    print()
    print("=" * 70)
    print("READY FOR M2-P6")
    print("=" * 70)


# ============================================================
# ARGUMENTS
# ============================================================

def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "SilentVox M2-P5 "
            "training robustness analysis"
        )
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--base-channels",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.02,
    )

    parser.add_argument(
        "--time-mask-width",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--frequency-mask-width",
        type=int,
        default=6,
    )

    return parser.parse_args()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    arguments = parse_arguments()
    run(arguments)