from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, TensorDataset

from cnn_model import (
    SmallCNN,
    count_parameters,
    estimate_model_size_kb,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

P2_ROOT = PROJECT_ROOT / "DATASET_PREPROCESSING"
P3_ROOT = PROJECT_ROOT / "FEATURE_EXTRACTION"
P4_ROOT = PROJECT_ROOT / "MODEL_TRAINING"

P3_MANIFEST = P3_ROOT / "metadata" / "feature_manifest.csv"
P3_CONFIG = P3_ROOT / "metadata" / "feature_config.json"

MODELS_DIR = P4_ROOT / "models"
METRICS_DIR = P4_ROOT / "metrics"
CONFUSION_DIR = P4_ROOT / "confusion_matrices"
BENCHMARK_DIR = P4_ROOT / "benchmarks"
METADATA_DIR = P4_ROOT / "metadata"

BEST_MODEL_PATH = MODELS_DIR / "cnn_logmel_best.pt"

EXPECTED_P3_VERSION = "P3-v0.1"
EXPECTED_P2_VERSION = "P2-v0.1"

EXPECTED_CLASSES = [
    "ENGINE",
    "RAIN",
    "SIREN",
    "WIND",
]

EXPECTED_LOGMEL_SHAPE = (64, 197)

SEED = 42


# ============================================================
# REPRODUCIBILITY
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
# DIRECTORIES
# ============================================================

def create_directories() -> None:
    for directory in [
        MODELS_DIR,
        METRICS_DIR,
        CONFUSION_DIR,
        BENCHMARK_DIR,
        METADATA_DIR,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


# ============================================================
# COLUMN HELPERS
# ============================================================

def find_column(
    df: pd.DataFrame,
    candidates: list[str],
    required: bool = True,
) -> str | None:
    normalized = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower()

        if key in normalized:
            return normalized[key]

    if required:
        raise ValueError(
            f"Could not find required column.\n"
            f"Expected one of: {candidates}\n"
            f"Available columns: {list(df.columns)}"
        )

    return None


# ============================================================
# P3 MANIFEST
# ============================================================

def load_p3_manifest() -> pd.DataFrame:
    if not P3_MANIFEST.exists():
        raise FileNotFoundError(
            f"P3 feature manifest not found:\n{P3_MANIFEST}"
        )

    df = pd.read_csv(P3_MANIFEST)

    if df.empty:
        raise ValueError(
            "P3 feature manifest is empty."
        )

    class_column = find_column(
        df,
        [
            "target_class",
            "class",
            "label",
        ],
    )

    split_column = find_column(
        df,
        [
            "split",
            "dataset_split",
        ],
    )

    feature_column = find_column(
        df,
        [
            "log_mel_path",
            "logmel_path",
            "log_mel_feature_path",
            "logmel_feature_path",
        ],
    )

    source_column = find_column(
        df,
        [
            "source_recording_id",
            "recording_id",
            "source_id",
        ],
    )

    filename_column = find_column(
        df,
        [
            "original_filename",
            "filename",
            "source_filename",
        ],
        required=False,
    )

    df["_class"] = (
        df[class_column]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["_split"] = (
        df[split_column]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["_logmel_path"] = (
        df[feature_column].astype(str)
    )

    df["_source_id"] = (
        df[source_column].astype(str)
    )

    if filename_column is not None:
        df["_recording_key"] = (
            df[source_column].astype(str)
            + "::"
            + df[filename_column].astype(str)
        )
    else:
        df["_recording_key"] = (
            df[source_column].astype(str)
        )

    expected_class_set = set(EXPECTED_CLASSES)
    actual_class_set = set(
        df["_class"].unique()
    )

    if actual_class_set != expected_class_set:
        raise ValueError(
            f"P3 class mismatch.\n"
            f"Expected: {EXPECTED_CLASSES}\n"
            f"Found: {sorted(actual_class_set)}"
        )

    expected_splits = {
        "train",
        "validation",
        "test",
    }

    actual_splits = set(
        df["_split"].unique()
    )

    if actual_splits != expected_splits:
        raise ValueError(
            f"P3 split mismatch.\n"
            f"Expected: {sorted(expected_splits)}\n"
            f"Found: {sorted(actual_splits)}"
        )

    if len(df) != 640:
        raise ValueError(
            f"Expected 640 P3 feature windows, "
            f"found {len(df)}."
        )

    return df


# ============================================================
# P3 VALIDATION
# ============================================================

def verify_p3_input(df: pd.DataFrame) -> None:
    train_df = df[
        df["_split"] == "train"
    ]

    validation_df = df[
        df["_split"] == "validation"
    ]

    test_df = df[
        df["_split"] == "test"
    ]

    train_sources = set(
        train_df["_source_id"]
    )

    validation_sources = set(
        validation_df["_source_id"]
    )

    test_sources = set(
        test_df["_source_id"]
    )

    leakage_train_validation = (
        train_sources & validation_sources
    )

    leakage_train_test = (
        train_sources & test_sources
    )

    leakage_validation_test = (
        validation_sources & test_sources
    )

    if (
        leakage_train_validation
        or leakage_train_test
        or leakage_validation_test
    ):
        raise ValueError(
            "Source-level leakage detected "
            "in P3 feature manifest."
        )

    print("P3 input verification:")
    print(f"  Feature windows: {len(df)}")
    print(f"  Classes: {EXPECTED_CLASSES}")
    print(f"  Train: {len(train_df)}")
    print(f"  Validation: {len(validation_df)}")
    print(f"  Test: {len(test_df)}")
    print(
        f"  Train sources: "
        f"{len(train_sources)}"
    )
    print(
        f"  Validation sources: "
        f"{len(validation_sources)}"
    )
    print(
        f"  Test sources: "
        f"{len(test_sources)}"
    )
    print("  Source-level leakage: NONE")


# ============================================================
# FEATURE PATH RESOLUTION
# ============================================================

def resolve_feature_path(
    path_value: str,
) -> Path:
    raw = Path(path_value)

    if raw.is_absolute():
        return raw

    normalized = (
        str(path_value)
        .replace("\\", os.sep)
        .replace("/", os.sep)
    )

    candidate_paths = [
        P3_ROOT / normalized,
        PROJECT_ROOT / normalized,
    ]

    for candidate in candidate_paths:
        if candidate.exists():
            return candidate

    return P3_ROOT / normalized


# ============================================================
# FEATURE LOADING
# ============================================================

def load_feature_array(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:

    features: list[np.ndarray] = []
    labels: list[int] = []

    class_to_index = {
        class_name: index
        for index, class_name
        in enumerate(EXPECTED_CLASSES)
    }

    for _, row in df.iterrows():

        feature_path = resolve_feature_path(
            row["_logmel_path"]
        )

        if not feature_path.exists():
            raise FileNotFoundError(
                f"P3 feature file not found:\n"
                f"{feature_path}"
            )

        feature = np.load(feature_path)

        if feature.shape != EXPECTED_LOGMEL_SHAPE:
            raise ValueError(
                f"Unexpected P3 Log-Mel shape "
                f"{feature.shape}.\n"
                f"Expected: "
                f"{EXPECTED_LOGMEL_SHAPE}\n"
                f"File: {feature_path}"
            )

        if not np.isfinite(feature).all():
            raise ValueError(
                f"Non-finite values found in:\n"
                f"{feature_path}"
            )

        features.append(
            feature.astype(np.float32)
        )

        labels.append(
            class_to_index[row["_class"]]
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
# TORCH DATA
# ============================================================

def create_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:

    X_tensor = torch.from_numpy(
        X[:, np.newaxis, :, :]
    )

    y_tensor = torch.from_numpy(y)

    dataset = TensorDataset(
        X_tensor,
        y_tensor,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


# ============================================================
# DEVICE
# ============================================================

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# MODEL EVALUATION
# ============================================================

@torch.no_grad()
def predict_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:

    model.eval()

    all_probabilities: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for X_batch, y_batch in loader:

        X_batch = X_batch.to(device)

        logits = model(X_batch)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        all_probabilities.append(
            probabilities.cpu().numpy()
        )

        all_targets.append(
            y_batch.numpy()
        )

    probabilities = np.concatenate(
        all_probabilities,
        axis=0,
    )

    targets = np.concatenate(
        all_targets,
        axis=0,
    )

    return targets, probabilities


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

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

def save_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:

    report = classification_report(
        y_true,
        y_pred,
        labels=list(
            range(len(EXPECTED_CLASSES))
        ),
        target_names=EXPECTED_CLASSES,
        output_dict=True,
        zero_division=0,
    )

    rows = []

    for class_name in EXPECTED_CLASSES:
        row = report[class_name]

        rows.append(
            {
                "class": class_name,
                "precision": row["precision"],
                "recall": row["recall"],
                "f1_score": row["f1-score"],
                "support": row["support"],
            }
        )

    macro = report["macro avg"]

    rows.append(
        {
            "class": "MACRO_AVG",
            "precision": macro["precision"],
            "recall": macro["recall"],
            "f1_score": macro["f1-score"],
            "support": macro["support"],
        }
    )

    pd.DataFrame(rows).to_csv(
        output_path,
        index=False,
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(
            range(len(EXPECTED_CLASSES))
        ),
    )

    output_df = pd.DataFrame(
        matrix,
        index=EXPECTED_CLASSES,
        columns=EXPECTED_CLASSES,
    )

    output_df.index.name = "true_class"

    output_df.to_csv(output_path)


# ============================================================
# RECORDING-LEVEL EVALUATION
# ============================================================

def recording_level_evaluation(
    df: pd.DataFrame,
    test_probabilities: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
]:

    test_df = (
        df[df["_split"] == "test"]
        .copy()
        .reset_index(drop=True)
    )

    if len(test_df) != len(
        test_probabilities
    ):
        raise ValueError(
            "Test probability count does not "
            "match test manifest."
        )

    test_df["_probabilities"] = list(
        test_probabilities
    )

    records = []

    for recording_key, group in (
        test_df.groupby(
            "_recording_key",
            sort=True,
        )
    ):

        probability_matrix = np.stack(
            group["_probabilities"].to_list()
        )

        mean_probability = (
            probability_matrix.mean(axis=0)
        )

        prediction = int(
            np.argmax(mean_probability)
        )

        true_classes = (
            group["_class"].unique()
        )

        if len(true_classes) != 1:
            raise ValueError(
                "Recording contains multiple "
                "target classes:\n"
                f"{recording_key}\n"
                f"Classes: {true_classes}"
            )

        true_class = true_classes[0]

        record = {
            "recording_key": recording_key,
            "true_class": true_class,
            "predicted_class":
                EXPECTED_CLASSES[prediction],
            "confidence": float(
                mean_probability[prediction]
            ),
            "window_count": len(group),
        }

        for index, class_name in enumerate(
            EXPECTED_CLASSES
        ):
            record[
                f"prob_{class_name}"
            ] = float(
                mean_probability[index]
            )

        records.append(record)

    recording_predictions = pd.DataFrame(
        records
    )

    class_to_index = {
        class_name: index
        for index, class_name
        in enumerate(EXPECTED_CLASSES)
    }

    y_true = (
        recording_predictions["true_class"]
        .map(class_to_index)
        .to_numpy()
    )

    y_pred = (
        recording_predictions[
            "predicted_class"
        ]
        .map(class_to_index)
        .to_numpy()
    )

    return (
        y_true,
        y_pred,
        recording_predictions,
    )


# ============================================================
# TRAINING
# ============================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        logits = model(X_batch)

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

        predictions = logits.argmax(
            dim=1
        )

        correct += (
            (predictions == y_batch)
            .sum()
            .item()
        )

        total += X_batch.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# INFERENCE BENCHMARK
# ============================================================

def benchmark_inference(
    model: nn.Module,
    sample: torch.Tensor,
    device: torch.device,
    repeats: int,
) -> dict[str, float | int | str]:

    model.eval()

    sample = sample.to(device)

    with torch.no_grad():

        for _ in range(10):
            model(sample)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        for _ in range(repeats):
            model(sample)

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = (
            time.perf_counter() - start
        )

    average_seconds = (
        elapsed / repeats
    )

    average_ms = (
        average_seconds * 1000.0
    )

    window_duration_seconds = 2.0

    real_time_factor = (
        window_duration_seconds
        / average_seconds
    )

    return {
        "device": str(device),
        "benchmark_repeats": repeats,
        "average_inference_seconds":
            average_seconds,
        "average_inference_ms":
            average_ms,
        "window_duration_seconds":
            window_duration_seconds,
        "real_time_factor":
            real_time_factor,
    }


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def train(args: argparse.Namespace) -> None:

    set_seed(args.seed)
    create_directories()

    print("=" * 70)
    print("SILENTVOX M2-P4 — LOG-MEL SMALL CNN")
    print("=" * 70)

    # --------------------------------------------------------
    # P3 CONFIG
    # --------------------------------------------------------

    if not P3_CONFIG.exists():
        raise FileNotFoundError(
            f"P3 config not found:\n{P3_CONFIG}"
        )

    with open(
        P3_CONFIG,
        "r",
        encoding="utf-8",
    ) as file:
        p3_config = json.load(file)

    feature_version = p3_config.get(
        "feature_version"
    )

    p2_version = p3_config.get(
        "p2_version"
    )

    if feature_version != EXPECTED_P3_VERSION:
        raise ValueError(
            f"Unexpected P3 version: "
            f"{feature_version}"
        )

    if p2_version != EXPECTED_P2_VERSION:
        raise ValueError(
            f"Unexpected P2 version: "
            f"{p2_version}"
        )

    # --------------------------------------------------------
    # LOAD MANIFEST
    # --------------------------------------------------------

    print()
    print("Loading P3 feature manifest...")

    df = load_p3_manifest()

    verify_p3_input(df)

    # --------------------------------------------------------
    # LOAD FEATURES
    # --------------------------------------------------------

    print()
    print("Loading P3 Log-Mel features...")

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

    X_train, y_train = load_feature_array(
        train_df
    )

    X_validation, y_validation = (
        load_feature_array(validation_df)
    )

    X_test, y_test = load_feature_array(
        test_df
    )

    print(
        f"  Train features:      "
        f"{X_train.shape}"
    )

    print(
        f"  Validation features: "
        f"{X_validation.shape}"
    )

    print(
        f"  Test features:       "
        f"{X_test.shape}"
    )

    # --------------------------------------------------------
    # DATA LOADERS
    # --------------------------------------------------------

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

    test_loader = create_loader(
        X_test,
        y_test,
        args.batch_size,
        shuffle=False,
    )

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device = get_device()

    print()
    print(f"Device: {device}")

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = SmallCNN(
        num_classes=len(EXPECTED_CLASSES),
        base_channels=args.base_channels,
    )

    model = model.to(device)

    parameter_count = count_parameters(
        model
    )

    model_size_kb = estimate_model_size_kb(
        model
    )

    print()
    print("Model:")
    print("  Architecture: SmallCNN")
    print("  Input shape: (1, 64, 197)")
    print(
        f"  Classes: "
        f"{len(EXPECTED_CLASSES)}"
    )

    print(
        f"  Trainable parameters: "
        f"{parameter_count:,}"
    )

    print(
        f"  Estimated FP32 size: "
        f"{model_size_kb:.2f} KB"
    )

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    best_validation_f1 = -np.inf
    best_epoch = 0

    history = []

    print()
    print("Training...")
    print()

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        epoch_start = time.perf_counter()

        train_loss, train_accuracy = (
            train_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
            )
        )

        validation_true, (
            validation_probabilities
        ) = predict_model(
            model,
            validation_loader,
            device,
        )

        validation_predictions = (
            validation_probabilities.argmax(
                axis=1
            )
        )

        validation_metrics = calculate_metrics(
            validation_true,
            validation_predictions,
        )

        epoch_time = (
            time.perf_counter()
            - epoch_start
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_accuracy":
                    validation_metrics[
                        "accuracy"
                    ],
                "validation_macro_precision":
                    validation_metrics[
                        "macro_precision"
                    ],
                "validation_macro_recall":
                    validation_metrics[
                        "macro_recall"
                    ],
                "validation_macro_f1":
                    validation_metrics[
                        "macro_f1"
                    ],
                "epoch_time_seconds":
                    epoch_time,
            }
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Loss {train_loss:.4f} | "
            f"Train Acc {train_accuracy:.4f} | "
            f"Val F1 "
            f"{validation_metrics['macro_f1']:.4f}"
        )

        current_f1 = validation_metrics[
            "macro_f1"
        ]

        if current_f1 > best_validation_f1:

            best_validation_f1 = current_f1
            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),
                    "num_classes":
                        len(EXPECTED_CLASSES),
                    "classes":
                        EXPECTED_CLASSES,
                    "input_shape":
                        [1, 64, 197],
                    "base_channels":
                        args.base_channels,
                    "epoch":
                        epoch,
                    "validation_macro_f1":
                        float(current_f1),
                    "seed":
                        args.seed,
                    "p3_version":
                        EXPECTED_P3_VERSION,
                },
                BEST_MODEL_PATH,
            )

    # --------------------------------------------------------
    # TRAINING HISTORY
    # --------------------------------------------------------

    history_df = pd.DataFrame(
        history
    )

    history_df.to_csv(
        METRICS_DIR / "training_history.csv",
        index=False,
    )

    # --------------------------------------------------------
    # LOAD BEST MODEL
    # --------------------------------------------------------

    checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print()
    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Best validation macro-F1: "
        f"{best_validation_f1:.4f}"
    )

    # --------------------------------------------------------
    # WINDOW-LEVEL TEST
    # --------------------------------------------------------

    test_true, test_probabilities = (
        predict_model(
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

    window_metrics = calculate_metrics(
        test_true,
        test_predictions,
    )

    save_classification_report(
        test_true,
        test_predictions,
        METRICS_DIR
        / "window_level_classification_report.csv",
    )

    save_confusion_matrix(
        test_true,
        test_predictions,
        CONFUSION_DIR
        / "window_level_confusion_matrix.csv",
    )

    # --------------------------------------------------------
    # RECORDING-LEVEL TEST
    # --------------------------------------------------------

    (
        recording_true,
        recording_pred,
        recording_prediction_df,
    ) = recording_level_evaluation(
        df,
        test_probabilities,
    )

    recording_metrics = calculate_metrics(
        recording_true,
        recording_pred,
    )

    save_classification_report(
        recording_true,
        recording_pred,
        METRICS_DIR
        / "recording_level_classification_report.csv",
    )

    save_confusion_matrix(
        recording_true,
        recording_pred,
        CONFUSION_DIR
        / "recording_level_confusion_matrix.csv",
    )

    recording_prediction_df.to_csv(
        METRICS_DIR
        / "recording_level_predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # INFERENCE BENCHMARK
    # --------------------------------------------------------

    benchmark_sample = torch.from_numpy(
        X_test[0:1, np.newaxis, :, :]
    )

    benchmark_results = benchmark_inference(
        model,
        benchmark_sample,
        device,
        args.benchmark_repeats,
    )

    benchmark_results.update(
        {
            "model": "SmallCNN",
            "p4_version": "P4-v0.1",
            "p3_version":
                EXPECTED_P3_VERSION,
            "input_shape":
                [1, 64, 197],
            "parameter_count":
                parameter_count,
            "estimated_model_size_kb":
                model_size_kb,
        }
    )

    with open(
        BENCHMARK_DIR
        / "cnn_inference_benchmark.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            benchmark_results,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # MODEL METADATA
    # --------------------------------------------------------

    label_metadata = {
        "p4_version": "P4-v0.1",
        "p3_version":
            EXPECTED_P3_VERSION,
        "classes":
            EXPECTED_CLASSES,
        "class_to_index": {
            class_name: index
            for index, class_name
            in enumerate(EXPECTED_CLASSES)
        },
        "input_shape": [
            1,
            64,
            197,
        ],
        "base_channels":
            args.base_channels,
        "epochs":
            args.epochs,
        "batch_size":
            args.batch_size,
        "learning_rate":
            args.learning_rate,
        "weight_decay":
            args.weight_decay,
        "seed":
            args.seed,
        "best_epoch":
            best_epoch,
        "best_validation_macro_f1":
            float(best_validation_f1),
    }

    with open(
        METADATA_DIR
        / "cnn_logmel_labels.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            label_metadata,
            file,
            indent=2,
        )

    model_metadata = {
        "p4_version": "P4-v0.1",
        "model_name": "SmallCNN",
        "feature_type":
            "Log-Mel Spectrogram",
        "p3_version":
            EXPECTED_P3_VERSION,
        "p2_version":
            EXPECTED_P2_VERSION,
        "input_shape": [
            1,
            64,
            197,
        ],
        "classes":
            EXPECTED_CLASSES,
        "parameter_count":
            parameter_count,
        "estimated_fp32_size_kb":
            model_size_kb,
        "device":
            str(device),
        "training": {
            "epochs":
                args.epochs,
            "batch_size":
                args.batch_size,
            "learning_rate":
                args.learning_rate,
            "weight_decay":
                args.weight_decay,
            "seed":
                args.seed,
        },
        "best_epoch":
            best_epoch,
        "best_validation_macro_f1":
            float(best_validation_f1),
    }

    with open(
        METADATA_DIR
        / "P4_CNN_MODEL_METADATA.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            model_metadata,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # FINAL RESULTS
    # --------------------------------------------------------

    results = {
        "p4_version":
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
            len(recording_prediction_df),
        "best_epoch":
            best_epoch,
        "best_validation_macro_f1":
            float(best_validation_f1),
        "window_accuracy":
            window_metrics["accuracy"],
        "window_macro_precision":
            window_metrics[
                "macro_precision"
            ],
        "window_macro_recall":
            window_metrics[
                "macro_recall"
            ],
        "window_macro_f1":
            window_metrics["macro_f1"],
        "recording_accuracy":
            recording_metrics["accuracy"],
        "recording_macro_precision":
            recording_metrics[
                "macro_precision"
            ],
        "recording_macro_recall":
            recording_metrics[
                "macro_recall"
            ],
        "recording_macro_f1":
            recording_metrics["macro_f1"],
        "parameter_count":
            parameter_count,
        "model_size_kb":
            model_size_kb,
        "inference_ms":
            benchmark_results[
                "average_inference_ms"
            ],
        "real_time_factor":
            benchmark_results[
                "real_time_factor"
            ],
        "device":
            str(device),
        "seed":
            args.seed,
    }

    pd.DataFrame(
        [results]
    ).to_csv(
        METRICS_DIR / "CNN_RESULTS.csv",
        index=False,
    )

    # --------------------------------------------------------
    # FINAL PRINT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("P4 MODEL TRAINING COMPLETED")
    print("=" * 70)

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
    print("Model:")

    print(
        f"  Parameters:      "
        f"{parameter_count:,}"
    )

    print(
        f"  Size:             "
        f"{model_size_kb:.2f} KB"
    )

    print()
    print("Inference:")

    print(
        f"  Average:          "
        f"{benchmark_results['average_inference_ms']:.3f} ms"
    )

    print(
        f"  Real-time factor: "
        f"{benchmark_results['real_time_factor']:.2f}"
    )

    print()
    print("Outputs:")

    print(f"  {BEST_MODEL_PATH}")

    print(
        f"  {METRICS_DIR / 'CNN_RESULTS.csv'}"
    )

    print(
        f"  {METRICS_DIR / 'training_history.csv'}"
    )

    print(
        f"  {METRICS_DIR / 'window_level_classification_report.csv'}"
    )

    print(
        f"  {METRICS_DIR / 'recording_level_classification_report.csv'}"
    )

    print(
        f"  {METRICS_DIR / 'recording_level_predictions.csv'}"
    )

    print(
        f"  {CONFUSION_DIR / 'window_level_confusion_matrix.csv'}"
    )

    print(
        f"  {CONFUSION_DIR / 'recording_level_confusion_matrix.csv'}"
    )

    print(
        f"  {BENCHMARK_DIR / 'cnn_inference_benchmark.json'}"
    )

    print(
        f"  {METADATA_DIR / 'cnn_logmel_labels.json'}"
    )

    print(
        f"  {METADATA_DIR / 'P4_CNN_MODEL_METADATA.json'}"
    )

    print()
    print("=" * 70)
    print("READY FOR M2-P5")
    print("=" * 70)


# ============================================================
# ARGUMENTS
# ============================================================

def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "SilentVox M2-P4 "
            "Log-Mel SmallCNN training"
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
        "--benchmark-repeats",
        type=int,
        default=100,
    )

    return parser.parse_args()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    arguments = parse_arguments()
    train(arguments)