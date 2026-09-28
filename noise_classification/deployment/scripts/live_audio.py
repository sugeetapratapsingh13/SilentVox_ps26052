from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path

import librosa
import numpy as np
import sounddevice as sd
import torch


# ============================================================
# SILENTVOX M2-P6 — LIVE AUDIO DEPLOYMENT
# ============================================================


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

P3_STATS_PATH = (
    PROJECT_ROOT
    / "noise_classification"
    / "features"
    / "metadata"
    / "feature_normalization_stats.npz"
)

P3_CONFIG_PATH = (
    PROJECT_ROOT
    / "noise_classification"
    / "features"
    / "metadata"
    / "feature_config.json"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "noise_classification"
    / "models"
    / "trained_models"
    / "cnn_logmel_augmented_best.pt"
)

CNN_MODEL_DIR = (
    PROJECT_ROOT
    / "noise_classification"
    / "models"
    / "architecture"
)

if str(CNN_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(CNN_MODEL_DIR))

from cnn_model import SmallCNN


# ============================================================
# BENCHMARK-COMPATIBLE PATH ALIASES
# ============================================================

P5_MODEL = MODEL_PATH
P3_NORMALIZATION_STATS = P3_STATS_PATH


# ============================================================
# CONFIGURATION
# ============================================================

# Keep the currently established deployment taxonomy unchanged
# until the actual P5 checkpoint taxonomy is verified.

CLASSES = [
    "ENGINE",
    "RAIN",
    "SIREN",
    "WIND",
]

SAMPLE_RATE = 16000
CHANNELS = 1

WINDOW_SECONDS = 2.0
WINDOW_SAMPLES = int(
    SAMPLE_RATE * WINDOW_SECONDS
)


# ============================================================
# LOG-MEL CONFIGURATION
# ============================================================

N_FFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 160
N_MELS = 64
POWER = 2.0
CENTER = False

EXPECTED_LOG_MEL_SHAPE = (64, 197)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cpu")


# ============================================================
# TEMPORAL SMOOTHING
# ============================================================

SMOOTHING_HISTORY = 5


# ============================================================
# UNKNOWN THRESHOLD
# ============================================================

# None = disabled.
UNKNOWN_THRESHOLD = None


# ============================================================
# VALIDATION
# ============================================================

def validate_paths() -> None:
    """
    Verify that all required SilentVox files exist.
    """

    required_paths = {
        "P3 normalization statistics": P3_STATS_PATH,
        "P3 feature configuration": P3_CONFIG_PATH,
        "P5 augmented CNN model": P5_MODEL,
    }

    missing = []

    for name, path in required_paths.items():
        if not path.exists():
            missing.append(
                f"{name}: {path}"
            )

    if missing:
        raise FileNotFoundError(
            "Required SilentVox files are missing:\n"
            + "\n".join(missing)
        )


# ============================================================
# MICROPHONE CAPTURE
# ============================================================

def capture_window(
    device=None,
    window_samples=WINDOW_SAMPLES,
) -> tuple[np.ndarray, bool]:
    """
    Capture exactly one fixed-length microphone window.

    This function is required by:
        temporal_smoothing_benchmark.py

    Parameters
    ----------
    device:
        Sounddevice input device.
        None = system default microphone.

    window_samples:
        Number of samples to capture.

    Returns
    -------
    audio:
        1-D float32 mono audio array.

    overflowed:
        True if sounddevice reports an input overflow.
    """

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        blocksize=window_samples,
        latency="low",
        device=device,
    )

    with stream:
        audio, overflowed = stream.read(
            window_samples
        )

    audio = np.asarray(
        audio,
        dtype=np.float32,
    ).reshape(-1)

    if len(audio) != window_samples:
        raise RuntimeError(
            "Microphone returned an unexpected "
            f"number of samples. "
            f"Expected {window_samples}, "
            f"received {len(audio)}."
        )

    return audio, bool(overflowed)


# ============================================================
# P3 NORMALIZATION
# ============================================================

def load_normalization_stats() -> tuple[np.ndarray, np.ndarray]:
    """
    Load authoritative P3 Log-Mel normalization statistics.

    P3 stores:

        log_mel_mean
        log_mel_std

    Both have shape:

        (64,)

    They are broadcast across the time dimension.
    """

    data = np.load(
        P3_NORMALIZATION_STATS
    )

    required_keys = {
        "log_mel_mean",
        "log_mel_std",
    }

    missing_keys = (
        required_keys - set(data.files)
    )

    if missing_keys:
        raise KeyError(
            "Missing required P3 normalization keys: "
            f"{sorted(missing_keys)}\n"
            f"Available keys: {data.files}"
        )

    mean = np.asarray(
        data["log_mel_mean"],
        dtype=np.float32,
    )

    std = np.asarray(
        data["log_mel_std"],
        dtype=np.float32,
    )

    if mean.shape != (N_MELS,):
        raise ValueError(
            "Unexpected Log-Mel mean shape. "
            f"Expected ({N_MELS},), "
            f"received {mean.shape}."
        )

    if std.shape != (N_MELS,):
        raise ValueError(
            "Unexpected Log-Mel std shape. "
            f"Expected ({N_MELS},), "
            f"received {std.shape}."
        )

    std = np.maximum(
        std,
        1e-8,
    )

    return mean, std


# ============================================================
# P3 FEATURE CONFIGURATION
# ============================================================

def load_feature_config() -> dict:
    """
    Load the authoritative P3 feature configuration.
    """

    with open(
        P3_CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


# ============================================================
# MODEL LOADING
# ============================================================

def load_model(
    model_path: str | Path | None = None,
) -> torch.nn.Module:
    """
    Load the SilentVox CNN checkpoint.

    Supports:

        load_model()

    and:

        load_model(P5_MODEL)

    The second form is required by the
    M2-P6 temporal smoothing benchmark.
    """

    if model_path is None:
        model_path = P5_MODEL

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"CNN model checkpoint not found: "
            f"{model_path}"
        )

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=False,
    )

    # --------------------------------------------------------
    # Extract state dictionary
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint[
                "state_dict"
            ]

        else:
            state_dict = checkpoint

    else:
        raise TypeError(
            "Unsupported checkpoint format: "
            f"{type(checkpoint)}"
        )

    # --------------------------------------------------------
    # Construct model
    # --------------------------------------------------------

    model = SmallCNN(
        num_classes=len(CLASSES),
        in_channels=1,
        base_channels=8,
    )

    # --------------------------------------------------------
    # Load trained weights
    # --------------------------------------------------------

    try:
        model.load_state_dict(
            state_dict
        )

    except RuntimeError as exc:
        raise RuntimeError(
            "\n"
            "============================================================\n"
            "CNN CHECKPOINT / DEPLOYMENT MISMATCH\n"
            "============================================================\n"
            f"Checkpoint:\n{model_path}\n\n"
            f"Deployment class count: {len(CLASSES)}\n"
            f"Deployment classes: {CLASSES}\n\n"
            "The checkpoint architecture does not match "
            "the current deployment configuration.\n\n"
            f"Original PyTorch error:\n{exc}\n"
            "============================================================"
        ) from exc

    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# AUDIO PREPROCESSING
# ============================================================

def preprocess_audio(
    audio: np.ndarray,
) -> np.ndarray:
    """
    Convert captured audio into the expected
    16 kHz mono float32 representation.
    """

    audio = np.asarray(
        audio,
        dtype=np.float32,
    ).reshape(-1)

    if len(audio) != WINDOW_SAMPLES:
        raise ValueError(
            f"Expected {WINDOW_SAMPLES} samples, "
            f"received {len(audio)}."
        )

    # Remove DC offset.
    audio = (
        audio - np.mean(audio)
    )

    # Protect against invalid numerical values.
    audio = np.nan_to_num(
        audio,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return audio.astype(
        np.float32
    )


# ============================================================
# LOG-MEL EXTRACTION
# ============================================================

def extract_log_mel(
    audio: np.ndarray,
) -> np.ndarray:
    """
    Extract the P3-compatible Log-Mel representation.
    """

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=POWER,
        center=CENTER,
    )

    log_mel = librosa.power_to_db(
        mel,
        ref=np.max,
    )

    log_mel = np.asarray(
        log_mel,
        dtype=np.float32,
    )

    if log_mel.shape != EXPECTED_LOG_MEL_SHAPE:
        raise ValueError(
            "Unexpected Log-Mel shape. "
            f"Expected {EXPECTED_LOG_MEL_SHAPE}, "
            f"received {log_mel.shape}."
        )

    return log_mel


# ============================================================
# NORMALIZE LOG-MEL
# ============================================================

def normalize_log_mel(
    log_mel: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """
    Apply authoritative P3 normalization.

    log_mel:
        (64, 197)

    mean:
        (64,)

    std:
        (64,)
    """

    log_mel = np.asarray(
        log_mel,
        dtype=np.float32,
    )

    if log_mel.ndim != 2:
        raise ValueError(
            "Expected Log-Mel feature shape "
            "(Mel, Time), received "
            f"{log_mel.shape}."
        )

    if log_mel.shape[0] != N_MELS:
        raise ValueError(
            "Unexpected number of Mel bins. "
            f"Expected {N_MELS}, "
            f"received {log_mel.shape[0]}."
        )

    mean = np.asarray(
        mean,
        dtype=np.float32,
    ).reshape(-1)

    std = np.asarray(
        std,
        dtype=np.float32,
    ).reshape(-1)

    if mean.shape != (N_MELS,):
        raise ValueError(
            f"Expected mean shape "
            f"({N_MELS},), "
            f"received {mean.shape}."
        )

    if std.shape != (N_MELS,):
        raise ValueError(
            f"Expected std shape "
            f"({N_MELS},), "
            f"received {std.shape}."
        )

    std = np.maximum(
        std,
        1e-8,
    )

    normalized = (
        log_mel - mean[:, None]
    ) / std[:, None]

    normalized = np.nan_to_num(
        normalized,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return normalized.astype(
        np.float32
    )


# ============================================================
# MAJORITY VOTING
# ============================================================

class MajorityVoteSmoother:
    """
    Temporal smoothing using majority voting.
    """

    def __init__(
        self,
        history_size: int = SMOOTHING_HISTORY,
    ) -> None:

        if history_size <= 0:
            raise ValueError(
                "history_size must be greater than zero."
            )

        self.history = deque(
            maxlen=history_size
        )

    def update(
        self,
        prediction: int,
    ) -> int:

        prediction = int(prediction)

        if prediction < 0 or prediction >= len(CLASSES):
            raise ValueError(
                f"Invalid class index: {prediction}"
            )

        self.history.append(
            prediction
        )

        counts = np.bincount(
            list(self.history),
            minlength=len(CLASSES),
        )

        return int(
            np.argmax(counts)
        )


# ============================================================
# PROBABILITY MOVING AVERAGE
# ============================================================

class ProbabilityMovingAverage:
    """
    Temporal smoothing using probability averaging.
    """

    def __init__(
        self,
        num_classes: int,
        history_size: int = SMOOTHING_HISTORY,
    ) -> None:

        if num_classes <= 0:
            raise ValueError(
                "num_classes must be greater than zero."
            )

        if history_size <= 0:
            raise ValueError(
                "history_size must be greater than zero."
            )

        self.num_classes = num_classes

        self.history = deque(
            maxlen=history_size
        )

    def update(
        self,
        probabilities: np.ndarray,
    ) -> tuple[int, np.ndarray]:

        probabilities = np.asarray(
            probabilities,
            dtype=np.float32,
        ).reshape(-1)

        if probabilities.shape != (
            self.num_classes,
        ):
            raise ValueError(
                "Probability vector shape mismatch. "
                f"Expected ({self.num_classes},), "
                f"received {probabilities.shape}."
            )

        self.history.append(
            probabilities
        )

        averaged = np.mean(
            np.stack(
                self.history
            ),
            axis=0,
        ).astype(
            np.float32
        )

        prediction = int(
            np.argmax(
                averaged
            )
        )

        return (
            prediction,
            averaged,
        )


# ============================================================
# CNN PREDICTION
# ============================================================

def predict(
    model: SmallCNN,
    log_mel: np.ndarray,
) -> tuple[int, np.ndarray]:
    """
    Run CNN inference.

    Returns exactly:

        class_index
        probabilities

    This two-value interface is required by
    temporal_smoothing_benchmark.py.
    """

    log_mel = np.asarray(
        log_mel,
        dtype=np.float32,
    )

    if log_mel.shape != EXPECTED_LOG_MEL_SHAPE:
        raise ValueError(
            "CNN input Log-Mel shape mismatch. "
            f"Expected {EXPECTED_LOG_MEL_SHAPE}, "
            f"received {log_mel.shape}."
        )

    tensor = torch.from_numpy(
        log_mel
    )

    # (64, 197)
    #      ↓
    # (1, 64, 197)
    #      ↓
    # (1, 1, 64, 197)

    tensor = tensor.unsqueeze(0)
    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(
        DEVICE,
        dtype=torch.float32,
    )

    with torch.inference_mode():

        logits = model(
            tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

    probabilities = (
        probabilities
        .cpu()
        .numpy()[0]
        .astype(np.float32)
    )

    if probabilities.shape != (
        len(CLASSES),
    ):
        raise ValueError(
            "CNN output class count mismatch. "
            f"Expected {len(CLASSES)} probabilities, "
            f"received {probabilities.shape}."
        )

    class_index = int(
        np.argmax(
            probabilities
        )
    )

    return (
        class_index,
        probabilities,
    )


# ============================================================
# COMPLETE PROCESSING PIPELINE
# ============================================================

class LiveProcessor:
    """
    Complete M2-P6 inference and temporal
    smoothing pipeline.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> None:

        self.model = model
        self.mean = mean
        self.std = std

        self.majority_smoother = (
            MajorityVoteSmoother(
                SMOOTHING_HISTORY
            )
        )

        self.probability_smoother = (
            ProbabilityMovingAverage(
                num_classes=len(CLASSES),
                history_size=SMOOTHING_HISTORY,
            )
        )

    def process(
        self,
        audio: np.ndarray,
    ) -> dict:

        # ----------------------------------------------------
        # 1. Preprocess audio
        # ----------------------------------------------------

        audio = preprocess_audio(
            audio
        )

        # ----------------------------------------------------
        # 2. Extract Log-Mel
        # ----------------------------------------------------

        log_mel = extract_log_mel(
            audio
        )

        # ----------------------------------------------------
        # 3. P3 normalization
        # ----------------------------------------------------

        normalized = normalize_log_mel(
            log_mel,
            self.mean,
            self.std,
        )

        # ----------------------------------------------------
        # 4. CNN inference
        # ----------------------------------------------------

        prediction, probabilities = predict(
            self.model,
            normalized,
        )

        # Confidence = highest raw CNN probability.
        confidence = float(
            np.max(
                probabilities
            )
        )

        # ----------------------------------------------------
        # 5. UNKNOWN threshold
        # ----------------------------------------------------

        if (
            UNKNOWN_THRESHOLD is not None
            and confidence < UNKNOWN_THRESHOLD
        ):
            raw_label = "UNKNOWN"
        else:
            raw_label = CLASSES[
                prediction
            ]

        # ----------------------------------------------------
        # 6. Majority voting
        # ----------------------------------------------------

        majority_prediction = (
            self.majority_smoother.update(
                prediction
            )
        )

        majority_label = CLASSES[
            majority_prediction
        ]

        # ----------------------------------------------------
        # 7. Probability moving average
        # ----------------------------------------------------

        (
            probability_prediction,
            averaged_probabilities,
        ) = self.probability_smoother.update(
            probabilities
        )

        probability_label = CLASSES[
            probability_prediction
        ]

        probability_confidence = float(
            averaged_probabilities[
                probability_prediction
            ]
        )

        # ----------------------------------------------------
        # 8. Return complete result
        # ----------------------------------------------------

        return {
            "raw_prediction": prediction,

            "raw_label": raw_label,

            "confidence": confidence,

            "probabilities": probabilities,

            "majority_prediction": (
                majority_prediction
            ),

            "majority_label": majority_label,

            "probability_prediction": (
                probability_prediction
            ),

            "probability_label": probability_label,

            "probability_confidence": (
                probability_confidence
            ),
        }


# ============================================================
# CONTINUOUS LIVE MODE
# ============================================================

def run_live() -> None:
    """
    Run continuous microphone inference.
    """

    validate_paths()

    print()
    print("=" * 70)
    print(
        "SILENTVOX M2-P6 — LIVE AUDIO INFERENCE"
    )
    print("=" * 70)
    print()

    print(
        "Loading P3 normalization statistics..."
    )

    mean, std = (
        load_normalization_stats()
    )

    print(
        "Loading P5 augmented model..."
    )

    model = load_model(
        P5_MODEL
    )

    print()

    print(
        f"Sample rate:       {SAMPLE_RATE} Hz"
    )

    print(
        f"Channels:          {CHANNELS}"
    )

    print(
        f"Window:            {WINDOW_SECONDS} s"
    )

    print(
        f"Window samples:    {WINDOW_SAMPLES}"
    )

    print(
        f"Log-Mel shape:     {EXPECTED_LOG_MEL_SHAPE}"
    )

    print(
        f"Smoothing history: {SMOOTHING_HISTORY}"
    )

    print()

    print(
        "Starting microphone..."
    )

    print(
        "Press Ctrl+C to stop."
    )

    print()

    processor = LiveProcessor(
        model=model,
        mean=mean,
        std=std,
    )

    try:

        while True:

            audio, overflowed = capture_window(
                device=None,
                window_samples=WINDOW_SAMPLES,
            )

            if overflowed:
                print(
                    "WARNING: Audio input overflow detected."
                )

            result = processor.process(
                audio
            )

            print(
                f"Raw: "
                f"{result['raw_label']:<10} "
                f"Conf: "
                f"{result['confidence']:.3f} | "
                f"Majority: "
                f"{result['majority_label']:<10} | "
                f"Probability MA: "
                f"{result['probability_label']:<10} "
                f"("
                f"{result['probability_confidence']:.3f}"
                f")"
            )

    except KeyboardInterrupt:

        print()
        print(
            "Live inference stopped."
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """
    Program entry point.
    """

    run_live()


if __name__ == "__main__":
    main()
