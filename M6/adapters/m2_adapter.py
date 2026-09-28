"""
SilentVox M6 -> M2 adapter.

Purpose:
    Provide a stable M6 integration boundary around the existing
    M2 deployment implementation.

Evidence status:
    VERIFIED when the adapter successfully loads the existing M2
    checkpoint and produces a valid four-class probability vector.

This adapter intentionally does not modify M2 source code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np


# ---------------------------------------------------------------------
# Repository paths
# ---------------------------------------------------------------------

M6_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = M6_DIR.parent

M2_DEPLOYMENT_DIR = (
    REPO_ROOT
    / "noise_classification"
    / "deployment"
    / "scripts"
)

M2_MODEL_PATH = (
    REPO_ROOT
    / "noise_classification"
    / "models"
    / "trained_models"
    / "cnn_logmel_best.pt"
)

M2_LABEL_PATH = (
    REPO_ROOT
    / "noise_classification"
    / "models"
    / "architecture"
    / "cnn_logmel_labels.json"
)


# ---------------------------------------------------------------------
# M2 deployment import
# ---------------------------------------------------------------------

if str(M2_DEPLOYMENT_DIR) not in sys.path:
    sys.path.insert(0, str(M2_DEPLOYMENT_DIR))

from live_audio import load_model, predict  # noqa: E402


# ---------------------------------------------------------------------
# M2 contract
# ---------------------------------------------------------------------

EXPECTED_SAMPLE_RATE_HZ = 16000
EXPECTED_LOG_MEL_SHAPE = (64, 197)
EXPECTED_CLASS_COUNT = 4

EXPECTED_CLASSES = (
    "ENGINE",
    "RAIN",
    "SIREN",
    "WIND",
)


@dataclass(frozen=True)
class M2Prediction:
    """Stable prediction object exposed to M6."""

    class_index: int
    class_name: str
    probabilities: np.ndarray
    confidence: float


class M2Adapter:
    """
    M6 wrapper around the existing M2 CNN deployment.

    The adapter owns the integration boundary, while inference remains
    implemented by M2's existing live_audio.py.
    """

    def __init__(
        self,
        model_path: str | Path = M2_MODEL_PATH,
    ) -> None:
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"M2 checkpoint not found: {self.model_path}"
            )

        self.model = load_model(self.model_path)

    @staticmethod
    def validate_log_mel(
        log_mel: np.ndarray,
    ) -> np.ndarray:
        """
        Validate and normalize the M6 -> M2 feature boundary.

        Required M2 input:
            float32 Log-Mel array with shape (64, 197)
        """

        array = np.asarray(
            log_mel,
            dtype=np.float32,
        )

        if array.shape != EXPECTED_LOG_MEL_SHAPE:
            raise ValueError(
                "M6 -> M2 Log-Mel shape mismatch: "
                f"expected {EXPECTED_LOG_MEL_SHAPE}, "
                f"received {array.shape}"
            )

        if not np.all(np.isfinite(array)):
            raise ValueError(
                "M6 -> M2 Log-Mel input contains "
                "NaN or infinite values."
            )

        return array

    def predict(
        self,
        log_mel: np.ndarray,
    ) -> M2Prediction:
        """
        Run one M2 inference through the existing M2 deployment code.
        """

        features = self.validate_log_mel(log_mel)

        class_index, probabilities = predict(
            self.model,
            features,
        )

        probabilities = np.asarray(
            probabilities,
            dtype=np.float32,
        )

        if probabilities.shape != (
            EXPECTED_CLASS_COUNT,
        ):
            raise RuntimeError(
                "M2 probability-vector contract violation: "
                f"expected {(EXPECTED_CLASS_COUNT,)}, "
                f"received {probabilities.shape}"
            )

        if not np.all(np.isfinite(probabilities)):
            raise RuntimeError(
                "M2 returned non-finite probabilities."
            )

        if class_index < 0 or class_index >= EXPECTED_CLASS_COUNT:
            raise RuntimeError(
                "M2 returned invalid class index: "
                f"{class_index}"
            )

        probability_sum = float(
            np.sum(probabilities)
        )

        if not np.isclose(
            probability_sum,
            1.0,
            atol=1e-5,
        ):
            raise RuntimeError(
                "M2 probabilities do not sum to 1: "
                f"{probability_sum}"
            )

        class_name = EXPECTED_CLASSES[class_index]
        confidence = float(probabilities[class_index])

        return M2Prediction(
            class_index=int(class_index),
            class_name=class_name,
            probabilities=probabilities,
            confidence=confidence,
        )


def create_adapter() -> M2Adapter:
    """Create the default M6 M2 adapter."""

    return M2Adapter()


if __name__ == "__main__":
    adapter = create_adapter()

    dummy_features = np.zeros(
        EXPECTED_LOG_MEL_SHAPE,
        dtype=np.float32,
    )

    result = adapter.predict(dummy_features)

    print("M6 M2 adapter: PASS")
    print(f"Checkpoint: {adapter.model_path}")
    print(f"Class index: {result.class_index}")
    print(f"Class name:  {result.class_name}")
    print(f"Confidence:  {result.confidence:.6f}")
    print(
        "Probabilities:",
        np.array2string(
            result.probabilities,
            precision=6,
        ),
    )
