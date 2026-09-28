from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


EXPECTED_SAMPLE_RATE = 16000
EXPECTED_FILTER_LENGTH = 128
EXPECTED_DELAY_SAMPLES = 32
EXPECTED_ATTENUATION = 0.65


@dataclass(frozen=True)
class M5SecondaryPath:
    impulse_response: np.ndarray
    sample_rate: int
    delay_samples: int
    attenuation: float
    status: str = "SIMULATED_SECONDARY_PATH_PHYSICAL_MEASUREMENT_PENDING"

    @property
    def filter_length(self) -> int:
        return int(self.impulse_response.shape[0])


class M5Adapter:
    """Adapter for the M5 simulated secondary-path model."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        expected_sample_rate: int = EXPECTED_SAMPLE_RATE,
    ) -> None:
        self.model_path = Path(model_path)
        self.expected_sample_rate = int(expected_sample_rate)

    def load(self) -> M5SecondaryPath:
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"M5 secondary-path model not found: {self.model_path}"
            )

        if self.model_path.suffix.lower() != ".npy":
            raise ValueError(
                f"M5 secondary-path model must be a .npy file, "
                f"got: {self.model_path.name}"
            )

        impulse_response = np.load(self.model_path, allow_pickle=False)

        if not isinstance(impulse_response, np.ndarray):
            raise ValueError("M5 secondary-path model is not a NumPy array")

        if impulse_response.ndim != 1:
            raise ValueError(
                f"M5 secondary-path model must be 1-D, got shape "
                f"{impulse_response.shape}"
            )

        if impulse_response.shape[0] != EXPECTED_FILTER_LENGTH:
            raise ValueError(
                f"M5 secondary-path model must contain "
                f"{EXPECTED_FILTER_LENGTH} samples, got "
                f"{impulse_response.shape[0]}"
            )

        impulse_response = np.asarray(impulse_response, dtype=np.float64)

        if not np.isfinite(impulse_response).all():
            raise ValueError(
                "M5 secondary-path model contains non-finite values"
            )

        if np.max(np.abs(impulse_response)) == 0.0:
            raise ValueError(
                "M5 secondary-path model is entirely zero"
            )

        return M5SecondaryPath(
            impulse_response=impulse_response,
            sample_rate=self.expected_sample_rate,
            delay_samples=EXPECTED_DELAY_SAMPLES,
            attenuation=EXPECTED_ATTENUATION,
        )

    def summary(self) -> dict[str, object]:
        model = self.load()

        return {
            "sample_rate": model.sample_rate,
            "filter_length": model.filter_length,
            "delay_samples": model.delay_samples,
            "delay_ms": 1000.0 * model.delay_samples / model.sample_rate,
            "attenuation": model.attenuation,
            "status": model.status,
            "model_path": str(self.model_path),
        }


def create_adapter(model_path: str | Path) -> M5Adapter:
    return M5Adapter(model_path)


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]

    # The M5 artifact is supplied in the repository ZIP and is extracted
    # temporarily for validation. Physical measurement remains pending.
    model_path = (
        repo_root
        / "M6"
        / "_m5_inspect"
        / "SilentVox_M5"
        / "models"
        / "secondary_path_estimate.npy"
    )

    adapter = create_adapter(model_path)
    summary = adapter.summary()

    print("M6 M5 adapter: PASS")
    print(f"Sample rate:      {summary['sample_rate']} Hz")
    print(f"Filter length:    {summary['filter_length']}")
    print(f"Delay:            {summary['delay_samples']} samples")
    print(f"Delay:            {summary['delay_ms']:.3f} ms")
    print(f"Attenuation:      {summary['attenuation']}")
    print(f"Status:           {summary['status']}")
