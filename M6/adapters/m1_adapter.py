from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import lfilter


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from M6.adapters.m5_adapter import M5SecondaryPath


M1_SAMPLE_RATE = 16000
M1_FILTER_LENGTH = 64
M1_MU = 1e-5

M1_NATIVE_DELAY_SAMPLES = 8
M1_NATIVE_ATTENUATION = 0.7
M1_NATIVE_SECONDARY_FILTER = np.array(
    [0.20, 0.30, 0.30, 0.15, 0.05],
    dtype=np.float64,
)


@dataclass(frozen=True)
class M1FxLMSResult:
    reference: np.ndarray
    control: np.ndarray
    residual: np.ndarray
    weights: np.ndarray
    sample_rate: int
    filter_length: int
    learning_rate: float
    secondary_path_mode: str


class M1Adapter:
    """M6 runtime adapter implementing the existing M1 FxLMS structure."""

    def __init__(
        self,
        *,
        sample_rate: int = M1_SAMPLE_RATE,
        filter_length: int = M1_FILTER_LENGTH,
        learning_rate: float = M1_MU,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.filter_length = int(filter_length)
        self.learning_rate = float(learning_rate)

        if self.sample_rate != M1_SAMPLE_RATE:
            raise ValueError(
                f"M1 FxLMS expects {M1_SAMPLE_RATE} Hz, "
                f"got {self.sample_rate}"
            )

        if self.filter_length <= 0:
            raise ValueError("filter_length must be positive")

        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")

    def process(
        self,
        reference: np.ndarray,
        disturbance: np.ndarray | None = None,
        *,
        secondary_path: M5SecondaryPath | None = None,
    ) -> M1FxLMSResult:
        reference = np.asarray(reference, dtype=np.float64)

        if reference.ndim != 1:
            raise ValueError("reference must be a 1-D array")

        if reference.size == 0:
            raise ValueError("reference must not be empty")

        if not np.isfinite(reference).all():
            raise ValueError("reference contains non-finite values")

        if disturbance is None:
            disturbance = reference.copy()
        else:
            disturbance = np.asarray(disturbance, dtype=np.float64)

            if disturbance.shape != reference.shape:
                raise ValueError(
                    "disturbance must have the same shape as reference"
                )

            if not np.isfinite(disturbance).all():
                raise ValueError(
                    "disturbance contains non-finite values"
                )

        if secondary_path is None:
            secondary_filter = M1_NATIVE_SECONDARY_FILTER
            delay_samples = M1_NATIVE_DELAY_SAMPLES
            attenuation = M1_NATIVE_ATTENUATION
            secondary_path_mode = "M1_NATIVE_SIMULATION"
        else:
            if secondary_path.sample_rate != self.sample_rate:
                raise ValueError(
                    "M5 secondary-path sample rate does not match M1"
                )

            secondary_filter = np.asarray(
                secondary_path.impulse_response,
                dtype=np.float64,
            )
            delay_samples = int(secondary_path.delay_samples)
            attenuation = float(secondary_path.attenuation)
            secondary_path_mode = "M5_SIMULATED_SECONDARY_PATH"

            if secondary_filter.ndim != 1:
                raise ValueError("secondary path must be 1-D")

            if not np.isfinite(secondary_filter).all():
                raise ValueError(
                    "secondary path contains non-finite values"
                )

        x_filtered = lfilter(
            secondary_filter,
            [1.0],
            reference,
        )

        weights = np.zeros(self.filter_length, dtype=np.float64)
        reference_buffer = np.zeros(
            self.filter_length,
            dtype=np.float64,
        )
        filtered_buffer = np.zeros(
            self.filter_length,
            dtype=np.float64,
        )
        secondary_buffer = np.zeros(
            delay_samples + 1,
            dtype=np.float64,
        )

        control = np.zeros_like(reference)
        residual = np.zeros_like(reference)

        for n in range(reference.size):
            reference_buffer[1:] = reference_buffer[:-1]
            reference_buffer[0] = reference[n]

            control[n] = np.dot(weights, reference_buffer)

            secondary_buffer[1:] = secondary_buffer[:-1]
            secondary_buffer[0] = control[n]

            secondary_output = 0.0

            for k in range(secondary_filter.size):
                buffer_index = delay_samples - k

                if buffer_index >= 0:
                    secondary_output += (
                        attenuation
                        * secondary_filter[k]
                        * secondary_buffer[buffer_index]
                    )

            residual[n] = disturbance[n] - secondary_output

            filtered_buffer[1:] = filtered_buffer[:-1]
            filtered_buffer[0] = x_filtered[n]

            weights += (
                self.learning_rate
                * residual[n]
                * filtered_buffer
            )

        return M1FxLMSResult(
            reference=reference,
            control=control,
            residual=residual,
            weights=weights,
            sample_rate=self.sample_rate,
            filter_length=self.filter_length,
            learning_rate=self.learning_rate,
            secondary_path_mode=secondary_path_mode,
        )


def create_adapter() -> M1Adapter:
    return M1Adapter()


if __name__ == "__main__":
    adapter = create_adapter()

    t = np.arange(0, 0.25, 1.0 / M1_SAMPLE_RATE)
    reference = (
        0.5 * np.sin(2 * np.pi * 300 * t)
        + 0.3 * np.sin(2 * np.pi * 1000 * t)
    )

    result = adapter.process(reference)

    print("M6 M1 adapter: PASS")
    print(f"Sample rate:       {result.sample_rate} Hz")
    print(f"Filter length:     {result.filter_length}")
    print(f"Learning rate:     {result.learning_rate}")
    print(f"Secondary path:    {result.secondary_path_mode}")
    print(f"Samples processed: {result.reference.size}")
    print(f"Residual finite:   {np.isfinite(result.residual).all()}")
