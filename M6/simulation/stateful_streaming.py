from __future__ import annotations

import json
import sys
import time
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf
from scipy.signal import lfilter

REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from M6.adapters.m5_adapter import M5Adapter


M4_AUDIO_DIR = REPO_ROOT / "M4" / "virtual_mics"

ZIP_PATH = REPO_ROOT / "SilentVox_M5_GITHUB_CLEAN.zip"

OUT_DIR = (
    REPO_ROOT
    / "M6"
    / "evidence"
    / "simulation"
)


SAMPLE_RATE = 16000
FILTER_LENGTH = 64
LEARNING_RATE = 1e-5


def rms(signal: np.ndarray) -> float:
    signal = np.asarray(signal, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(signal))))


def load_audio(path: Path) -> np.ndarray:

    audio, sample_rate = sf.read(
        str(path),
        dtype="float64",
    )

    if sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"Expected {SAMPLE_RATE} Hz, got {sample_rate} Hz: {path}"
        )

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    audio = np.asarray(audio, dtype=np.float64)

    if not np.isfinite(audio).all():
        raise ValueError(f"Non-finite audio detected: {path}")

    return audio


class StatefulFxLMS:
    """
    M6-owned stateful FxLMS runtime controller.

    State persists across every process_block() call.
    M1 source files are not modified.
    """

    def __init__(
        self,
        secondary_path,
        *,
        sample_rate: int = SAMPLE_RATE,
        filter_length: int = FILTER_LENGTH,
        learning_rate: float = LEARNING_RATE,
    ) -> None:

        if sample_rate != SAMPLE_RATE:
            raise ValueError(
                f"Expected {SAMPLE_RATE} Hz"
            )

        self.sample_rate = int(sample_rate)
        self.filter_length = int(filter_length)
        self.learning_rate = float(learning_rate)

        self.secondary_filter = np.asarray(
            secondary_path.impulse_response,
            dtype=np.float64,
        )

        if self.secondary_filter.ndim != 1:
            raise ValueError(
                "Secondary path must be 1-D"
            )

        if not np.isfinite(
            self.secondary_filter
        ).all():
            raise ValueError(
                "Secondary path contains non-finite values"
            )

        self.weights = np.zeros(
            self.filter_length,
            dtype=np.float64,
        )

        self.reference_buffer = np.zeros(
            self.filter_length,
            dtype=np.float64,
        )

        self.filtered_reference_buffer = np.zeros(
            self.filter_length,
            dtype=np.float64,
        )

        self.secondary_buffer = np.zeros(
            self.secondary_filter.size,
            dtype=np.float64,
        )

        self.filtered_reference_tail = np.zeros(
            self.secondary_filter.size - 1,
            dtype=np.float64,
        )

        self.total_samples = 0

    def process_block(
        self,
        reference: np.ndarray,
        disturbance: np.ndarray,
    ) -> np.ndarray:

        reference = np.asarray(
            reference,
            dtype=np.float64,
        )

        disturbance = np.asarray(
            disturbance,
            dtype=np.float64,
        )

        if reference.ndim != 1:
            raise ValueError(
                "reference must be 1-D"
            )

        if disturbance.shape != reference.shape:
            raise ValueError(
                "disturbance must match reference shape"
            )

        if not np.isfinite(reference).all():
            raise ValueError(
                "reference contains non-finite values"
            )

        if not np.isfinite(disturbance).all():
            raise ValueError(
                "disturbance contains non-finite values"
            )

        if reference.size == 0:
            return np.empty(
                0,
                dtype=np.float64,
            )

        # Filter the current reference block through
        # the complete secondary-path estimate.
        filtered_reference = lfilter(
            self.secondary_filter,
            [1.0],
            reference,
        )

        residual = np.zeros_like(
            reference,
            dtype=np.float64,
        )

        for n in range(reference.size):

            # Persistent reference history.
            self.reference_buffer[1:] = (
                self.reference_buffer[:-1]
            )

            self.reference_buffer[0] = (
                reference[n]
            )

            # Current control signal.
            control_sample = float(
                np.dot(
                    self.weights,
                    self.reference_buffer,
                )
            )

            # Persistent secondary-path history.
            self.secondary_buffer[1:] = (
                self.secondary_buffer[:-1]
            )

            self.secondary_buffer[0] = (
                control_sample
            )

            # Complete M5 secondary-path convolution.
            secondary_output = float(
                np.dot(
                    self.secondary_filter,
                    self.secondary_buffer,
                )
            )

            # ANC residual.
            residual[n] = (
                disturbance[n]
                - secondary_output
            )

            # Persistent filtered-reference history.
            self.filtered_reference_buffer[1:] = (
                self.filtered_reference_buffer[:-1]
            )

            self.filtered_reference_buffer[0] = (
                filtered_reference[n]
            )

            # FxLMS adaptation.
            self.weights += (
                self.learning_rate
                * residual[n]
                * self.filtered_reference_buffer
            )

        self.total_samples += reference.size

        if not np.isfinite(
            residual
        ).all():
            raise RuntimeError(
                "Stateful FxLMS produced non-finite residual"
            )

        if not np.isfinite(
            self.weights
        ).all():
            raise RuntimeError(
                "Stateful FxLMS produced non-finite weights"
            )

        return residual


def run_stream(
    reference: np.ndarray,
    disturbance: np.ndarray,
    secondary_path,
    block_size: int,
) -> tuple[np.ndarray, float, StatefulFxLMS]:

    controller = StatefulFxLMS(
        secondary_path,
    )

    outputs = []

    start = time.perf_counter()

    for start_idx in range(
        0,
        len(reference),
        block_size,
    ):

        end_idx = min(
            start_idx + block_size,
            len(reference),
        )

        ref_block = reference[
            start_idx:end_idx
        ]

        dist_block = disturbance[
            start_idx:end_idx
        ]

        residual_block = (
            controller.process_block(
                ref_block,
                dist_block,
            )
        )

        outputs.append(
            residual_block
        )

    elapsed = (
        time.perf_counter()
        - start
    )

    return (
        np.concatenate(outputs),
        elapsed,
        controller,
    )


def main() -> None:

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference = load_audio(
        M4_AUDIO_DIR / "reference.wav"
    )

    disturbance = load_audio(
        M4_AUDIO_DIR / "error.wav"
    )

    sample_count = min(
        len(reference),
        len(disturbance),
    )

    reference = reference[
        :sample_count
    ]

    disturbance = disturbance[
        :sample_count
    ]

    with TemporaryDirectory(
        prefix="silentvox_m5_stream_"
    ) as temp_dir:

        temp_root = Path(temp_dir)

        with zipfile.ZipFile(
            ZIP_PATH,
            "r",
        ) as archive:

            members = [
                name
                for name in archive.namelist()
                if name.endswith(
                    "secondary_path_estimate.npy"
                )
            ]

            if not members:
                raise FileNotFoundError(
                    "M5 secondary-path model not found"
                )

            archive.extract(
                members[0],
                temp_root,
            )

            model_path = (
                temp_root
                / members[0]
            )

        m5_adapter = M5Adapter(
            model_path
        )

        secondary_path = (
            m5_adapter.load()
        )

        results = []

        for block_size in (
            256,
            512,
            1024,
        ):

            residual, elapsed, controller = (
                run_stream(
                    reference,
                    disturbance,
                    secondary_path,
                    block_size,
                )
            )

            duration = (
                len(reference)
                / SAMPLE_RATE
            )

            rtf = (
                duration / elapsed
                if elapsed > 0
                else float("inf")
            )

            input_rms = rms(
                disturbance
            )

            residual_rms = rms(
                residual
            )

            attenuation_db = (
                20.0
                * np.log10(
                    max(
                        input_rms,
                        1e-12,
                    )
                    /
                    max(
                        residual_rms,
                        1e-12,
                    )
                )
            )

            results.append(
                {
                    "block_size": block_size,
                    "samples": len(reference),
                    "duration_seconds": duration,
                    "processing_seconds": elapsed,
                    "real_time_factor": rtf,
                    "input_rms": input_rms,
                    "residual_rms": residual_rms,
                    "attenuation_db": attenuation_db,
                    "finite_residual": bool(
                        np.isfinite(
                            residual
                        ).all()
                    ),
                    "state_samples_processed": (
                        controller.total_samples
                    ),
                    "final_weight_norm": float(
                        np.linalg.norm(
                            controller.weights
                        )
                    ),
                }
            )

    report = {
        "status": "SIMULATED",
        "physical_validation": False,
        "sample_rate_hz": SAMPLE_RATE,
        "samples": len(reference),
        "continuous_stateful_processing": True,
        "state_persistence": True,
        "m1_source_modified": False,
        "input_reference": (
            "M4/virtual_mics/reference.wav"
        ),
        "input_disturbance": (
            "M4/virtual_mics/error.wav"
        ),
        "m5_model": (
            "SilentVox_M5_GITHUB_CLEAN.zip::"
            "SilentVox_M5/models/"
            "secondary_path_estimate.npy"
        ),
        "block_sizes": [
            256,
            512,
            1024,
        ],
        "results": results,
        "note": (
            "Software-only M6 stateful FxLMS streaming "
            "simulation. Adaptive weights and signal "
            "histories persist across blocks. "
            "M4 virtual WAV artifacts are used directly. "
            "No Raspberry Pi or physical headset validation "
            "is claimed."
        ),
    }

    path = (
        OUT_DIR
        / "stateful_streaming.json"
    )

    path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "=== SilentVox M6 Stateful Streaming ==="
    )

    for row in results:
        print(
            f"Block {row['block_size']:4d}: "
            f"{row['processing_seconds']:.4f} s, "
            f"RTF {row['real_time_factor']:.2f}x, "
            f"attenuation "
            f"{row['attenuation_db']:.3f} dB, "
            f"weights "
            f"{row['final_weight_norm']:.6f}, "
            f"finite "
            f"{row['finite_residual']}"
        )

    print()
    print("Evidence:")
    print(path)


if __name__ == "__main__":
    main()
