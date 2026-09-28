from __future__ import annotations

import json
import sys
import time
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.signal import lfilter

REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from M6.adapters.m5_adapter import M5Adapter
from M6.simulation.stateful_streaming import StatefulFxLMS


SAMPLE_RATE = 16000
DURATION_SECONDS = 8.0
SAMPLES = int(SAMPLE_RATE * DURATION_SECONDS)

M5_ZIP = (
    REPO_ROOT
    / "SilentVox_M5_GITHUB_CLEAN.zip"
)

OUT_DIR = (
    REPO_ROOT
    / "M6"
    / "evidence"
    / "simulation"
)

REPORT_PATH = (
    OUT_DIR
    / "controlled_anc_validation.json"
)


def rms(signal: np.ndarray) -> float:
    signal = np.asarray(
        signal,
        dtype=np.float64,
    )

    return float(
        np.sqrt(
            np.mean(
                np.square(signal)
            )
        )
    )


def attenuation_db(
    input_signal: np.ndarray,
    output_signal: np.ndarray,
) -> float:

    input_rms = rms(input_signal)
    output_rms = rms(output_signal)

    return float(
        20.0
        * np.log10(
            max(input_rms, 1e-12)
            /
            max(output_rms, 1e-12)
        )
    )


def load_m5():

    with TemporaryDirectory(
        prefix="silentvox_m5_controlled_"
    ) as temp_dir:

        temp_root = Path(temp_dir)

        with zipfile.ZipFile(
            M5_ZIP,
            "r",
        ) as archive:

            member = next(
                (
                    name
                    for name in archive.namelist()
                    if name.endswith(
                        "secondary_path_estimate.npy"
                    )
                ),
                None,
            )

            if member is None:
                raise FileNotFoundError(
                    "M5 secondary-path model not found"
                )

            archive.extract(
                member,
                temp_root,
            )

            model_path = (
                temp_root
                / member
            )

            adapter = M5Adapter(
                model_path
            )

            return adapter.load()


def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    secondary_path = load_m5()

    rng = np.random.default_rng(
        20260928
    )

    # Controlled broadband reference.
    reference = rng.normal(
        0.0,
        0.20,
        SAMPLES,
    ).astype(
        np.float64
    )

    # Known disturbance-generation controller.
    # The adaptive FxLMS controller must learn an
    # equivalent response from the reference.
    true_controller = np.array(
        [
            0.35,
            -0.20,
            0.12,
            0.08,
            -0.06,
            0.04,
            -0.025,
            0.015,
        ],
        dtype=np.float64,
    )

    desired_control = lfilter(
        true_controller,
        [1.0],
        reference,
    )

    disturbance = lfilter(
        secondary_path.impulse_response,
        [1.0],
        desired_control,
    )

    # Measure the disturbance before ANC.
    input_rms = rms(
        disturbance
    )

    # Use the actual M6 stateful streaming controller.
    controller = StatefulFxLMS(
        secondary_path,
        filter_length=64,
        learning_rate=1e-5,
    )

    residual_parts = []

    block_size = 512

    start = time.perf_counter()

    for start_idx in range(
        0,
        SAMPLES,
        block_size,
    ):

        end_idx = min(
            start_idx + block_size,
            SAMPLES,
        )

        residual_block = (
            controller.process_block(
                reference[start_idx:end_idx],
                disturbance[start_idx:end_idx],
            )
        )

        residual_parts.append(
            residual_block
        )

    elapsed = (
        time.perf_counter()
        - start
    )

    residual = np.concatenate(
        residual_parts
    )

    # Evaluate convergence after the initial
    # adaptation period as well as over the
    # complete signal.
    warmup = SAMPLE_RATE

    full_input_rms = rms(
        disturbance
    )

    full_residual_rms = rms(
        residual
    )

    tail_input_rms = rms(
        disturbance[warmup:]
    )

    tail_residual_rms = rms(
        residual[warmup:]
    )

    full_attenuation = attenuation_db(
        disturbance,
        residual,
    )

    tail_attenuation = attenuation_db(
        disturbance[warmup:],
        residual[warmup:],
    )

    duration = (
        SAMPLES
        / SAMPLE_RATE
    )

    rtf = (
        duration / elapsed
        if elapsed > 0
        else float("inf")
    )

    report = {
        "status": "SIMULATED",
        "physical_validation": False,
        "validation_type": (
            "CONTROLLED_CORRELATED_ANC"
        ),
        "sample_rate_hz": SAMPLE_RATE,
        "samples": SAMPLES,
        "duration_seconds": duration,
        "block_size": block_size,
        "stateful": True,
        "state_samples_processed": (
            controller.total_samples
        ),
        "m1_source_modified": False,
        "m5": {
            "filter_length": (
                secondary_path.filter_length
            ),
            "delay_samples": (
                secondary_path.delay_samples
            ),
            "attenuation": (
                secondary_path.attenuation
            ),
            "status": secondary_path.status,
        },
        "controlled_disturbance": {
            "correlated_with_reference": True,
            "generation_filter": (
                true_controller.tolist()
            ),
            "description": (
                "Disturbance is generated by "
                "passing the reference through "
                "a known controller and the M5 "
                "secondary-path model."
            ),
        },
        "metrics": {
            "input_rms": full_input_rms,
            "residual_rms": full_residual_rms,
            "full_attenuation_db": (
                full_attenuation
            ),
            "tail_input_rms": tail_input_rms,
            "tail_residual_rms": (
                tail_residual_rms
            ),
            "tail_attenuation_db": (
                tail_attenuation
            ),
            "processing_seconds": elapsed,
            "real_time_factor": rtf,
            "finite_residual": bool(
                np.isfinite(
                    residual
                ).all()
            ),
            "finite_weights": bool(
                np.isfinite(
                    controller.weights
                ).all()
            ),
            "final_weight_norm": float(
                np.linalg.norm(
                    controller.weights
                )
            ),
        },
        "notes": [
            "Controlled software-only ANC validation.",
            "The disturbance is intentionally correlated "
            "with the reference so adaptive cancellation "
            "can be evaluated.",
            "M5 remains a simulated secondary-path estimate.",
            "No Raspberry Pi or physical headset validation "
            "is claimed.",
        ],
    }

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "=== SilentVox M6 Controlled ANC Validation ==="
    )
    print(
        f"Input RMS:       {full_input_rms:.8f}"
    )
    print(
        f"Residual RMS:    {full_residual_rms:.8f}"
    )
    print(
        f"Full attenuation:{full_attenuation:8.3f} dB"
    )
    print(
        f"Tail attenuation:{tail_attenuation:8.3f} dB"
    )
    print(
        f"Final weights:   "
        f"{np.linalg.norm(controller.weights):.6f}"
    )
    print(
        f"RTF:             {rtf:.2f}x"
    )
    print(
        f"Finite residual: "
        f"{np.isfinite(residual).all()}"
    )
    print()
    print(
        f"Evidence: {REPORT_PATH}"
    )


if __name__ == "__main__":
    main()
