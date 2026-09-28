from __future__ import annotations

import csv
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from M6.adapters.m1_adapter import M1Adapter
from M6.adapters.m5_adapter import M5Adapter
from M6.integration.virtual_audio import default_virtual_inputs


M5_ZIP = REPO_ROOT / "SilentVox_M5_GITHUB_CLEAN.zip"
M5_MEMBER = "SilentVox_M5/models/secondary_path_estimate.npy"

EVIDENCE_DIR = REPO_ROOT / "M6" / "evidence" / "simulation"

CSV_PATH = EVIDENCE_DIR / "secondary_path_sensitivity.csv"
JSON_PATH = EVIDENCE_DIR / "secondary_path_sensitivity.json"

SAMPLE_RATE = 16000

DELAYS_MS = (0.0, 1.0, 2.0, 5.0, 10.0)
ATTENUATIONS = (0.40, 0.65, 0.85)


def rms(signal: np.ndarray) -> float:
    signal = np.asarray(signal, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(signal))))


def attenuation_db(
    input_signal: np.ndarray,
    output_signal: np.ndarray,
) -> float:
    input_rms = rms(input_signal)
    output_rms = rms(output_signal)

    if input_rms <= 0.0:
        return float("-inf")

    if output_rms <= 0.0:
        return float("inf")

    return float(
        20.0 * np.log10(input_rms / output_rms)
    )


def load_base_model():
    if not M5_ZIP.is_file():
        raise FileNotFoundError(
            f"M5 ZIP not found: {M5_ZIP}"
        )

    temp_dir = tempfile.TemporaryDirectory()

    model_path = (
        Path(temp_dir.name)
        / "secondary_path_estimate.npy"
    )

    with zipfile.ZipFile(M5_ZIP) as archive:
        model_path.write_bytes(
            archive.read(M5_MEMBER)
        )

    model = M5Adapter(model_path).load()

    return model, temp_dir


def build_modified_ir(
    base_ir: np.ndarray,
    delay_ms: float,
    attenuation: float,
) -> np.ndarray:

    delay_samples = int(
        round(
            delay_ms
            * SAMPLE_RATE
            / 1000.0
        )
    )

    base_ir = np.asarray(
        base_ir,
        dtype=np.float64,
    )

    shifted = np.zeros_like(base_ir)

    if delay_samples >= len(base_ir):
        return shifted

    shifted[delay_samples:] = (
        base_ir[
            :len(base_ir) - delay_samples
        ]
    )

    max_abs = np.max(np.abs(shifted))

    if max_abs > 0.0:
        shifted *= (
            attenuation
            / max_abs
        )

    return shifted


def main() -> None:
    EVIDENCE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    streams = default_virtual_inputs()

    reference = np.asarray(
        streams["REF"],
        dtype=np.float64,
    )

    disturbance = np.asarray(
        streams["ERR"],
        dtype=np.float64,
    )

    base_model, temp_dir = load_base_model()

    try:
        base_ir = np.asarray(
            base_model.impulse_response,
            dtype=np.float64,
        )

        base_input_rms = rms(disturbance)

        rows = []

        print(
            "=== SilentVox M6 Secondary-Path "
            "Sensitivity Simulation ==="
        )

        print(
            f"Sample rate: {SAMPLE_RATE} Hz"
        )

        print(
            f"Base IR length: {len(base_ir)} samples"
        )

        for delay_ms in DELAYS_MS:

            for attenuation in ATTENUATIONS:

                modified_ir = build_modified_ir(
                    base_ir,
                    delay_ms,
                    attenuation,
                )

                # Create a simulation-only M5 model.
                simulated_path = type(base_model)(
                    impulse_response=modified_ir,
                    sample_rate=SAMPLE_RATE,
                    delay_samples=int(
                        round(
                            delay_ms
                            * SAMPLE_RATE
                            / 1000.0
                        )
                    ),
                    attenuation=attenuation,
                    status=(
                        "SIMULATION_ONLY"
                    ),
                )

                result = M1Adapter().process(
                    reference,
                    disturbance=disturbance,
                    secondary_path=simulated_path,
                )

                residual_rms = rms(
                    result.residual
                )

                attenuation_db_value = (
                    attenuation_db(
                        disturbance,
                        result.residual,
                    )
                )

                rows.append(
                    {
                        "delay_ms": delay_ms,
                        "delay_samples": int(
                            round(
                                delay_ms
                                * SAMPLE_RATE
                                / 1000.0
                            )
                        ),
                        "secondary_path_attenuation": attenuation,
                        "input_rms": base_input_rms,
                        "residual_rms": residual_rms,
                        "attenuation_db": (
                            attenuation_db_value
                        ),
                        "finite_control": bool(
                            np.isfinite(
                                result.control
                            ).all()
                        ),
                        "finite_residual": bool(
                            np.isfinite(
                                result.residual
                            ).all()
                        ),
                        "finite_weights": bool(
                            np.isfinite(
                                result.weights
                            ).all()
                        ),
                    }
                )

                print(
                    f"delay={delay_ms:>4.1f} ms "
                    f"attenuation={attenuation:.2f} "
                    f"-> "
                    f"residual={residual_rms:.6f}, "
                    f"ANC={attenuation_db_value:.3f} dB"
                )

    finally:
        temp_dir.cleanup()

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)

    report = {
        "status": "SIMULATED",
        "sample_rate_hz": SAMPLE_RATE,
        "delay_values_ms": list(
            DELAYS_MS
        ),
        "attenuation_values": list(
            ATTENUATIONS
        ),
        "base_secondary_path": {
            "filter_length": int(
                base_model.filter_length
            ),
            "delay_samples": int(
                base_model.delay_samples
            ),
            "attenuation": float(
                base_model.attenuation
            ),
        },
        "results": rows,
        "notes": [
            "Simulation-only secondary-path sensitivity experiment.",
            "Modified impulse responses are generated in memory.",
            "The M5 repository model is not modified.",
            "Results are not physical acoustic measurements.",
            "Results should not be interpreted as Raspberry Pi hardware performance.",
        ],
    }

    JSON_PATH.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== SENSITIVITY EXPERIMENT COMPLETE ===")
    print(f"CSV:  {CSV_PATH}")
    print(f"JSON: {JSON_PATH}")


if __name__ == "__main__":
    main()
