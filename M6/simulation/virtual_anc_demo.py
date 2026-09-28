from __future__ import annotations

import json
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from M6.adapters.m1_adapter import M1Adapter
from M6.adapters.m2_adapter import M2Adapter
from M6.adapters.m5_adapter import M5Adapter
from M6.integration.virtual_audio import default_virtual_inputs


M5_ZIP = REPO_ROOT / "SilentVox_M5_GITHUB_CLEAN.zip"
M5_MEMBER = "SilentVox_M5/models/secondary_path_estimate.npy"

M2_FEATURE_DIR = (
    REPO_ROOT
    / "noise_classification"
    / "features"
    / "extraction"
)

M2_STATS_PATH = (
    REPO_ROOT
    / "noise_classification"
    / "features"
    / "metadata"
    / "feature_normalization_stats.npz"
)

EVIDENCE_DIR = REPO_ROOT / "M6" / "evidence" / "simulation"
REPORT_PATH = EVIDENCE_DIR / "virtual_anc_demo.json"


def rms(signal: np.ndarray) -> float:
    signal = np.asarray(signal, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(signal))))


def attenuation_db(input_signal: np.ndarray, output_signal: np.ndarray) -> float:
    input_rms = rms(input_signal)
    output_rms = rms(output_signal)

    if output_rms <= 0.0:
        return float("inf")

    if input_rms <= 0.0:
        return float("-inf")

    return float(20.0 * np.log10(input_rms / output_rms))


def load_m5_model() -> tuple[M5Adapter, tempfile.TemporaryDirectory[str]]:
    if not M5_ZIP.is_file():
        raise FileNotFoundError(f"M5 ZIP not found: {M5_ZIP}")

    temp_dir = tempfile.TemporaryDirectory()
    model_path = Path(temp_dir.name) / "secondary_path_estimate.npy"

    with zipfile.ZipFile(M5_ZIP) as archive:
        model_path.write_bytes(archive.read(M5_MEMBER))

    return M5Adapter(model_path), temp_dir


def classify_with_m2(audio: np.ndarray) -> dict[str, object]:
    """
    Use the existing M2 feature-extraction implementation.

    This is an integration demonstration only; it is not an accuracy test.
    """
    if str(M2_FEATURE_DIR) not in sys.path:
        sys.path.insert(0, str(M2_FEATURE_DIR))

    from extract_features import extract_features

    stats = np.load(M2_STATS_PATH)

    _, log_mel = extract_features(
        np.asarray(audio[:32000], dtype=np.float32)
    )

    mean = np.asarray(stats["log_mel_mean"], dtype=np.float32)
    std = np.asarray(stats["log_mel_std"], dtype=np.float32)

    normalized = (
        np.asarray(log_mel, dtype=np.float32) - mean[:, None]
    ) / np.maximum(std[:, None], 1e-8)

    adapter = M2Adapter()
    prediction = adapter.predict(normalized)

    return {
        "class_index": prediction.class_index,
        "class_name": prediction.class_name,
        "confidence": prediction.confidence,
        "probabilities": prediction.probabilities.tolist(),
        "feature_shape": list(normalized.shape),
        "note": "Software integration result; not an M2 accuracy measurement.",
    }


def main() -> None:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    streams = default_virtual_inputs()

    reference = np.asarray(streams["REF"], dtype=np.float64)
    error = np.asarray(streams["ERR"], dtype=np.float64)
    speech = np.asarray(streams["SPCH"], dtype=np.float64)

    if reference.shape != error.shape:
        raise RuntimeError("REF and ERR lengths do not match")

    if reference.size < 32000:
        raise RuntimeError("Virtual input is shorter than the M2 2-second window")

    print("=== SilentVox M6 Virtual ANC Simulation ===")
    print(f"Repository:   {REPO_ROOT}")
    print(f"Samples:      {reference.size}")
    print("Sample rate:  16000 Hz")

    print("\n[1/3] M2 classification...")
    classification = classify_with_m2(speech)

    print(f"Class:        {classification['class_name']}")
    print(f"Confidence:   {classification['confidence']:.6f}")

    print("\n[2/3] Loading M5 secondary path...")
    m5_adapter, temp_dir = load_m5_model()

    try:
        secondary_path = m5_adapter.load()

        print(f"Filter length: {secondary_path.filter_length}")
        print(f"Delay:         {secondary_path.delay_samples} samples")
        print(f"Attenuation:   {secondary_path.attenuation}")
        print(f"Status:        {secondary_path.status}")

        print("\n[3/3] Running M1 FxLMS...")

        start = time.perf_counter()

        result = M1Adapter().process(
            reference,
            disturbance=error,
            secondary_path=secondary_path,
        )

        elapsed = time.perf_counter() - start

    finally:
        temp_dir.cleanup()

    input_rms = rms(error)
    residual_rms = rms(result.residual)
    attenuation = attenuation_db(error, result.residual)

    duration_seconds = reference.size / 16000.0
    processing_ms = elapsed * 1000.0
    rtf = duration_seconds / elapsed if elapsed > 0 else float("inf")

    report = {
        "status": "SIMULATED",
        "validation_boundary": {
            "software_only": True,
            "physical_headset_validation": False,
            "physical_secondary_path_measurement": False,
            "real_time_hardware_validation": False,
        },
        "audio": {
            "sample_rate_hz": 16000,
            "samples": int(reference.size),
            "duration_seconds": duration_seconds,
            "channels": ["REF", "ERR", "SPCH"],
        },
        "m2": classification,
        "m5": {
            "sample_rate_hz": secondary_path.sample_rate,
            "filter_length": secondary_path.filter_length,
            "delay_samples": secondary_path.delay_samples,
            "delay_ms": (
                1000.0
                * secondary_path.delay_samples
                / secondary_path.sample_rate
            ),
            "attenuation": secondary_path.attenuation,
            "status": secondary_path.status,
        },
        "m1": {
            "sample_rate_hz": result.sample_rate,
            "filter_length": result.filter_length,
            "learning_rate": result.learning_rate,
            "secondary_path_mode": result.secondary_path_mode,
        },
        "metrics": {
            "input_rms": input_rms,
            "residual_rms": residual_rms,
            "attenuation_db": attenuation,
            "processing_time_ms": processing_ms,
            "real_time_factor": rtf,
            "frame_duration_ms": 1000.0 * duration_seconds,
            "finite_control": bool(np.isfinite(result.control).all()),
            "finite_residual": bool(np.isfinite(result.residual).all()),
            "finite_weights": bool(np.isfinite(result.weights).all()),
        },
        "notes": [
            "This is a virtual software simulation.",
            "M5 is a simulated secondary-path estimate.",
            "Attenuation is a simulated numerical result, not a physical ANC measurement.",
            "Frame duration is not equivalent to end-to-end hardware latency.",
        ],
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n=== SIMULATION COMPLETE ===")
    print(f"Input RMS:        {input_rms:.8f}")
    print(f"Residual RMS:     {residual_rms:.8f}")
    print(f"Attenuation:      {attenuation:.3f} dB")
    print(f"Processing time:  {processing_ms:.3f} ms")
    print(f"Real-time factor: {rtf:.2f}x")
    print(f"Evidence:         {REPORT_PATH}")


if __name__ == "__main__":
    main()
