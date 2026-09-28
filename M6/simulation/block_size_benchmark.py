from __future__ import annotations

import csv
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
from M6.adapters.m5_adapter import M5Adapter
from M6.integration.virtual_audio import default_virtual_inputs


M5_ZIP = REPO_ROOT / "SilentVox_M5_GITHUB_CLEAN.zip"
M5_MEMBER = "SilentVox_M5/models/secondary_path_estimate.npy"

EVIDENCE_DIR = REPO_ROOT / "M6" / "evidence" / "simulation"
CSV_PATH = EVIDENCE_DIR / "block_size_benchmark.csv"
JSON_PATH = EVIDENCE_DIR / "block_size_benchmark.json"

BLOCK_SIZES = (256, 512, 1024)
SAMPLE_RATE = 16000


def rms(signal: np.ndarray) -> float:
    signal = np.asarray(signal, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(signal))))


def attenuation_db(input_signal: np.ndarray, output_signal: np.ndarray) -> float:
    input_rms = rms(input_signal)
    output_rms = rms(output_signal)

    if input_rms <= 0.0:
        return float("-inf")

    if output_rms <= 0.0:
        return float("inf")

    return float(20.0 * np.log10(input_rms / output_rms))


def load_m5():
    if not M5_ZIP.is_file():
        raise FileNotFoundError(f"M5 ZIP not found: {M5_ZIP}")

    temp_dir = tempfile.TemporaryDirectory()
    model_path = Path(temp_dir.name) / "secondary_path_estimate.npy"

    with zipfile.ZipFile(M5_ZIP) as archive:
        model_path.write_bytes(archive.read(M5_MEMBER))

    model = M5Adapter(model_path).load()
    return model, temp_dir


def main() -> None:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    streams = default_virtual_inputs()

    reference = np.asarray(streams["REF"], dtype=np.float64)
    disturbance = np.asarray(streams["ERR"], dtype=np.float64)

    secondary_path, temp_dir = load_m5()

    try:
        rows = []

        print("=== SilentVox M6 Block-Size Simulation Benchmark ===")
        print(f"Sample rate: {SAMPLE_RATE} Hz")
        print(f"Total samples: {reference.size}")

        for block_size in BLOCK_SIZES:
            block_duration_ms = 1000.0 * block_size / SAMPLE_RATE

            processing_times = []
            input_values = []
            residual_values = []

            for start in range(0, reference.size - block_size + 1, block_size):
                ref_block = reference[start:start + block_size]
                disturbance_block = disturbance[start:start + block_size]

                timer_start = time.perf_counter()

                result = M1Adapter().process(
                    ref_block,
                    disturbance=disturbance_block,
                    secondary_path=secondary_path,
                )

                elapsed = time.perf_counter() - timer_start

                processing_times.append(elapsed * 1000.0)
                input_values.append(rms(disturbance_block))
                residual_values.append(rms(result.residual))

            processing_ms = float(np.mean(processing_times))
            processing_max_ms = float(np.max(processing_times))
            input_rms = float(np.mean(input_values))
            residual_rms = float(np.mean(residual_values))

            attenuation = attenuation_db(
                np.asarray(input_values),
                np.asarray(residual_values),
            )

            rtf = block_duration_ms / processing_ms
            meets_block_budget = processing_ms < block_duration_ms

            row = {
                "block_size_samples": block_size,
                "block_duration_ms": block_duration_ms,
                "mean_processing_ms": processing_ms,
                "max_processing_ms": processing_max_ms,
                "real_time_factor": rtf,
                "input_rms": input_rms,
                "residual_rms": residual_rms,
                "attenuation_db": attenuation,
                "processing_below_block_duration": meets_block_budget,
                "blocks_tested": len(processing_times),
            }

            rows.append(row)

            print(
                f"\nBlock {block_size}:"
                f"\n  Duration:       {block_duration_ms:.3f} ms"
                f"\n  Mean processing:{processing_ms:.3f} ms"
                f"\n  Max processing: {processing_max_ms:.3f} ms"
                f"\n  RTF:             {rtf:.2f}x"
                f"\n  Attenuation:     {attenuation:.3f} dB"
                f"\n  Below budget:    {meets_block_budget}"
            )

        with CSV_PATH.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(rows)

        report = {
            "status": "SIMULATED",
            "sample_rate_hz": SAMPLE_RATE,
            "block_sizes_samples": list(BLOCK_SIZES),
            "secondary_path": {
                "filter_length": secondary_path.filter_length,
                "delay_samples": secondary_path.delay_samples,
                "attenuation": secondary_path.attenuation,
                "status": secondary_path.status,
            },
            "results": rows,
            "notes": [
                "This is a software block-processing benchmark.",
                "Each block is processed independently.",
                "This is not a continuous stateful hardware audio-stream test.",
                "Processing time below block duration is a preliminary software real-time feasibility indicator.",
                "It is not a physical Raspberry Pi latency measurement.",
            ],
        }

        JSON_PATH.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

        print("\n=== BENCHMARK COMPLETE ===")
        print(f"CSV:  {CSV_PATH}")
        print(f"JSON: {JSON_PATH}")

    finally:
        temp_dir.cleanup()


if __name__ == "__main__":
    main()
