import csv
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]

RESULTS_DIR = PROJECT_ROOT / "ROBUSTNESS_ANALYSIS" / "deployment" / "results"
REPORTS_DIR = PROJECT_ROOT / "ROBUSTNESS_ANALYSIS" / "deployment" / "reports"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SCRIPT_DIR))

import live_audio

NUM_WINDOWS = 20

CSV_PATH = RESULTS_DIR / "P6_live_inference_benchmark.csv"
REPORT_PATH = REPORTS_DIR / "P6_LIVE_INFERENCE_BENCHMARK.md"


def main():
    print("=" * 70)
    print("SILENTVOX M2-P6 — LIVE INFERENCE BENCHMARK")
    print("=" * 70)

    print()
    print("Loading P3 normalization statistics...")
    log_mel_mean, log_mel_std = live_audio.load_normalization_stats()

    print("Loading P5 augmented model...")
    model = live_audio.load_model(live_audio.P5_MODEL)

    print()
    print("Model:", live_audio.P5_MODEL)
    print("Sample rate:", live_audio.SAMPLE_RATE, "Hz")
    print("Channels:", live_audio.CHANNELS)
    print("Window:", live_audio.WINDOW_SECONDS, "seconds")
    print("Window samples:", live_audio.WINDOW_SAMPLES)
    print("Feature: 64-band Log-Mel")
    print("Feature shape: 64 x 197")
    print("Benchmark windows:", NUM_WINDOWS)

    print()
    print("Starting microphone benchmark...")
    print("Speak normally near the microphone.")
    print("The benchmark will stop automatically after 20 windows.")
    print()

    records = []

    stream = sd.InputStream(
        samplerate=live_audio.SAMPLE_RATE,
        channels=live_audio.CHANNELS,
        dtype="float32",
        blocksize=live_audio.WINDOW_SAMPLES,
        latency="low",
    )

    stream.start()

    try:
        for window_index in range(1, NUM_WINDOWS + 1):

            audio, overflowed = stream.read(
                live_audio.WINDOW_SAMPLES
            )

            audio = np.asarray(
                audio,
                dtype=np.float32,
            )

            if audio.ndim == 2:
                audio = audio[:, 0]

            processing_start = time.perf_counter()

            feature = live_audio.extract_log_mel(
                audio
            )

            feature = live_audio.normalize_log_mel(
                feature,
                log_mel_mean,
                log_mel_std,
            )

            index, probabilities = live_audio.predict(
                model,
                feature,
            )

            processing_end = time.perf_counter()

            processing_time_ms = (
                processing_end - processing_start
            ) * 1000.0

            predicted_class = live_audio.CLASSES[index]
            confidence = float(probabilities[index])

            rms = float(
                np.sqrt(
                    np.mean(
                        np.square(
                            audio.astype(np.float64)
                        )
                    )
                )
            )

            finite = bool(np.isfinite(audio).all())

            records.append(
                {
                    "window_index": window_index,
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "processing_time_ms": processing_time_ms,
                    "audio_rms": rms,
                    "audio_finite": finite,
                    "overflowed": bool(overflowed),
                    "prob_engine": float(probabilities[0]),
                    "prob_rain": float(probabilities[1]),
                    "prob_siren": float(probabilities[2]),
                    "prob_wind": float(probabilities[3]),
                }
            )

            print(
                f"Window {window_index:04d} | "
                f"Prediction={predicted_class:<6} | "
                f"Confidence={confidence:.3f} | "
                f"Processing={processing_time_ms:.2f} ms | "
                f"RMS={rms:.6f}"
            )

    finally:
        stream.stop()
        stream.close()

    if not records:
        print("No benchmark records collected.")
        return

    times = np.array(
        [r["processing_time_ms"] for r in records],
        dtype=np.float64,
    )

    confidences = np.array(
        [r["confidence"] for r in records],
        dtype=np.float64,
    )

    rms_values = np.array(
        [r["audio_rms"] for r in records],
        dtype=np.float64,
    )

    average_time = float(np.mean(times))
    median_time = float(np.median(times))
    minimum_time = float(np.min(times))
    maximum_time = float(np.max(times))

    average_confidence = float(
        np.mean(confidences)
    )

    average_rms = float(
        np.mean(rms_values)
    )

    overflow_count = sum(
        1 for r in records if r["overflowed"]
    )

    nonfinite_count = sum(
        1 for r in records if not r["audio_finite"]
    )

    real_time_factor = (
        live_audio.WINDOW_SECONDS
        / (average_time / 1000.0)
    )

    class_counts = {
        class_name: 0
        for class_name in live_audio.CLASSES
    }

    for record in records:
        class_counts[
            record["predicted_class"]
        ] += 1

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        fieldnames = list(records[0].keys())

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(records)

    distribution_lines = []

    for class_name in live_audio.CLASSES:
        distribution_lines.append(
            f"- {class_name}: {class_counts[class_name]}"
        )

    distribution = "\n".join(
        distribution_lines
    )

    report = f"""# SilentVox M2-P6 — Live Inference Benchmark

## Purpose

This test verifies continuous real-microphone inference through the P6 deployment pipeline.

This is a runtime verification test, not a classification-accuracy evaluation.

## Configuration

- Model: P5 augmented CNN
- Sample rate: {live_audio.SAMPLE_RATE} Hz
- Channels: {live_audio.CHANNELS}
- Window duration: {live_audio.WINDOW_SECONDS} seconds
- Window samples: {live_audio.WINDOW_SAMPLES}
- Feature: 64-band Log-Mel
- Feature shape: 64 x 197
- Normalization: P3 training-set statistics
- Windows processed: {len(records)}

## Runtime Results

- Minimum processing time: {minimum_time:.3f} ms
- Median processing time: {median_time:.3f} ms
- Average processing time: {average_time:.3f} ms
- Maximum processing time: {maximum_time:.3f} ms
- Real-time factor: {real_time_factor:.2f}

## Audio Integrity

- Overflow windows: {overflow_count}
- Non-finite windows: {nonfinite_count}
- Average RMS: {average_rms:.6f}

## Confidence

- Mean predicted confidence: {average_confidence:.6f}

Confidence is reported descriptively only.
No validated UNKNOWN threshold has been applied.

## Prediction Distribution

{distribution}

These predictions are outputs for live microphone recordings and do not have ground-truth labels.

## Verified Pipeline

Microphone
|
v
16 kHz mono capture
|
v
2-second window
|
v
P3 Log-Mel extraction
|
v
P3 training-set normalization
|
v
P5 CNN inference
|
v
Class probability output

## Limitations

This benchmark does not establish:

- classification accuracy
- classification correctness
- confidence calibration
- final UNKNOWN threshold
- controlled noise robustness
- end-to-end headset latency
- Raspberry Pi hardware performance

Physical Raspberry Pi validation remains future work because the physical Pi is not currently available.

## Output Files

- deployment/results/P6_live_inference_benchmark.csv
- deployment/reports/P6_LIVE_INFERENCE_BENCHMARK.md

## Status

P6 LIVE INFERENCE BENCHMARK COMPLETE
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("P6 LIVE INFERENCE BENCHMARK COMPLETE")
    print("=" * 70)

    print()
    print("Windows processed:", len(records))
    print(f"Average processing: {average_time:.3f} ms")
    print(f"Median processing:  {median_time:.3f} ms")
    print(f"Maximum processing: {maximum_time:.3f} ms")
    print(f"Real-time factor:   {real_time_factor:.2f}")
    print("Overflow windows:", overflow_count)
    print("Non-finite windows:", nonfinite_count)

    print()
    print("CSV saved:", CSV_PATH)
    print("Report saved:", REPORT_PATH)


if __name__ == "__main__":
    main()