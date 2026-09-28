import csv
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import sounddevice as sd

SCRIPT_DIR = Path(__file__).resolve().parent
DEPLOYMENT_DIR = SCRIPT_DIR.parent

RESULTS_DIR = DEPLOYMENT_DIR / "results"
REPORTS_DIR = DEPLOYMENT_DIR / "reports"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SCRIPT_DIR))

import live_audio


NUM_WINDOWS = 30


def transition_count(values):
    if len(values) < 2:
        return 0

    return sum(
        previous != current
        for previous, current in zip(values, values[1:])
    )


def class_distribution(values):
    counts = Counter(values)

    return {
        class_name: counts.get(class_name, 0)
        for class_name in live_audio.CLASSES
    }


def resolve_majority_output(value):
    """
    Convert MajorityVoteSmoother output into an integer class index.
    """

    if isinstance(value, (int, np.integer)):
        return int(value)

    if isinstance(value, str):

        if value in live_audio.CLASSES:
            return live_audio.CLASSES.index(value)

        return int(value)

    array = np.asarray(value)

    if array.ndim == 0:
        return int(array.item())

    if array.size == 1:
        return int(array.reshape(-1)[0])

    raise TypeError(
        f"Unexpected majority smoother output shape: {array.shape}"
    )


def resolve_probability_output(value):
    """
    Normalize ProbabilityMovingAverage output.

    Supported outputs:

    1. class index
    2. (class_index, probabilities)
    3. probability vector
    """

    if isinstance(value, tuple):

        if len(value) == 2:

            class_part, probability_part = value

            class_index = resolve_majority_output(
                class_part
            )

            probabilities = np.asarray(
                probability_part,
                dtype=np.float32
            ).reshape(-1)

            return class_index, probabilities

    array = np.asarray(value)

    if array.ndim == 0:

        return int(array.item()), None

    array = array.reshape(-1)

    if array.size == len(live_audio.CLASSES):

        probabilities = array.astype(
            np.float32
        )

        class_index = int(
            np.argmax(probabilities)
        )

        return class_index, probabilities

    if array.size == 1:

        return int(array[0]), None

    raise TypeError(
        f"Unexpected probability smoother output shape: {array.shape}"
    )


def main():

    print("=" * 70)
    print(
        "SILENTVOX M2-P6 — TEMPORAL SMOOTHING BENCHMARK"
    )
    print("=" * 70)

    print()
    print(
        "Loading P3 normalization statistics..."
    )

    mean, std = live_audio.load_normalization_stats()

    print(
        "Loading P5 augmented model..."
    )

    model = live_audio.load_model(
        live_audio.P5_MODEL
    )

    majority = live_audio.MajorityVoteSmoother(
        history_size=live_audio.SMOOTHING_HISTORY
    )

    probability = live_audio.ProbabilityMovingAverage(
        num_classes=len(live_audio.CLASSES),
        history_size=live_audio.SMOOTHING_HISTORY
    )

    print()

    print(
        f"Sample rate:       {live_audio.SAMPLE_RATE} Hz"
    )

    print(
        f"Channels:          {live_audio.CHANNELS}"
    )

    print(
        f"Window:            {live_audio.WINDOW_SECONDS} seconds"
    )

    print(
        f"Window samples:    {live_audio.WINDOW_SAMPLES}"
    )

    print(
        f"Smoothing history: {live_audio.SMOOTHING_HISTORY}"
    )

    print(
        f"Benchmark windows: {NUM_WINDOWS}"
    )

    print()

    print(
        "Starting microphone benchmark..."
    )

    print(
        "Speak normally near the microphone."
    )

    print(
        f"The benchmark will stop automatically after "
        f"{NUM_WINDOWS} windows."
    )

    print()

    rows = []

    raw_predictions = []
    majority_predictions = []
    probability_predictions = []

    processing_times = []

    overflow_count = 0
    nonfinite_count = 0

    for window_number in range(
        1,
        NUM_WINDOWS + 1
    ):

        start = time.perf_counter()

        audio, overflowed = live_audio.capture_window(
            device=None
        )

        if overflowed:
            overflow_count += 1

        audio = np.asarray(
            audio,
            dtype=np.float32
        ).reshape(-1)

        finite_audio = bool(
            np.isfinite(audio).all()
        )

        if not finite_audio:
            nonfinite_count += 1

        feature = live_audio.extract_log_mel(
            audio
        )

        feature = live_audio.normalize_log_mel(
            feature,
            mean,
            std
        )

        raw_index, raw_probabilities = (
            live_audio.predict(
                model,
                feature
            )
        )

        raw_index = int(raw_index)

        raw_class = live_audio.CLASSES[
            raw_index
        ]

        raw_confidence = float(
            raw_probabilities[raw_index]
        )

        # --------------------------------------------------
        # MAJORITY VOTING
        # --------------------------------------------------

        majority_raw = majority.update(
            raw_index
        )

        majority_index = resolve_majority_output(
            majority_raw
        )

        majority_index = max(
            0,
            min(
                majority_index,
                len(live_audio.CLASSES) - 1
            )
        )

        majority_class = live_audio.CLASSES[
            majority_index
        ]

        # --------------------------------------------------
        # PROBABILITY MOVING AVERAGE
        # --------------------------------------------------

        probability_raw = probability.update(
            np.asarray(
                raw_probabilities,
                dtype=np.float32
            )
        )

        (
            probability_index,
            probability_vector
        ) = resolve_probability_output(
            probability_raw
        )

        probability_index = max(
            0,
            min(
                probability_index,
                len(live_audio.CLASSES) - 1
            )
        )

        probability_class = live_audio.CLASSES[
            probability_index
        ]

        if probability_vector is not None:

            probability_vector = np.asarray(
                probability_vector,
                dtype=np.float32
            ).reshape(-1)

            if (
                probability_vector.size
                == len(live_audio.CLASSES)
            ):

                probability_confidence = float(
                    probability_vector[
                        probability_index
                    ]
                )

            else:

                probability_confidence = float(
                    np.max(probability_vector)
                )

        else:

            probability_confidence = float(
                raw_probabilities[
                    probability_index
                ]
            )

        # --------------------------------------------------
        # TIMING
        # --------------------------------------------------

        processing_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        processing_times.append(
            processing_ms
        )

        # --------------------------------------------------
        # RMS
        # --------------------------------------------------

        rms = float(
            np.sqrt(
                np.mean(
                    np.square(audio)
                )
            )
        )

        # --------------------------------------------------
        # STORE RESULTS
        # --------------------------------------------------

        raw_predictions.append(
            raw_class
        )

        majority_predictions.append(
            majority_class
        )

        probability_predictions.append(
            probability_class
        )

        rows.append(
            {
                "window": window_number,
                "raw_prediction": raw_class,
                "raw_confidence": raw_confidence,
                "majority_prediction": majority_class,
                "probability_average_prediction":
                    probability_class,
                "probability_average_confidence":
                    probability_confidence,
                "processing_ms": processing_ms,
                "rms": rms,
                "overflowed": bool(
                    overflowed
                ),
                "finite_audio": finite_audio,
            }
        )

        print(
            f"Window {window_number:04d} | "
            f"Raw={raw_class:<6} "
            f"Conf={raw_confidence:.3f} | "
            f"Majority={majority_class:<6} | "
            f"ProbAvg={probability_class:<6} "
            f"Conf={probability_confidence:.3f} | "
            f"Processing={processing_ms:.2f} ms | "
            f"RMS={rms:.6f}"
        )

    # ======================================================
    # OUTPUT PATHS
    # ======================================================

    csv_path = (
        RESULTS_DIR
        / "P6_temporal_smoothing_benchmark.csv"
    )

    report_path = (
        REPORTS_DIR
        / "P6_TEMPORAL_SMOOTHING_BENCHMARK.md"
    )

    # ======================================================
    # CSV
    # ======================================================

    fieldnames = (
        list(rows[0].keys())
        if rows
        else []
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    # ======================================================
    # DISTRIBUTIONS
    # ======================================================

    raw_distribution = class_distribution(
        raw_predictions
    )

    majority_distribution = class_distribution(
        majority_predictions
    )

    probability_distribution = class_distribution(
        probability_predictions
    )

    # ======================================================
    # REPORT
    # ======================================================

    report_lines = [

        "# P6 Temporal Smoothing Benchmark",

        "",

        "## Purpose",

        "",

        (
            "This benchmark verifies runtime operation "
            "of the P6 temporal smoothing mechanisms "
            "using live microphone input. It does not "
            "measure classification accuracy because "
            "live microphone windows have no ground-truth "
            "labels."
        ),

        "",

        "## Configuration",

        "",

        f"- Sample rate: {live_audio.SAMPLE_RATE} Hz",

        f"- Channels: {live_audio.CHANNELS}",

        (
            f"- Window duration: "
            f"{live_audio.WINDOW_SECONDS} s"
        ),

        (
            f"- Window samples: "
            f"{live_audio.WINDOW_SAMPLES}"
        ),

        (
            f"- Smoothing history: "
            f"{live_audio.SMOOTHING_HISTORY}"
        ),

        f"- Benchmark windows: {NUM_WINDOWS}",

        "- Model: P5 augmented CNN",

        "- UNKNOWN threshold: not configured",

        "",

        "## Runtime Results",

        "",

        f"- Windows processed: {len(rows)}",

        f"- Overflow windows: {overflow_count}",

        (
            f"- Non-finite windows: "
            f"{nonfinite_count}"
        ),

        (
            f"- Minimum processing time: "
            f"{min(processing_times):.3f} ms"
        ),

        (
            f"- Median processing time: "
            f"{np.median(processing_times):.3f} ms"
        ),

        (
            f"- Average processing time: "
            f"{np.mean(processing_times):.3f} ms"
        ),

        (
            f"- Maximum processing time: "
            f"{max(processing_times):.3f} ms"
        ),

        "",

        "## Prediction Transitions",

        "",

        (
            f"- Raw prediction transitions: "
            f"{transition_count(raw_predictions)}"
        ),

        (
            f"- Majority-vote transitions: "
            f"{transition_count(majority_predictions)}"
        ),

        (
            f"- Probability-average transitions: "
            f"{transition_count(probability_predictions)}"
        ),

        "",

        "## Raw Prediction Distribution",

        "",
    ]

    for class_name in live_audio.CLASSES:

        report_lines.append(
            f"- {class_name}: "
            f"{raw_distribution.get(class_name, 0)}"
        )

    report_lines.extend(
        [
            "",
            "## Majority-Vote Distribution",
            "",
        ]
    )

    for class_name in live_audio.CLASSES:

        report_lines.append(
            f"- {class_name}: "
            f"{majority_distribution.get(class_name, 0)}"
        )

    report_lines.extend(
        [
            "",
            "## Probability-Average Distribution",
            "",
        ]
    )

    for class_name in live_audio.CLASSES:

        report_lines.append(
            f"- {class_name}: "
            f"{probability_distribution.get(class_name, 0)}"
        )

    report_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "The benchmark compares raw predictions "
                "with majority-vote and probability-average "
                "outputs. Prediction-transition counts are "
                "used as a runtime stability indicator. "
                "A lower transition count may indicate "
                "greater temporal stability, but it does "
                "not establish classification accuracy "
                "or generalization performance."
            ),
            "",
            "## Limitations",
            "",
            (
                "- Live microphone windows have no "
                "ground-truth labels."
            ),
            (
                "- Classification accuracy is not "
                "measured here."
            ),
            (
                "- Confidence values are not calibrated "
                "probabilities."
            ),
            (
                "- UNKNOWN threshold selection remains "
                "pending validation."
            ),
            (
                "- Physical Raspberry Pi validation has "
                "not been performed."
            ),
            (
                "- Results are from the current Windows "
                "development environment."
            ),
            "",
            "## Status",
            "",
            (
                "P6 temporal smoothing runtime verification "
                "completed."
            ),
            "",
        ]
    )

    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8"
    )

    # ======================================================
    # FINAL SUMMARY
    # ======================================================

    print()

    print("=" * 70)

    print(
        "P6 TEMPORAL SMOOTHING BENCHMARK COMPLETE"
    )

    print("=" * 70)

    print()

    print(
        f"Windows processed: {len(rows)}"
    )

    print(
        f"Raw transitions: "
        f"{transition_count(raw_predictions)}"
    )

    print(
        f"Majority transitions: "
        f"{transition_count(majority_predictions)}"
    )

    print(
        f"Probability-average transitions: "
        f"{transition_count(probability_predictions)}"
    )

    print(
        f"Median processing: "
        f"{np.median(processing_times):.3f} ms"
    )

    print(
        f"Average processing: "
        f"{np.mean(processing_times):.3f} ms"
    )

    print(
        f"Overflow windows: "
        f"{overflow_count}"
    )

    print(
        f"Non-finite windows: "
        f"{nonfinite_count}"
    )

    print()

    print(
        f"CSV saved: {csv_path}"
    )

    print(
        f"Report saved: {report_path}"
    )


if __name__ == "__main__":
    main()