import os
import sys
import time
import csv
import psutil
import numpy as np
import soundfile as sf

# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from LMS import lms_filter
from NLMS import nlms_filter


# ============================================================
# FILES
# ============================================================

REFERENCE_FILE = os.path.join(
    PROJECT_DIR,
    "input_audio",
    "environmental_noise.wav"
)

DESIRED_FILE = os.path.join(
    PROJECT_DIR,
    "input_audio",
    "noisy_environmental.wav"
)

OUTPUT_FILE = os.path.join(
    PROJECT_DIR,
    "REALTIME_BENCHMARK.csv"
)


# ============================================================
# ANC PARAMETERS
# ============================================================

SAMPLE_RATE = 16000

FILTER_LENGTH = 128

LMS_MU = 0.0001

NLMS_MU = 0.001

EPSILON = 1e-8

BLOCK_SIZES = [256, 512, 1024]


# ============================================================
# LOAD AUDIO
# ============================================================

def load_audio(filename):

    audio, sr = sf.read(
        filename,
        dtype="float64"
    )

    if audio.ndim > 1:
        audio = np.mean(
            audio,
            axis=1
        )

    return audio, sr


# ============================================================
# RUN ONE BLOCK
# ============================================================

def run_block(
    algorithm,
    reference,
    desired,
    filter_length,
    step_size
):

    if algorithm == "LMS":

        result = lms_filter(
            reference,
            desired,
            SAMPLE_RATE,
            filter_length,
            step_size,
            EPSILON
        )

    elif algorithm == "NLMS":

        result = nlms_filter(
            reference,
            desired,
            SAMPLE_RATE,
            filter_length,
            step_size,
            EPSILON
        )

    else:

        raise ValueError(
            "Unknown algorithm"
        )

    return result


# ============================================================
# BENCHMARK
# ============================================================

def benchmark(
    algorithm,
    reference,
    desired,
    filter_length,
    step_size,
    block_size
):

    num_blocks = min(
        len(reference),
        len(desired)
    ) // block_size

    reference = reference[
        :num_blocks * block_size
    ]

    desired = desired[
        :num_blocks * block_size
    ]

    process = psutil.Process(
        os.getpid()
    )

    process.cpu_percent(
        interval=None
    )

    block_times = []

    total_start = time.perf_counter()

    for i in range(num_blocks):

        start_index = (
            i * block_size
        )

        end_index = (
            (i + 1) * block_size
        )

        ref_block = reference[
            start_index:end_index
        ]

        desired_block = desired[
            start_index:end_index
        ]

        start = time.perf_counter()

        run_block(
            algorithm,
            ref_block,
            desired_block,
            filter_length,
            step_size
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        block_times.append(
            elapsed
        )

    total_processing_time = (
        time.perf_counter()
        - total_start
    )

    cpu_percent = process.cpu_percent(
        interval=None
    )

    memory_mb = (
        process.memory_info().rss
        / (1024 * 1024)
    )

    # ========================================================
    # TIMING
    # ========================================================

    block_duration = (
        block_size / SAMPLE_RATE
    )

    avg_processing = np.mean(
        block_times
    )

    max_processing = np.max(
        block_times
    )

    latency_ms = (
        avg_processing * 1000
    )

    max_latency_ms = (
        max_processing * 1000
    )

    realtime_factor = (
        block_duration / avg_processing
    )

    audio_duration = (
        len(reference)
        / SAMPLE_RATE
    )

    # ========================================================
    # REAL-TIME TEST
    # ========================================================

    realtime_status = (
        "PASS"
        if avg_processing < block_duration
        else "FAIL"
    )

    return {

        "algorithm":
            algorithm,

        "filter_length":
            filter_length,

        "step_size":
            step_size,

        "epsilon":
            EPSILON,

        "block_size":
            block_size,

        "sampling_rate":
            SAMPLE_RATE,

        "audio_duration_s":
            round(
                audio_duration,
                4
            ),

        "processing_time_s":
            round(
                total_processing_time,
                6
            ),

        "avg_block_processing_ms":
            round(
                avg_processing * 1000,
                6
            ),

        "max_block_processing_ms":
            round(
                max_processing * 1000,
                6
            ),

        "block_duration_ms":
            round(
                block_duration * 1000,
                6
            ),

        "latency_ms":
            round(
                latency_ms,
                6
            ),

        "max_latency_ms":
            round(
                max_latency_ms,
                6
            ),

        "real_time_factor":
            round(
                realtime_factor,
                4
            ),

        "cpu_percent":
            round(
                cpu_percent,
                2
            ),

        "memory_mb":
            round(
                memory_mb,
                2
            ),

        "status":
            realtime_status
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("             REAL-TIME ANC BENCHMARK")
    print("=" * 65)

    print("\nReference:")
    print(REFERENCE_FILE)

    print("\nDesired:")
    print(DESIRED_FILE)

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    if not os.path.exists(
        REFERENCE_FILE
    ):

        print(
            "\nERROR: Reference file not found."
        )

        return

    if not os.path.exists(
        DESIRED_FILE
    ):

        print(
            "\nERROR: Desired file not found."
        )

        return

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    reference, ref_sr = load_audio(
        REFERENCE_FILE
    )

    desired, desired_sr = load_audio(
        DESIRED_FILE
    )

    print(
        f"\nReference sample rate: "
        f"{ref_sr} Hz"
    )

    print(
        f"Desired sample rate: "
        f"{desired_sr} Hz"
    )

    print(
        f"Reference samples: "
        f"{len(reference)}"
    )

    print(
        f"Desired samples: "
        f"{len(desired)}"
    )

    if ref_sr != SAMPLE_RATE:

        print(
            "\nWARNING: Reference sample "
            "rate is not 16 kHz."
        )

    if desired_sr != SAMPLE_RATE:

        print(
            "\nWARNING: Desired sample "
            "rate is not 16 kHz."
        )

    # --------------------------------------------------------
    # MAKE SAME LENGTH
    # --------------------------------------------------------

    min_length = min(
        len(reference),
        len(desired)
    )

    reference = reference[
        :min_length
    ]

    desired = desired[
        :min_length
    ]

    print(
        f"\nBenchmark samples: "
        f"{min_length}"
    )

    print(
        f"Duration: "
        f"{min_length / SAMPLE_RATE:.2f} s"
    )

    # --------------------------------------------------------
    # CONFIGURATIONS
    # --------------------------------------------------------

    configs = [

        (
            "LMS",
            FILTER_LENGTH,
            LMS_MU
        ),

        (
            "NLMS",
            FILTER_LENGTH,
            NLMS_MU
        )
    ]

    results = []

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    for algorithm, filter_length, step_size in configs:

        for block_size in BLOCK_SIZES:

            print(
                "\n" + "-" * 65
            )

            print(
                f"Algorithm      : "
                f"{algorithm}"
            )

            print(
                f"Filter length  : "
                f"{filter_length}"
            )

            print(
                f"Step size      : "
                f"{step_size}"
            )

            print(
                f"Epsilon        : "
                f"{EPSILON}"
            )

            print(
                f"Block size     : "
                f"{block_size}"
            )

            result = benchmark(

                algorithm,

                reference,

                desired,

                filter_length,

                step_size,

                block_size
            )

            results.append(
                result
            )

            print(
                f"Processing time: "
                f"{result['processing_time_s']} s"
            )

            print(
                f"Avg latency    : "
                f"{result['latency_ms']} ms"
            )

            print(
                f"Max latency    : "
                f"{result['max_latency_ms']} ms"
            )

            print(
                f"RTF            : "
                f"{result['real_time_factor']}"
            )

            print(
                f"CPU            : "
                f"{result['cpu_percent']} %"
            )

            print(
                f"RAM            : "
                f"{result['memory_mb']} MB"
            )

            print(
                f"Status         : "
                f"{result['status']}"
            )

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    if results:

        fieldnames = list(
            results[0].keys()
        )

        with open(
            OUTPUT_FILE,
            "w",
            newline=""
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames
            )

            writer.writeheader()

            writer.writerows(
                results
            )

    print(
        "\n" + "=" * 65
    )

    print(
        "REAL-TIME BENCHMARK COMPLETE"
    )

    print(
        "=" * 65
    )

    print(
        "\nCreated:"
    )

    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":

    main()