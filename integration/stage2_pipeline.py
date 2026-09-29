from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from integration.m3_adapter import create_adapter as create_m3_adapter
from M6.adapters.m1_adapter import create_adapter as create_m1_adapter
from noise_classification.deployment.scripts.live_audio import (
    LiveProcessor,
    P5_MODEL,
    SAMPLE_RATE,
    load_model,
    load_normalization_stats,
)


def rms_db(audio: np.ndarray) -> float:
    audio = np.asarray(audio, dtype=np.float64)

    rms = float(
        np.sqrt(
            np.mean(
                np.square(audio)
            )
        )
    )

    return 20.0 * np.log10(
        max(rms, 1e-12)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SilentVox Stage-2 M2 -> M1 -> M3 pipeline"
    )

    parser.add_argument(
        "input_wav",
        type=Path,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "./integration/stage2_output.wav"
        ),
    )

    args = parser.parse_args()

    print("=" * 70)
    print("SILENTVOX STAGE-2 M2 -> M1 -> M3 PIPELINE")
    print("=" * 70)

    # ------------------------------------------------------------
    # LOAD AUDIO
    # ------------------------------------------------------------

    audio, sample_rate = librosa.load(
        args.input_wav,
        sr=SAMPLE_RATE,
        mono=True,
    )

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    if audio.size == 0:
        raise RuntimeError(
            "Input WAV is empty."
        )

    if not np.isfinite(audio).all():
        raise RuntimeError(
            "Input WAV contains non-finite values."
        )

    print()
    print("INPUT")
    print("-" * 70)
    print(f"File:             {args.input_wav}")
    print(f"Sample rate:      {sample_rate} Hz")
    print(f"Samples:          {len(audio)}")
    print(
        f"Duration:         "
        f"{len(audio) / sample_rate:.3f} sec"
    )
    print(
        f"Input RMS:        "
        f"{rms_db(audio):.3f} dBFS"
    )

    # ------------------------------------------------------------
    # M2
    # ------------------------------------------------------------

    print()
    print("M2 CLASSIFICATION")
    print("-" * 70)

    m2_start = time.perf_counter()

    mean, std = load_normalization_stats()

    model = load_model(
        P5_MODEL
    )

    processor = LiveProcessor(
        model=model,
        mean=mean,
        std=std,
    )

    # M2 expects a 2-second window.
    window_samples = 32000

    if len(audio) < window_samples:
        m2_audio = np.pad(
            audio,
            (0, window_samples - len(audio)),
        )
    else:
        m2_audio = audio[
            :window_samples
        ]

    m2_result = processor.process(
        m2_audio
    )

    m2_time = (
        time.perf_counter()
        - m2_start
    )

    print(
        f"Raw label:        "
        f"{m2_result['raw_label']}"
    )

    print(
        f"Confidence:       "
        f"{m2_result['confidence']:.4f}"
    )

    print(
        f"Majority label:   "
        f"{m2_result['majority_label']}"
    )

    print(
        f"Probability label:"
        f" {m2_result['probability_label']}"
    )

    print(
        f"M2 time:          "
        f"{m2_time:.6f} sec"
    )

    # ------------------------------------------------------------
    # M1 FxLMS
    # ------------------------------------------------------------

    print()
    print("M1 FxLMS")
    print("-" * 70)

    m1_start = time.perf_counter()

    m1 = create_m1_adapter()

    m1_result = m1.process(
        audio
    )

    m1_time = (
        time.perf_counter()
        - m1_start
    )

    residual = np.asarray(
        m1_result.residual,
        dtype=np.float32,
    )

    if not np.isfinite(residual).all():
        raise RuntimeError(
            "M1 produced non-finite residual."
        )

    print(
        f"Algorithm:        FxLMS"
    )

    print(
        f"Filter length:    "
        f"{m1_result.filter_length}"
    )

    print(
        f"Learning rate:    "
        f"{m1_result.learning_rate}"
    )

    print(
        f"Secondary path:   "
        f"{m1_result.secondary_path_mode}"
    )

    print(
        f"Residual RMS:     "
        f"{rms_db(residual):.3f} dBFS"
    )

    print(
        f"M1 time:          "
        f"{m1_time:.6f} sec"
    )

    # ------------------------------------------------------------
    # M3
    # ------------------------------------------------------------

    print()
    print("M3 ENHANCEMENT")
    print("-" * 70)

    m3 = create_m3_adapter()

    m3_start = time.perf_counter()

    m3_result = m3.process(
        residual,
        SAMPLE_RATE,
    )

    m3_time = (
        time.perf_counter()
        - m3_start
    )

    enhanced = np.asarray(
        m3_result.enhanced_audio,
        dtype=np.float32,
    )

    if not np.isfinite(enhanced).all():
        raise RuntimeError(
            "M3 produced non-finite output."
        )

    print(
        f"Output samples:   "
        f"{len(enhanced)}"
    )

    print(
        f"Output RMS:       "
        f"{rms_db(enhanced):.3f} dBFS"
    )

    print(
        f"M3 inference:     "
        f"{m3_result.inference_time_sec:.6f} sec"
    )

    print(
        f"M3 wall time:     "
        f"{m3_time:.6f} sec"
    )

    # ------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sf.write(
        args.output,
        enhanced,
        SAMPLE_RATE,
    )

    total_time = (
        m2_time
        + m1_time
        + m3_time
    )

    print()
    print("=" * 70)
    print("STAGE-2 RESULT")
    print("=" * 70)

    print(
        f"M2:               "
        f"{m2_result['raw_label']} "
        f"({m2_result['confidence']:.4f})"
    )

    print(
        f"M1 residual:      "
        f"{len(residual)} samples"
    )

    print(
        f"M3 output:        "
        f"{len(enhanced)} samples"
    )

    print(
        f"Total stage time: "
        f"{total_time:.6f} sec"
    )

    print(
        f"Output WAV:       "
        f"{args.output}"
    )

    print()
    print("STAGE-2 M2 -> M1 -> M3 PIPELINE PASS")


if __name__ == "__main__":
    main()
