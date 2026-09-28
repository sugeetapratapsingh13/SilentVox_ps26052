import pandas as pd
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MANIFEST_PATH = PROJECT_ROOT / "manifests" / "split_manifest.csv"
DATASET_DIR = PROJECT_ROOT / "dataset"

TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
STORED_SUBTYPE = "PCM_16"
PREPROCESSING_VERSION = "P2-v0.1"

REQUIRED_COLUMNS = [
    "original_filename",
    "audio_path",
    "source_recording_id",
    "target_class",
    "split",
]


def process_audio(input_path, output_path):
    audio, sample_rate = sf.read(
        input_path,
        dtype="float32",
        always_2d=False
    )

    audio = np.asarray(audio, dtype=np.float32)

    if audio.size == 0:
        raise ValueError("EMPTY_AUDIO")

    if not np.isfinite(audio).all():
        raise ValueError(
            "AUDIO_CONTAINS_NAN_OR_INFINITE_VALUES"
        )

    if audio.ndim == 1:
        mono_audio = audio

    elif audio.ndim == 2:
        mono_audio = np.mean(
            audio,
            axis=1,
            dtype=np.float32
        )

    else:
        raise ValueError(
            f"UNSUPPORTED_AUDIO_DIMENSIONS: {audio.ndim}"
        )

    if sample_rate != TARGET_SAMPLE_RATE:
        gcd = np.gcd(
            int(sample_rate),
            TARGET_SAMPLE_RATE
        )

        up = TARGET_SAMPLE_RATE // gcd
        down = int(sample_rate) // gcd

        mono_audio = resample_poly(
            mono_audio,
            up,
            down
        ).astype(np.float32)

    if not np.isfinite(mono_audio).all():
        raise ValueError(
            "PROCESSED_AUDIO_CONTAINS_NAN_OR_INFINITE_VALUES"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    sf.write(
        output_path,
        mono_audio,
        TARGET_SAMPLE_RATE,
        subtype=STORED_SUBTYPE
    )


def main():

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Split manifest not found: {MANIFEST_PATH}"
        )

    df = pd.read_csv(MANIFEST_PATH)

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required manifest columns: {missing_columns}"
        )

    if df.empty:
        raise ValueError(
            "Split manifest is empty."
        )

    if df["split"].isna().any():
        raise ValueError(
            "Manifest contains missing split values."
        )

    if (df["split"] == "UNASSIGNED").any():
        raise ValueError(
            "Manifest still contains UNASSIGNED rows."
        )

    processed_paths = []
    failed_files = []

    total = len(df)

    for index, row in df.iterrows():

        input_path = PROJECT_ROOT / str(
            row["audio_path"]
        )

        target_class = str(
            row["target_class"]
        )

        split = str(
            row["split"]
        )

        source_id = str(
            row["source_recording_id"]
        )

        original_filename = str(
            row["original_filename"]
        )

        original_stem = Path(
            original_filename
        ).stem

        output_filename = (
            f"ESC50_{source_id}_"
            f"{original_stem}_"
            f"{target_class}.wav"
        )

        output_path = (
            DATASET_DIR
            / split
            / target_class
            / output_filename
        )

        try:

            if not input_path.exists():
                raise FileNotFoundError(
                    f"Missing input file: {input_path}"
                )

            process_audio(
                input_path,
                output_path
            )

            relative_output = (
                output_path
                .relative_to(PROJECT_ROOT)
                .as_posix()
            )

            processed_paths.append(
                relative_output
            )

            print(
                f"[{index + 1}/{total}] "
                f"Processed: {original_filename}"
            )

        except Exception as error:

            print(
                f"ERROR: {original_filename} -> {error}"
            )

            failed_files.append(
                original_filename
            )

            processed_paths.append("")

    df["processed_audio_path"] = processed_paths
    df["preprocessing_version"] = (
        PREPROCESSING_VERSION
    )

    df.to_csv(
        MANIFEST_PATH,
        index=False
    )

    print()
    print("=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"Total files: {total}")
    print(
        f"Successfully processed: "
        f"{total - len(failed_files)}"
    )
    print(
        f"Failed: {len(failed_files)}"
    )
    print(
        f"Sample rate: "
        f"{TARGET_SAMPLE_RATE} Hz"
    )
    print(
        f"Channels: {TARGET_CHANNELS}"
    )
    print(
        f"Stored format: {STORED_SUBTYPE}"
    )
    print(
        "Duration handling: source duration preserved"
    )
    print("Cropping: none")
    print("Padding: none")
    print(
        f"Version: {PREPROCESSING_VERSION}"
    )

    if failed_files:

        print()
        print("FAILED FILES:")

        for filename in failed_files:
            print(filename)

        raise RuntimeError(
            f"{len(failed_files)} files failed."
        )


if __name__ == "__main__":
    main()