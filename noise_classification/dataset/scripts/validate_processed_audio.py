import pandas as pd
import numpy as np
import soundfile as sf
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MANIFEST_PATH = (
    PROJECT_ROOT
    / "manifests"
    / "split_manifest.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "statistics"
    / "processed_audio_validation_report.csv"
)


# ============================================================
# EXPECTED CONFIGURATION
# ============================================================

EXPECTED_SAMPLE_RATE = 16000
EXPECTED_CHANNELS = 1
EXPECTED_SUBTYPE = "PCM_16"


# ============================================================
# VALIDATE ONE PROCESSED FILE
# ============================================================

def validate_processed_audio(processed_audio_path):

    result = {
        "exists": False,
        "readable": False,
        "sample_rate": None,
        "channels": None,
        "frames": None,
        "duration_seconds": None,
        "subtype": None,
        "has_nan": None,
        "has_inf": None,
        "peak_amplitude": None,
        "rms": None,
        "status": "INVALID",
        "reason": "",
    }

    path = PROJECT_ROOT / processed_audio_path

    if not path.exists():

        result["reason"] = "FILE_NOT_FOUND"
        return result

    result["exists"] = True

    try:

        info = sf.info(path)

        result["sample_rate"] = info.samplerate
        result["channels"] = info.channels
        result["frames"] = info.frames
        result["duration_seconds"] = (
            info.frames / info.samplerate
        )
        result["subtype"] = info.subtype

        audio, _ = sf.read(
            path,
            dtype="float32",
            always_2d=False
        )

        result["readable"] = True

        audio = np.asarray(
            audio,
            dtype=np.float32
        )

        if audio.ndim == 2:
            audio_for_stats = np.mean(
                audio,
                axis=1
            )
        else:
            audio_for_stats = audio

        result["has_nan"] = bool(
            np.isnan(audio_for_stats).any()
        )

        result["has_inf"] = bool(
            np.isinf(audio_for_stats).any()
        )

        if len(audio_for_stats) > 0:

            result["peak_amplitude"] = float(
                np.max(
                    np.abs(audio_for_stats)
                )
            )

            result["rms"] = float(
                np.sqrt(
                    np.mean(
                        audio_for_stats ** 2
                    )
                )
            )

        # ----------------------------------------------------
        # Specification checks
        # ----------------------------------------------------

        if info.samplerate != EXPECTED_SAMPLE_RATE:

            result["reason"] = (
                f"INVALID_SAMPLE_RATE: "
                f"{info.samplerate}"
            )

        elif info.channels != EXPECTED_CHANNELS:

            result["reason"] = (
                f"INVALID_CHANNELS: "
                f"{info.channels}"
            )

        elif info.subtype != EXPECTED_SUBTYPE:

            result["reason"] = (
                f"INVALID_SUBTYPE: "
                f"{info.subtype}"
            )

        elif len(audio_for_stats) == 0:

            result["reason"] = "EMPTY_AUDIO"

        elif result["has_nan"]:

            result["reason"] = "NAN_VALUES"

        elif result["has_inf"]:

            result["reason"] = "INFINITE_VALUES"

        elif result["duration_seconds"] <= 0:

            result["reason"] = "ZERO_DURATION"

        else:

            result["status"] = "VALID"
            result["reason"] = "OK"

    except Exception as error:

        result["reason"] = (
            f"READ_ERROR: {error}"
        )

    return result


# ============================================================
# LOAD MANIFEST
# ============================================================

if not MANIFEST_PATH.exists():
    raise FileNotFoundError(
        f"Manifest not found: {MANIFEST_PATH}"
    )

df = pd.read_csv(MANIFEST_PATH)

if "processed_audio_path" not in df.columns:
    raise ValueError(
        "processed_audio_path not found. "
        "Run preprocess.py first."
    )


# ============================================================
# VALIDATE ALL PROCESSED AUDIO
# ============================================================

results = []

for _, row in df.iterrows():

    validation = validate_processed_audio(
        row["processed_audio_path"]
    )

    validation.update({
        "source_recording_id":
            row.get(
                "source_recording_id",
                ""
            ),

        "original_filename":
            row.get(
                "original_filename",
                ""
            ),

        "target_class":
            row.get(
                "target_class",
                ""
            ),

        "split":
            row.get(
                "split",
                ""
            ),

        "processed_audio_path":
            row["processed_audio_path"],
    })

    results.append(validation)


# ============================================================
# SAVE REPORT
# ============================================================

report = pd.DataFrame(results)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

report.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n========================================")
print("PROCESSED AUDIO VALIDATION")
print("========================================")

print(
    report["status"]
    .value_counts()
)

print(
    f"\nTotal files checked: {len(report)}"
)

print(
    f"Valid files: "
    f"{(report['status'] == 'VALID').sum()}"
)

print(
    f"Invalid files: "
    f"{(report['status'] != 'VALID').sum()}"
)

print("\nSample rates:")

print(
    report["sample_rate"]
    .value_counts()
    .sort_index()
)

print("\nChannels:")

print(
    report["channels"]
    .value_counts()
    .sort_index()
)

print("\nWAV subtypes:")

print(
    report["subtype"]
    .value_counts()
)

print(
    f"\nSaved to: {OUTPUT_PATH}"
)

if (report["status"] != "VALID").any():

    raise RuntimeError(
        "Processed audio validation found invalid files."
    )

print("\nPROCESSED AUDIO VALIDATION PASSED.")