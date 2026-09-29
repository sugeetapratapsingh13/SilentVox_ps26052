from __future__ import annotations

import csv
import importlib.util
import os
import inspect
import json
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

M4_DIR = ROOT / "M4" / "virtual_mics"

M2_SCRIPT = (
    ROOT
    / "noise_classification"
    / "deployment"
    / "scripts"
    / "live_audio.py"
)

M3_MODEL_FILE = (
    ROOT
    / "M3-P4"
    / "models"
    / "cnn"
    / "model.py"
)

M3_CHECKPOINT = (
    ROOT
    / "M3-P4"
    / "model"
    / "trained_model.pth"
)

M3_STFT_FILE = (
    ROOT
    / "M3-P4"
    / "preprocessing"
    / "stft.py"
)

M5_ZIP = ROOT / "SilentVox_M5_GITHUB_CLEAN.zip"

OUT_DIR = (
    ROOT
    / "M6"
    / "evidence"
    / "integration"
)

OUT_WAV = OUT_DIR / "final_integrated_output.wav"
REPORT_JSON = OUT_DIR / "final_integration_report.json"
TRIAL_CSV = OUT_DIR / "final_integration_trials.csv"


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_RATE = 16000
BLOCK_SIZE = int(os.environ.get("SILENTVOX_BLOCK_SIZE", "512"))
M2_WINDOW = 32000

M1_FILTER_LENGTH = 64
M1_LEARNING_RATE = 1e-5

MAX_OUTPUT = 0.999


# ============================================================
# UTILITIES
# ============================================================

def load_module(module_name: str, file_path: Path):
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file not found: {file_path}"
        )

    spec = importlib.util.spec_from_file_location(
        module_name,
        str(file_path),
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load module: {file_path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


def rms(signal: np.ndarray) -> float:
    signal = np.asarray(signal, dtype=np.float64)

    if signal.size == 0:
        return 0.0

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

    input_rms = max(
        rms(input_signal),
        1e-12,
    )

    output_rms = max(
        rms(output_signal),
        1e-12,
    )

    return float(
        20.0
        * np.log10(
            input_rms / output_rms
        )
    )


def load_audio(path: Path) -> np.ndarray:

    if not path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {path}"
        )

    audio, sr = sf.read(
        str(path),
        dtype="float64",
    )

    if sr != SAMPLE_RATE:
        raise ValueError(
            f"{path} is {sr} Hz; "
            f"expected {SAMPLE_RATE} Hz."
        )

    if audio.ndim > 1:
        audio = np.mean(
            audio,
            axis=1,
        )

    audio = np.asarray(
        audio,
        dtype=np.float64,
    )

    if not np.isfinite(audio).all():
        raise ValueError(
            f"Non-finite samples detected: {path}"
        )

    return audio


def pad_to_length(
    signal: np.ndarray,
    length: int,
) -> np.ndarray:

    signal = np.asarray(
        signal,
        dtype=np.float64,
    )

    if len(signal) >= length:
        return signal[:length]

    output = np.zeros(
        length,
        dtype=np.float64,
    )

    output[:len(signal)] = signal

    return output


def clip_safe(
    signal: np.ndarray,
) -> np.ndarray:

    signal = np.asarray(
        signal,
        dtype=np.float64,
    )

    signal = np.nan_to_num(
        signal,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return np.clip(
        signal,
        -MAX_OUTPUT,
        MAX_OUTPUT,
    )


# ============================================================
# M1 + M5
# ============================================================

def load_m1_controller():

    stateful_module = load_module(
        "silentvox_m6_stateful",
        ROOT
        / "M6"
        / "simulation"
        / "stateful_streaming.py",
    )

    m5_adapter_module = load_module(
        "silentvox_m5_adapter",
        ROOT
        / "M6"
        / "adapters"
        / "m5_adapter.py",
    )

    if not M5_ZIP.exists():
        raise FileNotFoundError(
            "M5 package not found: "
            f"{M5_ZIP}"
        )

    M5Adapter = (
        m5_adapter_module.M5Adapter
    )

    with tempfile.TemporaryDirectory(
        prefix="silentvox_m6_m5_"
    ) as temp_dir:

        temp_root = Path(temp_dir)

        with zipfile.ZipFile(
            M5_ZIP,
            "r",
        ) as archive:

            candidates = [
                name
                for name in archive.namelist()
                if name.endswith(
                    "secondary_path_estimate.npy"
                )
            ]

            if not candidates:
                raise FileNotFoundError(
                    "secondary_path_estimate.npy "
                    "was not found inside "
                    f"{M5_ZIP}"
                )

            selected = candidates[0]

            archive.extract(
                selected,
                temp_root,
            )

            model_path = (
                temp_root / selected
            )

            adapter = M5Adapter(
                model_path
            )

            secondary_path = adapter.load()

            controller = (
                stateful_module.StatefulFxLMS(
                    secondary_path,
                    sample_rate=SAMPLE_RATE,
                    filter_length=M1_FILTER_LENGTH,
                    learning_rate=M1_LEARNING_RATE,
                )
            )

            return controller, secondary_path


# ============================================================
# M2
# ============================================================

def load_m2():

    module = load_module(
        "silentvox_m2_live",
        M2_SCRIPT,
    )

    if not hasattr(
        module,
        "LiveProcessor",
    ):
        raise AttributeError(
            "LiveProcessor was not found "
            "inside live_audio.py"
        )

    LiveProcessor = (
        module.LiveProcessor
    )

    if hasattr(module, "validate_paths"):
        module.validate_paths()

    if not hasattr(module, "load_model"):
        raise AttributeError(
            "M2 live_audio.py does not expose load_model()"
        )

    if not hasattr(module, "load_normalization_stats"):
        raise AttributeError(
            "M2 live_audio.py does not expose "
            "load_normalization_stats()"
        )

    model = module.load_model()
    mean, std = module.load_normalization_stats()

    processor = LiveProcessor(
        model=model,
        mean=mean,
        std=std,
    )

    return processor


def run_m2(
    processor,
    audio: np.ndarray,
) -> dict:

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    if len(audio) < M2_WINDOW:
        audio = np.pad(
            audio,
            (
                0,
                M2_WINDOW - len(audio),
            ),
        )

    audio = audio[
        :M2_WINDOW
    ]

    result = processor.process(
        audio
    )

    if not isinstance(
        result,
        dict,
    ):
        raise RuntimeError(
            "M2 LiveProcessor.process() "
            "did not return a dictionary."
        )

    return result


# ============================================================
# M3
# ============================================================

def load_m3():

    model_module = load_module(
        "silentvox_m3_model",
        M3_MODEL_FILE,
    )

    stft_module = load_module(
        "silentvox_m3_stft",
        M3_STFT_FILE,
    )

    if not hasattr(
        model_module,
        "SpeechEnhancementCNN",
    ):
        raise AttributeError(
            "SpeechEnhancementCNN was not "
            "found in M3 model.py"
        )

    model = (
        model_module.SpeechEnhancementCNN()
    )

    checkpoint = torch.load(
        str(M3_CHECKPOINT),
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(
        checkpoint,
        dict,
    ):
        if (
            "state_dict"
            in checkpoint
        ):
            state_dict = checkpoint[
                "state_dict"
            ]

        elif (
            "model_state_dict"
            in checkpoint
        ):
            state_dict = checkpoint[
                "model_state_dict"
            ]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    if not isinstance(
        state_dict,
        dict,
    ):
        raise RuntimeError(
            "Unsupported M3 checkpoint format."
        )

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        clean_key = key

        if clean_key.startswith(
            "module."
        ):
            clean_key = clean_key[
                len("module.") :
            ]

        cleaned_state_dict[
            clean_key
        ] = value

    model.load_state_dict(
        cleaned_state_dict,
        strict=True,
    )

    model.eval()

    return model, stft_module


def enhance_m3(
    model,
    stft_module,
    audio: np.ndarray,
) -> np.ndarray:

    audio = np.asarray(
        audio,
        dtype=np.float64,
    )

    if audio.size == 0:
        return audio.copy()

    mixture = torch.from_numpy(
        audio.astype(
            np.float32
        )
    )

    mixture_stft = (
        stft_module.stft(
            mixture
        )
    )

    mixture_mag, mixture_phase = (
        stft_module.magnitude_phase(
            mixture_stft
        )
    )

    log_mag = torch.log1p(
        mixture_mag
    )

    model_input = (
        log_mag
        .unsqueeze(0)
        .unsqueeze(0)
    )

    with torch.no_grad():

        mask = model(
            model_input
        )

    mask = mask.squeeze(
        0
    ).squeeze(
        0
    )

    enhanced_mag = (
        mixture_mag * mask
    )

    enhanced = (
        stft_module.reconstruct(
            enhanced_mag,
            mixture_phase,
            length=len(audio),
        )
    )

    enhanced = (
        enhanced.detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    return pad_to_length(
        enhanced,
        len(audio),
    )


# ============================================================
# MAIN ORCHESTRATION
# ============================================================

def run():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=== SilentVox M6 Final Integration ==="
    )

    print(
        "Status: SIMULATED"
    )

    print(
        "Physical validation: NOT PERFORMED"
    )

    print()

    # --------------------------------------------------------
    # Load M4 virtual channels
    # --------------------------------------------------------

    reference = load_audio(
        M4_DIR / "reference.wav"
    )

    disturbance = load_audio(
        M4_DIR / "error.wav"
    )

    speech = load_audio(
        M4_DIR / "speech.wav"
    )

    sample_count = min(
        len(reference),
        len(disturbance),
        len(speech),
    )

    reference = reference[
        :sample_count
    ]

    disturbance = disturbance[
        :sample_count
    ]

    speech = speech[
        :sample_count
    ]

    print(
        f"M4 samples: {sample_count}"
    )

    print(
        f"M4 duration: "
        f"{sample_count / SAMPLE_RATE:.3f} s"
    )

    # --------------------------------------------------------
    # Load M1 + M5
    # --------------------------------------------------------

    print(
        "Loading M1 stateful FxLMS..."
    )

    controller, secondary_path = (
        load_m1_controller()
    )

    print(
        "M1 stateful controller: OK"
    )

    print(
        "M5 secondary path: "
        "SIMULATED / PHYSICAL MEASUREMENT PENDING"
    )

    # --------------------------------------------------------
    # Load M2
    # --------------------------------------------------------

    print(
        "Loading M2 LiveProcessor..."
    )

    m2_processor = load_m2()

    print(
        "M2 LiveProcessor: OK"
    )

    # --------------------------------------------------------
    # Load M3
    # --------------------------------------------------------

    print(
        "Loading M3 enhancement model..."
    )

    m3_model, m3_stft = load_m3()

    print(
        "M3 model: OK"
    )

    # --------------------------------------------------------
    # M2 classification
    # --------------------------------------------------------

    print(
        "Running M2 classification..."
    )

    m2_result = run_m2(
        m2_processor,
        reference,
    )

    m2_raw_label = m2_result.get(
        "raw_label"
    )

    m2_majority_label = (
        m2_result.get(
            "majority_label"
        )
    )

    m2_probability_label = (
        m2_result.get(
            "probability_label"
        )
    )

    m2_confidence = float(
        m2_result.get(
            "confidence",
            0.0,
        )
    )

    print(
        f"M2 raw label: "
        f"{m2_raw_label}"
    )

    print(
        f"M2 majority label: "
        f"{m2_majority_label}"
    )

    print(
        f"M2 probability label: "
        f"{m2_probability_label}"
    )

    print(
        f"M2 confidence: "
        f"{m2_confidence:.6f}"
    )

    # --------------------------------------------------------
    # M1 stateful streaming
    # --------------------------------------------------------

    print()
    print(
        "Running stateful M1 FxLMS..."
    )

    residual_blocks = []

    start_time = (
        time.perf_counter()
    )

    for start_idx in range(
        0,
        sample_count,
        BLOCK_SIZE,
    ):

        end_idx = min(
            start_idx + BLOCK_SIZE,
            sample_count,
        )

        reference_block = (
            reference[
                start_idx:end_idx
            ]
        )

        disturbance_block = (
            disturbance[
                start_idx:end_idx
            ]
        )

        residual_block = (
            controller.process_block(
                reference_block,
                disturbance_block,
            )
        )

        residual_blocks.append(
            residual_block
        )

    processing_seconds = (
        time.perf_counter()
        - start_time
    )

    anc_residual = np.concatenate(
        residual_blocks
    )

    anc_residual = pad_to_length(
        anc_residual,
        sample_count,
    )

    if not np.isfinite(
        anc_residual
    ).all():
        raise RuntimeError(
            "M1 produced non-finite residual."
        )

    print(
        "M1 streaming: OK"
    )

    print(
        f"M1 processing: "
        f"{processing_seconds:.4f} s"
    )

    duration_seconds = (
        sample_count
        / SAMPLE_RATE
    )

    rtf = (
        duration_seconds
        / processing_seconds
        if processing_seconds > 0
        else float("inf")
    )

    print(
        f"M1 RTF: {rtf:.2f}x"
    )

    # --------------------------------------------------------
    # M3 enhancement
    # --------------------------------------------------------

    print(
        "Running M3 enhancement..."
    )

    m3_output = enhance_m3(
        m3_model,
        m3_stft,
        anc_residual,
    )

    m3_output = clip_safe(
        m3_output
    )

    if len(m3_output) != sample_count:
        raise RuntimeError(
            "M3 output length mismatch."
        )

    if not np.isfinite(
        m3_output
    ).all():
        raise RuntimeError(
            "M3 produced non-finite output."
        )

    print(
        "M3 enhancement: OK"
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    final_output = (
        speech + m3_output
    )

    final_output = clip_safe(
        final_output
    )

    if not np.isfinite(
        final_output
    ).all():
        raise RuntimeError(
            "Final output contains "
            "non-finite samples."
        )

    sf.write(
        str(OUT_WAV),
        final_output.astype(
            np.float32
        ),
        SAMPLE_RATE,
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    input_rms = rms(
        disturbance
    )

    anc_rms = rms(
        anc_residual
    )

    output_rms = rms(
        final_output
    )

    anc_attenuation = (
        attenuation_db(
            disturbance,
            anc_residual,
        )
    )

    trial = {
        "status": "SIMULATED",
        "physical_validation": False,
        "raspberry_pi_validation": False,
        "sample_rate_hz": SAMPLE_RATE,
        "block_size": BLOCK_SIZE,
        "samples": sample_count,
        "duration_seconds": duration_seconds,
        "m1_processing_seconds": processing_seconds,
        "m1_real_time_factor": rtf,
        "input_rms": input_rms,
        "anc_residual_rms": anc_rms,
        "final_output_rms": output_rms,
        "anc_attenuation_db": anc_attenuation,
        "m1_state_samples": (
            controller.total_samples
        ),
        "m1_final_weight_norm": float(
            np.linalg.norm(
                controller.weights
            )
        ),
        "m1_finite": bool(
            np.isfinite(
                anc_residual
            ).all()
        ),
        "m3_finite": bool(
            np.isfinite(
                m3_output
            ).all()
        ),
        "final_output_finite": bool(
            np.isfinite(
                final_output
            ).all()
        ),
        "m2_raw_label": m2_raw_label,
        "m2_majority_label": (
            m2_majority_label
        ),
        "m2_probability_label": (
            m2_probability_label
        ),
        "m2_confidence": m2_confidence,
        "m5_status": (
            "SIMULATED_SECONDARY_PATH_"
            "PHYSICAL_MEASUREMENT_PENDING"
        ),
        "output_file": str(
            OUT_WAV.relative_to(ROOT)
        ),
    }

    # --------------------------------------------------------
    # Evidence JSON
    # --------------------------------------------------------

    report = {
        "project": "SilentVox",
        "module": "M6",
        "status": "SIMULATED",
        "physical_validation": False,
        "raspberry_pi_validation": False,
        "architecture": [
            "M4 virtual audio",
            "M2 noise classification/control signal",
            "M1 stateful FxLMS ANC",
            "M3 speech enhancement",
            "final output",
        ],
        "m2": {
            "raw_label": m2_raw_label,
            "majority_label": (
                m2_majority_label
            ),
            "probability_label": (
                m2_probability_label
            ),
            "confidence": m2_confidence,
        },
        "m1": {
            "algorithm": "Stateful FxLMS",
            "filter_length": M1_FILTER_LENGTH,
            "learning_rate": M1_LEARNING_RATE,
            "block_size": BLOCK_SIZE,
            "state_persistence": True,
            "samples_processed": (
                controller.total_samples
            ),
            "final_weight_norm": float(
                np.linalg.norm(
                    controller.weights
                )
            ),
            "processing_seconds": (
                processing_seconds
            ),
            "real_time_factor": rtf,
        },
        "m3": {
            "model": (
                "M3-P4/model/trained_model.pth"
            ),
            "finite_output": bool(
                np.isfinite(
                    m3_output
                ).all()
            ),
            "output_samples": len(
                m3_output
            ),
        },
        "m5": {
            "source": (
                "SilentVox_M5_GITHUB_CLEAN.zip"
            ),
            "status": (
                "SIMULATED_SECONDARY_PATH_"
                "PHYSICAL_MEASUREMENT_PENDING"
            ),
        },
        "m4": {
            "reference": (
                "M4/virtual_mics/reference.wav"
            ),
            "disturbance": (
                "M4/virtual_mics/error.wav"
            ),
            "speech": (
                "M4/virtual_mics/speech.wav"
            ),
        },
        "metrics": trial,
        "notes": [
            "Software-only integration.",
            "M4 virtual WAV files are used.",
            "M5 secondary path is simulated.",
            "No Raspberry Pi hardware was tested.",
            "No physical headset validation is claimed.",
            "M2 classification is a control signal and "
            "is not treated as a serial audio-processing stage.",
        ],
    }

    REPORT_JSON.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    with TRIAL_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                trial.keys()
            ),
        )

        writer.writeheader()
        writer.writerow(
            trial
        )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print(
        "=== FINAL INTEGRATION RESULT ==="
    )

    print(
        f"M2 classification: {m2_probability_label}"
    )

    print(
        f"M2 confidence: {m2_confidence:.4f}"
    )

    print(
        f"M1 state samples: "
        f"{controller.total_samples}"
    )

    print(
        f"M1 RTF: {rtf:.2f}x"
    )

    print(
        f"M1 residual attenuation: "
        f"{anc_attenuation:.3f} dB"
    )

    print(
        f"M3 finite: "
        f"{np.isfinite(m3_output).all()}"
    )

    print(
        f"Final output finite: "
        f"{np.isfinite(final_output).all()}"
    )

    print()
    print(
        "Evidence:"
    )

    print(
        OUT_WAV
    )

    print(
        REPORT_JSON
    )

    print(
        TRIAL_CSV
    )


if __name__ == "__main__":
    run()
