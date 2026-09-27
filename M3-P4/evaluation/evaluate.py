import sys
import csv
import time
from pathlib import Path

import numpy as np
import torch
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "preprocessing"))
sys.path.insert(0, str(ROOT / "models" / "cnn"))

import stft
from model import SpeechEnhancementCNN


DATASET_DIR = Path(
    r"C:\Users\Tanisha\Downloads\M3-P2_dataset\dataset"
)

MIXTURE_DIR = DATASET_DIR / "mixtures"
SPEECH_DIR = DATASET_DIR / "speech"

MODEL_PATH = ROOT / "model" / "trained_model.pth"
OUTPUT_DIR = ROOT / "evaluation" / "enhanced_audio"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = ROOT / "evaluation" / "evaluation_results.csv"


def rms(x):
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def snr_db(clean, signal):
    n = min(len(clean), len(signal))
    clean = clean[:n]
    signal = signal[:n]
    noise = clean - signal
    return 20.0 * np.log10(
        rms(clean) / (rms(noise) + 1e-12)
    )


print("M3-P4 CNN EVALUATION")
print("====================")

device = torch.device("cpu")
print("Device:", device)

model = SpeechEnhancementCNN().to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.eval()

print("Model loaded:", MODEL_PATH)
print(
    "Parameters:",
    sum(p.numel() for p in model.parameters())
)

records = []

metadata = DATASET_DIR / "metadata" / "mixture_metadata.csv"

with open(metadata, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        mixture_path = MIXTURE_DIR / row["mixture_file"]
        speech_path = SPEECH_DIR / row["speech_file"]

        if mixture_path.exists() and speech_path.exists():
            records.append(row)

print("Evaluation records:", len(records))

records = records[:20]

results = []

for i, row in enumerate(records, 1):

    mixture_path = MIXTURE_DIR / row["mixture_file"]
    speech_path = SPEECH_DIR / row["speech_file"]

    mixture, sr = sf.read(
        str(mixture_path),
        dtype="float32"
    )

    clean, clean_sr = sf.read(
        str(speech_path),
        dtype="float32"
    )

    if mixture.ndim > 1:
        mixture = mixture.mean(axis=1)

    if clean.ndim > 1:
        clean = clean.mean(axis=1)

    mixture_stft = stft.stft(mixture)

    mixture_mag, mixture_phase = stft.magnitude_phase(
        mixture_stft
    )

    log_mag = np.log1p(mixture_mag)

    x = torch.as_tensor(
        log_mag,
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0)

    start = time.perf_counter()

    with torch.no_grad():
        mask = model(x)

    inference_time = time.perf_counter() - start

    mask = (
        mask.squeeze(0)
        .squeeze(0)
        .cpu()
        .numpy()
    )

    enhanced_mag = mixture_mag * mask

    enhanced = stft.reconstruct(
        enhanced_mag,
        mixture_phase,
        length=len(mixture)
    )

    enhanced = np.asarray(
        enhanced,
        dtype=np.float32
    )

    output_name = (
        Path(row["mixture_file"]).stem +
        "_enhanced.wav"
    )

    output_path = OUTPUT_DIR / output_name

    sf.write(
        str(output_path),
        enhanced,
        sr
    )

    n = min(len(clean), len(mixture), len(enhanced))

    noisy_snr = snr_db(
        clean[:n],
        mixture[:n]
    )

    enhanced_snr = snr_db(
        clean[:n],
        enhanced[:n]
    )

    improvement = enhanced_snr - noisy_snr

    results.append({
        "mixture_file": row["mixture_file"],
        "snr_db": row["snr_db"],
        "noisy_snr_db": round(noisy_snr, 4),
        "enhanced_snr_db": round(enhanced_snr, 4),
        "snr_improvement_db": round(improvement, 4),
        "inference_time_sec": round(
            inference_time,
            6
        ),
        "input_duration_sec": round(
            len(mixture) / sr,
            4
        ),
        "output_file": output_name
    })

    print(
        f"{i:02d}/{len(records)} | "
        f"SNR {row['snr_db']} dB | "
        f"Improvement: {improvement:.3f} dB | "
        f"Time: {inference_time:.4f} s"
    )


with open(
    RESULTS_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=results[0].keys()
    )

    writer.writeheader()
    writer.writerows(results)


improvements = [
    r["snr_improvement_db"]
    for r in results
]

times = [
    r["inference_time_sec"]
    for r in results
]

print()
print("EVALUATION COMPLETE")
print("===================")
print("Samples evaluated:", len(results))
print(
    "Average SNR improvement:",
    round(float(np.mean(improvements)), 4),
    "dB"
)
print(
    "Median SNR improvement:",
    round(float(np.median(improvements)), 4),
    "dB"
)
print(
    "Average inference time:",
    round(float(np.mean(times)), 6),
    "sec"
)
print("Enhanced audio:", OUTPUT_DIR)
print("Results:", RESULTS_PATH)
