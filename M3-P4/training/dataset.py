import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREPROCESSING_DIR = ROOT / "preprocessing"
sys.path.insert(0, str(PREPROCESSING_DIR))

import torch
from torch.utils.data import Dataset
import soundfile as sf
import numpy as np
import csv
import stft


class SpeechEnhancementDataset(Dataset):

    SEGMENT_FRAMES = 100

    def __init__(self, mixture_dir, speech_dir):

        self.mixture_dir = Path(mixture_dir)
        self.speech_dir = Path(speech_dir)

        metadata = self.mixture_dir.parent / "metadata" / "mixture_metadata.csv"

        self.records = []

        with open(metadata, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                mixture = self.mixture_dir / row["mixture_file"]
                speech = self.speech_dir / row["speech_file"]

                if mixture.exists() and speech.exists():
                    self.records.append((mixture, speech))

        print("Dataset records:", len(self.records))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):

        mixture_path, speech_path = self.records[index]

        mixture, mixture_sr = sf.read(
            str(mixture_path),
            dtype="float32"
        )

        speech, speech_sr = sf.read(
            str(speech_path),
            dtype="float32"
        )

        if mixture.ndim > 1:
            mixture = mixture.mean(axis=1)

        if speech.ndim > 1:
            speech = speech.mean(axis=1)

        mixture_stft = stft.stft(mixture)
        speech_stft = stft.stft(speech)

        mixture_mag, _ = stft.magnitude_phase(mixture_stft)
        speech_mag, _ = stft.magnitude_phase(speech_stft)

        mixture_mag = torch.as_tensor(
            mixture_mag, dtype=torch.float32
        )
        speech_mag = torch.as_tensor(
            speech_mag, dtype=torch.float32
        )

        # Ideal Ratio Mask
        irm = speech_mag / (mixture_mag + 1e-8)
        irm = torch.clamp(irm, 0.0, 1.0)

        # Log magnitude input
        mixture_mag = torch.log1p(mixture_mag)

        # Fixed-size temporal segment for batching
        frames = self.SEGMENT_FRAMES

        if mixture_mag.shape[1] >= frames:
            max_start = mixture_mag.shape[1] - frames

            # Deterministic segment selection
            start = (index * frames) % (max_start + 1)

            mixture_mag = mixture_mag[:, start:start + frames]
            irm = irm[:, start:start + frames]

        else:
            pad_width = frames - mixture_mag.shape[1]

            mixture_mag = torch.nn.functional.pad(
                mixture_mag,
                (0, pad_width)
            )

            irm = torch.nn.functional.pad(
                irm,
                (0, pad_width)
            )

        # CNN expects [channel, frequency, time]
        mixture_mag = mixture_mag.unsqueeze(0)
        irm = irm.unsqueeze(0)

        return mixture_mag, irm


if __name__ == "__main__":

    dataset = SpeechEnhancementDataset(
        str(Path(os.environ.get("M3_P2_DATASET_DIR", ROOT / "dataset" / "data")) / "mixtures"),
        str(Path(os.environ.get("M3_P2_DATASET_DIR", ROOT / "dataset" / "data")) / "speech")
    )

    print("Number of samples:", len(dataset))

    x, y = dataset[0]

    print("Input shape :", tuple(x.shape))
    print("Target shape:", tuple(y.shape))
    print("Input dtype :", x.dtype)
    print("Target dtype:", y.dtype)
    print("Input min/max:", float(x.min()), float(x.max()))
    print("Target min/max:", float(y.min()), float(y.max()))





