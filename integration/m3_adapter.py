from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
M3_ROOT = REPO_ROOT / "M3-P4"

sys.path.insert(0, str(M3_ROOT / "preprocessing"))
sys.path.insert(0, str(M3_ROOT / "models" / "cnn"))

import stft
from model import SpeechEnhancementCNN


M3_SAMPLE_RATE = 16000
M3_N_FFT = 512
M3_WIN_LENGTH = 512
M3_HOP_LENGTH = 128


@dataclass(frozen=True)
class M3Result:
    input_audio: np.ndarray
    enhanced_audio: np.ndarray
    inference_time_sec: float
    sample_rate: int
    model_name: str
    parameter_count: int


class M3Adapter:
    """
    Stable Stage-2 integration boundary for the existing M3 CNN.

    M3 owns:
        - STFT preprocessing
        - CNN inference
        - mask application
        - ISTFT reconstruction

    The integration layer only supplies and receives audio.
    """

    def __init__(
        self,
        model_path: str | Path = M3_ROOT / "model" / "trained_model.pth",
    ) -> None:

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"M3 model not found: {self.model_path}"
            )

        self.sample_rate = M3_SAMPLE_RATE
        self.device = torch.device("cpu")

        self.model = SpeechEnhancementCNN().to(self.device)

        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=True,
        )

        if (
            isinstance(checkpoint, dict)
            and "model_state_dict" in checkpoint
        ):
            self.model.load_state_dict(
                checkpoint["model_state_dict"]
            )
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()

        self.parameter_count = sum(
            p.numel()
            for p in self.model.parameters()
        )

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> M3Result:

        if sample_rate != self.sample_rate:
            raise ValueError(
                f"M3 expects {self.sample_rate} Hz audio, "
                f"got {sample_rate} Hz"
            )

        audio = np.asarray(
            audio,
            dtype=np.float32,
        )

        if audio.ndim != 1:
            raise ValueError(
                f"M3 expects mono 1-D audio, got shape {audio.shape}"
            )

        if audio.size == 0:
            raise ValueError("M3 input audio is empty")

        if not np.isfinite(audio).all():
            raise ValueError(
                "M3 input contains NaN or infinite values"
            )

        mixture_stft = stft.stft(audio)

        mixture_mag, mixture_phase = (
            stft.magnitude_phase(mixture_stft)
        )

        log_mag = np.log1p(
            mixture_mag.cpu().numpy()
        )

        x = torch.as_tensor(
            log_mag,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0).unsqueeze(0)

        start = time.perf_counter()

        with torch.no_grad():
            mask = self.model(x)

        inference_time = (
            time.perf_counter() - start
        )

        mask = (
            mask.squeeze(0)
            .squeeze(0)
            .cpu()
            .numpy()
        )

        if not np.isfinite(mask).all():
            raise RuntimeError(
                "M3 produced NaN or infinite mask values"
            )

        enhanced_mag = (
            mixture_mag.cpu().numpy() * mask
        )

        enhanced = stft.reconstruct(
            torch.as_tensor(enhanced_mag, dtype=torch.float32, device=self.device),
            mixture_phase,
            length=len(audio),
        )

        enhanced = np.asarray(
            enhanced.cpu().numpy(),
            dtype=np.float32,
        )

        if not np.isfinite(enhanced).all():
            raise RuntimeError(
                "M3 produced NaN or infinite output"
            )

        return M3Result(
            input_audio=audio,
            enhanced_audio=enhanced,
            inference_time_sec=inference_time,
            sample_rate=self.sample_rate,
            model_name="SpeechEnhancementCNN",
            parameter_count=self.parameter_count,
        )


def create_adapter() -> M3Adapter:
    return M3Adapter()


if __name__ == "__main__":

    adapter = create_adapter()

    test_audio = np.zeros(
        M3_SAMPLE_RATE,
        dtype=np.float32,
    )

    result = adapter.process(
        test_audio,
        M3_SAMPLE_RATE,
    )

    print("M3 integration adapter: PASS")
    print(f"Model:              {result.model_name}")
    print(f"Parameters:         {result.parameter_count}")
    print(f"Sample rate:        {result.sample_rate} Hz")
    print(f"Input samples:      {len(result.input_audio)}")
    print(f"Output samples:     {len(result.enhanced_audio)}")
    print(
        f"Inference time:     "
        f"{result.inference_time_sec:.6f} sec"
    )
    print(
        f"Output finite:      "
        f"{np.isfinite(result.enhanced_audio).all()}"
    )
