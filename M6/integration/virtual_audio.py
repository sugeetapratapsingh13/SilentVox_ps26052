"""M6 virtual audio interface.

Integration boundary between the validated M4 virtual audio layer and
the future SilentVox runtime.

This module performs software/virtual integration only. It does not
claim physical Raspberry Pi or physical audio-interface validation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Mapping
import sys

import numpy as np


# Allow both:
#   python -m M6.integration.virtual_audio
# and:
#   python M6\integration\virtual_audio.py
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from M4.audio_pipeline.pipeline import AudioConfig, VirtualAudioInterface
from M4.audio_pipeline.io import load_audio, validate_audio


@dataclass(frozen=True)
class M6AudioConfig:
    """M6 audio contract derived from the M4 -> M6 interface specification."""

    sample_rate: int = 16_000
    block_size: int = 512
    input_channels: tuple = ("REF", "ERR", "SPCH")
    output_channels: int = 1
    virtual_bit_depth: int = 16


class M6VirtualAudio:
    """M6 wrapper around the existing M4 virtual audio interface."""

    def __init__(self, config: M6AudioConfig | None = None):
        self.config = config or M6AudioConfig()

        if self.config.sample_rate != 16_000:
            raise ValueError("M6 currently requires a 16 kHz software boundary.")

        if self.config.block_size not in (256, 512, 1024):
            raise ValueError(
                "M6 block size must be one of the M4-tested values: 256, 512, 1024."
            )

        if len(self.config.input_channels) != 3:
            raise ValueError("M6 requires exactly three logical input channels.")

        self.interface = VirtualAudioInterface(
            config=AudioConfig(
                input_sample_rate=self.config.sample_rate,
                output_sample_rate=self.config.sample_rate,
                block_size=self.config.block_size,
                channels=self.config.input_channels,
                input_bit_depth=self.config.virtual_bit_depth,
                output_bit_depth=self.config.virtual_bit_depth,
            )
        )

    @property
    def frame_duration_ms(self) -> float:
        """Return block/frame duration, not total system latency."""
        return (
            self.config.block_size
            / self.config.sample_rate
            * 1000.0
        )

    def validate_streams(self, streams: Mapping[str, np.ndarray]) -> int:
        """Validate REF/ERR/SPCH streams and return common sample count."""
        missing = [
            channel
            for channel in self.config.input_channels
            if channel not in streams
        ]
        if missing:
            raise KeyError(f"Missing logical channels: {missing}")

        arrays = {}
        for channel in self.config.input_channels:
            array = np.asarray(streams[channel], dtype=np.float32)

            if array.ndim != 1:
                raise ValueError(
                    f"{channel} must be a mono 1-D stream; got shape {array.shape}"
                )

            if not np.isfinite(array).all():
                raise ValueError(f"{channel} contains non-finite values.")

            arrays[channel] = array

        lengths = {len(array) for array in arrays.values()}
        if len(lengths) != 1:
            raise ValueError(
                f"Logical input streams must have equal lengths; got {sorted(lengths)}"
            )

        n_samples = next(iter(lengths))

        if n_samples < self.config.block_size:
            raise ValueError(
                f"Input contains {n_samples} samples, fewer than one "
                f"{self.config.block_size}-sample block."
            )

        return n_samples

    def process_streams(
        self,
        streams: Mapping[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Apply the existing M4 virtual interface model to M6 inputs."""
        self.validate_streams(streams)

        processed = self.interface.process(dict(streams))

        for channel in self.config.input_channels:
            processed[channel] = np.asarray(
                processed[channel], dtype=np.float32
            )

        return processed

    def iter_blocks(
        self,
        streams: Mapping[str, np.ndarray],
    ) -> Iterator[tuple[int, Dict[str, np.ndarray]]]:
        """Yield complete fixed-size REF/ERR/SPCH processing blocks."""
        processed = self.process_streams(streams)
        n_samples = self.validate_streams(processed)
        block_size = self.config.block_size

        for start in range(0, n_samples - block_size + 1, block_size):
            yield (
                start,
                {
                    channel: processed[channel][start:start + block_size]
                    for channel in self.config.input_channels
                },
            )

    def load_wav_inputs(
        self,
        reference_path: Path,
        error_path: Path,
        speech_path: Path,
    ) -> Dict[str, np.ndarray]:
        """Load the three M4 virtual microphone WAV files."""
        paths = {
            "REF": Path(reference_path),
            "ERR": Path(error_path),
            "SPCH": Path(speech_path),
        }

        streams = {}

        for channel, path in paths.items():
            if not path.exists():
                raise FileNotFoundError(f"{channel} WAV not found: {path}")

            audio, sr = load_audio(
                path,
                target_sr=None,
                mono=True,
            )

            validate_audio(
                audio,
                sr,
                expected_sr=self.config.sample_rate,
                expected_channels=1,
            )

            streams[channel] = audio[:, 0].astype(np.float32)

        self.validate_streams(streams)
        return streams

    def simulate_output(
        self,
        output: np.ndarray,
        gain: float = 1.0,
    ) -> np.ndarray:
        """Apply the M4 virtual output boundary."""
        output = np.asarray(output, dtype=np.float32)

        if output.ndim != 1:
            raise ValueError(
                f"Output must be a mono 1-D stream; got shape {output.shape}"
            )

        if not np.isfinite(output).all():
            raise ValueError("Output contains non-finite values.")

        return self.interface.simulate_output(output, gain=gain)

    def contract_summary(self) -> dict:
        """Return machine-readable M6 software contract information."""
        return {
            "sample_rate_hz": self.config.sample_rate,
            "block_size_samples": self.config.block_size,
            "frame_duration_ms": self.frame_duration_ms,
            "input_channels": list(self.config.input_channels),
            "output_channels": self.config.output_channels,
            "internal_dtype": "float32",
            "virtual_input_format": "signed 16-bit PCM WAV",
            "physical_validation": "PENDING",
        }


def default_virtual_inputs() -> Dict[str, np.ndarray]:
    """Load the existing M4 virtual microphone inputs."""
    virtual_mics = REPO_ROOT / "M4" / "virtual_mics"

    return M6VirtualAudio().load_wav_inputs(
        virtual_mics / "reference.wav",
        virtual_mics / "error.wav",
        virtual_mics / "speech.wav",
    )


if __name__ == "__main__":
    audio = M6VirtualAudio()
    streams = default_virtual_inputs()

    print("M6 virtual audio interface: PASS")
    print(f"Sample rate: {audio.config.sample_rate} Hz")
    print(f"Block size: {audio.config.block_size} samples")
    print(f"Frame duration: {audio.frame_duration_ms:.3f} ms")
    print(f"Channels: {' | '.join(audio.config.input_channels)}")
    print(f"Samples: {len(streams['REF'])}")

    first_start, first_block = next(audio.iter_blocks(streams))
    print(f"First block start: {first_start}")
    print(
        "First block shapes:",
        {channel: block.shape for channel, block in first_block.items()},
    )
