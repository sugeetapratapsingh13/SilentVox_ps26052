from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class SafetyState(str, Enum):
    MONITOR = "MONITOR"
    SAFE_OUTPUT = "SAFE_OUTPUT"
    FAULT = "FAULT"


@dataclass
class SafetyResult:
    state: SafetyState
    output: np.ndarray
    fault: str | None


class SafetyController:
    """Simulation-level M6 safety controller."""

    def safe_output(
        self,
        samples: int,
        reason: str,
    ) -> SafetyResult:

        return SafetyResult(
            state=SafetyState.SAFE_OUTPUT,
            output=np.zeros(
                samples,
                dtype=np.float32,
            ),
            fault=reason,
        )

    def process(
        self,
        audio: np.ndarray,
        processor,
    ) -> SafetyResult:

        if not isinstance(audio, np.ndarray):
            return self.safe_output(
                0,
                "INVALID_AUDIO_TYPE",
            )

        if audio.ndim != 1:
            return self.safe_output(
                audio.size,
                "INVALID_AUDIO_SHAPE",
            )

        if audio.size == 0:
            return self.safe_output(
                0,
                "EMPTY_AUDIO",
            )

        if not np.all(np.isfinite(audio)):
            return self.safe_output(
                audio.size,
                "NON_FINITE_AUDIO",
            )

        try:
            output = processor(audio)
        except Exception as exc:
            return self.safe_output(
                audio.size,
                f"PROCESSING_FAILURE:{type(exc).__name__}",
            )

        if not isinstance(output, np.ndarray):
            return self.safe_output(
                audio.size,
                "INVALID_OUTPUT_TYPE",
            )

        if output.ndim != 1:
            return self.safe_output(
                audio.size,
                "INVALID_OUTPUT_SHAPE",
            )

        if output.size != audio.size:
            return self.safe_output(
                audio.size,
                "OUTPUT_LENGTH_MISMATCH",
            )

        if not np.all(np.isfinite(output)):
            return self.safe_output(
                audio.size,
                "NON_FINITE_OUTPUT",
            )

        return SafetyResult(
            state=SafetyState.MONITOR,
            output=output.astype(
                np.float32,
                copy=False,
            ),
            fault=None,
        )
