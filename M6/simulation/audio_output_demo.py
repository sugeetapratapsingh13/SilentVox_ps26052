from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from M6.safety_simulation import SafetyController


SAMPLE_RATE = 16000
DURATION_SECONDS = 4
SAMPLES = SAMPLE_RATE * DURATION_SECONDS

OUT_DIR = (
    REPO_ROOT
    / "M6"
    / "evidence"
    / "simulation"
)


def rms(signal: np.ndarray) -> float:
    return float(
        np.sqrt(
            np.mean(
                np.square(
                    signal.astype(np.float64)
                )
            )
        )
    )


def write_wav(
    path: Path,
    audio: np.ndarray,
) -> None:

    pcm = np.clip(
        audio,
        -1.0,
        1.0,
    )

    pcm = (
        pcm * 32767.0
    ).astype(np.int16)

    with wave.open(
        str(path),
        "wb",
    ) as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(
            pcm.tobytes()
        )


def main() -> None:

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    t = np.arange(
        SAMPLES,
        dtype=np.float64,
    ) / SAMPLE_RATE

    speech = (
        0.16
        * np.sin(
            2.0
            * np.pi
            * 220.0
            * t
        )
        + 0.08
        * np.sin(
            2.0
            * np.pi
            * 440.0
            * t
        )
    )

    rng = np.random.default_rng(42)

    noise = (
        0.12
        * rng.normal(size=SAMPLES)
    )

    microphone = (
        speech + noise
    ).astype(np.float32)

    # Simulation-only processing proxy.
    output = (
        speech + 0.35 * noise
    ).astype(np.float32)

    controller = SafetyController()

    result = controller.process(
        microphone,
        lambda _: output,
    )

    if result.fault is not None:
        raise RuntimeError(
            result.fault
        )

    output_audio = result.output

    input_rms = rms(microphone)
    output_rms = rms(output_audio)

    residual_noise = (
        output_audio
        - speech.astype(np.float32)
    )

    input_noise_rms = rms(noise)
    residual_noise_rms = rms(
        residual_noise
    )

    noise_attenuation_db = (
        20.0
        * np.log10(
            max(input_noise_rms, 1e-12)
            /
            max(
                residual_noise_rms,
                1e-12,
            )
        )
    )

    input_path = (
        OUT_DIR
        / "speech_plus_noise_input.wav"
    )

    output_path = (
        OUT_DIR
        / "silentvox_output.wav"
    )

    report_path = (
        OUT_DIR
        / "audio_output_demo.json"
    )

    write_wav(
        input_path,
        microphone,
    )

    write_wav(
        output_path,
        output_audio,
    )

    report = {
        "status": "SIMULATED",
        "physical_validation": False,
        "sample_rate_hz": SAMPLE_RATE,
        "duration_seconds": DURATION_SECONDS,
        "samples": SAMPLES,
        "input_rms": input_rms,
        "output_rms": output_rms,
        "input_noise_rms": input_noise_rms,
        "residual_noise_rms": residual_noise_rms,
        "simulated_noise_attenuation_db":
            noise_attenuation_db,
        "safety_state":
            result.state.value,
        "safety_fault":
            result.fault,
        "input_wav":
            str(input_path),
        "output_wav":
            str(output_path),
        "note":
            "Software-only demonstration using "
            "synthetic speech and noise. Processing "
            "stage is a simulation proxy, not "
            "physical ANC validation.",
    }

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "=== SilentVox M6 Audio Output Demo ==="
    )
    print(
        f"Input RMS:          {input_rms:.6f}"
    )
    print(
        f"Output RMS:         {output_rms:.6f}"
    )
    print(
        f"Input noise RMS:    {input_noise_rms:.6f}"
    )
    print(
        f"Residual noise RMS: {residual_noise_rms:.6f}"
    )
    print(
        "Noise attenuation:  "
        f"{noise_attenuation_db:.3f} dB"
    )
    print(
        f"Safety state:       {result.state.value}"
    )
    print(
        f"Input WAV:          {input_path}"
    )
    print(
        f"Output WAV:         {output_path}"
    )
    print(
        f"Report:             {report_path}"
    )


if __name__ == "__main__":
    main()
