import numpy as np
import soundfile as sf
import torch


SAMPLE_RATE = 16000
N_FFT = 512
WIN_LENGTH = 512
HOP_LENGTH = 128


def load_audio(path):
    audio, sr = sf.read(str(path), dtype="float32")

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != SAMPLE_RATE:
        raise ValueError(
            f"Expected {SAMPLE_RATE} Hz audio, got {sr} Hz: {path}"
        )

    return audio


def stft(audio):
    waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32))

    window = torch.hann_window(WIN_LENGTH)

    return torch.stft(
        waveform,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window,
        return_complex=True,
    )


def istft(spectrum, length=None):
    window = torch.hann_window(WIN_LENGTH)

    return torch.istft(
        spectrum,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window,
        length=length,
    )


def magnitude_phase(spectrum):
    magnitude = torch.abs(spectrum)
    phase = torch.angle(spectrum)

    return magnitude, phase


def reconstruct(magnitude, phase, length=None):
    spectrum = magnitude * torch.exp(1j * phase)

    return istft(spectrum, length=length)


if __name__ == "__main__":
    print("M3-P4 STFT preprocessing")
    print(f"Sample rate : {SAMPLE_RATE} Hz")
    print(f"FFT size    : {N_FFT}")
    print(f"Window size : {WIN_LENGTH}")
    print(f"Hop size    : {HOP_LENGTH}")
    print(f"Frequency bins: {N_FFT // 2 + 1}")
