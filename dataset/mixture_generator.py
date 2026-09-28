"""
mixture_generator.py
---------------------
M3-P2: builds Clean Speech + Environmental Noise -> Noisy Speech mixtures
for speech-enhancement training/evaluation, across a fixed SNR sweep, with
leakage prevented on BOTH axes (speaker AND noise source) per the spec's
"Critical" requirement.

This script is self-contained (stdlib + numpy + scipy + pandas only) since
M3-P2's boundary is explicitly "doesn't decide which enhancement algorithm
is best" -- it has no business depending on any particular model framework.

-----------------------------------------------------------------------
Why a DUAL split (not just a speaker split)?
-----------------------------------------------------------------------
Member 2's classifier spec calls out leakage prevention at the recording
level for noise classes. M3-P2 has the same requirement on the SPEECH side
(no speaker's voice in both train and test) but *also* needs it on the
NOISE side, or a model could learn to recognize a specific noise recording
rather than the noise class in general. This script partitions speaker IDs
and noise source files into disjoint train/val/test pools independently,
and only ever mixes speech from pool X with noise from pool X -- so neither
a speaker nor a specific noise recording can leak across splits.

-----------------------------------------------------------------------
Pipeline
-----------------------------------------------------------------------
dataset/metadata/speech_metadata.csv  (speaker_id, gender, speaking_style,
                                        speaking_rate, recording_condition,
                                        filename)
dataset/metadata/noise_metadata.csv   (noise_class, filename)
dataset/speech/*.wav
dataset/noise/*.wav
        |
        v
  assign_group_splits() on speaker_id and on noise filename (independently)
        |
        v
  for each (speech clip, split) x (one noise file per class in that split)
    x (each SNR in SNR_LEVELS_DB):
        mix_at_snr() -> dataset/mixtures/<split>/<noise_class>/<name>.wav
        |
        v
dataset_manifest.csv   (one row per mixture, full metadata)
snr_manifest.csv       (counts per split x noise_class x snr_db, for a
                         quick balance check)
"""

from __future__ import annotations

import argparse
import csv
import wave
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

SAMPLE_RATE = 16000

# Exactly the range specified: -10, -5, 0, +5, +10, +15, +20 dB.
# See DATASET_STRUCTURE.md "SNR range justification" for why this range
# (not a continuous sweep) was chosen for SilentVox's environment.
SNR_LEVELS_DB = [-10, -5, 0, 5, 10, 15, 20]

NOISE_CLASSES = ["engine", "rotor", "machinery", "wind", "broadband", "impulsive"]

VAL_FRAC = 0.15
TEST_FRAC = 0.15
SEED = 42


# --------------------------------------------------------------------------
# Audio I/O (stdlib + scipy only -- same approach as the P4 classifier
# pipeline, kept independent here since M3-P2 is a separate deliverable)
# --------------------------------------------------------------------------

def load_wav_mono(path: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load a PCM WAV file as float32 mono in [-1, 1], resampled to target_sr."""
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sr = wf.getframerate()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sampwidth == 1:
        audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"Unsupported sample width {sampwidth} bytes in {path}")

    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    if sr != target_sr:
        g = gcd(sr, target_sr)
        audio = resample_poly(audio, target_sr // g, sr // g).astype(np.float32)

    return audio


def save_wav_mono(path: str, audio: np.ndarray, sr: int = SAMPLE_RATE):
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


# --------------------------------------------------------------------------
# SNR mixing (same equation as the P4 classifier's augmentation.py, kept
# consistent across the project)
# --------------------------------------------------------------------------

def _match_length(audio: np.ndarray, target_len: int, rng: np.random.Generator) -> np.ndarray:
    if len(audio) == target_len:
        return audio
    if len(audio) > target_len:
        start = int(rng.integers(0, len(audio) - target_len + 1))
        return audio[start:start + target_len]
    reps = int(np.ceil(target_len / len(audio)))
    return np.tile(audio, reps)[:target_len]


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float,
                rng: np.random.Generator | None = None) -> np.ndarray:
    """y = speech + scale*noise, scaled so the mixture hits the target SNR."""
    rng = rng or np.random.default_rng()
    speech = speech.astype(np.float32)
    noise = _match_length(noise.astype(np.float32), len(speech), rng)

    speech_power = np.mean(speech ** 2) + 1e-12
    noise_power = np.mean(noise ** 2) + 1e-12
    target_noise_power = speech_power / (10.0 ** (snr_db / 10.0))
    scale = np.sqrt(target_noise_power / noise_power)
    mixed = speech + scale * noise
    return np.clip(mixed, -1.0, 1.0).astype(np.float32)


def achieved_snr_db(speech: np.ndarray, noise_component: np.ndarray) -> float:
    """For validation/testing: recompute the SNR actually present in a mix."""
    sp = np.mean(speech ** 2) + 1e-12
    np_ = np.mean(noise_component ** 2) + 1e-12
    return float(10.0 * np.log10(sp / np_))


# --------------------------------------------------------------------------
# Dual leakage-free split (speaker IDs and noise source files independently)
# --------------------------------------------------------------------------

def assign_group_splits(ids: list[str], val_frac: float = VAL_FRAC,
                         test_frac: float = TEST_FRAC, seed: int = SEED) -> dict[str, str]:
    """Randomly partitions a list of unique IDs into disjoint
    train/val/test pools. Used independently for speaker_id and for noise
    source filename -- see module docstring for why both are needed."""
    unique_ids = sorted(set(ids))  # sorted first for determinism before shuffling
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_ids)

    n = len(shuffled)
    n_test = max(1, int(round(n * test_frac))) if n >= 3 else 0
    n_val = max(1, int(round(n * val_frac))) if n >= 3 else 0
    n_test = min(n_test, n - 1) if n > 1 else 0
    n_val = min(n_val, n - n_test - 1) if n - n_test > 1 else 0

    test_ids = set(shuffled[:n_test])
    val_ids = set(shuffled[n_test:n_test + n_val])
    train_ids = set(shuffled[n_test + n_val:])

    split_map = {}
    for i in train_ids:
        split_map[i] = "train"
    for i in val_ids:
        split_map[i] = "val"
    for i in test_ids:
        split_map[i] = "test"
    return split_map


# --------------------------------------------------------------------------
# Manifest-driven mixture generation
# --------------------------------------------------------------------------

def load_speech_metadata(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"speaker_id", "gender", "speaking_style", "speaking_rate",
                "recording_condition", "filename"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"speech_metadata.csv missing columns: {missing}")
    return df


def load_noise_metadata(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"noise_class", "filename"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"noise_metadata.csv missing columns: {missing}")
    return df


def generate_mixtures(
    speech_metadata_csv: str,
    noise_metadata_csv: str,
    speech_dir: str,
    noise_dir: str,
    mixtures_dir: str,
    manifest_dir: str,
    snr_levels: list[int] = SNR_LEVELS_DB,
    noise_per_speech_per_class: int = 1,
    seed: int = SEED,
    write_audio: bool = True,
):
    speech_meta = load_speech_metadata(speech_metadata_csv)
    noise_meta = load_noise_metadata(noise_metadata_csv)

    speaker_split = assign_group_splits(speech_meta["speaker_id"].tolist(), seed=seed)
    noise_split = assign_group_splits(noise_meta["filename"].tolist(), seed=seed)

    speech_meta = speech_meta.copy()
    noise_meta = noise_meta.copy()
    speech_meta["split"] = speech_meta["speaker_id"].map(speaker_split)
    noise_meta["split"] = noise_meta["filename"].map(noise_split)

    # sanity: dual leakage check
    for split in ["train", "val", "test"]:
        spk = set(speech_meta.loc[speech_meta.split == split, "speaker_id"])
        other_spk = set(speech_meta.loc[speech_meta.split != split, "speaker_id"])
        assert not (spk & other_spk), f"speaker leakage touching split={split}"
        nz = set(noise_meta.loc[noise_meta.split == split, "filename"])
        other_nz = set(noise_meta.loc[noise_meta.split != split, "filename"])
        assert not (nz & other_nz), f"noise-source leakage touching split={split}"

    rng = np.random.default_rng(seed)
    mixtures_dir = Path(mixtures_dir)
    manifest_dir = Path(manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    mixture_id = 0

    for split in ["train", "val", "test"]:
        split_speech = speech_meta[speech_meta.split == split]
        split_noise = noise_meta[noise_meta.split == split]

        for _, srow in split_speech.iterrows():
            speech_path = Path(speech_dir) / srow["filename"]
            if write_audio and not speech_path.exists():
                print(f"[skip] missing speech file {speech_path}")
                continue
            speech_audio = load_wav_mono(str(speech_path)) if write_audio else None

            for noise_class in NOISE_CLASSES:
                class_noise = split_noise[split_noise.noise_class == noise_class]
                if len(class_noise) == 0:
                    continue  # no noise of this class in this split yet
                picks = class_noise.sample(
                    n=min(noise_per_speech_per_class, len(class_noise)),
                    random_state=int(rng.integers(0, 1_000_000)),
                )
                for _, nrow in picks.iterrows():
                    noise_path = Path(noise_dir) / nrow["filename"]
                    if write_audio and not noise_path.exists():
                        print(f"[skip] missing noise file {noise_path}")
                        continue
                    noise_audio = load_wav_mono(str(noise_path)) if write_audio else None

                    for snr in snr_levels:
                        mixture_id += 1
                        speaker_stem = Path(srow["filename"]).stem
                        noise_stem = Path(nrow["filename"]).stem
                        out_name = f"{speaker_stem}__{noise_stem}__snr{snr}.wav"
                        out_dir = mixtures_dir / split / noise_class
                        out_path = out_dir / out_name

                        if write_audio:
                            out_dir.mkdir(parents=True, exist_ok=True)
                            mixed = mix_at_snr(speech_audio, noise_audio, snr, rng)
                            save_wav_mono(str(out_path), mixed)

                        rows.append({
                            "mixture_id": mixture_id,
                            "speech_file": srow["filename"],
                            "speaker_id": srow["speaker_id"],
                            "gender": srow["gender"],
                            "speaking_style": srow["speaking_style"],
                            "speaking_rate": srow["speaking_rate"],
                            "recording_condition": srow["recording_condition"],
                            "noise_file": nrow["filename"],
                            "noise_class": noise_class,
                            "snr_db": snr,
                            "split": split,
                            "output_path": str(out_path.relative_to(mixtures_dir.parent))
                                            if write_audio else "",
                        })

    manifest_df = pd.DataFrame(rows)
    manifest_path = manifest_dir / "dataset_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    print(f"Wrote {len(manifest_df)} mixture rows to {manifest_path}")

    if len(manifest_df) > 0:
        snr_summary = (
            manifest_df.groupby(["split", "noise_class", "snr_db"])
            .size()
            .reset_index(name="count")
        )
    else:
        snr_summary = pd.DataFrame(columns=["split", "noise_class", "snr_db", "count"])
    snr_manifest_path = manifest_dir / "snr_manifest.csv"
    snr_summary.to_csv(snr_manifest_path, index=False)
    print(f"Wrote SNR/class balance summary to {snr_manifest_path}")

    return manifest_df, snr_summary


def main():
    parser = argparse.ArgumentParser(description="Generate speech+noise mixtures for M3-P2.")
    parser.add_argument("--speech_metadata", default="dataset/metadata/speech_metadata.csv")
    parser.add_argument("--noise_metadata", default="dataset/metadata/noise_metadata.csv")
    parser.add_argument("--speech_dir", default="dataset/speech")
    parser.add_argument("--noise_dir", default="dataset/noise")
    parser.add_argument("--mixtures_dir", default="dataset/mixtures")
    parser.add_argument("--manifest_dir", default="dataset/metadata")
    parser.add_argument("--noise_per_speech_per_class", type=int, default=1)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    generate_mixtures(
        args.speech_metadata, args.noise_metadata,
        args.speech_dir, args.noise_dir,
        args.mixtures_dir, args.manifest_dir,
        noise_per_speech_per_class=args.noise_per_speech_per_class,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
