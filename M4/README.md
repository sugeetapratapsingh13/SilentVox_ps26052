# SilentVox M4 — Audio Front-End

This module is the **complete software/digital M4 audio-front-end implementation** for SilentVox. It is reproducible and software-validated while physical microphones, a final USB audio interface, and Raspberry Pi 5 hardware remain unavailable for physical validation.

## Architecture

```text
REF/ERR/SPCH virtual streams
        ↓
Virtual audio-interface model
        ↓
SilentVox processing boundary
        ↓
OUT / virtual playback
```

The future physical path is: microphones -> USB audio interface -> Raspberry Pi 5 -> SilentVox -> audio-interface output -> speaker/headset.

## Requirements

- Python 3.10+ recommended
- Windows, Linux, or macOS for the software validation package
- Python packages listed in `requirements.txt`
- No physical hardware required for the software validation stage

## Quick start from this directory

```bash
python -m pip install -r requirements.txt
python -m analysis.run_analysis
python -m analysis.robustness
python benchmark.py
python -m unittest discover -s tests -p "test_*.py"
```

For the complete validation and evidence regeneration, use:

```bash
python run_all.py
```

The runner clears generated evidence first so stale output cannot make validation appear successful. It then runs analysis, robustness, block benchmark, unit tests, checks required evidence files, and writes `results/validation_summary.csv`.

## Expected evidence

- `virtual_mics/*.wav` — reproducible synthetic/generated digital streams.
- `plots/*_waveform.png` — waveform evidence.
- `plots/*_fft.png` — FFT evidence.
- `plots/*_spectrogram.png` — spectrogram evidence.
- `results/audio_metrics.csv` — RMS/peak/crest-factor/digital SNR evidence.
- `results/block_latency_benchmark.csv` — 256/512/1024 software benchmark.
- `results/robustness_tests.csv` — delay, gain, noise, quantization, clipping, frequency-limitation and DAC-style simulation evidence.
- `results/output_simulation.wav` — software playback/output-path demonstration.
- `results/validation_summary.csv` — machine-readable validation status.

## Block sizes and latency wording

At 16 kHz:

- 256 samples = 16 ms frame duration
- 512 samples = 32 ms frame duration
- 1024 samples = 64 ms frame duration

**Frame duration is not total system latency.** The benchmark separately reports software processing time. Physical microphone/interface/acoustic latency is not measured here.

## Browser demo

`browser_demo/` demonstrates browser-side block processing with an `AudioWorkletProcessor`. It is not Raspberry Pi emulation and does not measure Pi performance, USB-interface latency, microphone latency, acoustic latency, or physical channel synchronization.

Serve locally from this M4 directory:

```bash
python -m http.server 8000
```

Then open `http://localhost:8000/browser_demo/`.

## Claim control

The module uses these categories consistently:

- **DATASHEET SPECIFICATION** — manufacturer-published candidate specification.
- **SOFTWARE/DIGITAL RESULT** — measured by this repository's software.
- **SIMULATION** — deliberately modeled impairment or virtual hardware behavior.
- **PHYSICAL VALIDATION PENDING** — requires real microphones/interface/Pi/headset.

The module never treats a virtual WAV as a microphone measurement, a software benchmark as Pi latency, or a block duration as total system latency.

## Integration

See:

- `M4_AUDIO_FRONT_END_SPEC.md`
- `CHANNEL_MAP.md`
- `M4_TO_M1_AUDIO_CONSTRAINTS.md`
- `M4_TO_M6_INTERFACE_SPEC.md`
- `VALIDATION_STATUS.md`
- `MICROPHONE_COMPARISON.csv` / `MIC_SELECTION.md`
- `AUDIO_INTERFACE_COMPARISON.csv` / `AUDIO_INTERFACE_SELECTION.md`

## Physical transition

The logical contract remains REF / ERR / SPCH. Future physical interface channels can replace the virtual WAV sources without changing that downstream logical contract. Final physical microphone/interface selection, Pi 5 USB enumeration, ALSA/PipeWire mapping, physical synchronization, physical latency, long-duration hardware stability, and headset acoustic validation remain pending.
