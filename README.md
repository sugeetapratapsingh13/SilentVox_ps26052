# SilentVox ANC

SilentVox is an active noise cancellation (ANC) research and simulation project.

## ANC implementation

The project contains:

- LMS adaptive filtering
- NLMS adaptive filtering
- FxLMS adaptive filtering
- Simulated secondary-path modelling
- Multi-condition ANC simulation
- Parameter sweeps
- Real-time processing benchmarks

## Main algorithms

The primary ANC implementation is NLMS.

Run:

    python ANC/algorithms/NLMS.py

LMS:

    python ANC/algorithms/LMS.py

FxLMS:

    python ANC/algorithms/FxLMS.py

A second FxLMS implementation from the P5 work is preserved as:

    python ANC/algorithms/FxLMS_P5.py

## Simulation

Run the multi-condition simulation:

    python ANC/simulation/run_simulation.py

The simulation evaluates:

- stationary noise
- non-stationary noise
- periodic mechanical noise
- impulsive/transient noise
- speech plus noise

Results are written to:

    ANC/experiments/

Plots are stored under:

    ANC/plots/

## Secondary path

The current secondary-path model is SIMULATED.

Parameters:

- Sampling rate: 16 kHz
- Delay: 8 samples
- Delay: 0.5 ms
- Attenuation: 0.7
- FIR coefficients: [0.20, 0.30, 0.15, 0.05]

Run the secondary-path analysis with:

    python ANC/secondary_path/SECONDARY_PATH_ANALYSIS.py

No physical secondary-path measurement or closed-loop hardware validation is claimed in this repository.

## Benchmarks

Benchmark scripts are located in:

    ANC/benchmarks/

The repository includes parameter-sweep and real-time benchmark results.

## Research

Research and literature material is located in:

    ANC/research/

Theory and equations are located in:

    ANC/theory/

## Reproducibility

Python dependencies should be installed before running the scripts.

The repository intentionally does not include virtual environments, Python cache files, generated WAV files, API keys, passwords, or other local-machine artifacts.

## Status

The ANC algorithms and simulation pipeline have been executed locally as part of the P6 integration and benchmark process.

FxLMS has been executed with the simulated secondary path.

Physical ANC hardware validation is not claimed unless separately documented.
