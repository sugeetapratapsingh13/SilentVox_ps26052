# P6 Handoff Note

## Purpose

This repository is the consolidated ANC integration produced by P6.

## Primary ANC algorithm

NLMS is the designated primary implementation.

Run:

    python ANC/algorithms/NLMS.py

## Other algorithms

LMS:

    python ANC/algorithms/LMS.py

FxLMS:

    python ANC/algorithms/FxLMS.py

The P5 FxLMS implementation is also preserved as:

    python ANC/algorithms/FxLMS_P5.py

## Simulation

Run:

    python ANC/simulation/run_simulation.py

This generates multi-condition simulation results and plots.

## Secondary path

The current FxLMS secondary path is simulated.

It consists of an 8-sample delay, 0.7 attenuation and a five-tap FIR model.

No physical secondary-path measurement or hardware closed-loop validation is claimed.

## Repository structure

theory/
    ANC theory, equations and diagrams

research/
    literature and research-gap material

algorithms/
    LMS, NLMS and FxLMS implementations

simulation/
    ANC simulation components

secondary_path/
    secondary-path analysis and supporting material

experiments/
    benchmark and simulation results

plots/
    convergence plots, spectrograms and waveforms

benchmarks/
    benchmark scripts and real-time measurements

configuration/
    final configuration and runtime requirements

## Reproduction

Install the required Python dependencies, then run the algorithm or simulation commands documented in ANC/README.md.

## Integration rule

The final integration branch should be reviewed before merging into main.

Generated caches, virtual environments, large generated audio files, credentials and local-machine paths should not be committed.
