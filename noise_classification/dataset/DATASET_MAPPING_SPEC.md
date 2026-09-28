# SilentVox — Dataset Mapping Specification

## Purpose

This document defines how original dataset labels are mapped to the provisional SilentVox taxonomy.

Dataset labels must never be renamed into SilentVox classes without documenting the mapping decision.

## Mapping Types

### DIRECT
The original dataset label closely matches the SilentVox class definition.

### PARTIAL
The original label represents only part of the SilentVox class definition.

### INDIRECT
The recording contains related acoustic information but does not directly represent the SilentVox class.

### NONE
There is insufficient evidence to map the original label to the SilentVox class.

## Required Mapping Fields

- dataset
- original_filename
- original_label
- SilentVox_class
- mapping_type
- source_recording_id
- inclusion_decision
- mapping_reason
- taxonomy_version
- notes

## Mapping Rules

1. Never map a dataset label solely because its name sounds similar to a SilentVox class.
2. Preserve the original dataset label.
3. Never overwrite original dataset metadata.
4. Record every mapping decision explicitly.
5. Use PARTIAL when a source label covers only part of a SilentVox class.
6. Use NONE when there is no defensible mapping.
7. Do not force every dataset into every SilentVox class.
8. Do not merge classes merely to increase sample count.
9. Do not duplicate recordings across classes.
10. Maintain source-level identifiers for leakage prevention.
11. Dataset mapping must remain traceable to the original dataset.
12. Final taxonomy changes must be propagated consistently to P2-P6.

## Current Taxonomy Version

P1-v0.1-investigate

The taxonomy remains provisional until dataset coverage and experimental validation are completed.

## Current Candidate Classes

ENGINE
ROTOR
VEHICLE
MACHINERY
WIND
RAIN
SIREN
ALARM
CROWD
IMPACT/IMPULSIVE
SPEECH

MIXED and UNKNOWN are treated as special mechanisms rather than ordinary single-label classes.

## Important Principle

Do not force the data to fit the taxonomy.

Validate the taxonomy against the data, validate the model against unseen data, and validate the final system against the deployment domain.
