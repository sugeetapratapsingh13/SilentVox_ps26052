# SilentVox M6

## Purpose

M6 is the integration and virtual-development layer connecting the existing SilentVox modules with the future physical Raspberry Pi 5 hardware.

M6 is responsible for:

- software/system integration
- virtual audio end-to-end testing
- runtime integration
- reliability validation
- virtual hardware and simulation
- Raspberry Pi 5 development preparation
- electronics/schematic planning
- preparation for eventual physical hardware assembly

---

## M6 Architecture

M6
│
├── Software/System Integration
│   ├── M1 adapter
│   ├── M2 adapter
│   ├── M5 adapter
│   ├── Virtual audio E2E
│   ├── Runtime
│   └── Reliability
│
├── Virtual Hardware / Simulation
│   ├── Wokwi
│   ├── PiForge / PiSorch
│   ├── Raspberry Pi 5 virtual environment
│   ├── EasyEDA
│   └── Other useful simulation platforms
│
└── Physical Hardware Preparation
    ├── Pi 5
    ├── Audio interface
    ├── microphones
    ├── headset
    ├── power
    └── thermal system

---

## 1. Software/System Integration

M6 must provide the integration boundary between the existing modules and the final runtime environment.

### M1 adapter

Connect the M1 functionality to the M6 integration environment without changing the established M1 contract unnecessarily.

### M2 adapter

Connect M2 functionality into the M6 pipeline while preserving its expected inputs, outputs, and interfaces.

### M5 adapter

Connect M5 functionality into the integrated M6 runtime.

### Virtual audio E2E

The virtual audio path must allow the complete processing chain to be exercised before physical hardware is available.

The existing M4 virtual audio material can be used as controlled test input.

Important properties already verified for the virtual WAV set:

- 16 kHz
- mono
- 64,000 samples
- 4 seconds

Relevant signals include:

- reference.wav
- error.wav
- speech.wav
- speech_clean.wav
- speech_plus_noise.wav
- stationary_noise.wav
- rotor_noise.wav
- impulsive_noise.wav

M6 should use controlled and reproducible inputs for E2E validation rather than assuming relationships between signals without verification.

### Runtime

The integrated runtime must define:

- module startup
- configuration
- data flow
- error handling
- logging
- clean shutdown
- interface boundaries
- dependency handling

### Reliability

Reliability testing should cover:

- repeated execution
- missing or invalid inputs
- module failures
- runtime errors
- recovery behaviour
- reproducibility
- long-running operation

---

## 2. Virtual Hardware / Simulation

The purpose of this layer is to investigate and validate the future Raspberry Pi 5-based system before physical assembly.

### Wokwi

Use where applicable for embedded/software/hardware interaction experiments and supported simulation.

### PiForge / PiSorch

Investigate and use the relevant Raspberry Pi virtual/software environment for development and testing where it genuinely supports the M6 requirements.

### Raspberry Pi 5 virtual environment

Maintain a development environment that represents the intended Raspberry Pi 5 software/runtime environment as closely as practical.

### EasyEDA

Use for electronics architecture, schematics, PCB-related planning, and component visualization where applicable.

### Other simulation platforms

Additional platforms may be included when they provide a concrete benefit to:

- Raspberry Pi development
- electronics simulation
- signal/audio integration
- hardware planning
- system validation
- enclosure/component planning

Do not add tools merely for the sake of having more tools. Each platform should have a defined purpose.

---

## 3. Physical Hardware Preparation

M6 must prepare the software and virtual system for eventual physical hardware.

### Raspberry Pi 5

Primary processing hardware.

Consider:

- power
- USB connectivity
- audio interface connectivity
- cooling
- mounting
- enclosure clearance
- cable access

### Audio interface

Must provide the required audio input/output path for the intended architecture.

### Microphones

The architecture uses distinct logical microphone roles:

- REF — reference/environmental signal
- ERR — residual/error measurement
- SPCH — speech input

Physical placement remains subject to later experimentation.

### Headset

The output path must support the intended headset/transducer architecture.

Physical acoustic behaviour remains dependent on the actual:

- transducer
- microphone placement
- distance
- seal
- leakage
- enclosure/headset construction

### Power

The physical system must account for:

- Raspberry Pi power requirements
- powered peripherals
- audio hardware
- future hardware expansion

### Thermal system

The physical preparation must account for:

- Raspberry Pi cooling
- airflow
- ventilation
- enclosure thermal constraints
- sustained runtime

---

## 4. Hardware/Software Boundary

M6 must clearly separate:

### Virtual

- simulated hardware
- virtual audio
- software runtime
- simulation environments
- development/test environments

### Physical

- Raspberry Pi 5
- audio interface
- microphones
- headset
- power hardware
- cooling/thermal hardware

Virtual validation should be completed as far as practical before depending on physical hardware.

---

## 5. Integration Principles

1. Preserve existing module interfaces where practical.
2. Keep M1, M2, and M5 responsibilities clearly separated.
3. Make virtual E2E testing reproducible.
4. Do not assume signal relationships without testing them.
5. Keep simulation and physical hardware boundaries explicit.
6. Keep runtime failures observable through logging and diagnostics.
7. Avoid introducing tools that do not serve a defined M6 purpose.
8. Keep Raspberry Pi 5 deployment requirements visible during development.
9. Keep audio sample rate, channel configuration, and signal formats explicit.
10. Separate verified facts from assumptions and conceptual designs.
11. Do not treat simulated behaviour as proof of physical performance.
12. Physical acoustic performance must ultimately be validated on real hardware.

---

## 6. Evidence / Status Classification

M6 work should distinguish between:

### VERIFIED

Confirmed through actual testing, documentation, or available hardware evidence.

### ASSUMED

A working assumption that has not yet been experimentally verified.

### SIMULATED

Behaviour demonstrated in a virtual/simulation environment.

### PHYSICAL

Behaviour measured on the actual assembled hardware.

Simulation results must not automatically be labelled as physical results.

---

## 7. M6 Goal

The final objective is:

Existing SilentVox modules
        ↓
M6 software integration
        ↓
Virtual audio E2E
        ↓
Virtual hardware / simulation
        ↓
Raspberry Pi 5 runtime preparation
        ↓
Physical hardware integration
        ↓
Integrated SilentVox system

M6 therefore acts as the bridge between the existing software modules and the eventual physical Raspberry Pi 5 system.

---

## Status

M6 structure: READY

Software/System Integration: DEFINED

Virtual Hardware / Simulation: DEFINED

Physical Hardware Preparation: DEFINED

Detailed platform selection: TO BE VERIFIED

Physical hardware integration: PENDING
