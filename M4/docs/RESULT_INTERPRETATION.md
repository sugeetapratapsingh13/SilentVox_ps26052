# M4 Result Interpretation

## Frame duration

At 16 kHz:

- 256 samples = 16 ms
- 512 samples = 32 ms
- 1024 samples = 64 ms

These are the durations represented by individual frames. They are not end-to-end latency values.

## Processing latency

`block_latency_benchmark.csv` measures the time spent by the Python virtual-interface processing loop on the development machine. It is not a Raspberry Pi measurement and does not include microphone ADC latency, USB transport, OS scheduling, DAC latency, acoustic propagation, or the physical headset path.

## Robustness tests

`robustness_tests.csv` records the effect of intentionally introduced digital impairments. These tests demonstrate the software model, not the exact behavior of a particular physical microphone or converter.
