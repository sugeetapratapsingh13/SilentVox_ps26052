# M4 Claim-Control Audit

This file records the required wording boundaries.

| Do not claim | Correct M4 wording |
|---|---|
| Microphone latency = X ms | Simulated/software pipeline latency = X ms, if it was measured in software |
| Microphone SNR = X dB from WAV analysis | Digital test-stream SNR = X dB |
| Candidate microphone datasheet SNR = X dB as measured performance | Candidate microphone datasheet SNR = X dB |
| Interface works on Raspberry Pi 5 | Linux/USB compatibility identified where supported; Raspberry Pi 5 physical validation pending |
| 512 block = 32 ms system latency | 512-sample frame duration = 32 ms at 16 kHz |
| ANC headset achieved X dB reduction from M4 front-end tests | M4 front-end tests do not establish physical headset ANC reduction |
| Simulated ADC/DAC result is physical ADC/DAC performance | Digital ADC/DAC-style simulation result |
| Virtual microphone WAV is a microphone measurement | Virtual/synthetic digital test stream |

Before push, search the M4 tree for terms such as `microphone latency`, `microphone SNR`, `works on Raspberry Pi`, `system latency`, and `headset ANC` and verify that any occurrence is clearly negated, qualified, or attributed to a manufacturer/datasheet source.
