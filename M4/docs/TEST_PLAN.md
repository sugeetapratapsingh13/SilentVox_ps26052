# M4 Test Plan

| Test | Method | Expected evidence |
|---|---|---|
| Audio loading | Load every WAV with `audio_pipeline.io` | Successful validation |
| Resampling | Load at target 16 kHz | 16 kHz output |
| Channel roles | REF/ERR/SPCH mapping | `CHANNEL_MAP.md` |
| Blocks | 256/512/1024 | Benchmark CSV |
| Waveform | Generate plot | PNG |
| FFT | Generate plot | PNG |
| Spectrogram | Generate plot | PNG |
| Quantization | 16-bit simulation | Unit test |
| Clipping | Over-range input | Unit test / simulation |
| Delay | 1/5/10 ms model | Constraint documentation |
| Gain mismatch | Per-channel gain model | Constraint documentation |
| Noise | Additive-noise model | Metrics |
| Browser | AudioWorklet pass-through | Browser demo |
| Hardware | Pi/interface capture | Pending physical validation |
