# Microphone Selection Status

## Selection status

**FINAL PHYSICAL MICROPHONE = NOT SELECTED.**

Three concrete, currently documented candidates are compared in `MICROPHONE_COMPARISON.csv`. Their datasheet values are candidate specifications, not physical SilentVox measurements.

## Candidate evaluation matrix

| Candidate | REF | ERR | SPCH | Bandwidth | SNR evidence | SPL/AOP | Interface | Environmental/mechanical note | Status |
|---|---|---|---|---|---|---|---|---|---|
| Infineon IM63D135A | TBD | TBD | TBD | 7 Hz LF roll-off; high-SPL digital capture | 63.5 dB(A) datasheet SNR | 135 dBSPL AOP | PDM digital | 3.50 x 2.65 x 0.98 mm; IP57; -40 to 105 C; AEC-Q103-003 | Candidate only |
| Infineon IM73A135 | TBD | TBD | TBD | 20 Hz LF roll-off | 73 dB(A) datasheet SNR | 135 dBSPL AOP | Differential analog | 4.00 x 3.00 x 1.20 mm; IP57; ANC application documented | Candidate only |
| TDK InvenSense MMICT5837-00-012 | TBD | TBD | TBD | 27 Hz to >20 kHz | 68 dBA high-quality mode | 133 dBSPL AOP high-quality mode | PDM digital | 3.50 x 2.65 x 0.98 mm; -40 to +85 C | Candidate only |

## Engineering checks

- **REF suitability:** candidate must capture external noise with adequate bandwidth/headroom and stable phase characteristics after physical placement.
- **ERR suitability:** candidate must preserve residual/error information with low added noise and controlled gain/phase matching.
- **SPCH suitability:** candidate must provide adequate speech-band response and headroom at the selected physical location.
- **Bandwidth:** compare candidate response against the 16 kHz software sample-rate boundary and Nyquist limit.
- **SNR/self-noise:** use manufacturer values only for candidate comparison. They are not measured headset SNR.
- **Maximum SPL/AOP:** candidate headroom is documented from datasheets; actual operating SPL must be physically characterized.
- **Interface compatibility:** PDM candidates need an appropriate PDM clock/data acquisition path; the analog candidate needs a low-noise differential preamp/ADC/audio interface.
- **Synchronization:** simultaneous physical streams must share a known clock/timeline or a documented synchronization strategy.
- **Mechanical/environmental fit:** manufacturer package/environment ratings do not replace enclosure-level testing.

## Digital MEMS vs analog microphone consideration

Digital MEMS candidates can reduce analog pickup-path complexity but require suitable PDM clock/data acquisition and multi-microphone synchronization. The analog MEMS candidate can feed a conventional multichannel ADC/audio interface but adds analog gain, noise, grounding, and matching considerations. M4 therefore keeps the logical REF/ERR/SPCH contract independent of the microphone electrical interface.

## Selection process

`datasheet verification -> interface compatibility -> mechanical constraints -> candidate comparison -> physical prototype -> measured validation -> final selection`

## Future physical validation tests

- quiet-room noise floor;
- speech response;
- high-SPL response;
- clipping/headroom;
- channel gain consistency;
- phase/timing consistency across channels;
- relevant temperature/environmental tests.

**Selection Status: NOT SELECTED**
