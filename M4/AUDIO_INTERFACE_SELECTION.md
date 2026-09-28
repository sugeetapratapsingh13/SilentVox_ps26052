# Audio Interface Research and Selection Status

## Current decision

**NO FINAL PHYSICAL AUDIO INTERFACE HAS BEEN SELECTED.**

The candidates below are engineering comparison candidates only. Datasheet capability is not Raspberry Pi 5 physical validation.

## Candidate notes

### Behringer UMC404HD

Manufacturer documentation identifies a 4-input/4-output USB 2.0 interface with four mic preamps and 24-bit/192 kHz converters. The quick-start specification lists maximum power consumption of 2.5 W and dimensions of 45.81 x 292 x 130 mm. A current Linux kernel USB-audio quirk entry also identifies the UMC404HD, but that is not equivalent to a completed Raspberry Pi 5 integration test. The comparison therefore leaves Pi/ALSA/channel-order/latency validation pending.

### Audient EVO 8

Audient documents four analogue inputs and four digital outputs, 24-bit/96 kHz operation, USB 2.0 High Speed, and published latency figures for particular buffer/host configurations. Those figures are not the latency of a Raspberry Pi 5 SilentVox deployment. Physical target validation remains pending.

### Focusrite Scarlett 18i8 3rd Gen

The exact generation used here is **3rd Gen**. Focusrite documents 18 x 8 simultaneous I/O, four mic preamps, 24-bit/192 kHz operation, USB 2.0 Type-C, 241 x 61 x 159.5 mm dimensions, 1.335 kg weight, and an external 12 V DC 1.2 A adapter. Focusrite's Linux support page says its products are not supported on Linux, while noting that USB products are class-compliant and may work in Linux-based setups without Focusrite support. Therefore no Raspberry Pi/ALSA validation is claimed.

## Requirements retained

The eventual physical interface must provide:

- at least 3 simultaneous independent input channels;
- 16 kHz capture;
- output playback;
- stable operation;
- channel-order verification;
- latency measurement;
- long-duration stability.

It must also expose a practical Linux backend path (direct ALSA or through PipeWire/application backend) on the target Raspberry Pi 5 configuration.

## ALSA vs PipeWire vs application backend

- **ALSA** is the Linux audio API/device layer.
- **PipeWire** is a higher-level audio/session framework that can route and manage devices.
- **Application backend** is the library/API used by SilentVox to open and process the stream.

Raspberry Pi documentation states that Raspberry Pi OS uses PulseAudio or PipeWire by default and that applications can communicate directly with ALSA where needed. This does not validate any candidate interface on Pi 5.

## Power and size comparison

See `AUDIO_INTERFACE_COMPARISON.csv`. Power and dimensions are included where manufacturer documentation provides them. Retail prices are separate market observations and can change.

## Approximate India market observations (September 2026)

- UMC404HD: roughly ₹18,000–₹21,000 in current observed listings/price history.
- EVO 8: roughly ₹19,900–₹21,990 in current observed listings/price history.
- Scarlett 18i8 3rd Gen: roughly ₹49,457–₹53,487 in current observed listings.

These are **market observations, not manufacturer prices**, and are included only as a planning criterion.

## Final physical selection procedure

1. Obtain candidate hardware.
2. Connect it to Raspberry Pi 5.
3. Confirm USB enumeration.
4. Confirm ALSA/PipeWire visibility.
5. Verify input/output channel count.
6. Verify at least 3 simultaneous independent input channels.
7. Capture 16 kHz audio.
8. Inject known signals into each input.
9. Verify physical channel order.
10. Verify sustained capture.
11. Verify output playback.
12. Measure round-trip/audio-interface latency.
13. Run the 30-minute stability test and record CPU/RAM/underruns/overruns.
14. Only then make the final physical selection.

## Sources

Manufacturer sources are retained in `AUDIO_INTERFACE_COMPARISON.csv`. Additional current context: Raspberry Pi audio documentation, Linux kernel USB-audio device support, and market price observations are documented separately and are not treated as physical validation.
