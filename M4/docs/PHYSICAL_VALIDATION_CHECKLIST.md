# Physical Validation Checklist

Run only after the hardware is available.

1. Record the exact interface model and firmware/revision.
2. Connect the interface to the Raspberry Pi 5.
3. Confirm the device is visible to the Linux audio stack.
4. Record the reported capture/playback channel count.
5. Set 16 kHz capture and verify the actual rate.
6. Feed an identifiable test tone to each input one at a time.
7. Verify REF/ERR/SPCH channel mapping.
8. Measure capture-to-processing timing.
9. Measure output round-trip timing.
10. Run a sustained capture to identify XRUNs/overruns/underruns.
11. Measure actual input noise floor with the final microphone/enclosure.
12. Repeat at the selected block sizes.
13. Store raw measurements and hardware identifiers.

Do not replace software/simulation values in the repository with physical values unless the physical test has been recorded and documented.
