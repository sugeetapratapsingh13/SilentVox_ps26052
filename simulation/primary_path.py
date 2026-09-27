"""
primary_path.py

Simulates the acoustic PRIMARY PATH: the transfer function between the
noise source and the point where the noise actually disturbs the listener
(the error microphone / ear).

Pipeline position:

    Reference Noise --> [ PRIMARY PATH ] --> Noise-at-ear
                                                    +
                                              Desired Signal
                                                    =
                                              Noisy Signal (error mic)

We model the primary path as a short FIR impulse response: a pure
propagation delay followed by an exponentially decaying "reflection" tail
and an overall attenuation. This is a standard simplified way to represent
an acoustic path without needing measured hardware data.
"""

import numpy as np


def make_primary_path(delay_samples=5, tap_length=10, decay=0.6, attenuation=0.8):
    """
    Build the primary-path impulse response h[n].

    Parameters:
        delay_samples : propagation delay in samples before energy arrives
        tap_length     : number of decaying taps after the delay
        decay          : per-tap decay factor (0 < decay < 1)
        attenuation    : overall attenuation applied to the path

    Returns:
        h : 1D numpy array, the impulse response
    """
    h = np.zeros(delay_samples + tap_length)
    for i in range(tap_length):
        h[delay_samples + i] = attenuation * (decay ** i)
    return h


def apply_primary_path(signal, h):
    """
    Pass `signal` through the primary path impulse response `h`.
    Output is truncated back to the original signal length so
    everything downstream stays sample-aligned.
    """
    signal = np.asarray(signal, dtype=np.float64)
    out = np.convolve(signal, h)[: len(signal)]
    return out