"""
metrics.py

Computes the required per-condition measurements:
    input level, output level, noise reduction, residual error,
    convergence time, stability, processing time
"""

import numpy as np


def rms(x):
    x = np.asarray(x)
    return np.sqrt(np.mean(x ** 2) + 1e-20)


def noise_reduction_db(before, after):
    return 20 * np.log10(rms(before) / (rms(after) + 1e-20))


def find_convergence_index(convergence_curve, filter_length, tolerance=1.1,
                            window=100):
    """
    First iteration index at which the running error settles within
    `tolerance` x of its final steady-state level.
    """
    steady = np.mean(convergence_curve[-max(int(len(convergence_curve) * 0.1), 1):])
    threshold = steady * tolerance

    for i in range(filter_length, len(convergence_curve) - window):
        if np.mean(convergence_curve[i:i + window]) <= threshold:
            return i

    return len(convergence_curve) - 1


def check_stability(convergence_curve, filter_length, blowup_factor=3.0):
    """
    Rough stability check: compares mean squared error in the middle of
    the run against the final segment. If the tail is much worse than
    the middle, the filter is diverging / unstable.
    """
    n = len(convergence_curve)
    mid = convergence_curve[filter_length: n // 2]
    tail = convergence_curve[-max(int(n * 0.1), 1):]

    if len(mid) == 0 or len(tail) == 0:
        return True

    return bool(np.mean(tail) <= blowup_factor * np.mean(mid))


def compute_metrics(noisy_signal, error_signal, convergence_curve,
                     filter_length, sample_rate, processing_time):
    n = len(error_signal)
    steady_start = int(n * 0.7)

    input_level = rms(noisy_signal[filter_length:])
    output_level = rms(error_signal[steady_start:])
    nr_db = noise_reduction_db(noisy_signal[filter_length:], error_signal[steady_start:])
    residual_error = rms(error_signal[steady_start:])

    conv_idx = find_convergence_index(convergence_curve, filter_length)
    convergence_time_s = conv_idx / sample_rate

    stable = check_stability(convergence_curve, filter_length)

    return {
        "Input_Level_RMS": input_level,
        "Output_Level_RMS": output_level,
        "Noise_Reduction_dB": nr_db,
        "Residual_Error_RMS": residual_error,
        "Convergence_Time_s": convergence_time_s,
        "Stable": stable,
        "Processing_Time_s": processing_time,
    }