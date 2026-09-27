import time
import numpy as np


def benchmark_function(func, *args, **kwargs):
    start = time.perf_counter()

    result = func(*args, **kwargs)

    elapsed = time.perf_counter() - start

    return result, elapsed


def calculate_latency_ms(block_size, sampling_rate):
    return (block_size / sampling_rate) * 1000


def calculate_noise_reduction_db(input_signal, output_signal):
    input_power = np.mean(np.square(input_signal))
    output_power = np.mean(np.square(output_signal))

    if output_power <= 0:
        return float("inf")

    return 10 * np.log10(input_power / output_power)


def check_stability(signal):
    if not np.all(np.isfinite(signal)):
        return False, "NaN or infinite value detected"

    peak = np.max(np.abs(signal))

    if peak > 10:
        return False, "Output amplitude exceeded stability threshold"

    return True, "Stable"