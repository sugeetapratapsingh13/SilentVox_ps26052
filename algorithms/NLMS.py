import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import time
import os


def nlms_filter(
    reference_noise,
    desired_signal,
    sampling_rate,
    filter_length,
    step_size,
    epsilon
):
    """
    NLMS adaptive noise cancellation.

    Parameters:
        reference_noise : reference noise signal
        desired_signal  : noisy signal
        sampling_rate   : sampling frequency in Hz
        filter_length   : number of adaptive filter coefficients
        step_size       : NLMS learning rate (mu)
        epsilon         : small value preventing division by zero

    Returns:
        error_signal
        estimated_noise
        coefficient_history
        convergence_curve
        processing_time
    """

    reference_noise = np.asarray(
        reference_noise,
        dtype=np.float64
    ).flatten()

    desired_signal = np.asarray(
        desired_signal,
        dtype=np.float64
    ).flatten()

    # Make both signals the same length
    n_samples = min(
        len(reference_noise),
        len(desired_signal)
    )

    reference_noise = reference_noise[:n_samples]
    desired_signal = desired_signal[:n_samples]

    # Adaptive filter coefficients
    weights = np.zeros(filter_length)

    # Store coefficient history
    coefficient_history = np.zeros(
        (n_samples, filter_length)
    )

    # Estimated noise
    estimated_noise = np.zeros(n_samples)

    # Residual / cleaned signal
    error_signal = np.zeros(n_samples)

    start_time = time.perf_counter()

    # ==========================================================
    # NLMS ALGORITHM
    # ==========================================================

    for n in range(filter_length, n_samples):

        # Reference noise vector
        x = reference_noise[
            n - filter_length:n
        ][::-1]

        # Estimate noise
        y = np.dot(weights, x)

        # Desired / noisy signal
        d = desired_signal[n]

        # Error / residual
        e = d - y

        # Normalization term
        normalization = (
            epsilon + np.dot(x, x)
        )

        # NLMS coefficient update
        weights = weights + (
            step_size / normalization
        ) * e * x

        # Store results
        estimated_noise[n] = y
        error_signal[n] = e

        # Store adaptive coefficients
        coefficient_history[n] = weights

    # ==========================================================
    # PROCESSING TIME
    # ==========================================================

    processing_time = (
        time.perf_counter() - start_time
    )

    # ==========================================================
    # CONVERGENCE CURVE
    # ==========================================================

    convergence_curve = error_signal ** 2

    return (
        error_signal,
        estimated_noise,
        coefficient_history,
        convergence_curve,
        processing_time
    )


def run_nlms(
    noise_file,
    noisy_file,
    output_file,
    estimated_noise_file,
    coefficient_file,
    plot_file,
    filter_length=32,
    step_size=0.001,
    epsilon=1e-8
):

    # ==========================================================
    # READ INPUT FILES
    # ==========================================================

    reference_noise, sampling_rate = sf.read(
        noise_file
    )

    desired_signal, noisy_sampling_rate = sf.read(
        noisy_file
    )

    # Convert stereo to mono
    if reference_noise.ndim > 1:
        reference_noise = np.mean(
            reference_noise,
            axis=1
        )

    if desired_signal.ndim > 1:
        desired_signal = np.mean(
            desired_signal,
            axis=1
        )

    # Check sampling rates
    if sampling_rate != noisy_sampling_rate:
        raise ValueError(
            "Reference and noisy signals must "
            "have the same sampling rate."
        )

    # ==========================================================
    # RUN NLMS
    # ==========================================================

    (
        output_audio,
        estimated_noise,
        coefficients,
        convergence,
        processing_time
    ) = nlms_filter(
        reference_noise,
        desired_signal,
        sampling_rate,
        filter_length,
        step_size,
        epsilon
    )

    # ==========================================================
    # SAVE CLEANED / RESIDUAL AUDIO
    # ==========================================================

    sf.write(
        output_file,
        output_audio,
        sampling_rate
    )

    # ==========================================================
    # SAVE ESTIMATED NOISE
    # ==========================================================

    sf.write(
        estimated_noise_file,
        estimated_noise,
        sampling_rate
    )

    # ==========================================================
    # SAVE ADAPTIVE COEFFICIENT HISTORY
    # ==========================================================

    np.save(
        coefficient_file,
        coefficients
    )

    # ==========================================================
    # CONVERGENCE PLOT
    # ==========================================================

    plt.figure(figsize=(10, 5))

    # Prevent log(0)
    mse_curve = convergence + 1e-12

    plt.plot(
        10 * np.log10(mse_curve),
        linewidth=0.8
    )

    plt.xlabel("Iteration")
    plt.ylabel("Squared Error (dB)")

    plt.title(
        f"NLMS Convergence | "
        f"Filter Length={filter_length}, "
        f"μ={step_size}, "
        f"ε={epsilon}"
    )

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(plot_file)
    plt.close()

    # ==========================================================
    # FINAL MSE
    # ==========================================================

    final_mse = np.mean(
        convergence[filter_length:]
    )

    # ==========================================================
    # PRINT RESULTS
    # ==========================================================

    print("NLMS processing completed.")
    print(f"Sampling rate: {sampling_rate} Hz")
    print(f"Filter length: {filter_length}")
    print(f"Step size (μ): {step_size}")
    print(f"Epsilon: {epsilon}")
    print(f"Final MSE: {final_mse:.10f}")

    print(
        f"Processing time: "
        f"{processing_time:.6f} seconds"
    )

    print(
        f"Adaptive coefficients saved to: "
        f"{coefficient_file}"
    )

    return (
        output_audio,
        estimated_noise,
        coefficients,
        convergence,
        processing_time,
        final_mse
    )


# ==============================================================
# MAIN
# ==============================================================

if __name__ == "__main__":

    os.makedirs(
        "convergence_plots",
        exist_ok=True
    )

    run_nlms(
        noise_file="input_audio/sine_noise.wav",

        noisy_file="input_audio/noisy_sine.wav",

        output_file="nlms_output.wav",

        estimated_noise_file=
        "nlms_estimated_noise.wav",

        coefficient_file=
        "nlms_coefficients.npy",

        plot_file=
        "convergence_plots/nlms_sine.png",

        filter_length=32,

        step_size=0.001,

        epsilon=1e-8
    )