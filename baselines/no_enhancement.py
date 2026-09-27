"""
M3-P3 Classical Enhancement Baseline

No-enhancement baseline.

The noisy speech is passed through unchanged.
This establishes the reference performance before
classical enhancement.
"""

from pathlib import Path
import argparse

import numpy as np
import soundfile as sf


def no_enhancement(noisy):
    """
    Pass the noisy signal through unchanged.
    """
    return np.asarray(noisy, dtype=np.float64).copy()


def main():
    parser = argparse.ArgumentParser(
        description="M3-P3 no-enhancement baseline"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input noisy WAV file"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output WAV file"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    noisy, sr = sf.read(input_path)

    if noisy.ndim > 1:
        noisy = np.mean(noisy, axis=1)

    enhanced = no_enhancement(noisy)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    sf.write(
        output_path,
        enhanced,
        sr
    )

    print("No-enhancement processing completed.")
    print(f"Input:       {input_path}")
    print(f"Output:      {output_path}")
    print(f"Sample rate: {sr} Hz")
    print(f"Samples:     {len(enhanced)}")


if __name__ == "__main__":
    main()