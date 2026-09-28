from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Current software integration standard. Physical hardware values remain configurable/TBD.
INPUT_SAMPLE_RATE = 16_000
OUTPUT_SAMPLE_RATE = 16_000
SAMPLE_RATE = INPUT_SAMPLE_RATE
BLOCK_SIZES = (256, 512, 1024)
DEFAULT_BLOCK_SIZE = 512
LOGICAL_CHANNELS = ("REF", "ERR", "SPCH")
INPUT_CHANNELS = 3
OUTPUT_CHANNELS = 1
VIRTUAL_WAV_BIT_DEPTH = 16
PHYSICAL_INPUT_BIT_DEPTH = None
PHYSICAL_OUTPUT_BIT_DEPTH = None
INTERNAL_SAMPLE_TYPE = "float32"
REFERENCE_DELAYS_MS = (1, 5, 10)
SYNCHRONIZATION_TOLERANCE_MS = 0.25
BENCHMARK_TRIALS = 10
CLIPPING_THRESHOLD = 1.0
SYNTHETIC_GAIN_MISMATCH = {"REF": 1.0, "ERR": 0.85, "SPCH": 1.10}
SYNTHETIC_NOISE_RMS = 0.0
PHYSICAL_INTERFACE_NAME = None
EXPECTED_PHYSICAL_INPUT_CHANNELS = 3
EXPECTED_PHYSICAL_OUTPUT_CHANNELS = 1
AUDIO_BACKEND = "TBD: ALSA/PipeWire/application backend"
PHYSICAL_VALIDATION_STATUS = "PENDING"
MAX_INT16 = 32767

DIRS = {
    "virtual_mics": ROOT / "virtual_mics",
    "audio_pipeline": ROOT / "audio_pipeline",
    "analysis": ROOT / "analysis",
    "plots": ROOT / "plots",
    "tests": ROOT / "tests",
    "browser_demo": ROOT / "browser_demo",
    "results": ROOT / "results",
    "docs": ROOT / "docs",
}
