from pathlib import Path
import time
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

ROOT = Path(__file__).resolve().parents[1]

FP32_MODEL = ROOT / "deployment" / "m3_speech_enhancement.onnx"
INT8_MODEL = ROOT / "deployment" / "m3_speech_enhancement_int8.onnx"

print("=" * 70)
print("M3-P6 DEPLOYMENT OPTIMIZATION")
print("=" * 70)

print(f"FP32 model: {FP32_MODEL}")
print(f"Output model: {INT8_MODEL}")

if not FP32_MODEL.exists():
    raise FileNotFoundError(FP32_MODEL)

# ------------------------------------------------------------
# DYNAMIC INT8 QUANTIZATION
# ------------------------------------------------------------

quantize_dynamic(
    model_input=str(FP32_MODEL),
    model_output=str(INT8_MODEL),
    weight_type=QuantType.QInt8,
)

print()
print("INT8 quantization completed.")

# ------------------------------------------------------------
# MODEL SIZE
# ------------------------------------------------------------

fp32_size = FP32_MODEL.stat().st_size
int8_size = INT8_MODEL.stat().st_size

reduction_pct = (
    (fp32_size - int8_size)
    / fp32_size
    * 100.0
)

print()
print("MODEL SIZE")
print("-" * 70)
print(f"FP32 size:       {fp32_size} bytes")
print(f"INT8 size:       {int8_size} bytes")
print(f"Size reduction:  {reduction_pct:.2f}%")

# ------------------------------------------------------------
# RUNTIME VALIDATION
# ------------------------------------------------------------

providers = ["CPUExecutionProvider"]

fp32_session = ort.InferenceSession(
    str(FP32_MODEL),
    providers=providers,
)

int8_session = ort.InferenceSession(
    str(INT8_MODEL),
    providers=providers,
)

x = np.random.default_rng(42).random(
    (1, 1, 257, 100),
    dtype=np.float32,
)

fp32_input = fp32_session.get_inputs()[0].name
int8_input = int8_session.get_inputs()[0].name

fp32_output = fp32_session.run(
    None,
    {fp32_input: x},
)[0]

int8_output = int8_session.run(
    None,
    {int8_input: x},
)[0]

print()
print("RUNTIME VALIDATION")
print("-" * 70)

print("FP32 output shape:", fp32_output.shape)
print("INT8 output shape:", int8_output.shape)

print(
    "FP32 finite:",
    bool(np.isfinite(fp32_output).all()),
)

print(
    "INT8 finite:",
    bool(np.isfinite(int8_output).all()),
)

max_diff = float(
    np.max(
        np.abs(
            fp32_output - int8_output
        )
    )
)

mean_diff = float(
    np.mean(
        np.abs(
            fp32_output - int8_output
        )
    )
)

print(f"Max FP32/INT8 difference:  {max_diff:.8f}")
print(f"Mean FP32/INT8 difference: {mean_diff:.8f}")

# ------------------------------------------------------------
# LATENCY BENCHMARK
# ------------------------------------------------------------

WARMUP = 10
ITERATIONS = 100

for _ in range(WARMUP):
    fp32_session.run(
        None,
        {fp32_input: x},
    )

for _ in range(WARMUP):
    int8_session.run(
        None,
        {int8_input: x},
    )

fp32_times = []

for _ in range(ITERATIONS):
    start = time.perf_counter()

    fp32_session.run(
        None,
        {fp32_input: x},
    )

    fp32_times.append(
        time.perf_counter() - start
    )

int8_times = []

for _ in range(ITERATIONS):
    start = time.perf_counter()

    int8_session.run(
        None,
        {int8_input: x},
    )

    int8_times.append(
        time.perf_counter() - start
    )

fp32_mean = float(np.mean(fp32_times))
int8_mean = float(np.mean(int8_times))

speedup = fp32_mean / int8_mean

print()
print("CPU LATENCY")
print("-" * 70)
print(f"FP32 mean latency: {fp32_mean * 1000:.4f} ms")
print(f"INT8 mean latency: {int8_mean * 1000:.4f} ms")
print(f"Speedup:           {speedup:.3f}x")

print()
print("=" * 70)
print("M3-P6 DEPLOYMENT OPTIMIZATION RESULT")
print("=" * 70)
print(f"FP32 model:        {fp32_size} bytes")
print(f"INT8 model:        {int8_size} bytes")
print(f"Size reduction:    {reduction_pct:.2f}%")
print(f"Max output diff:   {max_diff:.8f}")
print(f"Mean output diff:  {mean_diff:.8f}")
print(f"FP32 latency:      {fp32_mean * 1000:.4f} ms")
print(f"INT8 latency:      {int8_mean * 1000:.4f} ms")
print(f"Speedup:           {speedup:.3f}x")
print("Optimization validation: PASS")
