from pathlib import Path
import subprocess
import sys
import time
import json
import os

ROOT = Path(__file__).resolve().parents[4]
ORCHESTRATOR = ROOT / "orchestrator.py"
OUTPUT_FILE = ROOT / "evidence" / "integration" / "software_platform_benchmark.json"

BLOCK_SIZES = [256, 512, 1024]

print("===================================")
print(" SilentVox Software Platform Benchmark")
print("===================================")
print("Target platform: Raspberry Pi 5")
print("Benchmark type: SOFTWARE-PLATFORM")
print("Physical Raspberry Pi: NOT CONNECTED")
print("-----------------------------------")

results = []

for block_size in BLOCK_SIZES:
    print(f"BLOCK SIZE: {block_size}")

    start = time.perf_counter()

    env = dict(os.environ)
    env["SILENTVOX_BLOCK_SIZE"] = str(block_size)

    result = subprocess.run(
        [sys.executable, str(ORCHESTRATOR)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env=env,
    )

    elapsed = time.perf_counter() - start
    passed = result.returncode == 0

    results.append({
        "block_size": block_size,
        "wall_time_seconds": elapsed,
        "orchestrator_pass": passed,
    })

    print(f"Wall time: {elapsed:.6f} s")
    print("Pipeline result:", "PASS" if passed else "FAIL")
    print("-----------------------------------")

    if not passed:
        print("ORCHESTRATOR STDOUT:")
        print(result.stdout)
        print("ORCHESTRATOR STDERR:")
        print(result.stderr)

output = {
    "benchmark_type": "SOFTWARE_PLATFORM_BENCHMARK",
    "target_platform": "Raspberry Pi 5",
    "physical_validation": False,
    "block_sizes": BLOCK_SIZES,
    "results": results,
    "notes": [
        "Each benchmark trial passes its block size to the M6 orchestrator through SILENTVOX_BLOCK_SIZE.",
        "The orchestrator therefore processes the corresponding audio block size for each trial.",
        "This remains a software/virtual benchmark and is not a physical Raspberry Pi 5 latency measurement."
    ],
}

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT_FILE.open("w", encoding="utf-8") as handle:
    json.dump(output, handle, indent=2)

if all(item["orchestrator_pass"] for item in results):
    print("Software-platform benchmark: PASS")
    print("Raspberry Pi 5 physical benchmark: PENDING")
else:
    print("Software-platform benchmark: FAIL")
    sys.exit(1)
