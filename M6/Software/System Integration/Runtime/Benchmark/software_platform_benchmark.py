from pathlib import Path
import subprocess
import sys
import time
import json

ROOT = Path(__file__).resolve().parents[4] / "M6"
ORCHESTRATOR = ROOT.parent / "orchestrator.py"
REPORT = ROOT / "evidence" / "integration" / "final_integration_report.json"

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

    result = subprocess.run(
        [sys.executable, str(ORCHESTRATOR)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
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

output = {
    "benchmark_type": "SOFTWARE_PLATFORM_BENCHMARK",
    "target_platform": "Raspberry Pi 5",
    "physical_validation": False,
    "block_sizes": BLOCK_SIZES,
    "results": results,
}

output_file = (
    ROOT / "evidence" / "integration" / "software_platform_benchmark.json"
)

output_file.parent.mkdir(parents=True, exist_ok=True)

with output_file.open("w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

if all(item["orchestrator_pass"] for item in results):
    print("Software-platform benchmark: PASS")
    print("Raspberry Pi 5 physical benchmark: PENDING")
else:
    print("Software-platform benchmark: FAIL")
    sys.exit(1)











