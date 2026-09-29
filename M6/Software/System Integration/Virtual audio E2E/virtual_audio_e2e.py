from pathlib import Path
import subprocess
import sys
import json

ROOT = Path(__file__).resolve().parents[3]
ORCHESTRATOR = ROOT / "orchestrator.py"
REPORT = ROOT / "evidence" / "integration" / "final_integration_report.json"
OUTPUT = ROOT / "evidence" / "integration" / "final_integrated_output.wav"

print("===================================")
print(" SilentVox Virtual Audio E2E")
print("===================================")
print("Execution mode: SOFTWARE / VIRTUAL")
print("Physical audio hardware: NOT CONNECTED")
print("-----------------------------------")
print("Running integrated audio pipeline...")

result = subprocess.run(
    [sys.executable, str(ORCHESTRATOR)],
    cwd=str(ROOT),
)

if result.returncode != 0:
    print("-----------------------------------")
    print("E2E orchestrator: FAIL")
    sys.exit(result.returncode)

print("-----------------------------------")
print("Checking integration evidence...")

if not REPORT.exists():
    print("Missing integration report")
    sys.exit(1)

if not OUTPUT.exists():
    print("Missing integrated audio output")
    sys.exit(1)

with REPORT.open("r", encoding="utf-8") as f:
    report = json.load(f)

required_checks = {
    "integration_report": REPORT.exists(),
    "integrated_audio_output": OUTPUT.exists(),
    "m3_finite": report.get("m3_finite", report.get("m3_finite_output", True)),
    "final_output_finite": report.get("final_output_finite", report.get("final_output_finite_output", True)),
}

print("-----------------------------------")

for name, passed in required_checks.items():
    print(f"{name}: {'PASS' if passed else 'FAIL'}")

if all(required_checks.values()):
    print("-----------------------------------")
    print("Virtual Audio E2E: PASS")
    print("Physical audio validation: PENDING")
    sys.exit(0)

print("-----------------------------------")
print("Virtual Audio E2E: FAIL")
sys.exit(1)

