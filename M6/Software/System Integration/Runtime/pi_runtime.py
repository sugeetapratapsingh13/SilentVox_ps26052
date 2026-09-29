from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
ORCHESTRATOR = ROOT / "orchestrator.py"

print("===================================")
print(" SilentVox Raspberry Pi Runtime")
print("===================================")
print("Platform target: Raspberry Pi 5")
print("Execution mode: SOFTWARE / VIRTUAL")
print("Physical hardware: NOT CONNECTED")
print("-----------------------------------")
print("Launching existing M6 E2E orchestrator...")
print("-----------------------------------")

result = subprocess.run(
    [sys.executable, str(ORCHESTRATOR)],
    cwd=str(ROOT),
)

print("-----------------------------------")

if result.returncode == 0:
    print("SilentVox Pi runtime: PASS")
else:
    print("SilentVox Pi runtime: FAIL")
    print("Orchestrator exit code:", result.returncode)

sys.exit(result.returncode)
