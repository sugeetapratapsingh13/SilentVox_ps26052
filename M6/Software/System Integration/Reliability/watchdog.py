from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
RUNTIME = ROOT / "Software" / "System Integration" / "Runtime" / "pi_runtime.py"

MAX_RESTARTS = 3

print("===================================")
print(" SilentVox Reliability Watchdog")
print("===================================")
print("Platform target: Raspberry Pi 5")
print("Execution mode: SOFTWARE / VIRTUAL")
print("Physical hardware: NOT CONNECTED")
print("-----------------------------------")

for attempt in range(1, MAX_RESTARTS + 1):
    print(f"Runtime attempt: {attempt}/{MAX_RESTARTS}")

    result = subprocess.run(
        [sys.executable, str(RUNTIME)],
        cwd=str(ROOT),
    )

    if result.returncode == 0:
        print("-----------------------------------")
        print("Runtime completed successfully")
        print("Watchdog result: PASS")
        sys.exit(0)

    print("Runtime failure detected")
    print("Exit code:", result.returncode)

    if attempt < MAX_RESTARTS:
        print("Watchdog action: RESTART")
        time.sleep(1)

print("-----------------------------------")
print("Maximum restart attempts reached")
print("Watchdog result: FAIL")
sys.exit(1)
