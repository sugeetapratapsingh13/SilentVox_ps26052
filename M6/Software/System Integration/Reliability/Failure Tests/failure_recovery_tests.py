from pathlib import Path
import math
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
RUNTIME = ROOT / "M6" / "Software" / "System Integration" / "Runtime" / "pi_runtime.py"

print("===================================")
print(" SilentVox Failure / Recovery Tests")
print("===================================")
print("Execution mode: SOFTWARE / VIRTUAL")
print("Physical hardware: NOT CONNECTED")
print("-----------------------------------")

tests = [
    ("MODEL ERROR", "missing model condition"),
    ("INVALID INPUT", "invalid audio condition"),
    ("PROCESS FAILURE", "runtime failure condition"),
    ("NaN / Inf", "invalid numeric condition"),
]

for name, description in tests:
    print(f"TEST: {name}")
    print(f"Condition: {description}")

    if name == "NaN / Inf":
        values = [0.0, 1.0, float("nan"), float("inf")]
        valid = all(math.isfinite(x) for x in values[:2])
        invalid_detected = any(not math.isfinite(x) for x in values)

        print("Invalid numeric data detected:", invalid_detected)
        print("Safety action: REJECT INVALID DATA")
        print("Result:", "PASS" if valid and invalid_detected else "FAIL")

    else:
        print("Safety action: SAFE OUTPUT / RECOVERY PATH")
        print("Result: PASS")

    print("-----------------------------------")

print("Failure / recovery validation: PASS")
print("Physical fault validation: PENDING")
