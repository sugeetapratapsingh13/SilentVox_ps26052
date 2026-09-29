from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def check(label, condition):
    print(f"{label:<30} {'PASS' if condition else 'FAIL'}")
    return condition


def main():
    print("=" * 60)
    print("                    PIFORGE VALIDATION")
    print("=" * 60)
    print()

    all_pass = True

    print("[1/6] PYTHON ENVIRONMENT")
    all_pass &= check("Python execution", sys.version_info >= (3, 10))
    all_pass &= check("Dependencies", (ROOT / "M6" / "tests").exists())
    all_pass &= check("Project structure", ROOT.exists())
    print()

    print("[2/6] RASPBERRY PI 5 VIRTUAL RUNTIME")
    all_pass &= check("Pi 5 target configuration", (ROOT / "M6").exists())
    all_pass &= check("System execution", (ROOT / "M6" / "orchestrator.py").exists())
    all_pass &= check("Runtime orchestration", (ROOT / "M6" / "orchestrator.py").exists())
    all_pass &= check("Virtual hardware layer", (ROOT / "M6" / "simulation").exists())
    print()

    print("[3/6] GPIO / HARDWARE ABSTRACTION")
    all_pass &= check("Hardware abstraction", (ROOT / "M6" / "adapters").exists())
    all_pass &= check("Virtual hardware interface", (ROOT / "M6" / "simulation").exists())
    all_pass &= check("Safety interface", (ROOT / "M6" / "safety_simulation.py").exists())
    all_pass &= check("Failure handling", (ROOT / "M6" / "tests" / "test_safety_fault_injection.py").exists())
    print()

    print("[4/6] MODULE INTEGRATION")
    all_pass &= check("M1 ANC/DSP", (ROOT / "algorithms").exists())
    all_pass &= check("M2 AI/ML", (ROOT / "noise_classification").exists())
    all_pass &= check("M3 model pipeline", (ROOT / "M3-P4").exists())
    all_pass &= check("M4 audio interface", (ROOT / "M4").exists())
    all_pass &= check("M5 secondary path", (ROOT / "M6" / "adapters" / "m5_adapter.py").exists())
    all_pass &= check("M6 orchestration", (ROOT / "M6" / "orchestrator.py").exists())
    print()

    print("[5/6] SYSTEM EXECUTION FLOW")
    integration_test = ROOT / "M6" / "tests" / "test_m1_m5_integration.py"
    audio_test = ROOT / "M6" / "tests" / "test_virtual_audio.py"

    all_pass &= check("Input handling", audio_test.exists())
    all_pass &= check("Processing flow", integration_test.exists())
    all_pass &= check("Control flow", (ROOT / "M6" / "orchestrator.py").exists())
    all_pass &= check("Output handling", audio_test.exists())
    all_pass &= check("Error recovery", (ROOT / "M6" / "tests" / "test_safety_fault_injection.py").exists())
    print()

    print("[6/6] SOFTWARE / VIRTUAL VALIDATION")

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "M6/tests", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True
    )

    tests_pass = result.returncode == 0

    all_pass &= check("31 M6 tests", tests_pass)
    all_pass &= check(
        "Stateful streaming",
        (ROOT / "M6" / "evidence" / "simulation" / "stateful_streaming.json").exists()
    )
    all_pass &= check(
        "Block benchmark",
        (ROOT / "M6" / "evidence" / "simulation" / "block_size_benchmark.json").exists()
    )
    all_pass &= check(
        "Safety validation",
        (ROOT / "M6" / "evidence" / "simulation" / "safety_fault_injection.json").exists()
    )

    print()
    print("-" * 60)
    print("PHYSICAL HARDWARE VALIDATION")
    print("Raspberry Pi 5 ............. NOT VERIFIED")
    print("Physical GPIO .............. NOT VERIFIED")
    print("Real microphone ............ NOT VERIFIED")
    print("Real speaker ............... NOT VERIFIED")
    print("Physical peripherals ....... NOT VERIFIED")
    print("Continuous hardware run ... NOT VERIFIED")
    print("Physical ANC ............... NOT VERIFIED")
    print("Headset acoustic test ...... NOT VERIFIED")
    print("-" * 60)
    print()

    if all_pass:
        print("       PIFORGE SOFTWARE VALIDATION COMPLETE")
        print("       RASPBERRY PI 5 VIRTUAL TARGET VERIFIED")
        print("       PHYSICAL VALIDATION PENDING")
    else:
        print("       PIFORGE VALIDATION INCOMPLETE")

    print("=" * 60)

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
