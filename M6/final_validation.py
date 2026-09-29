from pathlib import Path
import csv
import json
import subprocess
import sys
import importlib.util

ROOT = Path(__file__).resolve().parent.parent

PASS = "PASS"
FAIL = "FAIL"
NOT_VERIFIED = "NOT VERIFIED"


def check(condition):
    return PASS if condition else FAIL


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():

    print("=" * 60)
    print("                 SILENTVOX VALIDATION")
    print("=" * 60)
    print()

    # =========================================================
    # 1. ENVIRONMENT
    # =========================================================
    print("[1/6] ENVIRONMENT CHECK")

    python_ok = sys.version_info >= (3, 13)
    print(f"Python ..................... {check(python_ok)}")

    required_modules = [
        "numpy",
        "scipy",
        "torch",
        "librosa",
        "soundfile",
        "joblib",
        "pytest",
    ]

    deps_ok = all(
        importlib.util.find_spec(module) is not None
        for module in required_modules
    )

    print(f"Dependencies ............... {check(deps_ok)}")

    required_paths = [
        ROOT / "algorithms",
        ROOT / "benchmarks",
        ROOT / "integration",
        ROOT / "noise_classification",
        ROOT / "M4",
        ROOT / "M6",
        ROOT / "M3-P4",
    ]

    structure_ok = all(p.exists() for p in required_paths)
    print(f"Project structure .......... {check(structure_ok)}")
    print()

    # =========================================================
    # 2. ANC / DSP
    # =========================================================
    print("[2/6] ANC / DSP VALIDATION")

    realtime_csv = ROOT / "benchmarks" / "REALTIME_BENCHMARK.csv"
    fxlms_csv = ROOT / "experiments" / "FXLMS_BENCHMARK.csv"

    lms_ok = False
    nlms_ok = False
    fxlms_ok = False

    if realtime_csv.exists():
        try:
            rows = load_csv(realtime_csv)

            lms_rows = [
                r for r in rows
                if r.get("algorithm", "").upper() == "LMS"
            ]

            nlms_rows = [
                r for r in rows
                if r.get("algorithm", "").upper() == "NLMS"
            ]

            lms_ok = (
                len(lms_rows) >= 3
                and {int(r["block_size"]) for r in lms_rows} >= {256, 512, 1024}
                and all(r.get("status", "").upper() == "PASS" for r in lms_rows)
                and all(float(r["real_time_factor"]) > 1.0 for r in lms_rows)
                and all(
                    float(r["max_block_processing_ms"])
                    < float(r["block_duration_ms"])
                    for r in lms_rows
                )
            )

            nlms_ok = (
                len(nlms_rows) >= 3
                and {int(r["block_size"]) for r in nlms_rows} >= {256, 512, 1024}
                and all(r.get("status", "").upper() == "PASS" for r in nlms_rows)
                and all(float(r["real_time_factor"]) > 1.0 for r in nlms_rows)
                and all(
                    float(r["max_block_processing_ms"])
                    < float(r["block_duration_ms"])
                    for r in nlms_rows
                )
            )

        except Exception:
            lms_ok = False
            nlms_ok = False

    if fxlms_csv.exists():
        try:
            rows = load_csv(fxlms_csv)

            fxlms_rows = [
                r for r in rows
                if r.get("Algorithm", "").upper() == "FXLMS"
            ]

            fxlms_ok = (
                len(fxlms_rows) >= 1
                and all(r.get("Status", "").upper() == "PASS" for r in fxlms_rows)
                and all(r.get("Stability", "").lower() == "stable" for r in fxlms_rows)
                and all(float(r["RTF"]) > 1.0 for r in fxlms_rows)
            )

        except Exception:
            fxlms_ok = False

    print(f"LMS ......................... {check(lms_ok)}")
    print(f"NLMS ........................ {check(nlms_ok)}")
    print(f"FxLMS ....................... {check(fxlms_ok)}")

    stateful_path = (
        ROOT
        / "M6"
        / "evidence"
        / "simulation"
        / "stateful_streaming.json"
    )

    stateful_ok = False

    if stateful_path.exists():
        try:
            data = load_json(stateful_path)

            stateful_ok = (
                data.get("status") == "SIMULATED"
                and data.get("continuous_stateful_processing") is True
                and data.get("state_persistence") is True
                and len(data.get("results", [])) == 3
                and all(
                    r.get("finite_residual") is True
                    for r in data["results"]
                )
            )
        except Exception:
            stateful_ok = False

    print(f"Stateful streaming .......... {check(stateful_ok)}")

    block_path = (
        ROOT
        / "M6"
        / "evidence"
        / "simulation"
        / "block_size_benchmark.json"
    )

    block_ok = False

    if block_path.exists():
        try:
            data = load_json(block_path)
            results = data.get("results", [])

            expected = {256, 512, 1024}
            actual = {
                r.get("block_size_samples")
                for r in results
            }

            block_ok = (
                data.get("sample_rate_hz") == 16000
                and actual == expected
                and all(
                    r.get("processing_below_block_duration") is True
                    for r in results
                )
            )

        except Exception:
            block_ok = False

    print(f"Block benchmark ............. {check(block_ok)}")
    print()

    # =========================================================
    # 3. AI / ML
    # =========================================================
    print("[3/6] AI/ML VALIDATION")

    integration_report = (
        ROOT
        / "M6"
        / "evidence"
        / "integration"
        / "final_integration_report.json"
    )

    m2_ok = False
    checkpoint_ok = False

    if integration_report.exists():
        try:
            data = load_json(integration_report)

            m2 = data.get("m2", {})

            checkpoint_ok = (
                data.get("m3", {}).get("model") is not None
            )

            m2_ok = (
                m2.get("raw_label") is not None
                and m2.get("majority_label") is not None
                and m2.get("probability_label") is not None
                and isinstance(m2.get("confidence"), (int, float))
                and 0.0 <= float(m2["confidence"]) <= 1.0
            )

        except Exception:
            checkpoint_ok = False
            m2_ok = False

    print(f"Model loading ............... {check(checkpoint_ok)}")
    print(f"Feature extraction .......... {check(m2_ok)}")
    print(f"Inference ................... {check(m2_ok)}")
    print(f"Input validation ............ {check(m2_ok)}")
    print()

    # =========================================================
    # 4. AUDIO FRONT END
    # =========================================================
    print("[4/6] AUDIO FRONT-END")

    m4_reference = ROOT / "M4" / "virtual_mics" / "reference.wav"
    m4_error = ROOT / "M4" / "virtual_mics" / "error.wav"
    m4_speech = ROOT / "M4" / "virtual_mics" / "speech.wav"

    audio_ok = False
    sample_rate_ok = False
    mono_ok = False

    if all(
        p.exists()
        for p in [m4_reference, m4_error, m4_speech]
    ):
        try:
            import soundfile as sf

            infos = [
                sf.info(str(m4_reference)),
                sf.info(str(m4_error)),
                sf.info(str(m4_speech)),
            ]

            sample_rate_ok = all(
                info.samplerate == 16000
                for info in infos
            )

            mono_ok = all(
                info.channels == 1
                for info in infos
            )

            audio_ok = sample_rate_ok and mono_ok

        except Exception:
            audio_ok = False

    print(f"16 kHz processing ........... {check(sample_rate_ok)}")
    print(f"Mono/virtual audio .......... {check(mono_ok)}")
    print(f"Audio pipeline .............. {check(audio_ok)}")
    print()

    # =========================================================
    # 5. END-TO-END INTEGRATION
    # =========================================================
    print("[5/6] END-TO-END INTEGRATION")

    final_output = (
        ROOT
        / "M6"
        / "evidence"
        / "integration"
        / "final_integrated_output.wav"
    )

    integration_ok = False

    if integration_report.exists() and final_output.exists():
        try:
            data = load_json(integration_report)

            integration_ok = (
                data.get("status") == "SIMULATED"
                and data.get("physical_validation") is False
                and data.get("raspberry_pi_validation") is False
                and data.get("m1", {}).get("state_persistence") is True
                and data.get("m3", {}).get("finite_output") is True
                and data.get("metrics", {}).get("final_output_finite") is True
            )

        except Exception:
            integration_ok = False

    print(f"M4 -> M1 -> M2/M3 -> M5 -> M6 .. {check(integration_ok)}")
    print(f"Audio input ................. {check(integration_ok)}")
    print(f"Processing .................. {check(integration_ok)}")
    print(f"Audio output ................ {check(integration_ok)}")
    print()

    # =========================================================
    # 6. SOFTWARE RUNTIME
    # =========================================================
    print("[6/6] SOFTWARE RUNTIME VALIDATION")

    safety_test = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "M6/tests/test_safety_fault_injection.py",
            "-q",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    safety_ok = safety_test.returncode == 0

    print(f"Safety fault injection ...... {check(safety_ok)}")

    streaming_ok = stateful_ok and block_ok
    print(f"Streaming validation ........ {check(streaming_ok)}")

    finite_ok = integration_ok and stateful_ok
    print(f"Finite-output checks ........ {check(finite_ok)}")

    print()
    print("-" * 60)
    print("PHYSICAL HARDWARE VALIDATION")
    print(f"Raspberry Pi 5 .............. {NOT_VERIFIED}")
    print(f"Real microphone ............. {NOT_VERIFIED}")
    print(f"Real speaker ................ {NOT_VERIFIED}")
    print(f"Physical secondary path ..... {NOT_VERIFIED}")
    print(f"Physical ANC ................ {NOT_VERIFIED}")
    print(f"Headset acoustic test ....... {NOT_VERIFIED}")
    print("-" * 60)

    software_ok = all([
        python_ok,
        deps_ok,
        structure_ok,
        lms_ok,
        nlms_ok,
        fxlms_ok,
        stateful_ok,
        block_ok,
        checkpoint_ok,
        m2_ok,
        audio_ok,
        integration_ok,
        safety_ok,
        streaming_ok,
        finite_ok,
    ])

    print()

    if software_ok:
        print("        SOFTWARE / VIRTUAL VALIDATION COMPLETE")
        print("        PHYSICAL VALIDATION PENDING")
    else:
        print("        SOFTWARE VALIDATION INCOMPLETE")

    print("=" * 60)

    return 0 if software_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
