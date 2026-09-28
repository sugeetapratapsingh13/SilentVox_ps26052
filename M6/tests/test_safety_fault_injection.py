from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from M6.safety_simulation import SafetyController


REPO_ROOT = Path(__file__).resolve().parents[2]

EVIDENCE_DIR = (
    REPO_ROOT
    / "M6"
    / "evidence"
    / "simulation"
)


class TestSafetyFaultInjection(unittest.TestCase):

    def setUp(self):
        self.controller = SafetyController()
        self.audio = np.ones(
            512,
            dtype=np.float32,
        )

    def test_normal_processing(self):
        result = self.controller.process(
            self.audio,
            lambda x: x * 0.5,
        )

        self.assertEqual(
            result.state.value,
            "MONITOR",
        )
        self.assertIsNone(result.fault)
        self.assertTrue(
            np.allclose(
                result.output,
                0.5,
            )
        )

    def test_invalid_audio_shape(self):
        result = self.controller.process(
            np.ones(
                (2, 256),
                dtype=np.float32,
            ),
            lambda x: x,
        )

        self.assertEqual(
            result.state.value,
            "SAFE_OUTPUT",
        )
        self.assertEqual(
            result.fault,
            "INVALID_AUDIO_SHAPE",
        )

    def test_empty_audio(self):
        result = self.controller.process(
            np.array([], dtype=np.float32),
            lambda x: x,
        )

        self.assertEqual(
            result.fault,
            "EMPTY_AUDIO",
        )

    def test_nan_audio(self):
        audio = self.audio.copy()
        audio[100] = np.nan

        result = self.controller.process(
            audio,
            lambda x: x,
        )

        self.assertEqual(
            result.fault,
            "NON_FINITE_AUDIO",
        )
        self.assertTrue(
            np.all(result.output == 0)
        )

    def test_processing_failure(self):
        def failing_processor(_):
            raise RuntimeError("injected")

        result = self.controller.process(
            self.audio,
            failing_processor,
        )

        self.assertEqual(
            result.state.value,
            "SAFE_OUTPUT",
        )
        self.assertEqual(
            result.fault,
            "PROCESSING_FAILURE:RuntimeError",
        )

    def test_non_finite_output(self):
        result = self.controller.process(
            self.audio,
            lambda x: np.full_like(
                x,
                np.inf,
            ),
        )

        self.assertEqual(
            result.fault,
            "NON_FINITE_OUTPUT",
        )
        self.assertTrue(
            np.all(result.output == 0)
        )

    def test_output_length_mismatch(self):
        result = self.controller.process(
            self.audio,
            lambda x: x[:-1],
        )

        self.assertEqual(
            result.fault,
            "OUTPUT_LENGTH_MISMATCH",
        )
        self.assertEqual(
            len(result.output),
            len(self.audio),
        )

    def test_evidence_report(self):
        EVIDENCE_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        cases = [
            ("normal", "MONITOR", None),
            (
                "invalid_shape",
                "SAFE_OUTPUT",
                "INVALID_AUDIO_SHAPE",
            ),
            (
                "empty_audio",
                "SAFE_OUTPUT",
                "EMPTY_AUDIO",
            ),
            (
                "nan_audio",
                "SAFE_OUTPUT",
                "NON_FINITE_AUDIO",
            ),
            (
                "processing_failure",
                "SAFE_OUTPUT",
                "PROCESSING_FAILURE:RuntimeError",
            ),
            (
                "non_finite_output",
                "SAFE_OUTPUT",
                "NON_FINITE_OUTPUT",
            ),
            (
                "length_mismatch",
                "SAFE_OUTPUT",
                "OUTPUT_LENGTH_MISMATCH",
            ),
        ]

        report = {
            "status": "SIMULATED",
            "hardware_watchdog": False,
            "physical_validation": False,
            "cases": [
                {
                    "name": name,
                    "state": state,
                    "fault": fault,
                }
                for name, state, fault in cases
            ],
        }

        path = (
            EVIDENCE_DIR
            / "safety_fault_injection.json"
        )

        path.write_text(
            json.dumps(
                report,
                indent=2,
            ),
            encoding="utf-8",
        )

        self.assertTrue(path.is_file())


if __name__ == "__main__":
    unittest.main()
