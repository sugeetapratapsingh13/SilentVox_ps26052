from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DIR = REPO_ROOT / "M6" / "evidence" / "simulation"


class TestSimulationEvidence(unittest.TestCase):

    def test_virtual_anc_report_exists_and_is_valid(self):
        path = EVIDENCE_DIR / "virtual_anc_demo.json"

        self.assertTrue(path.is_file())

        report = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(report["status"], "SIMULATED")
        self.assertEqual(
            report["audio"]["sample_rate_hz"],
            16000,
        )

        self.assertTrue(
            report["metrics"]["finite_control"]
        )

        self.assertTrue(
            report["metrics"]["finite_residual"]
        )

        self.assertTrue(
            report["metrics"]["finite_weights"]
        )

        self.assertTrue(
            np.isfinite(
                report["metrics"]["attenuation_db"]
            )
        )

    def test_block_benchmark_exists_and_has_all_sizes(self):
        path = EVIDENCE_DIR / "block_size_benchmark.json"

        self.assertTrue(path.is_file())

        report = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(
            report["sample_rate_hz"],
            16000,
        )

        sizes = [
            row["block_size_samples"]
            for row in report["results"]
        ]

        self.assertEqual(
            sizes,
            [256, 512, 1024],
        )

    def test_block_processing_is_below_frame_duration(self):
        path = EVIDENCE_DIR / "block_size_benchmark.json"

        report = json.loads(path.read_text(encoding="utf-8"))

        for row in report["results"]:
            self.assertLess(
                row["mean_processing_ms"],
                row["block_duration_ms"],
            )

    def test_block_plot_exists(self):
        path = EVIDENCE_DIR / "block_size_processing.png"

        self.assertTrue(path.is_file())
        self.assertGreater(path.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
