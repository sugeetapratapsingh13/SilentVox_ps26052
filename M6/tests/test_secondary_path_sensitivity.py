from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]

EVIDENCE_DIR = (
    REPO_ROOT
    / "M6"
    / "evidence"
    / "simulation"
)


class TestSecondaryPathSensitivity(unittest.TestCase):

    def test_sensitivity_report_exists(self):
        path = (
            EVIDENCE_DIR
            / "secondary_path_sensitivity.json"
        )

        self.assertTrue(path.is_file())

        report = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(
            report["status"],
            "SIMULATED",
        )

        self.assertEqual(
            report["sample_rate_hz"],
            16000,
        )

    def test_all_15_conditions_exist(self):
        path = (
            EVIDENCE_DIR
            / "secondary_path_sensitivity.json"
        )

        report = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        results = report["results"]

        self.assertEqual(
            len(results),
            15,
        )

        delays = sorted(
            {
                row["delay_ms"]
                for row in results
            }
        )

        attenuations = sorted(
            {
                row[
                    "secondary_path_attenuation"
                ]
                for row in results
            }
        )

        self.assertEqual(
            delays,
            [0.0, 1.0, 2.0, 5.0, 10.0],
        )

        self.assertEqual(
            attenuations,
            [0.40, 0.65, 0.85],
        )

    def test_all_results_are_finite(self):
        path = (
            EVIDENCE_DIR
            / "secondary_path_sensitivity.json"
        )

        report = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        for row in report["results"]:

            for key in (
                "input_rms",
                "residual_rms",
                "attenuation_db",
            ):
                self.assertTrue(
                    np.isfinite(
                        row[key]
                    )
                )

            self.assertTrue(
                row["finite_control"]
            )

            self.assertTrue(
                row["finite_residual"]
            )

            self.assertTrue(
                row["finite_weights"]
            )

    def test_csv_and_plot_exist(self):
        csv_path = (
            EVIDENCE_DIR
            / "secondary_path_sensitivity.csv"
        )

        plot_path = (
            EVIDENCE_DIR
            / "secondary_path_sensitivity.png"
        )

        self.assertTrue(
            csv_path.is_file()
        )

        self.assertTrue(
            plot_path.is_file()
        )

        self.assertGreater(
            plot_path.stat().st_size,
            1000,
        )

        with csv_path.open(
            "r",
            encoding="utf-8",
        ) as handle:

            rows = list(
                csv.DictReader(handle)
            )

        self.assertEqual(
            len(rows),
            15,
        )


if __name__ == "__main__":
    unittest.main()
