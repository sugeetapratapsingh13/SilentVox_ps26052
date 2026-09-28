from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from M6.adapters.m1_adapter import M1Adapter
from M6.adapters.m5_adapter import M5Adapter


REPO_ROOT = Path(__file__).resolve().parents[2]
M5_ZIP = REPO_ROOT / "SilentVox_M5_GITHUB_CLEAN.zip"
M5_MEMBER = "SilentVox_M5/models/secondary_path_estimate.npy"


class TestM1M5Integration(unittest.TestCase):

    def extract_m5_model(self):
        with zipfile.ZipFile(M5_ZIP) as archive:
            data = archive.read(M5_MEMBER)

        temp_dir = tempfile.TemporaryDirectory()
        model_path = Path(temp_dir.name) / "secondary_path_estimate.npy"
        model_path.write_bytes(data)

        self.addCleanup(temp_dir.cleanup)
        return model_path

    def test_m5_model_loads(self):
        model_path = self.extract_m5_model()

        model = M5Adapter(model_path).load()

        self.assertEqual(model.sample_rate, 16000)
        self.assertEqual(model.filter_length, 128)
        self.assertEqual(model.delay_samples, 32)
        self.assertAlmostEqual(model.attenuation, 0.65)
        self.assertTrue(np.isfinite(model.impulse_response).all())

    def test_m5_rejects_non_npy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.zip"
            path.write_bytes(b"not an npy model")

            with self.assertRaises(ValueError):
                M5Adapter(path).load()

    def test_m1_native_processing(self):
        t = np.arange(0, 0.1, 1 / 16000)
        reference = np.sin(2 * np.pi * 300 * t)

        result = M1Adapter().process(reference)

        self.assertEqual(result.sample_rate, 16000)
        self.assertEqual(result.filter_length, 64)
        self.assertEqual(
            result.secondary_path_mode,
            "M1_NATIVE_SIMULATION",
        )
        self.assertTrue(np.isfinite(result.residual).all())
        self.assertTrue(np.isfinite(result.weights).all())

    def test_m1_with_m5_secondary_path(self):
        model_path = self.extract_m5_model()
        secondary_path = M5Adapter(model_path).load()

        t = np.arange(0, 0.25, 1 / 16000)
        reference = np.sin(2 * np.pi * 300 * t)

        result = M1Adapter().process(
            reference,
            secondary_path=secondary_path,
        )

        self.assertEqual(
            result.secondary_path_mode,
            "M5_SIMULATED_SECONDARY_PATH",
        )
        self.assertEqual(result.reference.shape, reference.shape)
        self.assertEqual(result.residual.shape, reference.shape)
        self.assertTrue(np.isfinite(result.residual).all())
        self.assertTrue(np.isfinite(result.weights).all())


if __name__ == "__main__":
    unittest.main()
