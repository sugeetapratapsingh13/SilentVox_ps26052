"""
M6 -> M2 adapter integration tests.
"""

from __future__ import annotations

import unittest

import numpy as np

from M6.adapters.m2_adapter import (
    EXPECTED_LOG_MEL_SHAPE,
    EXPECTED_CLASSES,
    M2Adapter,
)


class TestM2Adapter(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.adapter = M2Adapter()

    def test_real_checkpoint_loads(self) -> None:
        self.assertIsNotNone(self.adapter.model)

    def test_valid_log_mel_inference(self) -> None:
        features = np.zeros(
            EXPECTED_LOG_MEL_SHAPE,
            dtype=np.float32,
        )

        result = self.adapter.predict(features)

        self.assertIn(
            result.class_name,
            EXPECTED_CLASSES,
        )

        self.assertGreaterEqual(
            result.class_index,
            0,
        )

        self.assertLess(
            result.class_index,
            len(EXPECTED_CLASSES),
        )

        self.assertEqual(
            result.probabilities.shape,
            (4,),
        )

        self.assertTrue(
            np.all(np.isfinite(result.probabilities))
        )

        self.assertAlmostEqual(
            float(np.sum(result.probabilities)),
            1.0,
            places=5,
        )

        self.assertAlmostEqual(
            result.confidence,
            float(result.probabilities[result.class_index]),
            places=6,
        )

    def test_invalid_shape_is_rejected(self) -> None:
        invalid_features = np.zeros(
            (32, 197),
            dtype=np.float32,
        )

        with self.assertRaises(ValueError):
            self.adapter.predict(invalid_features)

    def test_nonfinite_input_is_rejected(self) -> None:
        features = np.zeros(
            EXPECTED_LOG_MEL_SHAPE,
            dtype=np.float32,
        )

        features[0, 0] = np.nan

        with self.assertRaises(ValueError):
            self.adapter.predict(features)


if __name__ == "__main__":
    unittest.main(verbosity=2)
