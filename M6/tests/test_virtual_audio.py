import unittest
from pathlib import Path

import numpy as np

from M6.integration.virtual_audio import M6AudioConfig, M6VirtualAudio


class TestM6VirtualAudio(unittest.TestCase):

    def setUp(self):
        self.audio = M6VirtualAudio()

    def test_contract(self):
        summary = self.audio.contract_summary()

        self.assertEqual(summary["sample_rate_hz"], 16000)
        self.assertEqual(summary["block_size_samples"], 512)
        self.assertEqual(summary["input_channels"], ["REF", "ERR", "SPCH"])
        self.assertEqual(summary["output_channels"], 1)
        self.assertEqual(summary["internal_dtype"], "float32")
        self.assertEqual(summary["physical_validation"], "PENDING")

    def test_existing_m4_virtual_wavs_load(self):
        root = Path(__file__).resolve().parents[2]
        virtual_mics = root / "M4" / "virtual_mics"

        streams = self.audio.load_wav_inputs(
            virtual_mics / "reference.wav",
            virtual_mics / "error.wav",
            virtual_mics / "speech.wav",
        )

        self.assertEqual(set(streams), {"REF", "ERR", "SPCH"})

        for channel in ("REF", "ERR", "SPCH"):
            self.assertEqual(streams[channel].dtype, np.float32)
            self.assertEqual(streams[channel].ndim, 1)
            self.assertTrue(np.isfinite(streams[channel]).all())

    def test_block_interface(self):
        n = 1024
        streams = {
            "REF": np.zeros(n, dtype=np.float32),
            "ERR": np.zeros(n, dtype=np.float32),
            "SPCH": np.zeros(n, dtype=np.float32),
        }

        blocks = list(self.audio.iter_blocks(streams))

        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0][0], 0)

        for _, block in blocks:
            for channel in ("REF", "ERR", "SPCH"):
                self.assertEqual(block[channel].shape, (512,))
                self.assertEqual(block[channel].dtype, np.float32)

    def test_missing_channel_rejected(self):
        streams = {
            "REF": np.zeros(512, dtype=np.float32),
            "ERR": np.zeros(512, dtype=np.float32),
        }

        with self.assertRaises(KeyError):
            self.audio.validate_streams(streams)

    def test_nonfinite_input_rejected(self):
        streams = {
            "REF": np.zeros(512, dtype=np.float32),
            "ERR": np.zeros(512, dtype=np.float32),
            "SPCH": np.zeros(512, dtype=np.float32),
        }
        streams["ERR"][10] = np.nan

        with self.assertRaises(ValueError):
            self.audio.validate_streams(streams)

    def test_output_boundary(self):
        output = np.linspace(-0.5, 0.5, 512, dtype=np.float32)
        result = self.audio.simulate_output(output)

        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(result.shape, (512,))
        self.assertTrue(np.isfinite(result).all())


if __name__ == "__main__":
    unittest.main()
