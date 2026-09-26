"""Small CPU checks for source-only packet-benefit diagnostics."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from demo.patch_prefix_probe import diagnostic_label, packet_region, verify_artifacts
from demo.scalable_codec import atomic_bytes, file_hash


class PacketLabelTests(unittest.TestCase):
    def setUp(self):
        self.packet = SimpleNamespace(
            meta=dict(start=1, count=2, roi=[2, 1, 3, 2]), wire=b"x"*100)
        self.source = np.full((4, 5, 6, 3), 100, dtype=np.uint8)
        self.base = np.full_like(self.source, 110)

    def test_only_packet_region_and_frames_count(self):
        full = np.zeros_like(self.source)
        full[packet_region(self.packet)] = 105
        result = diagnostic_label(self.source, self.base, full, self.packet)
        self.assertEqual(result["mse_before"], 100)
        self.assertEqual(result["mse_after"], 25)
        self.assertEqual(result["squared_error_reduction"], 75*2*3*2*3)
        self.assertAlmostEqual(result["local_psnr_gain_db"], 6.020599913)
        self.assertEqual(result["packet_bytes"], 100)
        self.assertTrue(result["label_requires_source"])

    def test_harmful_packet_is_not_clipped_to_zero(self):
        full = np.full_like(self.source, 120)
        result = diagnostic_label(self.source, self.base, full, self.packet)
        self.assertLess(result["local_psnr_gain_db"], 0)
        self.assertLess(result["squared_error_reduction_per_byte"], 0)

    def test_identical_output_has_zero_gain(self):
        result = diagnostic_label(self.source, self.base, self.base, self.packet)
        self.assertEqual(result["local_psnr_gain_db"], 0)
        self.assertEqual(result["squared_error_reduction_per_byte"], 0)

    def test_resume_rejects_changed_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "artifact.bin"
            atomic_bytes(path, b"original")
            hashes = {path.name: file_hash(path)}
            verify_artifacts(root, hashes)
            atomic_bytes(path, b"changed")
            with self.assertRaisesRegex(RuntimeError, "artifact changed"):
                verify_artifacts(root, hashes)


if __name__ == "__main__":
    unittest.main()
