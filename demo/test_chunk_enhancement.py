import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
import torch
from demo.chunk_enhancement_model import ChunkEnhancement, GaussianStreams
from demo.chunk_enhancement_codec import (
    pack_region, unpack_region, region_features, encode_enhancement, decode_enhancement,
)
from demo.scalable_format import base_container, frame_hash, parse, packet_bytes


class ChunkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_native_entropy_roundtrip(self):
        coder = GaussianStreams()
        symbols = torch.tensor([-127, -9, -2, -1, 0, 1, 2, 9, 127]).reshape(1, 1, 1, 9).float()
        scales = torch.linspace(0.11, 16, 9).reshape_as(symbols)
        for _ in range(2):
            torch.testing.assert_close(symbols, coder.decode(coder.encode(symbols, scales), scales), rtol=0, atol=0)
        with self.assertRaises(ValueError):
            coder.encode(symbols * 2, scales)

    def test_forward_backward_and_real_bits(self):
        torch.manual_seed(42)
        model = ChunkEnhancement(width=16, latent=8, hyper=4)
        source, base = torch.rand(1, 24, 64, 64), torch.rand(1, 24, 64, 64)
        features = torch.randn(1, 1024, 8, 8)
        value = model(source, base, features)
        ((value["reconstruction"]-source).square().mean() + value["bits"] * 1e-6).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        model.eval()
        for count in (1, 5, 8):
            for q in (0.5, 1, 2):
                stream, expected, costs = model.compress(source, base, features, q, count)
                result = model.decompress(stream, base, features, q, count)
                torch.testing.assert_close(result, expected, rtol=0, atol=0)
                self.assertEqual(len(stream), sum(costs.values()))

    def test_tail_rectangular_and_odd_region(self):
        rng = np.random.default_rng(0)
        frames = rng.integers(0, 256, (5, 96, 160, 3), dtype=np.uint8)
        roi = (8, 16, 91, 65)
        packed = pack_region(frames, 0, 5, roi, "cpu")
        self.assertEqual(tuple(packed.shape), (1, 24, 128, 128))
        np.testing.assert_array_equal(unpack_region(packed, 5, roi), frames[:, 16:81, 8:99])
        chunk = {"features": torch.zeros(1, 1024, 16, 24)}
        self.assertEqual(tuple(region_features(chunk, roi, "cpu").shape), (1, 1024, 16, 16))
        with self.assertRaises(ValueError):
            region_features(chunk, (1, 0, 64, 64), "cpu")

    def fixture(self):
        folder = tempfile.TemporaryDirectory(prefix="chunk_unit_")
        self.addCleanup(folder.cleanup)
        weights = Path(folder.name) / "unit-model.bin"
        weights.write_bytes(b"unit model identity, not a real checkpoint")
        rng = np.random.default_rng(19)
        base = rng.integers(32, 224, (6, 80, 136, 3), dtype=np.uint8)
        source = np.clip(base.astype(np.int16)+12, 0, 255).astype(np.uint8)
        chunks = [{"start": s, "count": n, "features": torch.zeros(1, 1024, 16, 24)}
                  for s, n in ((0, 1), (1, 5))]
        meta = {"base_codec": "dcvc_uf_hts_scalar", "base_qp": 8, "skip_thres": 0.0,
                "width": 136, "height": 80, "frame_count": 6, "display_format": "rgb_u8",
                "base_rgb_sha256": frame_hash(base), "model_i_sha256": "a"*64, "model_p_sha256": "b"*64}
        original = base_container(b"mock UF: container unit test only", meta)
        model = ChunkEnhancement(width=16, latent=8, hyper=4).eval()
        prefix, wires, expected, _ = encode_enhancement(
            model, weights, original, source, base, chunks, [[8, 8, 47, 65], [64, 0, 64, 64]], 1.0)
        return model, weights, base, chunks, prefix, wires, expected

    def test_neural_container_roundtrip_reordering_and_prefix(self):
        model, weights, base, chunks, prefix, wires, expected = self.fixture()
        # The native UF path is covered by real-video smoke; this isolates packet logic.
        with patch("demo.chunk_enhancement_codec.decode_features", return_value=(base, chunks)):
            for packets in (wires, list(reversed(wires))):
                output, report = decode_enhancement(model, weights, None, prefix+b"".join(packets))
                np.testing.assert_array_equal(output, expected)
                self.assertTrue(report["non_enhanced_exact"])
            bottom, _ = decode_enhancement(model, weights, None, prefix)
            np.testing.assert_array_equal(bottom, base)
            full = prefix+b"".join(wires)
            trimmed, _ = decode_enhancement(model, weights, None, full[:-7], allow_incomplete_tail=True)
            subset, _ = decode_enhancement(model, weights, None, prefix+b"".join(wires[:-1]))
            np.testing.assert_array_equal(trimmed, subset)

    def test_neural_packet_validation(self):
        model, weights, base, chunks, prefix, wires, _ = self.fixture()
        packet = parse(prefix+wires[0]).packets[0]
        with patch("demo.chunk_enhancement_codec.decode_features", return_value=(base, chunks)):
            for field, value in (("roi", [1, 0, 64, 64]), ("roi", [128, 0, 64, 64]),
                                 ("count", 8), ("start", 2), ("qstep", 0), ("codec", "wrong")):
                meta = dict(packet.meta, **{field: value})
                with self.assertRaises(ValueError):
                    decode_enhancement(model, weights, None, prefix+packet_bytes(meta, packet.payload))
            repeated = packet_bytes(dict(packet.meta, packet_id=99), packet.payload)
            with self.assertRaisesRegex(ValueError, "overlapping"):
                decode_enhancement(model, weights, None, prefix+wires[0]+repeated)
            weights.write_bytes(b"different model identity")
            with self.assertRaisesRegex(ValueError, "match"):
                decode_enhancement(model, weights, None, prefix+wires[0])


if __name__ == "__main__":
    unittest.main()
