import unittest
import numpy as np
import torch
from demo.chunk_enhancement_model import ChunkEnhancement, GaussianStreams
from demo.chunk_enhancement_codec import pack_region, unpack_region, region_features


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


if __name__ == "__main__":
    unittest.main()
