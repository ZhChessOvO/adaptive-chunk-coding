import unittest
from unittest.mock import patch
import numpy as np
from routervc.latent import generation as g, packet_format as p, chain_format as ch


class GenerationTests(unittest.TestCase):
    def stream(self):
        meta = dict(profile='RVLC1', identities=['ab'*32]*5, frames=17, height=576,
                    width=1024, q_star=48, bin_width=3, I_qp=32, I_parallel=1)
        base = ch.pack(meta, b'i'*4, [(b'z'*4, b'c'*4)]*2)
        return p.base_stream(base, 2, 'cd'*32)

    def test_header_prefix_and_no_assets_needed(self):
        base = self.stream()
        e = p.packet(1, 5, b'x'*4, p.digest(base))
        with patch.object(g, 'assets', side_effect=AssertionError('should not load G')):
            lower, upper = g.wrap(base, 'ef'*32), g.wrap(base+e, 'ef'*32)
            self.assertTrue(upper.startswith(lower))
            self.assertEqual(g.parse(upper)[0], base+e)
            self.assertEqual(len(upper), len(base)+len(e)+g.HEADER.size)

    def test_corrupt_and_truncated_envelope(self):
        data = g.wrap(self.stream(), 'ef'*32)
        for n in range(g.HEADER.size):
            with self.assertRaises(ValueError):
                g.parse(data[:n])
        bad = bytearray(data); bad[50] ^= 1
        with self.assertRaises(ValueError):
            g.parse(bytes(bad))

    def test_controls_and_feather(self):
        from demo.scalable_cooperation_format import weights, combine
        hashes = {k:'ab'*32 for k in ('dit', 'vae', 'positive', 'negative', 'lora', 'profile')}
        for shape in ((17, 512, 512, 3), (17, 576, 1024, 3)):
            c = g.control(shape, hashes)
            self.assertEqual(len(c['generate']), 4)
            self.assertEqual(c['protect'], [])
            alpha = weights(shape, c)
            self.assertGreater(alpha.sum(), 0)
            self.assertTrue(np.all(alpha[:, :shape[1]//4] == 0))
            base = np.zeros(shape, dtype=np.uint8)
            output = combine(base, np.full(shape, 255, dtype=np.uint8), alpha)
            np.testing.assert_array_equal(base[alpha == 0], output[alpha == 0])
        with self.assertRaises(ValueError):
            g.control((9, 512, 512, 3), hashes)


if __name__ == '__main__':
    unittest.main()
