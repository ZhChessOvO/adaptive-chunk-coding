import unittest
from unittest.mock import patch
import numpy as np

from routervc.latent import routing as r, packet_format as p, chain_format as ch


def bank():
    meta = dict(profile='RVLC1', identities=['ab'*32]*5, frames=17,
                height=576, width=1024, q_star=48, bin_width=3, I_qp=32, I_parallel=1)
    base = p.base_stream(ch.pack(meta, b'i'*4, [(b'z'*4,b'c'*4)]*2), 2, 'cd'*32)
    return base+b''.join(p.packet(j, i, bytes([i])*8, p.digest(base))
                         for i in range(16) for j in range(2))


class RoutingTests(unittest.TestCase):
    def test_live_profile_identity(self):
        self.assertEqual(len(r.identity()), 64)
        self.assertEqual(r.identity(), r.identity())

    def test_bundle_prefix_bytes_coverage(self):
        full = bank()
        empty, one, two = [r.subset(full, s) for s in ([], [5], [5, 9])]
        self.assertTrue(two.startswith(one) and one.startswith(empty))
        self.assertEqual(len(one)-len(empty), r.bundle_bytes(full)[5])
        np.testing.assert_array_equal(r.coverage(p.parse(one)), np.eye(16, dtype=np.float32)[5])
        partial = empty+p.packet(1,5,b'x'*8,p.digest(empty))
        self.assertEqual(r.coverage(p.parse(partial))[5], .5)
        with self.assertRaises(ValueError): r.subset(full, [5,5])
        with self.assertRaises(ValueError): r.subset(partial, [5])

    @patch.object(r, 'identity', return_value='ef'*32)
    def test_envelope_preserves_prefix_and_rejects_corruption(self, _):
        lo, hi = [r.wrap(r.subset(bank(), s), 'ab'*32, 'cd'*32) for s in ([],[5])]
        self.assertTrue(hi.startswith(lo))
        inner, config, parsed = r.parse(hi)
        self.assertEqual(len(hi)-len(inner), r.HEADER_BYTES)
        self.assertEqual(config['max_g'], 8)
        for index in (0,32,r.HEADER_BYTES-1):
            broken = bytearray(hi); broken[index] ^= 1
            with self.assertRaises(ValueError): r.parse(bytes(broken))
        for length in range(r.HEADER_BYTES):
            with self.assertRaises(ValueError): r.parse(hi[:length])

    def test_G_seed_is_spatial_not_selection_ordinal(self):
        a, b = [r.control((17,576,1024,3), {}, i) for i in (5,9)]
        self.assertEqual(b['seed']-a['seed'], 4*65536)
        self.assertEqual(a['generate'], [[0,17,256,144,256,144]])
        self.assertEqual(a['protect'], [])

    def test_render_always_reads_original_Y_and_order_is_irrelevant(self):
        from demo.scalable_cooperation_format import weights
        pixels = np.zeros((17,256,256,3), dtype=np.uint8)
        observed = []
        def generate(received, settings):
            observed.append(received is pixels)
            out = received.copy()
            out[weights(received.shape, settings)>0] = settings['seed']%251+1
            return out, {}
        left,_ = r.render(pixels, [0,15], {}, generate)
        right,_ = r.render(pixels, [15,0], {}, generate)
        np.testing.assert_array_equal(left,right)
        self.assertTrue(all(observed))


if __name__ == '__main__': unittest.main()
