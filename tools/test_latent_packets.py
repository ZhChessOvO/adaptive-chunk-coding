import unittest
import numpy as np
from routervc.latent import compact_entropy as e, entropy as old
from routervc.latent import packet_format as p, format as b, chain_format as ch
from routervc.latent.split import split_symbols


class CompactTests(unittest.TestCase):
    def test_all_symbols_scales(self):
        n = np.tile(np.arange(-128, 128, dtype=np.int16), (128, 1))
        scales = np.exp(np.linspace(np.log(.11), np.log(16), 128))[:, None]*np.ones_like(n)
        c, r = split_symbols(n, 3)
        data = e.encode(c, r, scales)
        np.testing.assert_array_equal(e.decode(data, c, scales), r)
        self.assertEqual(data, e.encode(c, r, scales))

    def test_exact_native_single_group(self):
        rng = np.random.default_rng(1008)
        for coarse in (-43, -12, 0, 12, 42):
            n = rng.integers(max(-128, coarse*3-1), min(127, coarse*3+1)+1, 1000, dtype=np.int16)
            c, r = split_symbols(n, 3)
            s = np.exp(rng.uniform(np.log(.11), np.log(16), n.shape))
            data = e.encode(c, r, s)
            native = old.encode_values(r, old.scale_indexes(s), old.tables(3, coarse))
            self.assertEqual(data, native)

    def test_rans_truncation_and_trailing(self):
        c, r = split_symbols(np.arange(-128, 128, dtype=np.int16), 3)
        s = np.ones_like(c)
        data = e.encode(c, r, s)
        for length in range(len(data)):
            with self.assertRaises(ValueError):
                e.decode(data[:length], c, s)
        with self.assertRaises(ValueError):
            e.decode(data+b'x', c, s)
        with self.assertRaises(ValueError):
            e.encode(c, r, np.zeros(1))


class PacketTests(unittest.TestCase):
    def bases(self):
        single = b.base_stream(qp=48, width=3, height=576, image_width=1024,
            i_parallel=1, identities=['ab'*32]*4, intra=b'i'*4, z=b'z'*4, coarse=b'c'*4)
        meta = dict(profile='RVLC1', identities=['ab'*32]*5, frames=41, height=576,
                    width=1024, q_star=48, bin_width=3, I_qp=32, I_parallel=1)
        chain = ch.pack(meta, b'i'*4, [(b'z'*4, b'c'*4)]*5)
        return [p.base_stream(single, 1, 'cd'*32), p.base_stream(chain, 2, 'cd'*32)]

    def test_order_repeat_binding(self):
        for base in self.bases():
            a = p.packet(0, 5, b'a'*4, p.digest(base))
            c = p.packet(0, 6, b'c'*4, p.digest(base))
            self.assertEqual(p.parse(base+a+c)['packets'], p.parse(base+c+a+a)['packets'])
            self.assertEqual(p.parse(base+c+a+a)['duplicates'], 1)
            for bad in (p.packet(0, 5, b'z'*4, p.digest(base)), p.packet(0, 5, b'a'*4, b'x'*32)):
                with self.assertRaises(ValueError):
                    p.parse(base+a+bad)

    def test_all_wire_truncations(self):
        for base in self.bases():
            packet = p.packet(0, 5, b'a'*4, p.digest(base))
            for length in range(len(base)):
                with self.assertRaises(ValueError):
                    p.parse(base[:length], allow_incomplete_tail=True)
            for length in range(1, len(packet)):
                with self.assertRaises(ValueError):
                    p.parse(base+packet[:length])
                self.assertEqual(p.parse(base+packet[:length], allow_incomplete_tail=True)['packets'], {})

    def test_chunk_address(self):
        single, chain = self.bases()
        with self.assertRaises(ValueError):
            p.parse(single+p.packet(1, 5, b'a'*4, p.digest(single)))
        result = p.parse(chain+p.packet(4, 5, b'a'*4, p.digest(chain)))
        self.assertEqual(set(result['packets']), {(4, 5)})


if __name__ == '__main__':
    unittest.main()
