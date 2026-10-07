import unittest
import numpy as np

from routervc.latent import regional_format as r, format as b


class RegionalTests(unittest.TestCase):
    def base(self):
        inner = b.base_stream(qp=48, width=3, height=576, image_width=1024,
            i_parallel=1, identities=['ab'*32]*4, intra=b'i'*4, z=b'z'*4, coarse=b'c'*4)
        return r.base_stream(inner, 'cd'*32)

    def test_complete_nonoverlapping_coverage(self):
        for shape in ((1, 256, 36, 64), (1, 256, 32, 32), (1, 256, 35, 53)):
            mask = np.zeros(shape, np.uint8)
            for index in range(16):
                mask[r.region_slice(shape, index)] += 1
            self.assertTrue(np.all(mask == 1))
        for index in (-1, 16, True):
            with self.assertRaises(ValueError):
                r.region_slice((1, 256, 36, 64), index)

    def test_order_duplicate_and_base_binding(self):
        base = self.base()
        a = r.packet(5, b'a'*8, r.digest(base))
        c = r.packet(2, b'c'*8, r.digest(base))
        left = r.parse(base+a+c)
        right = r.parse(base+c+a+a)
        self.assertEqual(left['regions'], right['regions'])
        self.assertEqual(right['duplicates'], 1)
        wrong = r.packet(5, b'a'*8, b'z'*32)
        with self.assertRaises(ValueError):
            r.parse(base+wrong)
        conflict = r.packet(5, b'b'*8, r.digest(base))
        with self.assertRaises(ValueError):
            r.parse(base+a+conflict)

    def test_all_truncation_boundaries(self):
        base = self.base()
        packet = r.packet(5, b'a'*8, r.digest(base))
        for index in range(len(base)):
            with self.assertRaises(ValueError):
                r.parse(base[:index], allow_incomplete_tail=True)
        for index in range(1, len(packet)):
            with self.assertRaises(ValueError):
                r.parse(base+packet[:index])
            parsed = r.parse(base+packet[:index], allow_incomplete_tail=True)
            self.assertEqual(parsed['regions'], {})
            self.assertEqual(parsed['ignored_tail_bytes'], index)
        self.assertEqual(len(base), len(r.parse(base)['base'])+r.HEADER.size)


if __name__ == '__main__':
    unittest.main()
