import unittest
import numpy as np
from routervc.latent.split import split_symbols, restore_symbols, coarse_representatives


class SplitTests(unittest.TestCase):
    def test_all_signed_symbols(self):
        symbols = np.arange(-128, 128, dtype=np.int16)
        for width in (1, 3, 5, 9, 17, 63, 127):
            c, r = split_symbols(symbols, width)
            np.testing.assert_array_equal(restore_symbols(c, r, width), symbols)
            rep = coarse_representatives(c, width)
            self.assertTrue(np.all(np.abs(rep-symbols) <= width/2))
            self.assertEqual(coarse_representatives(np.array([0]), width)[0], 0)

    def test_negative_floor(self):
        c, r = split_symbols(np.array([-5, -4, -3, -2, -1, 0, 1, 2]), 3)
        np.testing.assert_array_equal(c, [-2, -1, -1, -1, 0, 0, 0, 1])
        np.testing.assert_array_equal(c*3+r, [-5, -4, -3, -2, -1, 0, 1, 2])

    def test_invalid_values(self):
        for symbols, width in [(np.array([128]), 3), (np.array([-129]), 3),
                               (np.array([1.]), 3), (np.array([0]), 2)]:
            with self.assertRaises(ValueError): split_symbols(symbols, width)
        with self.assertRaises(ValueError): restore_symbols(np.array([50]), np.array([0]), 3)
        with self.assertRaises(ValueError): restore_symbols(np.array([0]), np.array([2]), 3)
        with self.assertRaises(ValueError): coarse_representatives(np.array([99]), 3)


if __name__ == '__main__': unittest.main()
