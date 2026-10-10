import unittest
import numpy as np
from routervc.fusion.boundaries import edges, measure


class BoundariesTest(unittest.TestCase):
    def test_types_and_i(self):
        result = edges((17, 256, 256, 3), [(0, 0)], [0, 1])
        e = [r for r in result if r['category'] == 'E_nonE']
        self.assertEqual(len(e), 2)
        self.assertTrue(all(r['frames'] == list(range(1, 9)) for r in e))
        self.assertEqual(len([r for r in result if r['category'] == 'G_G']), 1)
        self.assertEqual(len([r for r in result if r['category'] == 'G_nonG']), 3)

    def test_real_edge_is_not_a_seam(self):
        x = np.zeros((17, 256, 256, 3), np.uint8); x[:, :, 64:] = 200
        edge = next(e for e in edges(x.shape, [], [0]) if e['axis'] == 'x')
        result = measure(x, x, edge)
        self.assertEqual(result['crossing_gradient_mae'], 0)
        self.assertEqual(result['band_temporal_mae'], 0)
        self.assertGreater(max(result['normal_detail_profile']), .7)
        blurred = x.copy(); blurred[:, :, 63:65] = 100
        self.assertGreater(measure(x, blurred, edge)['crossing_gradient_mae'], .7)

    def test_all_or_none_has_no_E_boundary(self):
        for packets in ([], [(c, r) for c in range(2) for r in range(16)]):
            self.assertFalse(any(e['category'] == 'E_nonE' for e in edges((17, 256, 256, 3), packets, [])))


if __name__ == '__main__': unittest.main()
