"""CPU checks for summary semantics; no inference or saved evidence writes."""
import unittest

from tools.plot_latent_diagnostics import interpolate_native, native_means
from tools.latent_native_baseline import QPS
from tools.latent_regional_probe import regional_change
import numpy as np


class ReportTests(unittest.TestCase):
    def test_log_rate_interpolation_and_no_extrapolation(self):
        points = [{'bpp': .1, 'quality': 10}, {'bpp': .4, 'quality': 30}]
        self.assertAlmostEqual(interpolate_native(points, .2, 'quality'), 20)
        self.assertEqual(interpolate_native(points, .1, 'quality'), 10)
        self.assertIsNone(interpolate_native(points, .05, 'quality'))
        self.assertIsNone(interpolate_native(points, .8, 'quality'))

    def test_interpolation_invalid_rates(self):
        for rates in ([.1, .1], [0, .1], [-.1, .2]):
            with self.assertRaises(ValueError):
                interpolate_native([dict(bpp=r, q=1) for r in rates], .15, 'q')

    def test_dataset_means_do_not_pool_styles(self):
        rows = [dict(sample=f'{group}-{i}', qp=qp, bpp=value,
                     quality_all9=dict(psnr_db=value, lpips_alex=value/100))
                for group, value in [('reds', 20), ('uvg', 40)]
                for i in range(2) for qp in QPS]
        result = native_means(rows)
        self.assertEqual(result['REDS'][0]['psnr_db'], 20)
        self.assertEqual(result['UVG crop'][0]['psnr_db'], 40)
        self.assertEqual(result['REDS'][0]['count'], 2)
        with self.assertRaises(ValueError):
            native_means(rows + [rows[0]])

    def test_spatial_change_keeps_outside_effects_visible(self):
        base = np.zeros((9, 64, 64, 3), np.uint8)
        out = base.copy()
        out[1:, 16:32, 16:32] = 10
        metrics = regional_change(base, out, [5])
        self.assertAlmostEqual(metrics['nominal_coverage'], 1/16)
        self.assertEqual(metrics['inside_base_change_mae'], 10)
        self.assertEqual(metrics['outside_base_change_mae'], 0)
        self.assertIsNone(regional_change(base, out, [])['inside_base_change_mae'])
        self.assertIsNone(regional_change(base, out, list(range(16)))['outside_base_change_mae'])


if __name__ == '__main__':
    unittest.main()
