import unittest
from unittest.mock import patch
from tools import plot_latent_followup as report


class ReportTests(unittest.TestCase):
    def test_percentage_is_mean_of_windows_not_ratio_of_totals(self):
        cases = []
        for prefix in ('reds', 'uvg'):
            for i, (old, new, oe, ne) in enumerate(((100, 90, 80, 70), (200, 150, 100, 60))):
                point = dict(actual_bytes=new, old_bytes=old, E_wire_bytes=ne,
                    old_E_bytes=oe, bpp=new/1000, quality=dict(psnr_db=30., lpips_alex=.2))
                cases.append(dict(sample=f'{prefix}-{i}', points={f'E{c}':point for c in report.COUNTS}))
        with patch.object(report, 'read', return_value={'native_bytes':300}):
            result = report.grouped_single(cases)
        self.assertAlmostEqual(result['REDS'][16]['total_saving'], .175)
        self.assertAlmostEqual(result['UVG crops'][16]['E_saving'], .2625)
        self.assertIsNone(result['REDS'][0]['E_saving'])

    def test_global_local_and_peak_remain_separate(self):
        cases = []
        for prefix in ('reds', 'uvg'):
            for i, value in enumerate((.2, .4)):
                quality = {s:dict(psnr_db=30., lpips_alex=v, temporal_delta_mae=1.)
                           for s, v in (('whole', value), ('G_regions_mean', value/2))}
                p = dict(bpp=.01, quality=quality, G_off_quality=quality,
                         peak_cuda_allocated_bytes=(4+4*i)*2**30, seconds=10.)
                cases.append(dict(sample=f'{prefix}-{i}', points={f'E{c}':p for c in (0, 4, 8, 16)}))
        result = report.g_summary(cases)['REDS'][8]
        self.assertAlmostEqual(result['G_on']['whole']['lpips_alex'], .3)
        self.assertAlmostEqual(result['G_on']['G_regions_mean']['lpips_alex'], .15)
        self.assertEqual(result['max_GiB'], 8)
        self.assertEqual(result['mean_GiB'], 6)


if __name__ == '__main__':
    unittest.main()
