"""Pure CPU logic checks; no datasets, model weights, plotting, or GPU."""
import copy
import io
import unittest

import numpy as np

from demo.routervc_report import (bits_per_pixel, choose_operating_point, fixed_selection,
    grid_statistics, normalize_record, npz_shape, route_states, summarize)


def row(sid='reds1', dataset='REDS', method='context_raw', ratio=.5, max_g=4):
    return dict(sample_id=sid, dataset=dataset, method=method, ratio=ratio, max_g=max_g,
        bytes=128, quality=dict(lpips_alex=.2, psnr_db=30., temporal_delta_mae=4.),
        decode=dict(seconds=2., peak_cuda_allocated_bytes=1024,
                    route=dict(states=['B', 'E', 'G', 'EG']*4)))


class ReportLogicTest(unittest.TestCase):
    def test_npz_geometry_reads_header_without_loading_pixels(self):
        buffer = io.BytesIO()
        np.savez_compressed(buffer, reconstruction=np.zeros((9, 8, 16, 3), dtype=np.uint8))
        buffer.seek(0)
        self.assertEqual(npz_shape(buffer), (9, 8, 16, 3))
        wrong = io.BytesIO()
        np.savez_compressed(wrong, reconstruction=np.zeros((9, 8, 16, 3), dtype=np.float32))
        wrong.seek(0)
        with self.assertRaises(ValueError):
            npz_shape(wrong)

    def test_real_geometry_and_rgb_not_channel_denominator(self):
        self.assertEqual(bits_per_pixel(128, (2, 8, 8, 3)), 8.)
        self.assertEqual(bits_per_pixel(128, (4, 8, 8, 3)), 4.)
        with self.assertRaises(ValueError):
            bits_per_pixel(128., (2, 8, 8, 3))
        with self.assertRaises(ValueError):
            bits_per_pixel(0, (2, 8, 8, 3))

    def test_four_state_route_and_g_off_fallback(self):
        value = route_states(dict(coverage=[1.]*8+[0.]*8, indices=[0, 8]))
        self.assertEqual(value[:2], [3, 1])
        self.assertEqual(value[8:10], [2, 0])
        stats = grid_statistics(value)
        self.assertEqual(stats['components'], 2)
        self.assertEqual(stats['counts'], dict(B=7, E=7, G=1, EG=1))
        self.assertEqual(route_states({}), [0]*16)
        with self.assertRaises(ValueError):
            route_states(dict(states=['B']*15))

    def test_boundaries_do_not_count_frame_edges_or_e_boundaries(self):
        self.assertEqual(grid_statistics([3]*16)['boundary_edges'], 0)
        self.assertEqual(grid_statistics([3]*16)['components'], 1)
        self.assertEqual(grid_statistics([0, 1]*8)['boundary_edges'], 0)
        checker = [(x+y)%2*2 for y in range(4) for x in range(4)]
        self.assertEqual(grid_statistics(checker)['boundary_edges'], 24)
        self.assertEqual(grid_statistics(checker)['components'], 8)

    def test_normalization_rejects_inconsistent_or_invalid_evidence(self):
        raw = row(); original = copy.deepcopy(raw)
        result = normalize_record(raw, (2, 8, 8, 3))
        self.assertEqual(raw, original)
        self.assertEqual(result['boundary_edges'], 4)
        raw['decode']['route']['boundary_edges'] = 8
        with self.assertRaises(ValueError):
            normalize_record(raw, (2, 8, 8, 3))
        raw = row();raw['quality']['lpips_alex'] = float('nan')
        with self.assertRaises(ValueError):
            normalize_record(raw, (2, 8, 8, 3))

    def test_sample_means_pooled_bpp_and_budget_separation(self):
        a = normalize_record(row(), (2, 8, 8, 3))
        b = normalize_record(row('uvg1', 'UVG'), (4, 8, 8, 3))
        c = normalize_record(row(ratio=.25), (2, 8, 8, 3))
        summaries = summarize([a, b, c])
        all_mid = next(s for s in summaries if s['dataset']=='All' and s['ratio']==.5)
        self.assertEqual(all_mid['windows'], 2)
        self.assertEqual(all_mid['bpp_mean'], 6.)
        self.assertAlmostEqual(all_mid['bpp_pooled'], 2048/384)
        self.assertEqual(len(summaries), 5)
        with self.assertRaises(ValueError):
            summarize([a, a])

    def test_visual_selection_not_based_on_quality(self):
        records = [row('r_first'), row('r_better'), row('u_first', 'UVG')]
        records[0]['quality']['lpips_alex'] = .8
        records[1]['quality']['lpips_alex'] = .01
        self.assertEqual(fixed_selection(records), dict(REDS='r_first', UVG='u_first'))
        candidates = [row(ratio=.25, max_g=8), row(ratio=.5, max_g=4), row(ratio=.75, max_g=4)]
        self.assertIs(choose_operating_point(candidates, 'context_raw'), candidates[1])
        self.assertIsNone(choose_operating_point(candidates, 'context_smooth'))


if __name__ == '__main__':
    unittest.main()
