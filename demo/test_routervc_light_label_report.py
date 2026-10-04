import copy
import unittest
from demo.routervc_light_label_report import paired_regions, aggregate


class LightLabelReportTests(unittest.TestCase):
    def labels(self):
        old = dict(sample_id='x', dataset='REDS', sequence='0', router_split='train', regions=[
            dict(region=i, roi=[i, 0, 1, 1], costs=dict(e_packet_bytes=100),
                 quality={s: dict(lpips_alex=v) for s, v in zip(('B', 'E', 'G', 'EG'), (.5, .4, .3, .25))})
            for i in range(16)])
        new = copy.deepcopy(old)
        for region in new['regions']:
            region['costs']['e_packet_bytes'] = 60
            region['quality']['E']['lpips_alex'] = .42
            region['quality']['EG']['lpips_alex'] = .28
        return old, new

    def test_gains_and_costs_are_regional_not_stream_claims(self):
        rows = paired_regions(*self.labels()); summary = aggregate(rows)
        self.assertEqual(summary['windows'], 1)
        self.assertEqual(summary['regions'], 16)
        self.assertEqual(summary['q2']['e_packet_bytes'], 960)
        self.assertEqual(summary['q2']['EG_better_than_both'], 16)
        self.assertAlmostEqual(summary['q2']['mean_gains']['EG_over_G'], .02)
        self.assertAlmostEqual(summary['q2']['mean_gains']['EG_over_E'], .14)

    def test_identity_reuse_and_invalid_labels_fail(self):
        for mutation in ('identity', 'base', 'geometry', 'nan'):
            old, new = self.labels()
            if mutation == 'identity': new['router_split'] = 'validation'
            if mutation == 'base': new['regions'][0]['quality']['B']['lpips_alex'] = .6
            if mutation == 'geometry': new['regions'][0]['roi'] = [2, 2, 2, 2]
            if mutation == 'nan': new['regions'][0]['quality']['EG']['lpips_alex'] = float('nan')
            with self.assertRaises(ValueError): paired_regions(old, new)


if __name__ == '__main__': unittest.main()
