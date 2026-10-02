"""Crossed-model summaries must retain the measured four-clip comparison."""
import unittest

from demo.online_eg_cross_report import crossed_summary, COMBINATIONS
from demo.online_eg_eval_core import CLIPS, METRICS
from demo.online_eg_evaluate import jobs


class CrossSummaryTests(unittest.TestCase):
    def rows(self):
        rows = []
        for i, sid in enumerate(CLIPS):
            for job in jobs(i == 0):
                # Unrelated E-only/no-packet points must not enter full-G means.
                value = i if job['name'] in COMBINATIONS else 1000
                rows.append(dict(sample_id=sid, name=job['name'],
                    dataset='REDS' if i % 2 == 0 else 'UVG', bytes=100+value,
                    **{scope: {m: float(value) for m in METRICS}
                       for scope in ('quality', 'roi_quality')}))
        return rows

    def test_domain_and_clip_weights(self):
        result = crossed_summary(self.rows())
        for domain, expected in [('all', 1.5), ('REDS', 1.), ('UVG', 2.)]:
            self.assertEqual(set(result[domain]), set(COMBINATIONS))
            for row in result[domain].values():
                self.assertEqual(row['bytes'], 100+expected)
                self.assertEqual(row['roi_quality']['lpips_alex'], expected)
                self.assertEqual(row['quality']['psnr_db'], expected)

    def test_missing_or_duplicate_points_rejected(self):
        rows = self.rows()
        for bad in (rows[:-1], rows[:-1]+[rows[0]]):
            with self.assertRaises(ValueError): crossed_summary(bad)

    def test_mislabeled_domains_rejected(self):
        rows = self.rows()
        for row in rows:
            row['dataset'] = 'REDS'
        with self.assertRaises(AssertionError): crossed_summary(rows)


if __name__ == '__main__': unittest.main()
