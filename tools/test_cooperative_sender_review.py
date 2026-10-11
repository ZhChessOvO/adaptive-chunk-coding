"""Small CPU regression tests for the sender-only actual-byte comparison."""
from pathlib import Path
import tempfile
from unittest import TestCase
import numpy as np

from tools.cooperative_sender_review import ARMS, RATIOS, location, verified_file, verify_point
from tools.cooperative_sender_review_report import aggregate, region_mask
from demo.routervc_fullview_probe import digest


class SenderReviewTests(TestCase):
    def points(self):
        result=[]
        for cohort in ('diagnostic','grouped_validation'):
            for dataset in ('REDS','UVG'):
                for cap in (RATIOS if cohort=='diagnostic' else (.25,)):
                    for arm in ARMS:
                        q=dict(lpips_alex=.3+(0 if arm=='old_sender' else -.01 if dataset=='REDS' else .02),
                               psnr_db=25., temporal_delta_mae=2.)
                        result.append(dict(cohort=cohort,dataset=dataset,sample_id=dataset,cap=cap,arm=arm,
                            bpp=.1 if arm=='old_sender' else .09,actual_bytes=100,quality=q,received_quality=q,
                            fresh_seconds=30.,G_calls=8,peak_GiB=9.,reserved_GiB=12.))
        return result

    def test_aggregation_keeps_bytes_and_negative_results(self):
        groups=aggregate(self.points());self.assertEqual(len(groups),10)
        for g in groups:
            d=g['differences']['source']
            self.assertAlmostEqual(d['delta_bpp'],-.01)
            self.assertEqual(d['improved'],int(g['dataset']=='REDS'))
            self.assertEqual(d['worsened'],int(g['dataset']=='UVG'))

    def test_missing_or_duplicate_arm_fails(self):
        ps=self.points()
        with self.assertRaises(ValueError):aggregate(ps[:-1])
        with self.assertRaises(ValueError):aggregate(ps+[ps[0]])

    def test_receipt_tamper_is_not_silent_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'file';p.write_bytes(b'original');sha=digest(p)
            verified_file(p,sha);p.write_bytes(b'changed')
            with self.assertRaises(ValueError):verified_file(p,sha)
            with self.assertRaises(ValueError):verify_point(dict(artifacts={str(p):sha}))

    def test_cohort_path_separation(self):
        a=dict(cohort='diagnostic',row=dict(sample_id='x'),cap=.5)
        self.assertEqual(location(Path('/tmp'),a),Path('/tmp/diagnostic/x/e050'))
        self.assertNotEqual(location(Path('/tmp'),a),location(Path('/tmp'),dict(a,cohort='grouped_validation')))

    def test_equal_stopped_prefixes_are_legal(self):
        from demo.routervc_sender_router import prefix_under_budget
        costs=np.full(16,10,dtype=np.int64);costs[1]=60;order=[1]
        prefixes=[prefix_under_budget(order,costs,int(costs.sum()*r))['indices'] for r in RATIOS]
        self.assertEqual(prefixes,[[],[],[1],[1]])
        # A stopped positive-gain sender need not spend all E100 bytes.
        self.assertLess(sum(costs[prefixes[-1]]),costs.sum())

    def test_E_chunk_coordinates_are_not_spatial_indices(self):
        mask=region_mask([[0,5],[1,5],[0,12],[1,12]],packet_coordinates=True)
        self.assertEqual(np.flatnonzero(mask).tolist(),[5,12])
        self.assertEqual(np.flatnonzero(region_mask([5,12])).tolist(),[5,12])
        self.assertEqual(float(region_mask([],packet_coordinates=True).sum()),0.)
        with self.assertRaises(ValueError):region_mask([[0,5]])
        with self.assertRaises(ValueError):region_mask([5],packet_coordinates=True)

    def test_comparison_never_calls_optimizer(self):
        import inspect
        import tools.cooperative_sender_review as module
        source=inspect.getsource(module)
        self.assertNotIn('run_training(',source)
        self.assertNotIn('.backward(',source)
