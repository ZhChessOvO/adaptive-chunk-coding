"""CPU checks for fair coadaptation comparisons and each model's no-E fallback."""
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from copy import deepcopy
from unittest.mock import patch
import numpy as np
from demo import roi_condition_evaluate as evaluation
from demo.roi_condition_evaluate import ARMS, CLIPS, cases, exact_reference, grouped
from demo import roi_condition_report as report


class ComparisonTests(unittest.TestCase):
    def points(self):
        return [dict(sample_id=sid, candidate=k, mode=m, bytes=100)
                for i,sid in enumerate(CLIPS) for k,m in cases(i == 0)]

    def test_complete_unique_plan(self):
        self.assertEqual(len(grouped(self.points())), 58)
        self.assertEqual(ARMS, ('rgb','internal','zero'))

    def test_missing_and_duplicated_rejected(self):
        rows = self.points()
        for points in (rows[:-1], rows[:-1]+[rows[0]]):
            with self.assertRaises(ValueError):
                grouped(points)

    def test_identical_bytes_including_ablations(self):
        rows = self.points()
        rows[-1]['bytes'] += 1
        with self.assertRaises(AssertionError):
            grouped(rows)

    def test_no_E_compares_own_not_other_lora(self):
        self.assertEqual(exact_reference('off','none'), ('internal','none'))
        self.assertEqual(exact_reference('zero_off','none'), ('zero','none'))
        for arm in ARMS:
            self.assertIsNone(exact_reference(arm,'none'))
        self.assertEqual(exact_reference('repeat','full'), ('internal','full'))

    def test_references_always_precede_test(self):
        seen = set()
        for candidate,mode in cases(True):
            ref = exact_reference(candidate,mode)
            if ref:
                self.assertIn(ref, seen)
            if candidate not in ('rgb','G_off'):
                self.assertIn(('rgb',mode), seen)
            seen.add((candidate,mode))

    def test_completed_points_skip_metrics_and_preserve_formal_summary(self):
        # Distinct no-E values in different trained arms are legal. Each off
        # fallback has to match its own arm, not the RGB-control arm.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            refs = [dict(sample=dict(sample_id=sid,dataset='REDS'),source_hash='source',
                         points={v[1]:dict(bytes=90) for v in evaluation.MODES.values()}) for sid in CLIPS]
            for i,row in enumerate(refs):
                sid = row['sample']['sample_id']
                for k,m in cases(i == 0):
                    p = root/'evaluation'/sid/f'{k}_{m}'
                    p.mkdir(parents=True)
                    (p/'result.json').write_text(json.dumps(dict(sample_id=sid,candidate=k,mode=m,
                        bytes=100,fresh_decode=dict(bytes=100),direct=dict(bytes=90),artifacts={})))
            def read(path):
                if path == root/'queue_protocol.json':
                    return dict(code={})
                if path == evaluation.OLD/'summary.json':
                    return dict(results=refs)
                return json.loads(path.read_text())
            def frames(path):
                name = path.parent.name.rsplit('_',1)[0]
                value = 12 if name in ('zero','zero_off') else 13 if name == 'rgb' else 11
                return np.full((1,1,1,3),value,dtype=np.uint8)
            def decode(dest,row,k,m,adapter,run,disabled):
                return dest/row['sample']['sample_id']/f'{k}_{m}', dict(bytes=100)
            run = SimpleNamespace(started=time.monotonic(),check=lambda:None,update=lambda **kw:None)
            with patch.object(evaluation,'read',side_effect=read), \
                 patch.object(evaluation,'audit',return_value=dict(complete=True)), \
                 patch.object(evaluation,'file_hash',return_value='pinned'), \
                 patch.object(evaluation,'variant',side_effect=lambda source,target,mode:target), \
                 patch.object(evaluation,'identities',return_value={}), \
                 patch.object(evaluation,'load_source',return_value=None), \
                 patch.object(evaluation,'frame_hash',return_value='source'), \
                 patch.object(evaluation,'decode_point',side_effect=decode), \
                 patch.object(evaluation,'load_frames',side_effect=frames), \
                 patch.object(evaluation,'assert_noise'), \
                 patch.object(evaluation,'verify_artifacts'), \
                 patch.object(evaluation,'resources',return_value={}), \
                 patch.object(evaluation,'LPIPSAlex',side_effect=AssertionError('metric load')), \
                 patch.object(evaluation,'quality',side_effect=AssertionError('recompute')), \
                 patch.object(evaluation,'region_metrics',side_effect=AssertionError('recompute')):
                evaluation.evaluate(root,run)
                path = root/'evaluation/summary.json'
                first = path.read_bytes()
                evaluation.evaluate(root,run)
                self.assertEqual(path.read_bytes(),first)


class HistoricalComparisonTests(unittest.TestCase):
    def points(self):
        old,new = {},{}
        for i,sid in enumerate(CLIPS):
            for arm,mode in [(k,m) for k in ARMS for m in evaluation.MODES]+[('off','full')]:
                row = dict(bytes=100,dataset='REDS' if i % 2 == 0 else 'UVG',
                    roi_quality=dict(lpips_alex=.2,psnr_db=30.,temporal_delta_mae=1.),
                    quality=dict(lpips_alex=.1,psnr_db=31.,temporal_delta_mae=2.))
                key = sid,mode,arm
                old[key] = deepcopy(row)
                row['roi_quality'].update(lpips_alex=.19,psnr_db=29.9,temporal_delta_mae=1.02)
                new[key] = row
        return old,new

    def test_equal_budget_signs_and_scopes(self):
        old,new = self.points()
        value = report.paired_comparison(new,old)
        self.assertEqual(value['reused_points'],40)
        self.assertEqual(len(value['points']),40)
        for domain in ('all','REDS','UVG'):
            row = value['domains'][domain]['internal_full']
            self.assertAlmostEqual(row['roi_quality']['lpips_gain'],.01)
            self.assertAlmostEqual(row['roi_quality']['lpips_reduction_percent'],5.)
            self.assertAlmostEqual(row['roi_quality']['psnr_change_db'],-.1)
            self.assertAlmostEqual(row['roi_quality']['temporal_change'],.02)
            self.assertEqual(row['quality']['lpips_gain'],0.)

    def test_no_invented_partial_off_curve(self):
        old,new = self.points()
        value = report.paired_comparison(new,old)
        self.assertEqual({r['mode'] for r in value['points'] if r['arm']=='off'},{'full'})
        self.assertNotIn('off_partial',value['domains']['all'])

    def test_different_byte_budget_rejected(self):
        old,new = self.points()
        next(iter(new.values()))['bytes'] += 1
        with self.assertRaises(AssertionError): report.paired_comparison(new,old)

    def test_changed_historical_summary_rejected(self):
        with patch.object(report,'file_hash',return_value='different'):
            with self.assertRaises(AssertionError): report.previous_joint('pinned',{})


if __name__ == '__main__':
    unittest.main()
