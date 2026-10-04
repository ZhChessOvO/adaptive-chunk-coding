"""Pure CPU tests for saved-point analysis; no real data or model invocation."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

from demo import routervc_visual_report as r


class ReportTests(unittest.TestCase):
    def test_rate_interpolation_is_log_rate_bounded_and_exact_at_endpoints(self):
        rows=[dict(bpp=1.,quality=4.),dict(bpp=4.,quality=2.)]
        self.assertEqual(r.interpolate_uf(rows,2.,'quality'),3.)
        self.assertEqual(r.interpolate_uf(rows,1.,'quality'),4.)
        self.assertEqual(r.interpolate_uf(rows,4.,'quality'),2.)
        for value in (.9,4.1,float('nan')):
            self.assertIsNone(r.interpolate_uf(rows,value,'quality'))
        with self.assertRaises(ValueError):
            r.interpolate_uf([rows[0],rows[0]],1.,'quality')

    def test_curve_names_keep_g_budgets_and_arms_separate(self):
        self.assertEqual(r.curve_name('global_local_e0.5_g8'),'global_local_g8')
        self.assertEqual(r.curve_name('local_e0.25_g4'),'local_g4')
        self.assertEqual(r.curve_name('uf_qp32'),'uf')

    def test_comparison_never_compares_different_samples_or_extrapolates(self):
        rows=[]
        for group in r.GROUPS:
            for name,rate,loss in [('uf_qp8',1.,.8),('uf_qp32',4.,.4),
                                   ('global_local_e0.25_g4',2.,.5),('local_e0.25_g4',5.,.3)]:
                rows.append(dict(sample_id=group,group=group,point=name,bpp=rate,
                    lpips_alex=loss,psnr_db=20.,temporal_delta_mae=2.))
        report=r.comparisons(rows)
        for result in report['groups'].values():
            self.assertEqual(result['within_measured_UF_support'],1)
            self.assertEqual(result['router_lower_lpips_within_support'],1)
        self.assertEqual(len(report['matched_UF']),4)
        self.assertTrue(all(p['delta']['lpips_alex'] is None for p in report['matched_UF'] if not p['covered']))
        with self.assertRaises(ValueError):
            r.comparisons(rows+[rows[0]])

    def test_snapshot_detects_content_and_timing_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);r.save(root/'timing.json',dict(seconds=123.))
            before=r.snapshot(root)
            self.assertEqual(before,r.snapshot(root))
            r.save(root/'timing.json',dict(seconds=124.))
            self.assertNotEqual(before,r.snapshot(root))

    def test_audit_forbids_inference_and_preserves_original_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'formal';root.mkdir();out=Path(temp)/'analysis';out.mkdir()
            r.save(root/'protocol.json',dict(code={},models={'model':'checked'}))
            run=SimpleNamespace(root=out,check=Mock(),update=Mock())
            def validate(proxy,protocol,verify_only=False):
                from demo.conditioned_generation_pipeline import execute
                self.assertTrue(verify_only)
                self.assertEqual(proxy.root,root)
                with self.assertRaises(AssertionError):execute(None,None,None,None)
                return dict(complete=True)
            with patch('demo.routervc_visual_evaluate.code_hashes',return_value={}), \
                 patch('demo.routervc_visual_evaluate.completed_models',return_value={'model':'checked'}), \
                 patch('demo.routervc_visual_evaluate.evaluate',side_effect=validate):
                self.assertTrue(r.audit(root,run)['complete'])
            self.assertTrue(r.read(out/'audit.json')['original_records_and_timing_unchanged'])


if __name__=='__main__':unittest.main()
