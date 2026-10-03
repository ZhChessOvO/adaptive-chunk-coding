"""Pure CPU checks for matched support, paired signs and non-extrapolation."""
import copy
import unittest

from demo.routervc_analysis import (UF_METHODS,common_support,matched_points,
    paired_summary,interpolate_uf,comparisons,indexed,generation_timing)


def record(sample='r1',dataset='REDS',method='context_smooth',ratio=.5,max_g=4,bpp=.02,lpips=.2):
    return dict(sample_id=sample,dataset=dataset,method=method,ratio=ratio,max_g=max_g,
        bytes=200,bpp=bpp,quality=dict(lpips_alex=lpips,psnr_db=30.,temporal_delta_mae=2.),
        decode_seconds=4.,boundary_edges=4,components=2,actual_G_roi_calls=4,
        peak_cuda_allocated_bytes=100,native_bytes=100,selected_E=[1,2])


class AnalysisTests(unittest.TestCase):
    def test_smoke_filters_extra_main_UVG_video(self):
        rows=[record(method=m,ratio=0.,max_g=0) for m in UF_METHODS]
        rows += [record(),record('u1','UVG')]
        self.assertEqual(common_support(rows,4),[('REDS','r1')])
        points,_=matched_points(rows,common_support(rows,4),4)
        self.assertTrue(all(p['windows']==1 and p['samples']==[['REDS','r1']] for p in points))

    def test_support_intersects_every_rate_and_model(self):
        rows=[record(sample=s,method=m,ratio=0.,max_g=0) for s in ('r1','r2') for m in UF_METHODS]
        rows += [record(sample=s,ratio=r) for s in ('r1','r2') for r in (.25,.5,.75)]
        rows += [record(method='local_raw')]
        self.assertEqual(common_support(rows,4),[('REDS','r1')])
        with self.assertRaises(ValueError):common_support([record()],4)

    def test_paired_direction_and_no_unpaired_average(self):
        a=record();b=record(method='route_no_g',lpips=.35,max_g=0)
        b['bytes']=220
        out=paired_summary([(a,b)],'remove G')
        self.assertAlmostEqual(out['groups']['All']['mean_delta']['lpips_alex'],.15)
        self.assertEqual(out['groups']['All']['mean_delta']['bytes'],20)
        self.assertEqual(out['groups']['All']['variant_lower_lpips'],0)
        with self.assertRaises(ValueError):paired_summary([(a,record('different'))],'bad')

    def test_interpolation_inside_interval_only(self):
        points=[record(method='uf_qp8',bpp=.01,lpips=.4),record(method='uf_qp16',bpp=.03,lpips=.2)]
        estimate=interpolate_uf(points,.02)
        self.assertAlmostEqual(estimate['quality']['lpips_alex'],.3)
        self.assertEqual(estimate['interval'],['uf_qp8','uf_qp16'])
        for target in (.001,.04):self.assertEqual(interpolate_uf(points,target)['status'],'outside_measured_range')
        self.assertEqual(interpolate_uf(points,.01)['status'],'exact_measured_rate')
        self.assertEqual(interpolate_uf([points[0],copy.deepcopy(points[0]),points[1]],.01)['status'],'ambiguous_duplicate_rate')
        self.assertEqual(interpolate_uf([points[0],copy.deepcopy(points[0]),points[1]],.02)['status'],'ambiguous_duplicate_rate')

    def test_G_timing_measured_windows_not_area_or_network_call_count(self):
        decode=dict(generation_executed=True,generation_runtime=dict(model_load_seconds=8.,
            seconds_model_load_excluded=2.5,windows=[
                dict(runtime=dict(seconds_model_load_excluded=1.)),
                dict(runtime=dict(seconds_model_load_excluded=1.5))]))
        result=generation_timing(decode)
        self.assertEqual(result['g_execution_seconds_sum'],2.5)
        self.assertEqual(result['g_model_load_seconds'],8.)
        self.assertEqual(result['g_restore_window_calls'],2)
        decode['generation_runtime']['seconds_model_load_excluded']=123.
        with self.assertRaises(ValueError):generation_timing(decode)

    def test_G_off_is_zero_but_missing_runtime_is_unknown(self):
        self.assertEqual(generation_timing(dict(generation_executed=False))['g_execution_seconds_sum'],0.)
        missing=generation_timing(dict(generation_executed=True))
        self.assertIsNone(missing['g_execution_seconds_sum'])
        self.assertIsNone(missing['g_restore_window_calls'])

    def test_fixed_route_ablation_is_not_reallocated_e_only(self):
        reference=record();without=record(method='route_no_g',max_g=0,lpips=.3)
        reallocated=record(method='e_only',max_g=0,lpips=.9)
        result=comparisons([reference,without,reallocated],[('REDS','r1')])
        self.assertAlmostEqual(result['route_no_g']['groups']['All']['mean_delta']['lpips_alex'],.1)
        self.assertFalse(result['route_no_e']['groups'])

    def test_duplicate_operating_point_rejected(self):
        with self.assertRaises(ValueError):indexed([record(),record()])


if __name__=='__main__':unittest.main()
