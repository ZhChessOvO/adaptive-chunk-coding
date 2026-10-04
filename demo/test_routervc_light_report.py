import unittest
from demo.routervc_light_report import interpolate,comparisons,METRICS


class LightReportTests(unittest.TestCase):
    def test_duplicate_prefix_rates_are_not_extrapolated(self):
        curve=[dict(bpp=.01,lpips_alex=.4),dict(bpp=.01,lpips_alex=.4)]
        self.assertEqual(interpolate(curve,.01,'lpips_alex'),.4)
        self.assertIsNone(interpolate(curve,.02,'lpips_alex'))
        self.assertIsNone(interpolate(curve,.005,'lpips_alex'))
        with self.assertRaises(ValueError):
            interpolate(curve+[dict(bpp=.01,lpips_alex=.5)],.01,'lpips_alex')

    def test_log_rate_interpolation_and_scope(self):
        self.assertAlmostEqual(interpolate([dict(bpp=1.,q=0.),dict(bpp=4.,q=2.)],2.,'q'),1.)
        rows=[];old=[]
        for ds in ('REDS','UVG'):
            old += [dict(sample_id=ds,point=f'uf_qp{i}',bpp=b,**{k:.6 for k in METRICS}) for i,b in enumerate((.01,.1))]
            for arm in ('global_local','local'):
                for version in ('old','new'):
                    for ratio,bpp in ((0.,.01),(.25,.02),(.5,.04)):
                        rows.append(dict(sample_id=ds,dataset=ds,arm=arm,version=version,ratio=ratio,bpp=bpp,
                            E_indices=[] if ratio==0 else [0],G_indices=[1],
                            **{k:.5 if version=='old' else .4 for k in METRICS}))
        result=comparisons(rows,old)
        for ds in ('REDS','UVG'):
            for arm in ('global_local','local'):
                group=result['groups'][ds][arm]
                self.assertEqual(group['points'],2)  # E0 is excluded from aggregate adaptation counts.
                self.assertEqual(group['matched']['old_q2']['covered'],2)
                self.assertEqual(group['matched']['old_q2']['lower_lpips'],2)
                self.assertAlmostEqual(group['matched']['old_q2']['mean_delta']['lpips_alex'],-.1)


if __name__=='__main__':unittest.main()
