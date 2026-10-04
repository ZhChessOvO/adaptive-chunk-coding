import unittest
from demo import routervc_efficiency_report as r


class ReportTests(unittest.TestCase):
    def test_rate_matching_is_within_each_video_not_group_means(self):
        old=[]
        for name,bpp,quality in [('global_local_e0_g8',.01,.5),
                                 ('global_local_e0.25_g8',.02,.4),
                                 ('global_local_e0.5_g8',.04,.3),
                                 ('uf_qp8',.01,.6),('uf_qp32',.04,.4)]:
            old.append(dict(sample_id='a',point=name,bpp=bpp,bytes=int(bpp*10000),
                            **{k:quality for k in r.METRICS}))
        row=dict(sample_id='a',dataset='REDS',point='global_local_e0.5_g8',
                 bpp=.02,bytes=200,quality={k:.35 for k in r.METRICS})
        # Include one UVG row because formal grouping always keeps both groups.
        other=dict(row,sample_id='b',dataset='UVG')
        result=r.pair_q2([row,other],old+[dict(x,sample_id='b') for x in old])
        self.assertAlmostEqual(result['rows'][0]['rate_matched']['old_q1_G8']['lpips_alex'],-.05)
        self.assertAlmostEqual(result['groups']['REDS']['mean_file_byte_ratio'],.5)

    def test_no_extrapolation_for_unsupported_rate(self):
        old=[];new=[]
        for sid,ds in [('a','REDS'),('b','UVG')]:
            for name,bpp in [('global_local_e0_g8',.01),('global_local_e0.5_g8',.02),('uf_qp8',.01),('uf_qp32',.02)]:
                old.append(dict(sample_id=sid,point=name,bpp=bpp,bytes=200,**{k:.4 for k in r.METRICS}))
            new.append(dict(sample_id=sid,dataset=ds,point='global_local_e0.5_g8',bpp=.03,bytes=300,quality={k:.3 for k in r.METRICS}))
        result=r.pair_q2(new,old)
        self.assertIsNone(result['rows'][0]['rate_matched']['native_UF']['lpips_alex'])
        self.assertEqual(result['groups']['REDS']['matched'],{})


if __name__=='__main__':unittest.main()
