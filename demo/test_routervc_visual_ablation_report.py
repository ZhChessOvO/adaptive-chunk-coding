"""Small pure-data checks for the immutable multi-run report."""
import unittest
from demo import routervc_visual_ablation_report as r


class AblationReportTests(unittest.TestCase):
    def test_processing_volume_counts_overlap_not_unique_pixels(self):
        windows=[dict(runtime=dict(processing_shape=[17,256,256]))]*8
        self.assertEqual(r.processing_ratio(windows,17*512*512),2.)
        self.assertEqual(r.processing_ratio([],17*512*512),0.)
        with self.assertRaises(ValueError):r.processing_ratio([],0)

    def test_join_reuses_old_points_once(self):
        a=dict(sample_id='a',point='local_e0.5_g8')
        b=dict(sample_id='a',point='local_e0_g8')
        self.assertEqual(r.joined_rows([a],[a,b]),[a,b])
        with self.assertRaises(ValueError):r.joined_rows([a,b],[b])

    def test_paired_differences_keep_clip_and_arm(self):
        rows=[]
        for sid in ('a','b'):
            for arm in ('local','global_local'):
                for level in ('0','0.25','0.5'):
                    x=float(level)
                    rows.append(dict(sample_id=sid,group='REDS_fullview',point=f'{arm}_e{level}_g8',
                        bpp=x+1,lpips_alex=1-x,psnr_db=20+x,temporal_delta_mae=1-x))
        changes=r.paired_changes(rows)
        self.assertEqual(len(changes),8)
        self.assertTrue(all(x['delta']['lpips_alex']==-.25 for x in changes))

    def test_aggregation_does_not_mix_datasets(self):
        rows=[dict(sample_id=str(i),point='p',group=group,bpp=value,
            lpips_alex=value,psnr_db=value,temporal_delta_mae=value,receiver_seconds=value)
            for i,(group,value) in enumerate([('REDS_fullview',1),('REDS_fullview',3),('UVG_crop',8)])]
        result=r.aggregates(rows)
        self.assertEqual(result['REDS_fullview']['p']['bpp'],2)
        self.assertEqual(result['UVG_crop']['p']['bpp'],8)


if __name__=='__main__':unittest.main()
