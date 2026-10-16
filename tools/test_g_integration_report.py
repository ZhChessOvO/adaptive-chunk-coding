import unittest
import numpy as np
from tools.g_integration_report import group,canvas,NAMES


class ReportTests(unittest.TestCase):
    def test_cohorts_separate_and_no_hiding_regression(self):
        cases=[]
        for cohort,dataset,delta in (('diagnostic','REDS',.01),('diagnostic','UVG',-.02),
                                     ('grouped_validation','UVG',.03)):
            metrics={n:dict(quality=dict(lpips_alex=.2+(delta if n=='control' else 0),psnr_db=25.,
                temporal_delta_mae=3.),G_core_phase=dict(per_frame_rms_mean=1.)) for n in NAMES}
            cases.append(dict(cohort=cohort,dataset=dataset,bpp=.1,metrics=metrics))
        rows=group(cases);self.assertEqual(len(rows),3)
        for row in rows:
            self.assertEqual(row['count'],1)
            self.assertEqual(row['control_better_than_legacy'],int(row['dataset']=='UVG' and row['cohort']=='diagnostic'))

    def test_full_view_aspect_no_crop(self):
        frames=np.zeros((17,576,1024,3),np.uint8);frames[:,:,0]=255;frames[:,:,-1]=255
        im=canvas([('Full',frames)]*6,8,width=768)
        self.assertEqual(im.size,(2304,2*(432+54)))
        self.assertGreater(np.array(im)[54:486,0].mean(),100)


if __name__=='__main__':unittest.main()
