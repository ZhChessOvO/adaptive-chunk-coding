import unittest
from demo import routervc_content_review as r


class ReviewTests(unittest.TestCase):
    def test_unknown_detections_do_not_become_positive_or_negative_labels(self):
        regions=[dict(region=0,roi=[0,0,10,10]),dict(region=1,roi=[10,0,10,10])]
        coverage=dict(frames=[dict(text_digits=[dict(status='present',box=[1,1,3,3]),
            dict(status='unknown',box=[11,1,3,3])],face=[])])
        self.assertEqual(r.positive_regions(coverage,regions),[0])
        self.assertFalse(r.intersects([0,0,10,10],[10,0,10,10]))

    def test_empty_measurements_remain_unknown(self):
        record=dict(sample_id='x',source_role='train',paired_roi=[0,0,10,10],paired_region=0,
            frames=[dict(frame_index=0,text_digits=[],face=[dict(status='present',box=[0,0,10,10],
                states={s:{} for s in ('B','E','G','EG')})])])
        result=r.diagnostics_summary([record])
        self.assertFalse(result['trained']);self.assertFalse(result['character_errors_trainable'])
        self.assertEqual(result['categories']['face']['counts']['B_G_unknown'],1)
        self.assertEqual(result['categories']['face']['paired_deltas'],[])


if __name__=='__main__':unittest.main()
