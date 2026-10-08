import copy
import unittest
from tools.latent_receiver_review import summarize, COUNTS


def fixture():
    return [dict(sample_id=f'{d}{i}',dataset=d,arm=a,count=c,bpp=.1+c/100,
                 total_bytes=100+c,quality=dict(lpips_alex=.3,psnr_db=25.,temporal_delta_mae=2.),
                 generated=[] if a == 'G_off' else [0,5],seconds=10.,peak_cuda_allocated_bytes=2**30)
            for d,n in [('REDS',6),('UVG',7)] for i in range(n)
            for a in ('old_core','adapted_core','G_off') for c in COUNTS]


class ReviewTests(unittest.TestCase):
    def test_grouping_not_pooled(self):
        summary = summarize(fixture())
        self.assertEqual(summary['REDS']['adapted_core']['8']['samples'],6)
        self.assertEqual(summary['UVG']['adapted_core']['8']['samples'],7)
        self.assertEqual(summary['UVG']['G_off']['8']['mean_G_calls'],0)

    def test_incomplete_duplicates_or_changed_byte_caps_rejected(self):
        points=fixture()
        with self.assertRaises(ValueError): summarize(points[:-1])
        with self.assertRaises(ValueError): summarize(points+[points[0]])
        points[0]['total_bytes'] += 1
        with self.assertRaises(ValueError): summarize(points)


if __name__=='__main__': unittest.main()
