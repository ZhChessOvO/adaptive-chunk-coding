import copy
import unittest

from tools.latent_system_review import summarize, panel_label
from tools.latent_receiver_audit import paired_summary


class SystemReviewTests(unittest.TestCase):
    def test_panel_label_fits_single_header_line(self):
        point=dict(arm='zero_source',bpp=.030123,quality={'lpips_alex':.23456})
        label=panel_label(point)
        self.assertNotIn('\n',label)
        self.assertIn('0.0301 bpp',label)
        self.assertIn('L 0.235',label)
        self.assertLess(len(label),48)

    def points(self):
        return [dict(sample_id=d,dataset=d,arm=arm,budget=b,bpp=.03,actual_bytes=300,
                quality=dict(lpips_alex=.2,psnr_db=30.,temporal_delta_mae=5.),
                G_calls=0 if arm in ('UF','source_Goff') else 8,peak_GiB=9.,fresh_seconds=50.)
            for d in ('REDS','UVG') for arm in ('source','zero_source','fixed_order','source_Goff','UF')
            for b in ((8,48) if arm=='UF' else (0.,.5))]

    def test_summarize_complete_separated_datasets(self):
        result=summarize(self.points(),(0.,.5),(8,48),2)
        self.assertEqual(result['REDS']['source']['0.5']['samples'],1)
        self.assertEqual(result['UVG']['UF']['48']['mean_G_calls'],0)

    def test_incomplete_and_duplicate_rejected(self):
        points=self.points()
        with self.assertRaises(ValueError):summarize(points[:-1],(0.,.5),(8,48),2)
        with self.assertRaises(ValueError):summarize(points[:-1]+points[:1],(0.,.5),(8,48),2)

    def test_receiver_pairing_requires_real_equal_bytes(self):
        points=[dict(sample_id=d,dataset=d,arm=a,count=c,total_bytes=100,
            quality=dict(lpips_alex=.3-(.01 if a=='adapted_core' else 0),psnr_db=30.,temporal_delta_mae=4.))
            for d in ('REDS','UVG') for a in ('old_core','adapted_core','G_off') for c in (0,4,8,16)]
        result=paired_summary(points)
        self.assertAlmostEqual(result['REDS']['8']['lpips_gain'],.01)
        changed=copy.deepcopy(points)
        next(p for p in changed if p['arm']=='adapted_core')['total_bytes']+=1
        with self.assertRaises(ValueError):paired_summary(changed)


if __name__=='__main__':unittest.main()
