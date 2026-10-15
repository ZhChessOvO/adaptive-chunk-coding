import copy
import unittest
from tools.context_training_report import compare_cases, grouped


class ReportTests(unittest.TestCase):
    def item(self,dataset='REDS',group='internal_validation'):
        item=dict(case=dict(sample_id=dataset,dataset=dataset,evaluation_group=group),results=[])
        for name,value in [('initial',.4),('local',.3),('mixed',.2)]:
            for size in (256,512):
                item['results'].append(dict(name=name,size=size,
                    quality=dict(lpips_alex=value+(size==256),psnr_db=30-value,temporal_delta_mae=value),
                    phase_error=dict(phase_rms_u8=value)))
        return item

    def test_same_view_pairing(self):
        pairs=compare_cases([self.item()])
        self.assertEqual(len(pairs),2)
        for row in pairs:self.assertAlmostEqual(row['mixed_minus_local']['lpips'],-.1)

    def test_group_and_dataset_stay_separate(self):
        result=grouped(compare_cases([self.item(),self.item('UVG'),self.item('UVG','known_artifact')]))
        self.assertEqual(len(result),6)
        for row in result:
            self.assertEqual(row['count'],1)
            self.assertEqual(row['mixed_better_than_local'],dict(lpips=1,psnr=1,phase=1,temporal_mae=1))

    def test_missing_endpoint_rejected(self):
        item=self.item();item['results'].pop()
        with self.assertRaises(ValueError):compare_cases([item])

    def test_duplicate_endpoint_rejected(self):
        item=self.item();item['results'][-1]=copy.deepcopy(item['results'][0])
        with self.assertRaises(ValueError):compare_cases([item])


if __name__=='__main__':unittest.main()
