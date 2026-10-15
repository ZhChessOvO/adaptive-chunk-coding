import copy
import unittest

from tools import test_g_objective_trial as trial_tests
from tools.g_objective_trial_report import comparisons, grouped


class ReportTests(unittest.TestCase):
    def item(self):
        item=trial_tests.TrialTests().item();item['case']['sample_id']='test'
        for row in item['results']:
            row['quality']=dict(row['quality']);row['wide_384_quality']=dict(row['wide_384_quality'])
            row['wide_384_quality']['lpips_alex'] = .4
            if row['name']=='control':row['wide_384_quality']['lpips_alex']=.3
            if row['name']=='relaxed_rgb_temporal':row['wide_384_quality']['lpips_alex']=.31
        return item

    def test_scope_and_causal_contrasts_stay_separate(self):
        item=self.item(); before=copy.deepcopy(item);pairs=comparisons([item])
        self.assertEqual(len(pairs),3);self.assertEqual(item,before)
        wide=next(r for r in pairs if r['score_scope']=='interior384')
        self.assertNotIn('phase',wide['initial'])
        self.assertAlmostEqual(wide['control_minus_initial']['lpips'],-.1)
        self.assertAlmostEqual(wide['relaxed_minus_control']['lpips'],.01)
        self.assertAlmostEqual(wide['relaxed_minus_initial']['lpips'],-.09)
        self.assertEqual(pairs[0]['control_minus_initial']['lpips'],0.)

    def test_group_counts_and_signs(self):
        groups=grouped(comparisons([self.item(),self.item()]))
        self.assertEqual(len(groups),3)
        wide=next(g for g in groups if g['score_scope']=='interior384')
        self.assertEqual(wide['count'],2)
        self.assertEqual(wide['improvements']['control_minus_initial']['lpips'],2)
        self.assertEqual(wide['improvements']['relaxed_minus_control']['lpips'],0)

    def test_incomplete_or_invalid_pair_rejected(self):
        for invalid in ('missing','changed','nonfinite'):
            item=self.item()
            if invalid=='missing':item['results'].pop()
            elif invalid=='changed':item['results'][-1]['observation']['noise']='different'
            else:item['results'][-1]['wide_384_quality']['lpips_alex']=float('nan')
            with self.assertRaises(ValueError):comparisons([item])


if __name__=='__main__':unittest.main()
