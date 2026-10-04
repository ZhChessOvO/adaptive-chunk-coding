import unittest
from demo import routervc_generation_schedule as s


class ScheduleTests(unittest.TestCase):
    def setUp(self):
        self.rois=[[x*128,y*128,128,128] for y in range(4) for x in range(4)]

    def test_only_adjacent_selected_cells_merge(self):
        p=s.schedule([0,1,5,15],self.rois,(512,512))
        members=[i for g in p['groups'] for i in g['members']]
        self.assertEqual(sorted(members),[0,1,5,15])
        self.assertTrue(p['selected_core_area_unchanged'])
        self.assertEqual(p['groups'][-1]['members'],[15])
        self.assertEqual(p['groups'][-1]['seed_slot'],3)
        self.assertLess(p['total_area_ratio'],1)
        self.assertLessEqual(p['max_area_ratio'],1.5)

    def test_diagonal_and_gaps_not_filled(self):
        self.assertIsNone(s.adjacent_union(self.rois[0],self.rois[5]))
        self.assertIsNone(s.adjacent_union(self.rois[0],self.rois[2]))
        self.assertEqual(s.schedule([0,5],self.rois,(512,512))['merged_calls'],2)

    def test_cap_can_preserve_original_peak_area(self):
        p=s.schedule([5,6,9,10],self.rois,(512,512),area_multiplier=1.)
        self.assertEqual(p['merged_calls'],4)
        self.assertEqual(p['max_area_ratio'],1.)

    def test_bounds_empty_and_determinism(self):
        with self.assertRaises(ValueError):s.schedule([0,0],self.rois,(512,512))
        with self.assertRaises(ValueError):s.schedule([0],self.rois,(512,512),area_multiplier=.9)
        self.assertEqual(s.schedule([],self.rois,(512,512))['merged_calls'],0)
        a=s.schedule(list(range(8)),self.rois,(512,512))
        self.assertEqual(a,s.schedule(list(range(8)),self.rois,(512,512)))
        self.assertTrue(all(len(g['members'])<=2 for g in a['groups']))


if __name__=='__main__':unittest.main()
