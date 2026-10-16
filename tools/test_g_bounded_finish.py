import unittest
from tools.g_bounded_finish import recipe, deadline_seconds, needs_fit


class BoundedTests(unittest.TestCase):
    def test_same_objective_and_fixed_horizon(self):
        old=dict(loss={'rgb_temporal':.1,'lpips':.5},phase_weight=1.,rows=[{'dataset':'REDS'},{'dataset':'UVG'}])
        cfg=recipe(old,{'path':'initial','sha256':'x'})
        self.assertEqual(cfg['steps'],800);self.assertEqual(cfg['loss'],old['loss'])
        self.assertEqual(cfg['phase_weight'],1.);self.assertEqual(len(recipe(old,{},True)['rows']),2)
        self.assertNotIn('steps',old)

    def test_deadline_timezone(self):
        self.assertEqual(deadline_seconds('2026-10-16T08:30:00+00:00'),deadline_seconds('2026-10-16T16:30:00+08:00'))
        with self.assertRaises(ValueError):deadline_seconds('2026-10-16T08:30:00')

    def test_no_silent_recipe_change(self):
        with self.assertRaises(ValueError):recipe(dict(loss={'rgb_temporal':.025},phase_weight=1.),{})

    def test_review_restart_does_not_repeat_or_extend_training(self):
        self.assertFalse(needs_fit(400,800,400,False))
        self.assertFalse(needs_fit(25,800,25,False))
        self.assertTrue(needs_fit(375,800,400,False))
        self.assertTrue(needs_fit(800,800,800,False))
        self.assertFalse(needs_fit(800,800,800,True))


if __name__=='__main__':unittest.main()
