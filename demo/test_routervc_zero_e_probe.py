from pathlib import Path
import tempfile
import unittest
from demo import routervc_zero_e_probe as z


class ZeroTests(unittest.TestCase):
    def test_two_arms_same_fixed_g_budget_no_new_samples(self):
        entries=[dict(sample=dict(sample_id=str(i))) for i in range(13)]
        plan=z.zero_plan(dict(sources=entries))
        self.assertEqual(len(plan),26)
        self.assertTrue(all(p['ratio']==0 and p['max_g']==8 for _,p in plan))
        self.assertEqual({p['arm'] for _,p in plan},{'local','global_local'})

    def test_reused_links_cannot_silently_point_elsewhere(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=root/'a';b=root/'b';a.mkdir();b.mkdir()
            z.link_verified(root/'link',a);z.link_verified(root/'link',a)
            with self.assertRaises(ValueError):z.link_verified(root/'link',b)


if __name__=='__main__':unittest.main()
