import unittest
import itertools

import numpy as np
import torch

from demo.four_state_router import UtilityBackbone, gain_targets, make_views, split
from demo.four_state_router_evaluate import frontier,solve


class RouterTests(unittest.TestCase):
    def test_conditional_gain_is_not_independent_sum(self):
        scores={s:dict(lpips_alex=v,psnr_db=25.,temporal_delta_mae=3.)
                for s,v in zip(('B','E','G','EG'),(.5,.4,.3,.05))}
        without=gain_targets(scores,False);with_e=gain_targets(scores,True)
        self.assertAlmostEqual(float(with_e[0]+with_e[1]),.45,places=6)
        self.assertNotAlmostEqual(float(with_e[0]+without[1]),.45,places=6)

    def test_shapes_source_free_and_localized_supervision(self):
        base=np.zeros((17,512,512,3),np.uint8)
        enhanced=np.full_like(base,20)
        scores={s:dict(lpips_alex=.5,psnr_db=25.,temporal_delta_mae=3.) for s in ('B','E','G','EG')}
        x,y,m=make_views(base,enhanced,[scores]*16)
        self.assertEqual(x.shape,(17,16,37));self.assertEqual(y.shape,(17,16,6))
        self.assertEqual(m.sum(),32)
        self.assertEqual(x[...,36].sum(),16)
        self.assertTrue(np.isfinite(x).all())
        np.testing.assert_array_equal(x[1,1:],x[0,1:])

    def test_same_capacity_context_ablation_and_no_E_exact_zero(self):
        opts=(np.zeros(37),np.ones(37),np.zeros(6),np.ones(6))
        a,b=UtilityBackbone(True,*opts),UtilityBackbone(False,*opts)
        self.assertEqual(sum(p.numel() for p in a.parameters()),sum(p.numel() for p in b.parameters()))
        x=torch.randn(2,16,37);x[...,36]=0
        for model in (a,b):
            out=model(x)
            torch.testing.assert_close(out[...,[0,2,4]],torch.zeros(2,16,3),rtol=0,atol=0)
            out.sum().backward()

    def test_group_split(self):
        records=[dict(dataset=d,sequence=str(s)) for d in ('REDS','UVG') for s in range(10) for _ in range(2)]
        tr,va=split(records)
        groups=lambda ids:{(records[i]['dataset'],records[i]['sequence']) for i in ids}
        self.assertFalse(groups(tr)&groups(va))
        self.assertEqual(len(va),8)

    def test_four_state_budget_solver_matches_exhaustive(self):
        rng=np.random.default_rng(42)
        for _ in range(5):
            u=rng.normal(size=(4,4));u[:,0]=0
            eb=[4,7,5,3];shared=2;per=1;budget=13;limit=2
            result=solve(frontier(u,eb,shared,per),budget,limit)
            best=-float('inf')
            for states in itertools.product(range(4),repeat=4):
                ng=sum(s>=2 for s in states)
                rate=sum(b for b,s in zip(eb,states) if s in (1,3))+(shared+ng*per if ng else 0)
                if ng<=limit and rate<=budget:
                    best=max(best,float(u[np.arange(4),states].sum()))
            self.assertAlmostEqual(result['predicted_utility'],best,places=10)
            self.assertLessEqual(result['extra_bytes'],budget)
            self.assertLessEqual(result['g_calls'],limit)


if __name__=='__main__':unittest.main()
