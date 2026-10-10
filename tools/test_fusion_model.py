import unittest
import numpy as np
import torch

from routervc.fusion.model import PrecisionFusion
from routervc.fusion.inputs import maps, tensors


class FusionModelTest(unittest.TestCase):
    def inputs(self):
        rng = np.random.default_rng(7)
        rgb = rng.integers(0, 256, (17, 256, 256, 3), dtype=np.uint8)
        arrays = {k:rgb.copy() for k in ('base', 'received', 'current', 'multiband')}
        arrays['multiband'] = np.flip(rgb, 2).copy()
        arrays.update(maps(rgb.shape, [(0, 0), (1, 0)], [0, 1]))
        return tensors(arrays, [7, 8, 9], crop=(32, 0, 96, 96))

    def test_initial_identity(self):
        v = self.inputs(); model = PrecisionFusion()
        torch.testing.assert_close(model(v), v['current'][:, 3:6], rtol=0, atol=0)
        self.assertLess(sum(p.numel() for p in model.parameters()), 35000)

    def test_generation_cannot_escape_and_interiors_stay(self):
        v = self.inputs(); model = PrecisionFusion()
        for p in model.parameters(): torch.nn.init.normal_(p, std=.02)
        output = model(v)
        no_band = (v['e_band'] == 0) & (v['g_band'] == 0)
        mask = no_band.expand(-1, 3, -1, -1)
        torch.testing.assert_close(output[mask], v['current'][:, 3:6][mask], rtol=0, atol=0)
        changed = {k:x.clone() for k,x in v.items()}
        changed['multiband'].normal_(); changed['current'].normal_()
        # If no G, current is exactly received. Perturb only G candidates then.
        v['g_support'].zero_(); changed['g_support'].zero_()
        v['current'] = v['received'].clone(); changed['current'] = v['current'].clone()
        torch.testing.assert_close(model(v), model(changed), rtol=0, atol=0)

    def test_full_precision_g_off_bypass(self):
        a = maps((17, 256, 256, 3), [(c, i) for c in range(2) for i in range(16)], [])
        self.assertEqual(a['e_band'].max(), 0)
        self.assertEqual(a['g_band'].max(), 0)
        self.assertEqual(a['precision'][0].max(), 0)

    def test_objective_backward(self):
        from routervc.fusion.training import objective
        v = self.inputs(); model = PrecisionFusion()
        # Unit test substitutes only the expensive frozen perceptual function.
        loss, parts = objective(model(v), torch.zeros_like(v['current'][:, 3:6]), v,
                                lambda x,y: (x-y).square().mean())
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(model.head[-1].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.rec[-1].weight.grad.abs().sum().item(), 0)


if __name__ == '__main__': unittest.main()
