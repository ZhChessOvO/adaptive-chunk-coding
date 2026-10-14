from collections import Counter
import unittest
import torch
from demo.roi_condition_data import core_image_terms
from routervc.generation.context_training import choose_view,image_objectives
from routervc.generation.context_geometry import master_noise,noise_crop


def metric(a,b):return (a-b).square().mean((1,2,3),keepdim=True)


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        self.rows=[dict(dataset=d,sample_id=d,shape=[17,h,w,3]) for d,h,w in [('REDS',576,1024),('UVG',512,512)]]

    def test_pairing_and_stratified_three_quarters_wide(self):
        counts=Counter()
        for step in range(1,401):
            row,small,large,crop=choose_view(self.rows,step,123,'mixed')
            row2,small2,large2,crop2=choose_view(self.rows,step,123,'local')
            self.assertEqual((row,small,large),(row2,small2,large2));self.assertEqual(crop2,small)
            counts[row['dataset'],crop[2]]+=1
            noise=noise_crop(master_noise(step),crop,large)
            self.assertEqual(noise.shape[1],crop[2]//8)
        self.assertEqual(counts,{('REDS',256):75,('UVG',256):25,('REDS',512):225,('UVG',512):75})

    def test_original_local_image_objective_exact(self):
        torch.manual_seed(2);a=torch.randn(17,3,256,256);b=torch.randn_like(a)
        original=core_image_terms(a,b,metric,1);new,phase=image_objectives(a,b,metric,1)
        for k in original:torch.testing.assert_close(original[k],new[k],rtol=0,atol=0)
        self.assertTrue(torch.isfinite(phase))

    def test_wide_perceptual_and_periodic_gradients(self):
        torch.manual_seed(3);a=torch.randn(17,3,512,512,requires_grad=True);b=torch.randn_like(a)
        terms,phase=image_objectives(a,b,metric,2)
        (terms['lpips']+phase).backward();self.assertTrue(torch.isfinite(a.grad).all())
        self.assertGreater(float(a.grad.abs().sum()),0)
        self.assertEqual(float(a.grad[...,:64,:].abs().sum()),0)
        self.assertGreater(float(a.grad[...,64:128,64:128].abs().sum()),0)

    def test_saved_tensor_cpu_strategy_preserves_gradient(self):
        torch.manual_seed(4);a=torch.randn(8,8,requires_grad=True);b=a.detach().clone().requires_grad_(True)
        a.sin().square().sum().backward()
        with torch.autograd.graph.save_on_cpu():v=b.sin().square().sum()
        v.backward();torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
