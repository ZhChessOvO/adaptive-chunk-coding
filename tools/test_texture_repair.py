"""CPU contract checks for the new loss, paired schedule and source separation."""
import unittest
import torch
from routervc.generation.texture_losses import (phase_bias_loss, perceptual_view,
                                                transform_spec, paired_lpips)
from routervc.generation.texture_data import choose


class RepairTests(unittest.TestCase):
    def test_correct_periodic_texture_is_not_penalized(self):
        x=torch.arange(128).reshape(1,1,1,128).float()
        target=torch.sin(x*2*torch.pi/8).expand(4,3,128,128)
        self.assertLess(abs(float(phase_bias_loss(target,target))),1e-9)

    def test_added_periodic_error_has_gradient(self):
        target=torch.zeros(4,3,128,128)
        wave=torch.sin(torch.arange(128).float()*2*torch.pi/16)[None,None,None,:]
        pred=(target+.1*wave).requires_grad_()
        large=phase_bias_loss(pred,target)
        small=phase_bias_loss(.5*pred,target)
        self.assertGreater(float(large.detach()),float(small.detach())*1.9)
        large.backward(); self.assertGreater(float(pred.grad.norm()),0)

    def test_translation_scale_paired_and_repeatable(self):
        image=torch.rand(17,3,256,256,requires_grad=True)
        metric=lambda a,b:(a-b).square().mean((1,2,3))
        value,spec=paired_lpips(image,image.detach(),metric,4,123)
        self.assertEqual(spec,transform_spec(123))
        self.assertEqual(float(value.detach()),0)
        for size in (112,120,128,136,144):
            out=perceptual_view(image,dict(size=size,dx=8,dy=-8))
            self.assertEqual(out.shape,(17,3,128,128))
            out.mean().backward(retain_graph=True)
        self.assertTrue(torch.isfinite(image.grad).all())

    def test_identity_transform_is_old_core(self):
        image=torch.rand(2,3,256,256)
        self.assertTrue(torch.equal(perceptual_view(image,dict(size=128,dx=0,dy=0)),image[...,64:192,64:192]))

    def test_deterministic_resize_matches_bilinear_value_and_gradient(self):
        import torch.nn.functional as F
        for size in (112,120,136,144):
            x=torch.randn(2,3,256,256,requires_grad=True)
            y=x.detach().clone().requires_grad_()
            edge=(256-size)//2
            actual=perceptual_view(x,dict(size=size,dx=0,dy=0))
            expected=F.interpolate(y[...,edge:edge+size,edge:edge+size],(128,128),
                                   mode='bilinear',align_corners=False,antialias=True)
            torch.testing.assert_close(actual,expected,atol=5e-7,rtol=1e-5)
            actual.square().sum().backward(); expected.square().sum().backward()
            torch.testing.assert_close(x.grad,y.grad,atol=3e-6,rtol=1e-5)

    def test_paired_schedule(self):
        rows=[dict(sample_id=d,dataset=d,shape=[17,576,1024,3]) for d in ('REDS','UVG')]
        selections=[choose(rows,s) for s in range(1,21)]
        self.assertEqual(selections,[choose(rows,s) for s in range(1,21)])
        self.assertEqual(sum(r['dataset']=='UVG' for r,c in selections),5)
        self.assertTrue(all(c[0]%16==c[1]%16==0 for r,c in selections))


if __name__=='__main__': unittest.main()
