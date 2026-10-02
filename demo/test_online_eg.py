"""CPU regressions for packet geometry, byte-rounding and training pairing."""
import copy
import random
import unittest

import numpy as np
import torch
from torch import nn

from demo.online_eg_data import round_pixels, normalize_pixels, intersection, choose_crop, online_rgb
from demo.online_eg_pipeline import exact
from demo.online_eg_train import lr
from demo.internal_condition_model import ConditionBranch,make_bundle,validate_bundle


class TinyE(nn.Module):
    spatial_alignment = 16
    feature_halo = 4

    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(.08))

    def forward(self, source, base, features, qstep, valid_count):
        return dict(reconstruction=base+self.gain*(source-base),
                    bits=self.gain.square()+10, saturated=base.new_zeros(()))


class OnlineEGTests(unittest.TestCase):
    def setUp(self):
        self.model = TinyE()
        self.base = np.full((17,32,32,3),40,np.uint8)
        self.source = np.full_like(self.base,190)
        self.chunks = [dict(start=t,count=n,features=torch.zeros(1,1024,4,4))
                       for t,n in ((0,1),(1,8),(9,8))]
        self.packets = [dict(packet_id=1,start=1,count=8,roi=[0,0,16,16],qstep=1.)]

    def test_round_forward_and_backward(self):
        x=torch.tensor([.02,.41,.88],requires_grad=True)
        y=round_pixels(x)
        torch.testing.assert_close(y,x.detach().mul(255).round(),rtol=0,atol=0)
        y.sum().backward();torch.testing.assert_close(x.grad,torch.full_like(x,255.))

    def test_clamp_gradient(self):
        x=torch.tensor([-1.,2.],requires_grad=True)
        round_pixels(x).sum().backward()
        torch.testing.assert_close(x.grad,torch.zeros_like(x))

    def test_byte_normalization_all_levels_and_gradient(self):
        x=torch.arange(256,dtype=torch.float32,requires_grad=True)
        expected=torch.arange(256,dtype=torch.float32).div_(255).mul_(2).sub_(1)
        actual=normalize_pixels(x)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        actual.sum().backward()
        torch.testing.assert_close(x.grad,torch.full_like(x,2/255))

    def test_intersection(self):
        self.assertEqual(intersection([0,0,16,16],[8,8,16,16]),(8,8,8,8))
        self.assertIsNone(intersection([0,0,16,16],[16,0,16,16]))

    def test_no_e_exact(self):
        pixels,terms=online_rgb(self.model,self.source,self.base,self.chunks,[],[0,0,32,32])
        torch.testing.assert_close(pixels,torch.full_like(pixels,40.))
        self.assertEqual(terms['packets'],[]);self.assertEqual(float(terms['bpp']),0.)
        self.assertFalse(pixels.requires_grad)

    def test_packet_scope_and_gradient(self):
        pixels,terms=online_rgb(self.model,self.source,self.base,self.chunks,self.packets,[0,0,32,32])
        self.assertEqual(terms['coded_pixels'],8*16*16)
        self.assertEqual(terms['packets'],[1])
        expected=torch.full_like(pixels,40.);expected[1:9,:,:16,:16]=52.
        torch.testing.assert_close(pixels,expected,rtol=0,atol=0)
        pixels.sum().backward();self.assertGreater(float(self.model.gain.grad),0.)

    def test_partial_intersection_pays_whole_packet(self):
        pixels,terms=online_rgb(self.model,self.source,self.base,self.chunks,self.packets,[8,8,16,16])
        self.assertEqual(terms['coded_pixels'],8*16*16)
        expected=torch.full_like(pixels,40.);expected[1:9,:,:8,:8]=52.
        torch.testing.assert_close(pixels,expected,rtol=0,atol=0)

    def test_nonintersecting_packet_ignored(self):
        pixels,terms=online_rgb(self.model,self.source,self.base,self.chunks,self.packets,[16,16,16,16])
        self.assertEqual(terms['packets'],[])
        self.assertFalse(pixels.requires_grad)

    def test_fixed_branch_no_e_gradients(self):
        self.model.requires_grad_(False)
        pixels,terms=online_rgb(self.model,self.source,self.base,self.chunks,self.packets,[0,0,32,32],differentiable=False)
        self.assertFalse(pixels.requires_grad);self.assertFalse(terms['fidelity'].requires_grad)

    def test_crop_is_paired_and_intersects_p(self):
        packet=dict(start=1,roi=[0,0,256,256])
        a=choose_crop(random.Random(73),[packet]);b=choose_crop(random.Random(73),[packet])
        self.assertEqual(a,b);self.assertIsNotNone(intersection(a,packet['roi']))
        self.assertEqual(a[0]%16,0);self.assertEqual(a[1]%16,0)

    def test_independent_i_frame(self):
        packets=[dict(self.packets[0],start=0,count=1)]
        pixels,terms=online_rgb(self.model,self.source,self.base,self.chunks,packets,[0,0,32,32])
        self.assertEqual(terms['coded_pixels'],256)
        torch.testing.assert_close(pixels[1:],torch.full_like(pixels[1:],40.))

    def test_recursive_checkpoint_comparison(self):
        a=dict(model={'w':torch.ones(3)},optimizer={'state':{1:{'v':torch.ones(4)}}},step=3)
        b=copy.deepcopy(a);exact(a,b)
        b['model']['w'][0]=2
        with self.assertRaises(AssertionError):exact(a,b)

    def test_learning_rate_boundaries(self):
        self.assertAlmostEqual(lr(100,3000),1e-5)
        self.assertAlmostEqual(lr(3000,3000),2e-6)
        self.assertTrue(0 < lr(1,3000) < lr(100,3000))

    def test_bundle_uses_receiver_metadata_contract(self):
        config=dict(initial_adapter='a'*64,assets=dict(dit='b'*64))
        bundle=make_bundle(dict(state_dict={}),ConditionBranch('internal','off'),config,3)
        self.assertEqual(validate_bundle(bundle),('internal','off'))
        self.assertEqual(bundle['metadata']['initial_rgb_lora_sha256'],config['initial_adapter'])


if __name__=='__main__':unittest.main()
