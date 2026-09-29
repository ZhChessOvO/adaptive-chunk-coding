import unittest
import copy
from unittest.mock import patch

import torch
from torch import nn

from demo.feature_condition_model import FeatureCondition, conditioned_latent
from demo.feature_condition_cache import selected


def packet(start=1,count=8,roi=(0,0,16,16),ident=1):
    return dict(start=start,count=count,roi=list(roi),packet_id=ident,
                delta=torch.ones(1,512,roi[3]//8,roi[2]//8,dtype=torch.float16))


class Phases(nn.Module):
    def forward(self, value):
        return torch.arange(1,9,dtype=value.dtype).repeat_interleave(16).reshape(1,128,1,1).expand(1,128,*value.shape[-2:])


class FirstFeature(nn.Module):
    def forward(self,value):
        return value[:,:16] / 10


class FeatureTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.cond = torch.randn(5,4,4,16)
        self.net = FeatureCondition()

    def test_zero_initialization(self):
        actual,side,coverage = conditioned_latent(self.net,self.cond,[packet()])
        torch.testing.assert_close(actual,self.cond,rtol=0,atol=0)
        self.assertEqual(torch.count_nonzero(side),0)
        self.assertGreater(float(coverage.sum()),0)

    def test_absent_and_I_zero_even_after_learning(self):
        nn.init.normal_(self.net.fuse[-1].weight)
        nn.init.normal_(self.net.fuse[-1].bias)
        for packets in ([],[dict(packet(start=0,count=1),delta=None)]):
            actual,_,_=conditioned_latent(self.net,self.cond,packets)
            torch.testing.assert_close(actual,self.cond,rtol=0,atol=0)

    def test_phase_pool_and_partial_coverage(self):
        self.net.unpack,self.net.fuse=Phases(),FirstFeature()
        side,mask = self.net(self.cond,[packet(count=5)])
        self.assertEqual(mask[0].sum(),0)
        self.assertEqual(float(mask[1,0,0,0]),1)
        self.assertEqual(float(mask[2,0,0,0]),.25)
        self.assertAlmostEqual(float(side[1,0,0,0]),.25*torch.tanh(torch.tensor(.25)).item(),places=6)
        self.assertAlmostEqual(float(side[2,0,0,0]),.25*.25*torch.tanh(torch.tensor(.125)).item(),places=6)
        self.assertEqual(torch.count_nonzero(side[:,:,2:]),0)

    def test_non_aligned_long_window_tail(self):
        self.net.unpack,self.net.fuse=Phases(),FirstFeature()
        side,mask=self.net(self.cond,[packet(start=9,count=8),packet(start=17,count=1,ident=2)],start=16)
        self.assertAlmostEqual(float(side[0,0,0,0]),.25*torch.tanh(torch.tensor(.8)).item(),places=6)
        self.assertEqual(float(mask[1,0,0,0]),.25)
        self.assertEqual(mask[2:].sum(),0)

    def test_crop(self):
        _,mask=self.net(self.cond,[packet(roi=(32,16,16,16))],crop=(16,0,32,32))
        self.assertEqual(float(mask[:,0,:2,:].sum()),0)
        self.assertEqual(float(mask[:,0,:,0:2].sum()),0)
        self.assertGreater(float(mask.sum()),0)

    def test_reject_overlap_and_bad_geometry(self):
        with self.assertRaises(ValueError):self.net(self.cond,[packet(),packet(ident=2)])
        with self.assertRaises(ValueError):self.net(self.cond,[],crop=(1,0,32,32))
        with self.assertRaises(ValueError):self.net(self.cond[:4],[])

    def test_received_selection_only(self):
        rows=[packet(ident=i) for i in range(3)]
        self.assertEqual([p['packet_id'] for p in selected(rows,[2,0])],[2,0])
        self.assertEqual(selected(rows,[]),[])
        with self.assertRaises(ValueError):selected(rows,[4])

    def test_finite_gradient_and_bound(self):
        side,mask=self.net(self.cond,[packet()])
        side.sum().backward()
        self.assertGreater(float(self.net.fuse[-1].weight.grad.norm()),0)
        self.assertTrue(torch.isfinite(self.net.fuse[-1].weight.grad).all())
        nn.init.normal_(self.net.fuse[-1].weight,std=100)
        side,_=self.net(self.cond,[packet()])
        self.assertLessEqual(float(side.abs().max().detach()),.25)


class ReportTests(unittest.TestCase):
    @staticmethod
    def points():
        rows=[]
        for prefix,mode in enumerate(('none','partial','full')):
            for clip in range(4):
                for candidate in ('rgb','feature'):
                    value=prefix*100+clip*10+(1 if candidate=='feature' else 0)
                    def quality(v):
                        return dict(lpips_alex=v,psnr_db=v+1,temporal_delta_mae=v+2)
                    row=dict(sample_id=f'clip{clip}',mode=mode,candidate=candidate,
                        dataset='REDS' if clip<2 else 'UVG',roi_quality=quality(value),
                        quality=quality(value+1000))
                    for key,offset in [('direct',10),('previous',20)]:
                        row[key]=dict(roi_quality=quality(prefix*100+clip*10+offset),
                                      quality=quality(prefix*100+clip*10+offset+1000))
                    rows.append(row)
        return rows

    def test_pair_domain_prefix_and_scope(self):
        from demo.feature_condition_report import aggregate
        points=self.points()
        local=aggregate(points,'roi_quality')
        whole=aggregate(points,'quality')
        self.assertEqual(local['none']['all']['rgb']['lpips_alex'],15)
        self.assertEqual(local['full']['REDS']['feature']['lpips_alex'],206)
        self.assertEqual(local['partial']['UVG']['previous']['lpips_alex'],145)
        self.assertEqual(whole['full']['all']['feature']['lpips_alex'],1216)

    def test_missing_or_duplicate_clip_rejected(self):
        from demo.feature_condition_report import aggregate
        rows=self.points()
        with self.assertRaises(ValueError):aggregate(rows[:-1],'roi_quality')
        rows[-1]=copy.deepcopy(rows[-3])
        with self.assertRaises(ValueError):aggregate(rows,'roi_quality')


if __name__ == '__main__':
    unittest.main()
