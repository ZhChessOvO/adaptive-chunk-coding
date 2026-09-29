import copy
import random
import unittest

import torch

from demo.feature_interface_model import FeatureInterface, packet_view, condition_statistics, validate_bundle
from demo.feature_condition_model import conditioned_latent
from demo.feature_interface_train import crop_for_packets, learning_rate, make_bundle
from demo.feature_interface_report import paired_clips, aggregate, CLIPS, METRICS, MODES


def packet(ident=2):
    return dict(packet_id=ident,start=1,count=8,roi=[0,0,32,32],
                delta=torch.arange(512*16).reshape(1,512,4,4).float())


class InterfaceTests(unittest.TestCase):
    def test_views_preserve_geometry_and_do_not_mutate(self):
        p=packet(); before=p['delta'].clone()
        zero=packet_view([p],'zero')[0]
        self.assertEqual(zero['roi'],p['roi']); self.assertEqual(zero['start'],p['start'])
        self.assertEqual(zero['delta'].count_nonzero(),0)
        shuffled=packet_view([p],'shuffle')[0]['delta']
        self.assertFalse(torch.equal(shuffled,before))
        torch.testing.assert_close(shuffled.flatten(-2).sort(-1).values,
                                   before.flatten(-2).sort(-1).values,rtol=0,atol=0)
        torch.testing.assert_close(shuffled,packet_view([p],'shuffle')[0]['delta'],rtol=0,atol=0)
        torch.testing.assert_close(p['delta'],before,rtol=0,atol=0)

    def test_zero_control_cannot_read_feature_values(self):
        torch.manual_seed(1); net=FeatureInterface('zero')
        torch.nn.init.normal_(net.fuse[-1].weight,std=.01)
        cond=torch.randn(5,4,4,16); p=packet(); q=copy.deepcopy(p); q['delta'].normal_()
        a,m=net(cond,[p]); b,n=net(cond,[q])
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        torch.testing.assert_close(m,n,rtol=0,atol=0)

    def test_no_E_and_I_exact_after_training_like_change(self):
        cond=torch.randn(5,4,4,16,dtype=torch.bfloat16)
        for mode in ('actual','zero','shuffle','off'):
            net=FeatureInterface(mode); torch.nn.init.normal_(net.fuse[-1].bias,std=.2)
            for received in ([],[dict(packet(),delta=None,start=0,count=1)]):
                effective,side,_=conditioned_latent(net,cond,received)
                torch.testing.assert_close(effective,cond,rtol=0,atol=0)
                self.assertEqual(side.count_nonzero(),0)

    def test_statistics_measure_rounding_after_addition(self):
        raw=torch.ones(5,2,2,16,dtype=torch.bfloat16)
        side=torch.full_like(raw,.0001); coverage=torch.ones(5,1,2,2)
        stats=condition_statistics(raw,raw+side,side,coverage)
        self.assertGreater(stats['side_rms'],0)
        self.assertEqual(stats['effective_rms'],0)
        self.assertEqual(stats['rounded_away_fraction'],1)

    def test_frozen_network_passes_input_gradient(self):
        net=FeatureInterface(); frozen=torch.nn.Linear(16,1).requires_grad_(False)
        raw=torch.randn(5,4,4,16)
        cond,_,_=conditioned_latent(net,raw,[packet()]); cond.retain_grad()
        frozen(cond).square().mean().backward()
        self.assertGreater(cond.grad.norm(),0)
        self.assertGreater(net.fuse[-1].weight.grad.norm(),0)
        self.assertTrue(all(p.grad is None for p in frozen.parameters()))

    def test_crop_is_deterministic_and_intersects_received_P(self):
        a=crop_for_packets(random.Random(2),[packet()])
        self.assertEqual(a,crop_for_packets(random.Random(2),[packet()]))
        self.assertTrue(a[0]<32 and a[1]<32 and a[0]%16==a[1]%16==0)
        with self.assertRaises(ValueError): crop_for_packets(random.Random(2),[])

    def test_learning_rate_and_bundle_identity(self):
        self.assertEqual(learning_rate(100,3000),1e-4)
        self.assertAlmostEqual(learning_rate(3000,3000),2e-5)
        self.assertEqual(learning_rate(1,3000),1e-6)
        initial={'state_dict':{'weight':torch.randn(2)},'metadata':{'old':True}}
        config=dict(initial_adapter='abc',assets=dict(dit='def'))
        bundle=make_bundle(initial,FeatureInterface(),'zero',config,3)
        torch.testing.assert_close(bundle['state_dict']['weight'],initial['state_dict']['weight'],rtol=0,atol=0)
        self.assertEqual(validate_bundle(bundle),'zero')
        bundle['feature_enabled']=False
        with self.assertRaises(ValueError): validate_bundle(bundle)


class ReportTests(unittest.TestCase):
    def points(self):
        result=[]
        for index,sid in enumerate(CLIPS):
            for mode in MODES:
                for candidate in ('actual','zero'):
                    values={m:float(index+1) for m in METRICS}
                    point=dict(bytes=100,roi_quality=values,quality=values)
                    result.append(dict(point,sample_id=sid,mode=mode,candidate=candidate,
                        dataset='REDS' if 'reds' in sid else 'UVG',direct=point,
                        previous={k:point for k in ('rgb','feature')}))
        return result

    def test_paired_aggregation_and_domain(self):
        groups=aggregate(paired_clips(self.points()),'roi_quality')
        self.assertEqual(groups['full']['all']['actual']['lpips_alex'],2.5)
        self.assertEqual(groups['full']['REDS']['actual']['lpips_alex'],2.)
        self.assertEqual(groups['full']['UVG']['actual']['lpips_alex'],3.)

    def test_missing_duplicate_and_changed_bytes_rejected(self):
        points=self.points()
        with self.assertRaises(ValueError): paired_clips(points[:-1])
        with self.assertRaises(ValueError): paired_clips(points[:-1]+points[:1])
        points[0]['bytes']=200
        with self.assertRaises(AssertionError): paired_clips(points)


if __name__ == '__main__': unittest.main()
