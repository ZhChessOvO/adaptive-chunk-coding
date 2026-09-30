import copy
from types import SimpleNamespace
import unittest
import torch
from torch import nn

from demo.internal_condition_model import ConditionBranch, injection, validate_bundle, make_bundle
from demo.test_feature_interface import packet


class FakeBlock(nn.Module):
    def forward(self, vid, **kwargs):
        return vid * 2


class InternalTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(10)
        self.raw = torch.randn(5,4,4,16,dtype=torch.bfloat16)

    def test_zero_initial_rgb_unchanged_and_backward(self):
        net=ConditionBranch(width=8)
        raw,side,mask=net(self.raw,[packet()])
        self.assertIs(raw,self.raw)
        self.assertEqual(side.shape,(20,8))
        self.assertEqual(side.count_nonzero(),0)
        side.sum().backward()
        self.assertGreater(net.projection.weight.grad.norm(),0)

    def test_absent_and_I_and_off_exact_after_training(self):
        net=ConditionBranch(width=8)
        nn.init.normal_(net.projection.weight); nn.init.normal_(net.projection.bias)
        for packets in ([],[dict(packet(),start=0,count=1,delta=None)]):
            self.assertEqual(net(self.raw,packets)[1].count_nonzero(),0)
        net.mode='off'
        self.assertEqual(net(self.raw,[packet()])[1].count_nonzero(),0)

    def test_zero_content_control_and_position(self):
        net=ConditionBranch(mode='zero',width=8)
        nn.init.normal_(net.projection.weight)
        p=packet();q=copy.deepcopy(p);q['delta'].normal_()
        torch.testing.assert_close(net(self.raw,[p])[1],net(self.raw,[q])[1],rtol=0,atol=0)
        net.mode='actual'
        self.assertFalse(torch.equal(net(self.raw,[p])[1],net(self.raw,[q])[1]))
        _,side,mask=net(self.raw,[p])
        self.assertEqual(side[:4].count_nonzero(),0)
        self.assertTrue((mask[4:12] == 1).all())
        self.assertEqual(side[12:].count_nonzero(),0)

    def test_hook_dtype_coverage_gradient_and_cleanup(self):
        net=ConditionBranch(width=8);nn.init.normal_(net.projection.weight,std=.1)
        _,side,mask=net(self.raw,[packet()])
        dit=SimpleNamespace(blocks=nn.ModuleList([FakeBlock() for _ in range(32)]),
                            vid_in=SimpleNamespace(patch_size=(1,2,2)))
        hidden=torch.randn(20,8,dtype=torch.bfloat16); before=hidden.clone(); records=[]
        with injection(dit,net,side,mask,records):
            output=dit.blocks[24](vid=hidden,vid_shape=torch.tensor([[5,2,2]]))
        self.assertFalse(dit.blocks[24]._forward_pre_hooks)
        torch.testing.assert_close(hidden,before,rtol=0,atol=0)
        torch.testing.assert_close(output[mask[:,0]==0],hidden[mask[:,0]==0]*2,rtol=0,atol=0)
        self.assertEqual(output.dtype,hidden.dtype)
        output.float().sum().backward()
        self.assertGreater(net.projection.weight.grad.norm(),0)
        self.assertGreater(records[0]['effective_rms'],0)
        with self.assertRaises(RuntimeError):
            with injection(dit,net,side,mask):
                raise RuntimeError('intentional')
        self.assertFalse(dit.blocks[24]._forward_pre_hooks)

    def test_geometry_dtype_and_bundle_rejected(self):
        net=ConditionBranch(width=8)
        with self.assertRaises(ValueError): net(self.raw.float(),[packet()])
        with self.assertRaises(ValueError): net(self.raw,[packet()],crop=(1,0,32,32))
        bundle=make_bundle({'state_dict':{}},net,dict(initial_adapter='x',assets={'dit':'y'}),0)
        self.assertEqual(validate_bundle(bundle),('internal','actual'))
        bundle['internal_condition_format']='bad'
        with self.assertRaises(ValueError): validate_bundle(bundle)

    def test_input_is_legacy_architecture(self):
        from demo.feature_interface_model import FeatureInterface
        from demo.feature_condition_model import conditioned_latent
        net=ConditionBranch('input');old=FeatureInterface()
        old.load_state_dict(net.encoder.state_dict())
        self.assertEqual(sum(p.numel() for p in net.parameters()),39984)
        a=net(self.raw,[packet()]);b=conditioned_latent(old,self.raw,[packet()])
        for x,y in zip(a,b): torch.testing.assert_close(x,y,rtol=0,atol=0)

    def test_internal_parameter_count_and_paired_initialization(self):
        torch.manual_seed(260930);a=ConditionBranch('internal','actual')
        torch.manual_seed(260930);b=ConditionBranch('internal','zero')
        self.assertEqual(sum(p.numel() for p in a.parameters()),124992)
        for key,value in a.state_dict().items():
            torch.testing.assert_close(value,b.state_dict()[key],rtol=0,atol=0)

    def test_noise_pair_rejects_actual_noise_mismatch(self):
        from demo.internal_condition_pipeline import assert_noise
        x=dict(before_vae=1,after_vae=2,before_diffusion=3,conditions=['same'],
               diffusion_noise={'dtype':'torch.bfloat16','sha256':'actual'})
        a={'generation_runtime':{'condition_windows':[x]}}
        assert_noise(a,copy.deepcopy(a))
        b=copy.deepcopy(a);b['generation_runtime']['condition_windows'][0]['diffusion_noise']['sha256']='changed'
        with self.assertRaises(AssertionError): assert_noise(a,b)

    def test_report_rejects_missing_duplicate_or_unpaired_bytes(self):
        from demo.internal_condition_report import grouped,CLIPS,MODES,ARMS
        rows=[dict(sample_id=s,mode=m,candidate=k,bytes=100) for s in CLIPS for m in MODES for k in ARMS]
        rows += [dict(sample_id=s,mode='full',candidate=k,bytes=100) for s in CLIPS for k in ('without','shuffled')]
        rows += [dict(sample_id=next(iter(CLIPS)),mode='full',candidate=k,bytes=100) for k in ('repeat','G_off')]
        self.assertEqual(len(grouped(rows)),58)
        with self.assertRaises(ValueError): grouped(rows[:-1])
        with self.assertRaises(ValueError): grouped(rows[:-1]+rows[:1])
        rows[0]['bytes']=101
        with self.assertRaises(AssertionError): grouped(rows)


if __name__=='__main__': unittest.main()
