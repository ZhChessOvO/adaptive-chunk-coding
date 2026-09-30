"""Small CPU checks for LoRA plus independent packet-prompt coadaptation."""
import copy
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from demo.internal_condition_model import ConditionBranch, injection
from demo.internal_condition_pipeline import assert_resume
from demo.stage_c_seedvr2_lora_utils import LoRALinear
from demo.test_feature_interface import packet


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = LoRALinear(nn.Linear(8,8),2,2.)

    def forward(self, vid, **kwargs):
        return self.linear(vid)


class JointTests(unittest.TestCase):
    def test_both_branches_receive_gradients_and_base_stays_frozen(self):
        torch.manual_seed(19)
        branch = ConditionBranch(width=8)
        block = Block()
        dit = SimpleNamespace(blocks=nn.ModuleList([nn.Identity() for _ in range(24)]
            + [block] + [nn.Identity() for _ in range(7)]),
            vid_in=SimpleNamespace(patch_size=(1,2,2)))
        raw = torch.randn(5,4,4,16,dtype=torch.bfloat16)
        hidden = torch.randn(20,8)
        parameters = [block.linear.lora_a,block.linear.lora_b,*branch.parameters()]
        optimizer = torch.optim.AdamW(parameters,lr=.01)
        frozen = block.linear.base.weight.detach().clone()
        _,side,mask = branch(raw,[packet()])
        with injection(dit,branch,side,mask):
            output = block(vid=hidden,vid_shape=torch.tensor([[5,2,2]]))
        output.square().mean().backward()
        self.assertGreater(branch.projection.weight.grad.norm(),0)
        self.assertGreater(block.linear.lora_b.grad.norm(),0)
        self.assertIsNone(block.linear.base.weight.grad)
        optimizer.step()
        torch.testing.assert_close(block.linear.base.weight,frozen,rtol=0,atol=0)
        _,empty,mask = branch(raw,[])
        with injection(dit,branch,empty,mask):
            no_E = block(vid=hidden,vid_shape=torch.tensor([[5,2,2]]))
        torch.testing.assert_close(no_E,block(vid=hidden),rtol=0,atol=0)

    def test_paired_internal_initialization(self):
        torch.manual_seed(260930); real=ConditionBranch(mode='actual',width=8)
        torch.manual_seed(260930); zero=ConditionBranch(mode='zero',width=8)
        for key,value in real.state_dict().items():
            torch.testing.assert_close(value,zero.state_dict()[key],rtol=0,atol=0)

    def test_resume_compares_lora_branch_and_optimizer(self):
        value = dict(adapter=dict(state_dict={'lora':torch.ones(2)},branch_state={'prompt':torch.ones(2)}),
            optimizer=dict(param_groups=[{'lr_scale':.2}],state={0:{'exp_avg':torch.ones(2)}}))
        assert_resume(value,copy.deepcopy(value))
        for section,key in [('state_dict','lora'),('branch_state','prompt')]:
            bad=copy.deepcopy(value);bad['adapter'][section][key][0] += 1
            with self.assertRaises(AssertionError): assert_resume(value,bad)
        bad=copy.deepcopy(value);bad['optimizer']['state'][0]['exp_avg'][0] += 1
        with self.assertRaises(AssertionError): assert_resume(value,bad)


if __name__ == '__main__': unittest.main()
