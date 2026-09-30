"""Independent received-packet prompt at the start of SeedVR2's last 8 blocks.

No changes to the pinned input-addition model, codec or upstream DiT. The
temporal alignment remains the documented P8-phase heuristic, not exact VAE
receptive-field inversion. Single video, native scale, (1,2,2) DiT patches.
"""
from contextlib import contextmanager
import copy

import torch
from torch import nn
import torch.nn.functional as F

from demo.feature_condition_model import FeatureCondition
from demo.feature_interface_model import FeatureInterface, packet_view

FORMAT = 'received_delta_internal_block24_mean_bf16_v1'
KINDS = ('input', 'internal')
BLOCK = 24
WIDTH = 2560


class PacketEncoder(FeatureCondition):
    def __init__(self):
        super().__init__()
        # Reuse the tested packet assembly, but retain 32 separate channels.
        self.fuse[-1] = nn.Conv3d(32, 32, 1)


class ConditionBranch(nn.Module):
    def __init__(self, kind='internal', mode='actual', width=WIDTH):
        super().__init__()
        if kind not in KINDS or mode not in ('actual', 'zero', 'shuffle', 'off'):
            raise ValueError('unknown branch')
        self.kind, self.mode, self.width = kind, mode, width
        self.encoder = FeatureInterface(mode) if kind == 'input' else PacketEncoder()
        if kind == 'internal':
            self.projection = nn.Linear(32, width)
            nn.init.zeros_(self.projection.weight)
            nn.init.zeros_(self.projection.bias)

    def forward(self, raw, packets, *, start=0, crop=None):
        if raw.dtype != torch.bfloat16:
            raise ValueError('this new profile requires BF16 posterior-mean conditions')
        with torch.autocast(raw.device.type, enabled=False):
            if self.kind == 'input':
                self.encoder.mode = self.mode
                side, mask = self.encoder(raw, packets, start=start, crop=crop)
                return raw + side, side, mask
            field, coverage = self.encoder(raw.float(), packet_view(packets, self.mode),
                                           start=start, crop=crop)
            t, h, w, _ = field.shape
            if h % 2 or w % 2:
                raise ValueError('DiT patch alignment requires even latent dimensions')
            # Same T,H,W row-major token ordering as upstream NaPatchIn.
            field = F.avg_pool2d(field.permute(0,3,1,2), 2).permute(0,2,3,1)
            mask = F.avg_pool2d(coverage, 2).permute(0,2,3,1).reshape(-1,1)
            side = self.projection(field.reshape(-1,32))
            side = .25 * side.tanh() * mask
            return raw, side, mask


@contextmanager
def injection(dit, branch, side, mask, records=None):
    """Closure owns this call's tensors; never stores mutable per-window state.

    Upstream DiT does not checkpoint blocks. The training checkpoint is only
    around VAE decode, outside this scope. Recheck this if block checkpointing
    is enabled in a future profile.
    """
    if branch.kind == 'input':
        yield
        return
    if tuple(dit.vid_in.patch_size) != (1,2,2) or len(dit.blocks) != 32:
        raise ValueError('unsupported DiT layout')
    called = []

    def apply(_module, args, kwargs):
        hidden = kwargs['vid']
        shape = kwargs['vid_shape']
        if len(shape) != 1 or hidden.shape != side.shape or hidden.shape[-1] != branch.width:
            raise ValueError('packet prompt and DiT tokens are misaligned')
        # Relative scale avoids guessing the hidden-state magnitude. Scale is
        # detached: it is normalization, not a route for training frozen DiT.
        scale = hidden.detach().float().square().mean(-1,keepdim=True).add(1e-8).sqrt()
        effective = (hidden.float() + side.float()*scale).to(hidden.dtype)
        outside = (mask[:,0] == 0)
        if not torch.equal(hidden[outside], effective[outside]):
            raise RuntimeError('prompt changed an uncovered token')
        called.append(True)
        if records is not None:
            with torch.no_grad():
                records.append(dict(block=BLOCK, tokens=len(hidden), dtype=str(hidden.dtype),
                    hidden_rms=float(hidden.float().square().mean().sqrt()),
                    relative_prompt_rms=float(side.float().square().mean().sqrt()),
                    effective_rms=float((effective.float()-hidden.float()).square().mean().sqrt()),
                    outside_coverage_exact=True))
        return args, dict(kwargs, vid=effective)

    handle = dit.blocks[BLOCK].register_forward_pre_hook(apply, with_kwargs=True)
    try:
        yield
        if not called:
            raise RuntimeError('internal prompt was never applied')
    finally:
        handle.remove()


def make_bundle(initial, branch, config, step):
    bundle = copy.deepcopy(initial)
    bundle.update(internal_condition_format=FORMAT, branch_kind=branch.kind,
        branch_mode=branch.mode, branch_state={k:v.detach().cpu().clone()
                                            for k,v in branch.state_dict().items()})
    bundle['metadata'] = dict(step=step, inference_strength=1., config=config,
        frozen_lora=True, initial_rgb_lora_sha256=config['initial_adapter'],
        base_dit_sha256=config['assets']['dit'],
        interface_parameters=sum(p.numel() for p in branch.parameters()))
    return bundle


def validate_bundle(bundle):
    if (bundle.get('internal_condition_format') != FORMAT or
            bundle.get('branch_kind') not in KINDS or
            bundle.get('branch_mode') not in ('actual','zero','shuffle','off')):
        raise ValueError('invalid internal-conditioning bundle')
    return bundle['branch_kind'], bundle['branch_mode']
