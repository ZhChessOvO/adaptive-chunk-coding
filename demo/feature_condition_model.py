"""Small, receiver-only adapter from decoded P8 corrections to VAE conditions.

The feature lattice has stride eight. A learned projection unpacks each joint
P8 feature into eight phases, then pools [0], [1:5], ... in the *local* 17-frame
VAE window. This is an explicit alignment heuristic, not an inverse UF head.
"""
import torch
from torch import nn

FORMAT = 'uf_delta_to_seed_condition_v1'


class FeatureCondition(nn.Module):
    def __init__(self):
        super().__init__()
        self.unpack = nn.Sequential(nn.Conv2d(513, 32, 1), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 8*16, 1))
        self.fuse = nn.Sequential(nn.Conv3d(33, 32, (1,3,3), padding=(0,1,1)),
            nn.GELU(), nn.Conv3d(32, 16, 1))
        nn.init.zeros_(self.fuse[-1].weight)
        nn.init.zeros_(self.fuse[-1].bias)

    def forward(self, condition, packets, *, start=0, crop=None):
        """condition: [5,H/8,W/8,16]; packet delta: CPU FP16 [1,512,h,w]."""
        if condition.ndim != 4 or condition.shape[0] != 5 or condition.shape[-1] != 16:
            raise ValueError('expected a 17-frame/5-latent native-scale condition')
        _, h, w, _ = condition.shape
        crop = crop or (0, 0, w*8, h*8)
        x0, y0, cw, ch = crop
        if any(v % 8 for v in crop) or (cw, ch) != (w*8, h*8):
            raise ValueError('feature condition requires aligned native-scale geometry')
        field = condition.new_zeros((17, 16, h, w), dtype=torch.float32)
        mask = condition.new_zeros((17, 1, h, w), dtype=torch.float32)
        for packet in packets:
            if packet['delta'] is None:
                continue  # I-frame RGB fallback has no invented P8 feature.
            t, n = packet['start'], packet['count']
            x, y, pw, ph = packet['roi']
            if any(v % 8 for v in (x,y,pw,ph)) or not 1 <= n <= 8 or t < 1:
                raise ValueError('invalid feature-packet geometry')
            a, b = max(t, start), min(t+n, start+17)
            l, r, u, d = max(x,x0), min(x+pw,x0+cw), max(y,y0), min(y+ph,y0+ch)
            if a >= b or l >= r or u >= d:
                continue
            delta = packet['delta'].to(device=condition.device, dtype=torch.float32)
            rms = delta.square().mean(1, keepdim=True).add(1e-8).sqrt()
            value = self.unpack(torch.cat((delta/rms, rms.log1p()), 1))
            value = value.reshape(8, 16, *value.shape[-2:])
            dst = (slice(a-start,b-start), slice(None), slice((u-y0)//8,(d-y0)//8),
                   slice((l-x0)//8,(r-x0)//8))
            if mask[dst].any():
                raise ValueError('overlapping feature packets')
            field[dst] = value[a-t:b-t, :, (u-y)//8:(d-y)//8, (l-x)//8:(r-x)//8]
            mask[dst] = 1
        groups = [slice(0,1)] + [slice(i,i+4) for i in (1,5,9,13)]
        pooled = torch.stack([field[g].mean(0) for g in groups])
        coverage = torch.stack([mask[g].mean(0) for g in groups])
        inputs = torch.cat((pooled, condition.float().permute(0,3,1,2), coverage), 1)
        side = self.fuse(inputs.permute(1,0,2,3).unsqueeze(0))[0].permute(1,0,2,3)
        side = .25 * side.float().tanh() * coverage
        return side.permute(0,2,3,1).to(condition.dtype), coverage


def conditioned_latent(adapter, condition, packets, *, start=0, crop=None):
    side, coverage = adapter(condition, packets, start=start, crop=crop)
    return condition + side, side, coverage
