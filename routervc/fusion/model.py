"""Small precision-aware temporal display fusion; strictly separated G support."""
import torch
from torch import nn
from torch.nn import functional as F

FORMAT = 'routervc_precision_temporal_fusion_v1'


class PrecisionFusion(nn.Module):
    """Three adjacent input frames, two spatial scales, no codec reference writes.

    rec cannot read G pixels; generated corrections are gated inside local G.
    Zero heads give exact current reconstruction, including quantized interiors.
    """
    def __init__(self):
        super().__init__()
        self.rec = nn.Sequential(nn.Conv2d(22, 16, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(16, 16, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(16, 3, 3, padding=1))
        self.stem = nn.Sequential(nn.Conv2d(36, 16, 3, padding=1), nn.SiLU())
        self.low = nn.Sequential(nn.Conv2d(16, 24, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(24, 24, 3, padding=1), nn.SiLU())
        self.head = nn.Sequential(nn.Conv2d(40, 16, 3, padding=1), nn.SiLU(),
                                  nn.Conv2d(16, 5, 3, padding=1))
        for head in (self.rec[-1], self.head[-1]):
            nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)
        x = torch.arange(-4, 5, dtype=torch.float32)
        kernel = torch.exp(-x.square()/8); kernel /= kernel.sum()
        self.register_buffer('blur', (kernel[:, None]*kernel[None, :])[None, None].repeat(3, 1, 1, 1))

    def forward(self, v):
        # Channel-major triplets contain previous, current, next RGB / precision.
        b, y, c, m, e = (v[k] for k in ('base', 'received', 'current', 'multiband', 'precision'))
        eb, gb, gs = (v[k] for k in ('e_band', 'g_band', 'g_support'))
        rec = .02*torch.tanh(self.rec(torch.cat([b, y, e, eb], 1)))*eb
        feature = self.stem(torch.cat([y, c, m, b[:, 3:6], e, gs, gb, eb], 1))
        low = self.low(F.avg_pool2d(feature, 2))
        # CUDA bilinear backward is nondeterministic in the installed PyTorch.
        # Nearest upsampling followed by the 3x3 fusion convolution keeps exact
        # per-update resume without changing the frozen G/codec paths.
        low = F.interpolate(low, size=feature.shape[-2:], mode='nearest')
        pred = torch.tanh(self.head(torch.cat([feature, low], 1)))
        delta = m[:, 3:6]-c[:, 3:6]
        smooth = F.conv2d(F.pad(delta, (4, 4, 4, 4), mode='replicate'), self.blur, groups=3)
        generated = pred[:, :1]*smooth + pred[:, 1:2]*(delta-smooth) + .02*pred[:, 2:]
        # gb already tapers to zero both outside G and away from the boundary.
        return (c[:, 3:6] + rec + gb*gs*generated).clamp(0, 1)
