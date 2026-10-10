"""One core R_g backbone conditioned on its own selected G set, never source."""
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from demo import routervc_receiver_router as old

FORMAT = 'routervc_set_conditioned_receiver_v1'


def identity():
    return hashlib.sha256((Path(__file__).read_bytes() + json.dumps(
        old.code_identity(), sort_keys=True).encode())).hexdigest()


class Receiver(nn.Module):
    """Same global/local visual encoder; selection enters the single main body."""
    def __init__(self, config=None):
        super().__init__()
        core = old.ReceiverGUtilityRouter(config)
        self.config = core.config
        self.encoder, self.body, self.output = core.encoder, core.body, core.output
        previous = self.body[0]
        # Entire 4x4 local selection, four immediate neighbours and call fraction.
        self.body[0] = nn.Linear(previous.in_features + 21, previous.out_features)

    def encode_time(self, frames):
        shape = frames.shape
        values = self.encoder(frames.reshape(-1, *shape[-3:])).flatten(1)
        values = values.reshape(*shape[:-3], self.config.channels)
        return torch.cat((values.mean(-2), (values[..., -1, :] - values[..., 0, :]).abs()), -1)

    def encode(self, inputs):
        if set(inputs) != {'local_pairs', 'global_pairs', 'geometry', 'coverage'}:
            raise ValueError('only decoded B/Y observations accepted')
        local = self.encode_time(inputs['local_pairs'])
        global_ = (self.encode_time(inputs['global_pairs']) if self.config.use_global else
                   local.new_zeros((len(local), self.config.channels*2)))
        return torch.cat((local, global_[:, None].expand(-1, 16, -1),
                          inputs['geometry'], inputs['coverage']), -1)

    def from_embedding(self, embedding, selected):
        if (selected.shape != (embedding.shape[0], 16, 1)
                or not torch.isfinite(selected).all()
                or not torch.all((selected == 0) | (selected == 1))):
            raise ValueError('local G state must be binary [batch,16,1]')
        s = selected.reshape(-1, 4, 4)
        pad = torch.nn.functional.pad(s, (1, 1, 1, 1))
        neighbours = torch.stack((pad[:, :-2, 1:-1], pad[:, 2:, 1:-1],
                                   pad[:, 1:-1, :-2], pad[:, 1:-1, 2:]), -1).reshape(-1, 16, 4)
        global_state = selected[:, :, 0][:, None].expand(-1, 16, -1)
        fraction = selected.mean(1, keepdim=True).expand(-1, 16, -1)
        return self.output(self.body(torch.cat((embedding, global_state, neighbours, fraction), -1)))

    def forward(self, inputs):
        if set(inputs) != {'local_pairs', 'global_pairs', 'geometry', 'coverage', 'selected'}:
            raise ValueError('unexpected receiver inputs')
        if not all(torch.isfinite(v).all() for v in inputs.values()):
            raise ValueError('nonfinite receiver input')
        return self.from_embedding(self.encode({k:v for k,v in inputs.items() if k != 'selected'}),
                                   inputs['selected'])


def initialize(path, expected):
    initial, _ = old.load_model(path, expected_sha256=expected)
    with torch.random.fork_rng(devices=[]):
        model = Receiver(initial.config)
    state = model.state_dict()
    for key, value in initial.state_dict().items():
        if key == 'body.0.weight':
            state[key].zero_(); state[key][:, :value.shape[1]] = value
        else:
            state[key] = value.clone()
    # Only an initialization approximation: local gains -> whole-picture units.
    # Measured final-picture labels, not this scaling, are the supervision.
    for key in ('output.weight', 'output.bias'):
        state[key] = state[key]/16
    model.load_state_dict(state)
    return model


def payload(model, binding, epoch, score):
    return dict(format=FORMAT, code=identity(), config=asdict(model.config),
                model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                binding=binding, epoch=epoch, score=score, masks_sent=False,
                semantic_supervision=False)


def load(path, expected):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('receiver checkpoint hash differs')
    p = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
    if (p['format'] != FORMAT or p['code'] != identity() or p['masks_sent'] is not False
            or p['semantic_supervision'] is not False):
        raise ValueError('receiver profile differs')
    with torch.random.fork_rng(devices=[]): model = Receiver(old.Config(**p['config']))
    model.load_state_dict(p['model'], strict=True)
    if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError('nonfinite model')
    return model.eval().requires_grad_(False), p


@torch.no_grad()
def select(model, base, received, coverage, max_g=8):
    """Encode once on CPU, recompute cheap conditional heads after each choice."""
    if type(max_g) is not int or not 0 <= max_g <= 16:
        raise ValueError('invalid G budget')
    if any(v.device.type != 'cpu' for v in model.parameters()):
        raise ValueError('deployment router uses CPU')
    torch.set_num_threads(4); model.eval()
    inputs = old.build_inputs(base, received, coverage, halo=0, config=model.config)
    embedding = model.encode(inputs)
    selected = torch.zeros(1, 16, 1)
    order, trace = [], []
    for step in range(max_g):
        prediction = model.from_embedding(embedding, selected)[0].numpy()
        if not np.isfinite(prediction).all(): raise ValueError('nonfinite routing gain')
        candidates = [i for i in range(16) if i not in order]
        i = max(candidates, key=lambda j:(float(prediction[j, 0]), -j))
        trace.append(dict(selected=order.copy(), predictions=prediction.tolist(), candidate=i))
        if prediction[i, 0] <= 0: break
        order.append(i); selected[0, i, 0] = 1
    return dict(indices=order, trace=trace, mask_bytes=0, source_frames_used=False,
                state_source='receiver-local decisions', max_g=max_g)
