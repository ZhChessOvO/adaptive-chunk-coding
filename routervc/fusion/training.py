"""Fixed-data fusion pilot, deterministic per-update resume, frozen perceptual loss."""
from collections import OrderedDict
import gc
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from demo.routervc_fullview_probe import read, save, digest, immutable, verify_artifacts
from demo.chunk_enhancement_codec import atomic_torch
from demo.scalable_codec import atomic_npz
from routervc.latent import router_data
from routervc.fusion.model import PrecisionFusion, FORMAT
from routervc.fusion.inputs import maps, tensors
from routervc.fusion.blend import fuse
from tools.fusion_pilot import load_capture

SEED = 20261010
CODE = ('routervc/fusion/model.py', 'routervc/fusion/inputs.py', 'routervc/fusion/training.py',
        'routervc/fusion/blend.py', 'tools/fusion_train.py')


class Samples:
    def __init__(self, root, *, max_bytes=12 << 30):
        self.root = Path(root)
        self.protocol = read(self.root/'protocol.json')
        self.done = read(self.root/'complete.json')
        if not self.done['complete'] or self.done['protocol_sha256'] != digest(self.root/'protocol.json'):
            raise ValueError('data preparation is incomplete or changed')
        self.cache = OrderedDict(); self.bytes = 0; self.max_bytes = max_bytes

    def get(self, row):
        sid = row['sample_id']
        if sid in self.cache:
            self.cache.move_to_end(sid); return self.cache[sid]
        folder = self.root/'samples'/sid/'capture'
        receipt = next(r for r in self.done['samples'] if r['sample_id'] == sid)
        if digest(folder/'complete.json') != receipt['receipt_sha256']:
            raise ValueError('capture receipt changed')
        base, received, current, patches, done = load_capture(folder)
        controls = fuse(received, current, patches, done['generated'])
        detail = read(folder/'received.json')['detail']
        arrays = dict(base=base, received=received, current=current, multiband=controls['multiband'],
                      source=router_data.source(row))
        arrays.update(maps(base.shape, detail['received_regions'], done['generated']))
        size = sum(v.nbytes for v in arrays.values())
        while self.cache and self.bytes+size > self.max_bytes:
            _, old = self.cache.popitem(last=False); self.bytes -= sum(v.nbytes for v in old.values())
        self.cache[sid] = arrays; self.bytes += size
        return arrays


def patch_batch(arrays, step, device):
    rng = np.random.default_rng(SEED+step)
    _, h, w, _ = arrays['received'].shape
    center = int(rng.integers(1, 16)); centers = [center-1, center, center+1]
    eligible = arrays['g_band'][center] + arrays['e_band'][center]
    locations = np.argwhere(eligible > .1)
    if len(locations): y, x = locations[int(rng.integers(len(locations)))]
    else: y, x = h//2, w//2
    ch = cw = 192
    x = int(np.clip(x-cw//2+int(rng.integers(-32, 33)), 0, w-cw))
    y = int(np.clip(y-ch//2+int(rng.integers(-32, 33)), 0, h-ch))
    v = tensors(arrays, centers, (x, y, cw, ch), device)
    source = torch.from_numpy(arrays['source'][centers, y:y+ch, x:x+cw].copy()).permute(0, 3, 1, 2)
    source = source.to(device=device, dtype=torch.float32)/255.
    return v, source


def objective(output, target, inputs, perceptual):
    lp = perceptual(2*output-1, 2*target-1).mean()
    pixel = F.l1_loss(output, target)
    mask = torch.maximum(inputs['e_band'], inputs['g_band'])
    err = output-target
    dx, dy = err[:, :, :, 1:]-err[:, :, :, :-1], err[:, :, 1:]-err[:, :, :-1]
    mx, my = torch.maximum(mask[:, :, :, 1:], mask[:, :, :, :-1]), torch.maximum(mask[:, :, 1:], mask[:, :, :-1])
    gradient = ((dx.abs()*mx).sum()/(3*mx.sum()).clamp_min(1) +
                (dy.abs()*my).sum()/(3*my.sum()).clamp_min(1))/2
    temporal = (err[1:]-err[:-1]).abs().mean()
    total = lp + .15*pixel + .02*gradient + .05*temporal
    return total, {k:float(v.detach()) for k, v in dict(lpips=lp, pixel=pixel, gradient=gradient, temporal=temporal).items()}


@torch.no_grad()
def validate(model, rows, samples, perceptual, check, *, all_frames=False):
    model.eval(); records = []
    centers = list(range(17)) if all_frames else [0, 4, 8, 12, 16]
    for row in rows:
        check(); arrays = samples.get(row); scores = {'current':[], 'multiband':[], 'learned':[]}
        mse = {k:[] for k in scores}; outputs = []
        for frame in centers:
            v = tensors(arrays, [frame], device='cuda')
            output = model(v).mul(255).round().clamp(0, 255).div(255)
            target = torch.from_numpy(arrays['source'][frame].copy()).permute(2, 0, 1)[None].cuda().float()/255
            for name, value in (('current', v['current'][:, 3:6]), ('multiband', v['multiband'][:, 3:6]), ('learned', output)):
                scores[name].append(float(perceptual(2*value-1, 2*target-1).mean()))
                mse[name].append(float((value-target).square().mean()))
            if all_frames: outputs.append(output[0].permute(1, 2, 0).mul(255).round().byte().cpu().numpy())
        record = dict(sample_id=row['sample_id'], dataset=row['dataset'], frames=centers,
            quality={name:dict(lpips_alex=float(np.mean(scores[name])),
                              psnr_db=-10*math.log10(max(float(np.mean(mse[name])), 1e-12))) for name in scores})
        if all_frames:
            pred = np.stack(outputs); source = arrays['source'].astype(np.float32)
            for name, value in (('current', arrays['current']), ('multiband', arrays['multiband']), ('learned', pred)):
                record['quality'][name]['temporal_delta_mae'] = float(np.abs(np.diff(value.astype(np.float32), axis=0)-np.diff(source, axis=0)).mean())
            # Returning pixels to callers is optional; normal validation remains small.
        records.append(record)
    groups = {}
    for dataset in ('REDS', 'UVG'):
        subset = [r for r in records if r['dataset'] == dataset]
        groups[dataset] = {name:{key:float(np.mean([r['quality'][name][key] for r in subset]))
                        for key in subset[0]['quality'][name]} for name in ('current', 'multiband', 'learned')}
    score = float(np.mean([groups[d]['learned']['lpips_alex'] for d in groups]))
    model.train()
    return dict(records=records, groups=groups, selection_score=score,
                selection='equal REDS/UVG mean LPIPS; fixed five-frame validation' if not all_frames else 'full17 descriptive final evaluation')


def fit(root, data, run, *, epochs=60, stop_after=0):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    samples = Samples(data); rows = samples.protocol['rows']
    train = [r for r in rows if r['router_split'] == 'train']
    valid = [r for r in rows if r['router_split'] == 'validation']
    protocol = dict(format=FORMAT, data_complete_sha256=digest(Path(data)/'complete.json'),
        code={n:digest(repo/n) for n in CODE}, epochs=epochs, steps=epochs*len(train), seed=SEED,
        learning_rate=3e-4, patch=192, temporal_input=3, temporal_outputs=3,
        loss_weights=dict(lpips=1., pixel=.15, gradient=.02, temporal=.05),
        validation_every_epochs=10, validation_frames=[0, 4, 8, 12, 16],
        selection='equal dataset mean LPIPS', frozen='UF, E, R_s, R_g, G', no_new_masks=True)
    immutable(root/'protocol.json', protocol)
    if (root/'complete.json').exists():
        done = read(root/'complete.json'); verify_artifacts(root, done['artifacts']); return
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    import lpips
    metric = lpips.LPIPS(net='alex', verbose=False).cuda().eval().requires_grad_(False)
    model = PrecisionFusion().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    step, best = 0, float('inf'); history = []
    if (root/'resume.pt').exists():
        state = torch.load(root/'resume.pt', map_location='cpu', weights_only=True)
        if state['protocol_sha256'] != digest(root/'protocol.json'): raise ValueError('training recipe changed')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['cpu_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
        step, best, history = state['step'], state['best'], state['history']
    if stop_after and step >= stop_after:
        return
    start = time.monotonic(); losses = []

    def checkpoint():
        atomic_torch(root/'resume.pt', dict(format=FORMAT, protocol_sha256=digest(root/'protocol.json'),
            model=model.state_dict(), optimizer=optimizer.state_dict(), step=step, best=best, history=history,
            cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()))

    while step < protocol['steps']:
        run.check()
        epoch, position = divmod(step, len(train))
        order = np.random.default_rng(SEED+epoch).permutation(len(train))
        row = train[int(order[position])]
        arrays = samples.get(row); v, target = patch_batch(arrays, step, 'cuda')
        model.train(); optimizer.zero_grad(set_to_none=True)
        output = model(v); loss, parts = objective(output, target, v, metric)
        if not torch.isfinite(loss): raise ValueError('nonfinite fusion objective')
        loss.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(norm): raise ValueError('nonfinite fusion gradient')
        optimizer.step(); step += 1; losses.append(parts)
        if step % len(train) == 0 and ((epoch+1) % 10 == 0 or step == protocol['steps']):
            run.update(phase='fusion_validation', step=step, epoch=epoch+1)
            result = validate(model, valid, samples, metric, run.check)
            result.update(epoch=epoch+1, step=step)
            save(root/f'validation_{epoch+1:03d}.json', result)
            history.append(dict(epoch=epoch+1, step=step, score=result['selection_score'],
                                file=f'validation_{epoch+1:03d}.json'))
            if result['selection_score'] < best:
                best = result['selection_score']
                atomic_torch(root/'best.pt', dict(format=FORMAT, model=model.state_dict(), step=step,
                    epoch=epoch+1, score=best, protocol_sha256=digest(root/'protocol.json'),
                    data_complete_sha256=protocol['data_complete_sha256'], code=protocol['code']))
        checkpoint()  # A completed optimizer update is the resumable unit.
        with (root/'train.jsonl').open('a') as f:
            import json
            f.write(json.dumps(dict(step=step, epoch=epoch+1, sample=row['sample_id'], **parts))+'\n')
        run.update(phase='fusion_fit', step=step, total_steps=protocol['steps'], epoch=epoch+1,
                   loss=parts, model_parameters=sum(p.numel() for p in model.parameters()))
        if stop_after and step >= stop_after:
            save(root/'intentional_stop.json', dict(step=step, resume_sha256=digest(root/'resume.pt'))); return
    state = torch.load(root/'best.pt', map_location='cpu', weights_only=True); model.load_state_dict(state['model'])
    result = validate(model, valid, samples, metric, run.check, all_frames=True)
    save(root/'validation_final_full17.json', result)
    save(root/'complete.json', dict(complete=True, step=step, selected_epoch=state['epoch'],
        model_parameters=sum(p.numel() for p in model.parameters()), seconds_this_process=time.monotonic()-start,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        scope='bounded fixed-policy fusion pilot; no model promotion or new stream profile yet',
        artifacts={n:digest(root/n) for n in ('protocol.json', 'best.pt', 'resume.pt', 'validation_final_full17.json')}))
    run.update(phase='fusion_fit_complete', step=step); run.log_resources()
