"""Conditional R_g regression/ranking with atomic per-update exact restart."""
from pathlib import Path
import time
import numpy as np
import torch
from demo.routervc_fullview_probe import read, save, digest, immutable, verify_artifacts
from demo.chunk_enhancement_codec import atomic_torch
from demo.routervc_receiver_train import cpu_tree, _precision
from routervc.cooperation import receiver


def batch(data, device):
    return {part:{k:v.to(device) for k,v in data[part].items()} for part in ('inputs', 'targets')}


def scales(rows, get):
    square, count = torch.zeros(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64)
    for r in rows:
        if r['router_split'] != 'train': raise ValueError('validation must not set scales')
        t = get(r)['targets']; w = t['weight'].double()
        v = torch.where(w > 0, t['value'].double(), 0.)
        square += (v.square()*w).sum((0, 1)); count += w.sum((0, 1))
    if not torch.all(count > 0): raise ValueError('no measured TRAIN target')
    return (square/count).sqrt().clamp_min(1e-5).float()


@torch.no_grad()
def validate(model, rows, get, scale, device, check=lambda: None):
    groups = {}; model.eval()
    for row in rows:
        check(); b = batch(get(row), device)
        with _precision(device):
            p = model(b['inputs'])
            loss, _ = receiver.old.training_loss(p, b['targets'], gain_scale=scale, ranking_weight=.1)
        truth, known, pred = (b['targets']['value'][..., 0].cpu().numpy(),
            b['targets']['weight'][..., 0].cpu().numpy() > 0, p[..., 0].cpu().numpy())
        regrets = []
        for t, k, v in zip(truth, known, pred):
            ids = np.flatnonzero(k)
            if not len(ids): continue
            best = max(0., float(t[ids].max()))
            chosen = max(ids, key=lambda i:(float(v[i]), -int(i)))
            actual = float(t[chosen]) if v[chosen] > 0 else 0.
            regrets.append(best-actual)
        groups.setdefault(row['dataset'], []).append(dict(loss=float(loss), regret=float(np.mean(regrets))))
    by_group = {g:{k:float(np.mean([r[k] for r in rows])) for k in ('loss', 'regret')}
                for g, rows in groups.items()}
    return dict(groups=by_group, **{k:float(np.mean([v[k] for v in by_group.values()]))
                for k in ('loss', 'regret')}, scope='equal datasets; measured candidate one-step regret, not RD')


def fit(output, protocol, get, scale, *, check=lambda: None, progress=lambda **kw: None,
        stop_after=0, device='cuda'):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    binding = dict(protocol=protocol, scale=scale.tolist())
    immutable(output/'config.json', binding)
    if (output/'complete.json').exists():
        done = read(output/'complete.json'); verify_artifacts(output, done['artifacts']); return done
    train = [r for r in protocol['rows'] if r['router_split'] == 'train']
    valid = train if protocol.get('smoke') else [r for r in protocol['rows'] if r['router_split'] == 'validation']
    model = receiver.initialize(protocol['initial_path'], protocol['initial_sha256']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=protocol['learning_rate'], weight_decay=protocol['weight_decay'])
    scale = scale.to(device)
    state = dict(epoch=0, cursor=0, updates=0, epoch_loss=0., best=None, history=[], initial=None)
    resume = output/'resume.pt'
    if resume.exists():
        loaded = torch.load(resume, map_location='cpu', weights_only=True)
        if loaded['binding'] != binding: raise ValueError('training binding changed')
        model.load_state_dict(loaded['model']); optimizer.load_state_dict(loaded['optimizer'])
        state = loaded['state']; torch.set_rng_state(loaded['rng_cpu'])
        if str(device).startswith('cuda'): torch.cuda.set_rng_state_all(loaded['rng_cuda'])
    else:
        torch.manual_seed(protocol['seed'])
        if str(device).startswith('cuda'): torch.cuda.manual_seed_all(protocol['seed'])
    def checkpoint():
        atomic_torch(resume, dict(binding=binding, model=cpu_tree(model.state_dict()),
            optimizer=cpu_tree(optimizer.state_dict()), state=cpu_tree(state),
            rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all()
            if str(device).startswith('cuda') else []))
    began = time.monotonic()
    for epoch in range(state['epoch'], protocol['epochs']):
        order = torch.randperm(len(train), generator=torch.Generator().manual_seed(protocol['seed']+epoch)).tolist()
        # Calibration samples first in epoch 1: formal updates are real even while
        # later windows' labels are measured lazily. Not a queued-only handoff.
        if epoch == 0:
            calibration = [next(i for i,r in enumerate(train) if r['dataset'] == d) for d in ('REDS', 'UVG')]
            order = calibration+[i for i in order if i not in calibration]
        for position in range(state['cursor'], len(order)):
            check(); row = train[order[position]]; b = batch(get(row), device)
            with _precision(device):
                model.train(); optimizer.zero_grad(set_to_none=True)
                loss, _ = receiver.old.training_loss(model(b['inputs']), b['targets'], gain_scale=scale,
                                                      ranking_weight=protocol['ranking_weight'])
                if not torch.isfinite(loss): raise RuntimeError('nonfinite loss')
                loss.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                if not torch.isfinite(norm): raise RuntimeError('nonfinite gradients')
                optimizer.step()
            state.update(cursor=position+1, updates=state['updates']+1,
                         epoch_loss=state['epoch_loss']+float(loss.detach()))
            checkpoint()
            status = dict(phase='smoke_optimization' if protocol.get('smoke') else 'formal_receiver_optimization',
                epoch=epoch+1, epochs=protocol['epochs'], window=position+1, train_windows=len(train),
                updates=state['updates'], loss=float(loss.detach()), checkpoint_loadable=True,
                sample=row['sample_id'], seconds_this_process=time.monotonic()-began)
            save(output/'progress.json', status); progress(**status)
            print('COOPERATIVE_TRAIN', status, flush=True)
            if stop_after and state['updates'] >= stop_after: return None
        measured = validate(model, valid, get, scale, device, check)
        if state['initial'] is None:
            baseline = receiver.initialize(protocol['initial_path'], protocol['initial_sha256']).to(device)
            state['initial'] = validate(baseline, valid, get, scale, device, check); del baseline
        if state['best'] is None or (measured['regret'], measured['loss']) < (
                state['best']['score']['regret'], state['best']['score']['loss']):
            state['best'] = dict(epoch=epoch+1, score=measured, weights=cpu_tree(model.state_dict()))
        state['history'].append(dict(epoch=epoch+1, train_loss=state['epoch_loss']/len(train), validation=measured))
        state.update(epoch=epoch+1, cursor=0, epoch_loss=0.); checkpoint()
        current = cpu_tree(model.state_dict()); model.load_state_dict(state['best']['weights'])
        atomic_torch(output/'best.pt', receiver.payload(model, binding, state['best']['epoch'], state['best']['score']))
        model.load_state_dict(current); save(output/'history.json', dict(initial=state['initial'], epochs=state['history']))
    done = dict(complete=True, config=digest(output/'config.json'), updates=state['updates'],
        best={k:v for k,v in state['best'].items() if k != 'weights'},
        artifacts={n:digest(output/n) for n in ('resume.pt', 'best.pt', 'history.json')},
        sender_training_started=False, paired_picture_review_pending=True)
    save(output/'complete.json', done); progress(phase='receiver_fit_complete', updates=state['updates'])
    return done
