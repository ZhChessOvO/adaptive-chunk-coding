"""Single core R_g adaptation with new TRAIN-only scales and exact restart."""
from pathlib import Path
import time

import torch

from demo import routervc_receiver_router as receiver
from demo.routervc_receiver_train import cpu_tree, arm_batch, validate, _precision
from demo.routervc_fullview_probe import read, digest, save, immutable, verify_artifacts
from demo.chunk_enhancement_codec import atomic_torch


def loss_scales(rows, get):
    """Never read validation labels while estimating normalization."""
    square, count = torch.zeros(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64)
    for row in rows:
        if row['router_split'] != 'train':
            continue
        targets = get(row)['targets']
        value, weight = targets['value'].double(), targets['weight'].double()
        if not torch.isfinite(value[weight > 0]).all():
            raise ValueError('nonfinite training labels')
        square += (torch.where(weight > 0, value, 0).square()*weight).sum((0, 1))
        count += weight.sum((0, 1))
    if not torch.all(count > 0):
        raise ValueError('no TRAIN supervision for a receiver metric')
    return (square/count).sqrt().clamp_min(1e-5).float()


def fit(output, protocol, get, *, check=lambda: None, progress=lambda **kw: None,
        stop_after=0, device='cuda'):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    binding = dict(protocol=protocol, role='new latent B/E conditional G receiver only')
    immutable(output/'config.json', binding)
    if (output/'complete.json').exists():
        done = read(output/'complete.json')
        if done['config'] != digest(output/'config.json'):
            raise ValueError('completed training config changed')
        verify_artifacts(output, done['artifacts'])
        return done
    train = [r for r in protocol['rows'] if r['router_split'] == 'train']
    valid = protocol['rows'] if protocol['smoke'] else [r for r in protocol['rows']
                                                     if r['router_split'] == 'validation']
    initial, _ = receiver.load_model(protocol['initial_path'], expected_sha256=protocol['initial_sha256'])
    with torch.random.fork_rng(devices=[]):
        model = receiver.ReceiverGUtilityRouter(initial.config).to(device)
    model.load_state_dict(initial.state_dict(), strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=protocol['learning_rate'],
                                 weight_decay=protocol['weight_decay'])
    resume = output/'resume.pt'
    scale_path = output/'train_scales.json'
    if scale_path.exists():
        scale_record = read(scale_path)
        if scale_record['config'] != digest(output/'config.json'):
            raise ValueError('loss scale binding changed')
        scale = torch.tensor(scale_record['scale'])
    else:
        scale = loss_scales(train, get)
        save(scale_path, dict(config=digest(output/'config.json'), scale=scale.tolist(),
                             split='train only', windows=len(train)))
    scale = scale.to(device)
    state = dict(epoch=0, cursor=0, updates=0, epoch_loss=0., history=[], best=None, initial=None)
    if resume.exists():
        loaded = torch.load(resume, weights_only=True, map_location='cpu')
        if loaded['binding'] != binding:
            raise ValueError('resume configuration changed')
        torch.testing.assert_close(loaded['scale'], scale.cpu(), rtol=0, atol=0)
        model.load_state_dict(loaded['model']); optimizer.load_state_dict(loaded['optimizer'])
        state = loaded['state']
        torch.set_rng_state(loaded['rng_cpu'])
        if str(device).startswith('cuda'):
            torch.cuda.set_rng_state_all(loaded['rng_cuda'])
    else:
        torch.manual_seed(protocol['seed'])
        if str(device).startswith('cuda'):
            torch.cuda.manual_seed_all(protocol['seed'])

    def checkpoint():
        atomic_torch(resume, dict(binding=binding, model=cpu_tree(model.state_dict()),
            optimizer=cpu_tree(optimizer.state_dict()), state=cpu_tree(state), scale=scale.cpu(),
            rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all()
                if str(device).startswith('cuda') else []))

    if state['initial'] is None:
        initial = initial.to(device)
        state['initial'] = validate(initial, 'core', valid, get, scale, device, check)
        checkpoint()
    del initial
    began = time.monotonic()
    for epoch in range(state['epoch'], protocol['epochs']):
        order = torch.randperm(len(train), generator=torch.Generator().manual_seed(protocol['seed']+epoch)).tolist()
        for position in range(state['cursor'], len(order)):
            check()
            row = train[order[position]]
            batch = arm_batch(get(row), 'core', device)
            with _precision(device):
                model.train(); optimizer.zero_grad(set_to_none=True)
                loss, _ = receiver.training_loss(model(batch['inputs']), batch['targets'],
                    gain_scale=scale, ranking_weight=protocol['ranking_weight'])
                if not torch.isfinite(loss):
                    raise RuntimeError('nonfinite receiver loss')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                if not torch.isfinite(norm):
                    raise RuntimeError('nonfinite receiver gradient')
                optimizer.step()
            state['cursor'] = position+1
            state['updates'] += 1
            state['epoch_loss'] += float(loss.detach())
            checkpoint()
            status = dict(phase='smoke_optimization' if protocol['smoke'] else 'formal_receiver_optimization',
                epoch=epoch+1, epochs=protocol['epochs'], window=position+1, train_windows=len(train),
                updates=state['updates'], loss=float(loss.detach()), checkpoint_loadable=True,
                seconds_this_process=time.monotonic()-began)
            save(output/'progress.json', status); progress(**status)
            print('LATENT_RECEIVER_TRAIN', status, flush=True)
            if stop_after and state['updates'] >= stop_after:
                return None
        measured = validate(model, 'core', valid, get, scale, device, check, protocol['ranking_weight'])
        previous = state['best']
        if previous is None or (measured['regret'], measured['loss']) < (
                previous['score']['regret'], previous['score']['loss']):
            state['best'] = dict(epoch=epoch+1, score=measured, weights=cpu_tree(model.state_dict()))
        state['history'].append(dict(epoch=epoch+1, train_loss=state['epoch_loss']/len(train), validation=measured))
        state.update(epoch=epoch+1, cursor=0, epoch_loss=0.)
        checkpoint()
        save(output/'history.json', dict(initial=state['initial'], epochs=state['history']))
    model.load_state_dict(state['best']['weights'])
    atomic_torch(output/'core/best.pt', receiver.export_payload(model, 'core', binding,
        state['best']['epoch'], state['best']['score']))
    # resume.pt, not a possibly stale sidecar, is authoritative after a crash.
    save(output/'history.json', dict(initial=state['initial'], epochs=state['history']))
    done = dict(complete=True, config=digest(output/'config.json'), updates=state['updates'],
        initial=state['initial'], best={k:v for k,v in state['best'].items() if k != 'weights'},
        role='new latent B/E conditional G receiver only', sender_training_complete=False,
        artifacts={n:digest(output/n) for n in ('resume.pt', 'core/best.pt', 'history.json', 'train_scales.json')})
    save(output/'complete.json', done)
    return done
