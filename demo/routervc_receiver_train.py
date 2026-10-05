"""Train receiver-only conditional-G predictors, not the future sender Router.

Two input-context candidates start from identical three-G-channel weights.
Missing fixed-teacher views are measured lazily before a window's first update;
later epochs replay those labels. Atomic paired checkpoints retain the exact
within-epoch cursor. Validation is local G ranking, never whole-video RD.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict
import os
from pathlib import Path
import signal
import sys
import time

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import torch

from demo import routervc_receiver_router as receiver
from demo import routervc_visual_router as visual
from demo.routervc_fullview_probe import read, digest, save, immutable, verify_artifacts
from demo.chunk_enhancement_codec import atomic_torch, configure_torch
from demo.scalable_experiment import check_space

ROOT = Path('/root/autodl-fs/DCVC/runs/routervc_receiver_20261005')
CACHE = Path('/root/autodl-tmp/DCVC/cache/routervc_receiver_20261005')
MIXED = Path('/root/autodl-fs/DCVC/runs/routervc_mixed_router_20261004')
OLD = Path('/root/autodl-fs/DCVC/runs/routervc_light_router_20261004')
INITIAL = MIXED / 'formal/router/mixed_core/best.pt'
SCALE = OLD / 'router/global_local/resume.pt'
VIEW_NAMES = ('e0', 'e2', 'e4', 'e8', 'e12', 'e16')
CODE = ('routervc_receiver_router.py', 'routervc_receiver_data.py',
        'routervc_receiver_train.py', 'routervc_receiver_queue.py',
        'routervc_receiver_smoke.py', 'routervc_receiver_format.py',
        'routervc_receiver_policy.py', 'routervc_receiver_decode.py',
        'run_routervc_receiver.sh', 'routervc_visual_router.py',
        'routervc_mixed_router.py', 'routervc_mixed_data.py',
        'routervc_light_packets.py', 'four_state_receive.py',
        'internal_condition_decode.py', 'online_eg_decode.py')


def make_protocol(smoke=False):
    from demo.routervc_receiver_data import make_rows
    rows = make_rows(smoke)
    if not smoke and (Counter(r['dataset'] for r in rows) != dict(REDS=90, UVG=30)
            or Counter(r['router_split'] for r in rows) != dict(train=96, validation=24)):
        raise ValueError('preserve the approved 120-window sequence split')
    previous = read(OLD / 'teacher/protocol.json')
    return dict(format='routervc_receiver_training_v1', smoke=smoke, rows=rows,
        code={name: digest(REPO / 'demo' / name) for name in CODE}, qstep=2.,
        arms=list(receiver.ARMS), epochs=2 if smoke else 120,
        learning_rate=1e-4, weight_decay=1e-4, ranking_weight=.1, seed=20261005,
        view_names=list(VIEW_NAMES), initial_model=str(INITIAL),
        initial_sha256=digest(INITIAL), scale_checkpoint=str(SCALE),
        scale_sha256=digest(SCALE),
        teacher_profile=previous['profile'], enhancement=previous['enhancement'],
        adapter=previous['adapter'],
        role='receiver-only R_g; independent sender R_s is not trained here',
        initialization='same mixed_core best encoder/body and conditional G rows [1,3,5]',
        loss='masked standardized three-metric Huber + 0.1 same-view LPIPS pair ranking',
        metric_weights=[1., .1, .1],
        loss_scales='frozen prior q2 TRAIN-only scales [1,3,5], loss-only normalization',
        configurations='fixed nested 0/2/4/8/12/16 REGION subsets, not byte caps',
        labels='actual conditional G(Y); existing e0/e4/e8 reused after authentication',
        optimization='one paired update per training window per epoch; all six states equally weighted',
        best_selection='dataset-equal then six-state-equal local G4/G8 regret; loss tie-break',
        initial_validation='after epoch-one labels are ready; diagnostic only, not eligible best',
        freeze_UF_E_G=True, semantic_supervision=False, transmitted_masks=False,
        receiver_reads_source=False, sender_training_complete=False,
        deployment_pending=True, whole_video_RD_pending=True)


def initialize():
    """Prune the old shared head; do not retain E or untrained semantic outputs."""
    from demo.routervc_mixed_router import load_model
    legacy, _ = load_model(INITIAL)
    with torch.random.fork_rng(devices=[]):
        model = receiver.ReceiverGUtilityRouter(legacy.config)
    receiver.initialize_from_mixed(model, legacy.state_dict())
    scales = torch.load(SCALE, weights_only=True, map_location='cpu')['scale'][[1, 3, 5]].clone()
    return cpu_tree(model.state_dict()), scales, legacy.config


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def arm_batch(data, arm, device):
    if arm not in receiver.ARMS or tuple(data['view_names']) != VIEW_NAMES:
        raise ValueError('unknown receiver arm or changed view ordering')
    inputs = dict(data['inputs'])
    if arm == 'halo':
        inputs['local_pairs'] = data['halo_pairs']
    targets = data['targets']
    if set(targets) != {'value', 'weight'} or targets['value'].shape != (6, 16, 3):
        raise ValueError('expected six states, sixteen regions and three G-only targets')
    if targets['weight'].shape != targets['value'].shape:
        raise ValueError('mismatched receiver supervision weights')
    return dict(inputs={key: value.to(device) for key, value in inputs.items()},
        targets={key: value.to(device) for key, value in targets.items()})


def local_regret(prediction, target, caps=(4, 8)):
    """Best positive-gain top-k minus predicted positive-gain top-k, per 16 cells."""
    prediction, target = np.asarray(prediction), np.asarray(target)
    if prediction.shape != (16,) or target.shape != (16,) or not (
            np.isfinite(prediction).all() and np.isfinite(target).all()):
        raise ValueError('finite sixteen-cell LPIPS gains required')
    values = []
    for cap in caps:
        best = [i for i in np.argsort(-target, kind='stable') if target[i] > 0][:cap]
        chosen = [i for i in np.argsort(-prediction, kind='stable') if prediction[i] > 0][:cap]
        values.append(float((target[best].sum() - target[chosen].sum()) / 16))
    return float(np.mean(values))


def _precision(device):
    if str(device).startswith('cuda'):
        from demo.four_state_receive import codec_precision
        return codec_precision()
    return nullcontext()


@torch.no_grad()
def validate(model, arm, rows, get, scale, device, check=lambda: None, ranking_weight=.1):
    model.eval()
    groups = {}
    for row in rows:
        check()
        batch = arm_batch(get(row), arm, device)
        with _precision(device):
            prediction = model(batch['inputs'])
            state_scores = {}
            for index, name in enumerate(VIEW_NAMES):
                targets = {key: value[index:index+1] for key, value in batch['targets'].items()}
                if not torch.all(targets['weight'][..., 0] > 0):
                    raise ValueError('local ranking requires measured LPIPS for every region')
                loss, _ = receiver.training_loss(prediction[index:index+1], targets,
                    gain_scale=scale, ranking_weight=ranking_weight)
                state_scores[name] = dict(loss=float(loss), regret=local_regret(
                    prediction[index, :, 0].cpu().numpy(), targets['value'][0, :, 0].cpu().numpy()))
        groups.setdefault(row['dataset'], []).append(state_scores)
    if not groups:
        raise ValueError('empty receiver validation')
    by_dataset = {}
    for dataset, samples in groups.items():
        by_state = {name: {key: float(np.mean([s[name][key] for s in samples]))
                          for key in ('loss', 'regret')} for name in VIEW_NAMES}
        by_dataset[dataset] = dict(by_state=by_state, samples=len(samples),
            **{key: float(np.mean([s[key] for s in by_state.values()])) for key in ('loss', 'regret')})
    by_state = {name: {key: float(np.mean([d['by_state'][name][key] for d in by_dataset.values()]))
                      for key in ('loss', 'regret')} for name in VIEW_NAMES}
    return dict(groups=by_dataset, by_state=by_state,
        **{key: float(np.mean([s[key] for s in by_state.values()])) for key in ('loss', 'regret')},
        scope='dataset-equal and state-equal conditional local G4/G8 ranking; not whole-video RD')


def run_training(output, protocol, get, *, initial_state, scale, config=None,
                 device='cuda', check=lambda: None, stop_after=0, progress=lambda **kw: None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = config or visual.VisualRouterConfig()
    binding = dict(protocol=protocol, architecture=asdict(config))
    immutable(output / 'config.json', binding)
    done = output / 'complete.json'
    if done.exists():
        result = read(done)
        if result['config'] != digest(output / 'config.json'):
            raise ValueError('completed receiver config changed')
        verify_artifacts(output, result['artifacts'])
        return result
    rows, smoke = protocol['rows'], protocol['smoke']
    train = rows if smoke else [r for r in rows if r['router_split'] == 'train']
    valid = rows if smoke else [r for r in rows if r['router_split'] == 'validation']
    if not train or not valid:
        raise ValueError('empty receiver train or validation split')
    models, optimizers = {}, {}
    for arm in receiver.ARMS:
        with torch.random.fork_rng(devices=[]):
            model = receiver.ReceiverGUtilityRouter(config).to(device)
        model.load_state_dict(initial_state, strict=True)
        models[arm] = model
        optimizers[arm] = torch.optim.AdamW(model.parameters(), lr=protocol['learning_rate'],
                                           weight_decay=protocol.get('weight_decay', 1e-4))
    scale = torch.as_tensor(scale, dtype=torch.float32, device=device)
    if scale.shape != (3,) or not torch.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError('three positive TRAIN-only loss scales required')
    rank_weight = protocol.get('ranking_weight', .1)
    state = dict(epoch=0, cursor=0, updates=0, history=[], best={}, initial_validation=None,
                 epoch_losses={arm: 0. for arm in receiver.ARMS})
    last = output / 'resume.pt'
    if last.exists():
        loaded = torch.load(last, weights_only=True, map_location=device)
        if loaded['binding'] != binding:
            raise ValueError('receiver resume binding changed')
        torch.testing.assert_close(loaded['scale'], scale, rtol=0, atol=0)
        for arm in receiver.ARMS:
            models[arm].load_state_dict(loaded['models'][arm])
            optimizers[arm].load_state_dict(loaded['optimizers'][arm])
        state = cpu_tree(loaded['state'])

    def checkpoint():
        atomic_torch(last, dict(binding=binding, scale=scale.detach().cpu(), state=cpu_tree(state),
            models={a: cpu_tree(m.state_dict()) for a, m in models.items()},
            optimizers={a: cpu_tree(o.state_dict()) for a, o in optimizers.items()}))

    began = time.monotonic()
    for epoch in range(state['epoch'], protocol['epochs']):
        order = torch.randperm(len(train), generator=torch.Generator().manual_seed(protocol['seed']+epoch)).tolist()
        for position in range(state['cursor'], len(order)):
            check()
            row = train[order[position]]
            data = get(row)
            for arm in receiver.ARMS:
                with _precision(device):
                    model, optimizer = models[arm], optimizers[arm]
                    model.train()
                    batch = arm_batch(data, arm, device)
                    optimizer.zero_grad(set_to_none=True)
                    loss, _ = receiver.training_loss(model(batch['inputs']), batch['targets'],
                        gain_scale=scale, ranking_weight=rank_weight)
                    if not torch.isfinite(loss):
                        raise RuntimeError('nonfinite receiver loss')
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                    if not torch.isfinite(norm):
                        raise RuntimeError('nonfinite receiver gradient')
                    optimizer.step()
                    state['epoch_losses'][arm] += float(loss.detach())
            state['cursor'] = position + 1
            state['updates'] += 1
            checkpoint()
            status = dict(phase='smoke_optimization' if smoke else 'formal_receiver_optimization',
                epoch=epoch+1, epochs=protocol['epochs'], window=position+1,
                train_windows=len(train), updates_per_arm=state['updates'], sample_id=row['sample_id'],
                both_receiver_optimizers_updated=True, checkpoint_loadable=True,
                sender_training_complete=False, seconds_this_process=time.monotonic()-began)
            save(output / 'progress.json', status)
            progress(**status)
            print('RECEIVER_TRAIN', status, flush=True)
            if stop_after and state['updates'] >= stop_after:
                return None
        measured = {}
        for arm in receiver.ARMS:
            check()
            current = validate(models[arm], arm, valid, get, scale, device, check, rank_weight)
            measured[arm] = current
            previous = state['best'].get(arm)
            if previous is None or (current['regret'], current['loss']) < (
                    previous['score']['regret'], previous['score']['loss']):
                state['best'][arm] = dict(epoch=epoch+1, score=current,
                                        weights=cpu_tree(models[arm].state_dict()))
        if state['initial_validation'] is None:
            # This diagnostic waits until epoch-one validation has populated all
            # fixed teachers; it never postpones the first actual optimizer step.
            with torch.random.fork_rng(devices=[]):
                baseline = receiver.ReceiverGUtilityRouter(config).to(device)
            baseline.load_state_dict(initial_state, strict=True)
            state['initial_validation'] = {arm: validate(baseline, arm, valid, get, scale,
                device, check, rank_weight) for arm in receiver.ARMS}
            del baseline
        state['history'].append(dict(epoch=epoch+1,
            train_loss={a: value/len(train) for a, value in state['epoch_losses'].items()},
            validation=measured))
        state.update(epoch=epoch+1, cursor=0, epoch_losses={a: 0. for a in receiver.ARMS})
        checkpoint()
        for arm in receiver.ARMS:
            current = cpu_tree(models[arm].state_dict())
            best = state['best'][arm]
            models[arm].load_state_dict(best['weights'])
            atomic_torch(output / arm / 'best.pt', receiver.export_payload(models[arm], arm,
                binding, best['epoch'], best['score']))
            models[arm].load_state_dict(current)
        save(output / 'history.json', state['history'])
        save(output / 'initial_validation.json', state['initial_validation'])
        print(f'RECEIVER_EPOCH {epoch+1}/{protocol["epochs"]} best=' +
              str({a: value['epoch'] for a, value in state['best'].items()}), flush=True)
    # The authoritative epoch checkpoint precedes the human-readable exports.
    # A restart after the FINAL checkpoint skips the loop above, so regenerate
    # these JSONs from checkpoint state before authenticating completion. This
    # also repairs a missing file or an export still reflecting epoch N-1.
    save(output / 'history.json', state['history'])
    save(output / 'initial_validation.json', state['initial_validation'])
    for arm in receiver.ARMS:
        best = state['best'][arm]
        atomic_torch(output / arm / 'last.pt', receiver.export_payload(models[arm], arm,
            binding, protocol['epochs'], state['history'][-1]['validation'][arm]))
        models[arm].load_state_dict(best['weights'])
        atomic_torch(output / arm / 'best.pt', receiver.export_payload(models[arm], arm,
            binding, best['epoch'], best['score']))
        reloaded, _ = receiver.load_model(output / arm / 'best.pt')
        for key, value in reloaded.state_dict().items():
            torch.testing.assert_close(value, best['weights'][key], rtol=0, atol=0)
    names = ['resume.pt', 'history.json', 'initial_validation.json'] + [
        f'{arm}/{name}.pt' for arm in receiver.ARMS for name in ('best', 'last')]
    result = dict(complete=True, config=digest(output / 'config.json'), epochs=protocol['epochs'],
        updates_per_arm=state['updates'], parameters=sum(p.numel() for p in models[receiver.ARMS[0]].parameters()),
        selected={a: dict(epoch=v['epoch'], validation=v['score']) for a, v in state['best'].items()},
        artifacts={name: digest(output / name) for name in names}, role='receiver-only conditional G',
        deployment_pending=True, whole_video_RD_pending=True, sender_training_complete=False,
        semantic_supervision=False)
    save(done, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stop-after', type=int, default=0)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if not os.environ.get('TMUX') or not os.environ.get('ROUTERVC_RECEIVER_PARENT'):
        raise RuntimeError('supervised tmux receiver worker required')
    protocol = read(args.root / 'protocol.json')
    for name, expected in protocol['code'].items():
        if digest(REPO / 'demo' / name) != expected:
            raise ValueError('pinned receiver code changed: ' + name)
    stopped = False

    def stop(*_):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def check():
        check_space()
        if stopped:
            raise InterruptedError('resume from the last paired receiver update')

    torch.set_num_threads(4)
    configure_torch()
    from demo.routervc_receiver_data import Samples
    samples = Samples(args.root, args.cache, digest(args.root / 'protocol.json'), check)
    if digest(INITIAL) != protocol['initial_sha256'] or digest(SCALE) != protocol['scale_sha256']:
        raise ValueError('receiver initial weights or training scales changed')
    try:
        if args.verify_only:
            result = read(args.output / 'complete.json')
            if result['config'] != digest(args.output / 'config.json'):
                raise ValueError('completed receiver config changed')
            verify_artifacts(args.output, result['artifacts'])
            for row in protocol['rows']:
                samples.get(row, generate=False)
            print('RECEIVER_TRAINING_VERIFIED_READ_ONLY', flush=True)
            return
        initial_state, scale, config = initialize()
        result = run_training(args.output, protocol, samples.get, initial_state=initial_state,
            scale=scale, config=config, device='cuda', check=check, stop_after=args.stop_after)
        if result:
            immutable(args.root / 'labels.complete.json', dict(complete=True,
                protocol=digest(args.root / 'protocol.json'), samples={r['sample_id']: digest(
                    args.root / 'samples' / r['sample_id'] / 'complete.json') for r in protocol['rows']},
                regions=len(protocol['rows'])*96, measured_regions=len(protocol['rows'])*48,
                reused_regions=len(protocol['rows'])*48,
                no_UF_E_G_updates=True, sender_training_complete=False))
    except BaseException as error:
        save(args.output / 'last_failure.json', dict(error=repr(error)))
        raise
    finally:
        samples.release_generator()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
