"""Three paired Router-only continuations with lazy, measured mixed teachers.

Epoch one prepares each missing sample before optimizing it. Subsequent epochs
replay exactly those fixed labels; this is not pseudo-label/on-policy training.
Atomic checkpoints contain all arms/optimizers and the within-epoch cursor.
Best selection uses held-out conditional G ranking, not diagnostic test videos.
"""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict
import math
import os
from pathlib import Path
import signal
import sys
import time

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))
import numpy as np
import torch

from demo import routervc_visual_router as visual
from demo import routervc_mixed_router as mixed
from demo.routervc_mixed_data import OLD, Samples, manifest
from demo.routervc_fullview_probe import read, digest, save, immutable, verify_artifacts
from demo.chunk_enhancement_codec import atomic_torch, configure_torch
from demo.scalable_experiment import check_space

ROOT = Path('/root/autodl-fs/DCVC/runs/routervc_mixed_router_20261004')
CACHE = Path('/root/autodl-tmp/DCVC/cache/routervc_mixed_20261004')
INITIAL = OLD/'router/global_local/model.pt'
RESUME = OLD/'router/global_local/resume.pt'
CODE = ('routervc_mixed_router.py', 'routervc_mixed_data.py', 'routervc_mixed_train.py',
        'routervc_mixed_queue.py', 'run_routervc_mixed.sh', 'routervc_visual_router.py',
        'routervc_visual_train.py', 'routervc_visual_policy.py', 'routervc_light_packets.py',
        'four_state_receive.py', 'internal_condition_decode.py', 'online_eg_decode.py')


def make_protocol(smoke=False):
    rows = manifest(smoke)
    if not smoke and (Counter(r['dataset'] for r in rows) != dict(REDS=90, UVG=30)
            or Counter(r['router_split'] for r in rows) != dict(train=96, validation=24)):
        raise ValueError('preserve the approved mixed-view sequence split')
    previous = read(OLD/'teacher/protocol.json')
    return dict(format='routervc_mixed_training_v1', smoke=smoke, rows=rows,
        code={n:digest(REPO/'demo'/n) for n in CODE}, qstep=2., arms=list(mixed.ARMS),
        epochs=2 if smoke else 120, learning_rate=1e-4, seed=20261004,
        initial_model=str(INITIAL), initial_sha256=digest(INITIAL),
        scale_checkpoint=str(RESUME), scale_sha256=digest(RESUME),
        teacher_profile=previous['profile'], enhancement=previous['enhancement'], adapter=previous['adapter'],
        label_source='actual decoded multi-E Y, independently measured conditional G',
        configurations='fixed seeded nested 4/8 region subsets; NOT 25/50 percent bytes',
        best_selection='equal-dataset held-out conditional G4/G8 local gain regret; loss tie-break',
        loss_scales='frozen prior q2 TRAIN-only gain standard deviations, shared by all arms',
        initialization='all arms continue the same completed q2 global/local Router',
        optimization='one update per window/arm/epoch, equal 32 supervised cells',
        epoch_one='lazy fixed-teacher preparation before each update; no teacher depends on Router',
        freeze_UF_E_G=True, semantic_supervision=False, transmitted_masks=False,
        real_stream_evaluation='next session; training labels/ranking are not whole-video RD')


def arm_batch(data, arm, device):
    source = data['isolated'] if arm == 'isolated_core' else data
    inputs = dict(source['inputs'])
    if arm == 'mixed_halo': inputs['local_pairs'] = data['halo_pairs']
    return dict(inputs={k:v.to(device) for k,v in inputs.items()},
        targets={k:{f:v.to(device) for f,v in item.items()} for k,item in source['targets'].items()})


@torch.no_grad()
def validate(model, arm, rows, get, scale, device, check=lambda: None):
    # Every arm is selected on the SAME actual mixed validation reconstructions,
    # even the isolated-training control. Validation never updates parameters.
    model.eval(); groups = {}
    for row in rows:
        check(); data = get(row)
        batch = arm_batch(data, 'mixed_halo' if arm == 'mixed_halo' else 'mixed_core', device)
        from demo.four_state_receive import codec_precision
        with codec_precision() if device == 'cuda' else nullcontext():
            prediction = model(batch['inputs'])
            loss, _ = visual.masked_training_loss(prediction, batch['targets'], gain_scale=scale)
        pred = prediction['gains'][..., 1].cpu().numpy()
        truth = batch['targets']['gains']['value'][..., 1].cpu().numpy()
        item = dict(loss=float(loss), regret=float(np.mean([mixed.regret(p,t) for p,t in zip(pred,truth)])))
        groups.setdefault(row['dataset'], []).append(item)
    grouped = {d:{k:float(np.mean([r[k] for r in items])) for k in ('loss','regret')} for d,items in groups.items()}
    return dict(groups=grouped, loss=float(np.mean([r['loss'] for r in grouped.values()])),
        regret=float(np.mean([r['regret'] for r in grouped.values()])),
        scope='conditional local G ranking on Router validation, not full-video RD')


def cpu_tree(value):
    if torch.is_tensor(value): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value, list): return [cpu_tree(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu_tree(v) for v in value)
    return value


def run_training(output, protocol, get, *, initial_state, scale, config=None,
                 device='cuda', check=lambda: None, stop_after=0, progress=lambda **kw: None):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    config = config or visual.VisualRouterConfig()
    binding = dict(protocol=protocol, architecture=asdict(config))
    immutable(output/'config.json', binding)
    done = output/'complete.json'
    if done.exists():
        result = read(done)
        if result['config'] != digest(output/'config.json'): raise ValueError('completed config changed')
        verify_artifacts(output, result['artifacts']); return result
    rows = protocol['rows']; smoke = protocol['smoke']
    train = rows if smoke else [r for r in rows if r['router_split'] == 'train']
    valid = rows if smoke else [r for r in rows if r['router_split'] == 'validation']
    if not train or not valid: raise ValueError('empty train or validation split')
    models, optimizers = {}, {}
    for arm in mixed.ARMS:
        model = visual.VisualUtilityRouter(config).to(device)
        model.load_state_dict(initial_state)
        models[arm] = model
        optimizers[arm] = torch.optim.AdamW(model.parameters(), lr=protocol['learning_rate'], weight_decay=1e-4)
    scale = torch.as_tensor(scale, dtype=torch.float32, device=device)
    state = dict(epoch=0, cursor=0, updates=0, history=[], best={}, epoch_losses={a:0. for a in mixed.ARMS})
    last = output/'resume.pt'
    if last.exists():
        loaded = torch.load(last, weights_only=True, map_location=device)
        if loaded['binding'] != binding: raise ValueError('resume binding changed')
        torch.testing.assert_close(loaded['scale'], scale, rtol=0, atol=0)
        for arm in mixed.ARMS:
            models[arm].load_state_dict(loaded['models'][arm])
            optimizers[arm].load_state_dict(loaded['optimizers'][arm])
        state = cpu_tree(loaded['state'])

    def checkpoint():
        atomic_torch(last, dict(binding=binding, scale=scale.detach().cpu(), state=cpu_tree(state),
            models={a:cpu_tree(m.state_dict()) for a,m in models.items()},
            optimizers={a:cpu_tree(o.state_dict()) for a,o in optimizers.items()}))
    began = time.monotonic()
    for epoch in range(state['epoch'], protocol['epochs']):
        order = torch.randperm(len(train), generator=torch.Generator().manual_seed(protocol['seed']+epoch)).tolist()
        for position in range(state['cursor'], len(order)):
            check(); row = train[order[position]]; data = get(row)
            # Restore generator precision flags after each Router update; never
            # let interleaved optimization change the frozen G teacher arithmetic.
            from demo.four_state_receive import codec_precision
            for arm in mixed.ARMS:
                with codec_precision() if device == 'cuda' else nullcontext():
                    model, optimizer = models[arm], optimizers[arm]; model.train()
                    batch = arm_batch(data, arm, device)
                    if visual.supervision_metadata(batch['targets'])['semantic_supervision']:
                        raise ValueError('content supervision is outside this stage')
                    optimizer.zero_grad(set_to_none=True)
                    loss, _ = visual.masked_training_loss(model(batch['inputs']), batch['targets'], gain_scale=scale)
                    if not torch.isfinite(loss): raise RuntimeError('nonfinite mixed Router loss')
                    loss.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                    if not torch.isfinite(norm): raise RuntimeError('nonfinite Router gradient')
                    optimizer.step(); state['epoch_losses'][arm] += float(loss.detach())
            state['cursor'] = position+1; state['updates'] += 1
            checkpoint()
            status = dict(phase='formal_router_optimization' if not smoke else 'smoke_optimization',
                epoch=epoch+1, epochs=protocol['epochs'], window=position+1, train_windows=len(train),
                updates_per_arm=state['updates'], sample_id=row['sample_id'],
                all_three_optimizers_updated=True, checkpoint_loadable=True,
                seconds_this_process=time.monotonic()-began)
            save(output/'progress.json', status); progress(**status)
            print('MIXED_TRAIN', status, flush=True)
            if stop_after and state['updates'] >= stop_after: return None
        measured = {}
        for arm in mixed.ARMS:
            check(); measured[arm] = validate(models[arm], arm, valid, get, scale, device, check)
            current = measured[arm]
            previous = state['best'].get(arm)
            if previous is None or (current['regret'],current['loss']) < (previous['score']['regret'],previous['score']['loss']):
                state['best'][arm] = dict(epoch=epoch+1, score=current, weights=cpu_tree(models[arm].state_dict()))
        state['history'].append(dict(epoch=epoch+1,
            train_loss={a:v/len(train) for a,v in state['epoch_losses'].items()}, validation=measured))
        state.update(epoch=epoch+1, cursor=0, epoch_losses={a:0. for a in mixed.ARMS})
        checkpoint()
        # Checkpoint contains the authoritative best state, so interrupted exports
        # can always be reconstructed without selecting a different epoch.
        for arm in mixed.ARMS:
            current = cpu_tree(models[arm].state_dict()); best = state['best'][arm]
            models[arm].load_state_dict(best['weights'])
            atomic_torch(output/arm/'best.pt', mixed.export_payload(models[arm],arm,binding,
                best['epoch'],best['score']))
            models[arm].load_state_dict(current)
        save(output/'history.json', state['history'])
        print(f'MIXED_EPOCH {epoch+1}/{protocol["epochs"]} best='+
              str({a:v['epoch'] for a,v in state['best'].items()}), flush=True)
    for arm in mixed.ARMS:
        best = state['best'][arm]
        atomic_torch(output/arm/'last.pt', mixed.export_payload(models[arm],arm,binding,
            protocol['epochs'],state['history'][-1]['validation'][arm]))
        models[arm].load_state_dict(best['weights'])
        atomic_torch(output/arm/'best.pt', mixed.export_payload(models[arm],arm,binding,best['epoch'],best['score']))
        reloaded,payload = mixed.load_model(output/arm/'best.pt')
        for key,value in reloaded.state_dict().items():
            torch.testing.assert_close(value,best['weights'][key],rtol=0,atol=0)
    paths = ['resume.pt','history.json']+[f'{a}/{n}.pt' for a in mixed.ARMS for n in ('best','last')]
    result = dict(complete=True, config=digest(output/'config.json'), epochs=protocol['epochs'],
        updates_per_arm=state['updates'], parameters=sum(p.numel() for p in models[mixed.ARMS[0]].parameters()),
        selected={a:dict(epoch=v['epoch'],validation=v['score']) for a,v in state['best'].items()},
        artifacts={n:digest(output/n) for n in paths}, deployment_pending=True, semantic_supervision=False)
    save(done,result); return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--stop-after',type=int,default=0)
    p.add_argument('--verify-only',action='store_true'); args = p.parse_args()
    if not os.environ.get('TMUX') or not os.environ.get('ROUTERVC_MIXED_PARENT'):
        raise RuntimeError('supervised tmux worker required')
    protocol = read(args.root/'protocol.json')
    for name,expected in protocol['code'].items():
        if digest(REPO/'demo'/name) != expected: raise ValueError('pinned code changed: '+name)
    stopped = False
    def stop(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def check():
        check_space()
        if stopped: raise InterruptedError('resume from the last complete three-arm update')
    torch.set_num_threads(4); configure_torch()
    samples = Samples(args.root,args.cache,digest(args.root/'protocol.json'),check)
    if digest(INITIAL) != protocol['initial_sha256'] or digest(RESUME) != protocol['scale_sha256']:
        raise ValueError('initial weights or training scales changed')
    from demo.routervc_visual_policy import load_model
    model = load_model(INITIAL); initial = model.state_dict()
    scale = torch.load(RESUME,weights_only=True,map_location='cpu')['scale']; del model
    try:
        if args.verify_only:
            record = read(args.output/'complete.json');verify_artifacts(args.output,record['artifacts'])
            for row in protocol['rows']: samples.get(row,generate=False)
            print('MIXED_TRAINING_VERIFIED_READ_ONLY',flush=True);return
        result = run_training(args.output,protocol,samples.get,initial_state=initial,
            scale=scale,device='cuda',check=check,stop_after=args.stop_after)
        if result:
            immutable(args.root/'labels.complete.json',dict(complete=True,
                protocol=digest(args.root/'protocol.json'),samples={r['sample_id']:digest(
                    args.root/'samples'/r['sample_id']/'complete.json') for r in protocol['rows']},
                regions=len(protocol['rows'])*32, no_UF_E_G_updates=True))
    except BaseException as error:
        save(args.output/'last_failure.json',dict(error=repr(error)));raise
    finally:
        samples.release_generator()
        if torch.distributed.is_initialized(): torch.distributed.destroy_process_group()


if __name__ == '__main__': main()
