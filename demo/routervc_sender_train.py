"""Independent paired source-aware R_s training on measured FINAL marginals.

Finish the fixed R_g/G teacher first, fit one robust TRAIN-only scalar scale,
then train source/zero_source twins from identical random weights. Validation
ranks only three measured candidates at two sampled states; it is not an
exhaustive allocation oracle or an end-to-end RD evaluation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import torch

from demo import routervc_sender_router as sender
from demo.routervc_fullview_probe import read, digest, save, immutable, verify_artifacts
from demo.chunk_enhancement_codec import atomic_torch
from demo.four_state_receive import codec_precision

ROOT = Path('/root/autodl-fs/DCVC/runs/routervc_sender_20261005')
CACHE = Path('/root/autodl-tmp/DCVC/cache/routervc_sender_20261005')
ARMS = ('source', 'zero_source')
STATES = ('empty', 'partial')
FORMAT = 'routervc_sender_training_v1'
CODE = ('routervc_sender_router.py', 'routervc_sender_data.py',
        'routervc_sender_train.py', 'routervc_sender_queue.py',
        'routervc_sender_checks.py', 'routervc_sender_encode.py', 'run_routervc_sender.sh',
        'routervc_receiver_data.py', 'routervc_mixed_data.py',
        'routervc_light_teacher.py', 'routervc_light_packets.py',
        'routervc_visual_policy.py', 'routervc_fullview_probe.py',
        'four_state_receive.py', 'scalable_experiment.py',
        'stage_c_three_path_roi_probe.py', 'stage_c_a800_teacher.py')


def semantic_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def make_protocol(smoke=False, receiver_checkpoint=None, expected_sha256=None,
                  receiver_complete=None):
    from demo import routervc_sender_data as data
    from demo import routervc_receiver_format as fmt
    if type(smoke) is not bool or receiver_checkpoint is None or expected_sha256 is None:
        raise ValueError('an explicit fixed R_g checkpoint and hash are required')
    rows = data.make_rows(smoke)
    if not smoke and (Counter(r['dataset'] for r in rows) != dict(REDS=90, UVG=30)
            or Counter(r['router_split'] for r in rows) != dict(train=96, validation=24)):
        raise ValueError('preserve the approved 120-window, 96/24 sequence ledger')
    teacher = data.make_settings(receiver_checkpoint, expected_sha256, smoke=smoke,
                                receiver_complete=receiver_complete, seed=20261005)
    return dict(format=FORMAT, smoke=smoke, rows=rows, teacher=teacher,
        code={name: digest(REPO/'demo'/name) for name in sorted(set(CODE) | set(fmt.CODE))},
        arms=list(ARMS), epochs=2 if smoke else 120, seed=20261005,
        learning_rate=1e-4, weight_decay=1e-4, ranking_weight=.1,
        role='independent source-aware R_s final-marginal predictor; fixed completed R_g',
        initialization='same seeded random R_s weights for both arms; no R_g weights or scales',
        ablation='zero X and X-B/X-Y channels only; equal architecture and non-source inputs',
        teacher_preparation='ALL fixed final renderings complete before scale fitting or any optimizer step',
        loss='masked scalar final-LPIPS Huber + 0.1 measured same-state gain/byte pair ranking',
        loss_scale='75th percentile of abs(measured TRAIN final LPIPS delta), floor 1e-6; loss-only',
        optimization='one atomic paired update per train window; two measured states equally weighted',
        best_selection='REDS/UVG equal, then empty/partial equal, measured-candidate normalized density regret; loss tie-break',
        validation_scope='three measured candidates per sampled state, positive top1 or no-add; NOT full allocation oracle or RD',
        initial_validation='fixed independent random twins before optimization, after all labels are ready',
        negative_marginals_retained=True, unknown_candidates_masked=True,
        receiver_weights_shared=False, freeze_UF_E_G_and_Rg=True,
        transmitted_masks=False, semantic_supervision=False,
        full_allocation_evaluation_pending=True, whole_video_RD_pending=True)


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


def capture_rng(device='cpu'):
    numpy = np.random.get_state()
    return dict(python=random.getstate(), torch=torch.get_rng_state().clone(),
        numpy=dict(name=numpy[0], keys=torch.from_numpy(numpy[1].astype(np.int64)),
                   position=numpy[2], has_gauss=numpy[3], cached_gauss=numpy[4]),
        cuda=[v.cpu().clone() for v in torch.cuda.get_rng_state_all()]
             if str(device).startswith('cuda') else [])


def restore_rng(value, device='cpu'):
    random.setstate(value['python'])
    numpy = value['numpy']
    np.random.set_state((numpy['name'], numpy['keys'].cpu().numpy().astype(np.uint32),
                        numpy['position'], numpy['has_gauss'], numpy['cached_gauss']))
    torch.set_rng_state(value['torch'].cpu())
    if str(device).startswith('cuda'):
        if len(value['cuda']) != torch.cuda.device_count():
            raise ValueError('CUDA RNG device topology changed')
        torch.cuda.set_rng_state_all([v.cpu() for v in value['cuda']])
    elif value['cuda']:
        raise ValueError('cannot exactly resume a CUDA training checkpoint on CPU')


@contextmanager
def deterministic_training(seed, device):
    """Use deterministic FP32 for CPU AND GPU; restore the caller's policy/RNG."""
    if str(device).startswith('cuda') and os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8', ':16:8'):
        raise ValueError('set CUBLAS_WORKSPACE_CONFIG=:4096:8 in the worker before CUDA initialization')
    previous = capture_rng(device)
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    threads = torch.get_num_threads()
    try:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        if str(device).startswith('cuda'):
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        with codec_precision():
            yield
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.set_num_threads(threads)
        restore_rng(previous, device)


def split_rows(protocol):
    rows = protocol['rows']
    if not rows or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('unique nonempty sender sample ledger required')
    if any(r['dataset'] not in ('REDS', 'UVG') or r['router_split'] not in ('train', 'validation') for r in rows):
        raise ValueError('unknown sender dataset/split')
    train = [r for r in rows if r['router_split'] == 'train']
    valid = rows if protocol['smoke'] else [r for r in rows if r['router_split'] == 'validation']
    if not train or not valid:
        raise ValueError('nonempty sender train/validation sets required')
    return train, valid


def arm_batch(data, arm, device):
    if arm not in ARMS or set(data['inputs']) != sender.INPUT_KEYS:
        raise ValueError('unknown sender arm or inputs')
    target = data['targets']
    if (set(target) != {'value', 'weight', 'label_scope'} or target['label_scope'] != sender.LABEL_SCOPE
            or target['value'].shape != (2,16) or target['weight'].shape != (2,16)):
        raise ValueError('two states of scalar FINAL marginal targets are required')
    value, weight = target['value'], target['weight']
    known = weight > 0
    if (not torch.isfinite(weight).all() or torch.any(weight < 0)
            or not torch.isfinite(value[known]).all() or not torch.all(known.sum(1) == 3)):
        raise ValueError('exactly three finite measured candidates per state are required')
    costs = data['packet_bytes']
    if (costs.dtype != torch.int64 or costs.shape != (2,16) or torch.any(costs <= 0)
            or not torch.equal(costs, data['inputs']['packet_bytes'].squeeze(-1))):
        raise ValueError('candidate costs must be actual positive integer packet bytes')
    coverage = data['inputs']['coverage']
    if coverage.shape != (2,16,1) or torch.any(coverage.squeeze(-1)[known] != 0):
        raise ValueError('measured additions must be unreceived candidates')
    plans = data['plans']
    if len(plans) != 2 or plans[0]['selected'] or not plans[1]['selected']:
        raise ValueError('expected empty and nonempty fixed teacher states')
    for state, plan in enumerate(plans):
        if sorted(plan['candidates']) != torch.where(known[state])[0].tolist():
            raise ValueError('known candidate mask differs from measured teacher plans')
        if sorted(plan['selected']) != torch.where(coverage[state,:,0] == 1)[0].tolist():
            raise ValueError('received input coverage differs from measured teacher state')
    inputs = {key: tensor.to(device) for key, tensor in data['inputs'].items()}
    if arm == 'zero_source':
        # Clone only the two source-bearing tensors, never mutate compact cache.
        inputs['global_video'] = inputs['global_video'].clone()
        inputs['local_video'] = inputs['local_video'].clone()
        inputs['global_video'][:, :3] = 0
        inputs['global_video'][:, 12:18] = 0
        inputs['local_video'][:, :, :3] = 0
        inputs['local_video'][:, :, 12:18] = 0
    return dict(inputs=inputs, targets=dict(value=value.to(device), weight=weight.to(device),
        label_scope=sender.LABEL_SCOPE), packet_bytes=costs.to(device))


def fit_scale(protocol, get, check=lambda: None):
    """Never inspect validation labels when fitting the common final-Delta scale."""
    train, _ = split_rows(protocol)
    observations, ledger = [], []
    for row in train:
        check()
        batch = arm_batch(get(row), 'source', 'cpu')
        known = batch['targets']['weight'] > 0
        values = batch['targets']['value'][known].double().tolist()
        observations.extend(values)
        ledger.append(dict(sample_id=row['sample_id'], values=values,
                           known=known.nonzero().tolist()))
    values = np.asarray(observations, dtype=np.float64)
    raw = float(np.quantile(np.abs(values), .75, method='linear'))
    return dict(format='sender_train_final_delta_scale_v1', value=max(raw, 1e-6),
        statistic='linear 0.75 quantile of abs(measured signed FINAL LPIPS delta)',
        raw_quantile=raw, floor=1e-6, floor_applied=raw < 1e-6,
        train_samples=[r['sample_id'] for r in train], measured_marginals=len(values),
        observations_sha256=semantic_hash(ledger), label_scope=sender.LABEL_SCOPE,
        negative=int((values < 0).sum()), positive=int((values > 0).sum()),
        zero=int((values == 0).sum()), minimum=float(values.min()), maximum=float(values.max()),
        loss_only_normalization=True, validation_used=False, receiver_scale_reused=False)


def prepare_labels(root, protocol, samples, check=lambda: None, progress=lambda **kw: None):
    """Atomic all-teacher boundary. Partial per-state renders already resume in Samples."""
    root = Path(root)
    if read(root/'protocol.json') != protocol:
        raise ValueError('sender preparation protocol differs from immutable file')
    records = {}
    for index, row in enumerate(protocol['rows']):
        check()
        value = samples.get(row, generate=True)
        arm_batch(value, 'source', 'cpu')
        path = root/'samples'/row['sample_id']/'complete.json'
        result = read(path)
        if (not result.get('complete') or result.get('measured_final_renderings') != 8
                or result.get('measured_marginals') != 6 or result.get('label_scope') != sender.LABEL_SCOPE
                or result.get('current_sender_policy_used') is not False):
            raise ValueError('incomplete or wrong-scope sender final teacher')
        records[row['sample_id']] = digest(path)
        status = dict(phase='sender_final_teacher', completed_windows=index+1,
            windows=len(protocol['rows']), measured_final_renderings=8*(index+1),
            measured_marginals=6*(index+1), sample_id=row['sample_id'], optimizer_started=False)
        save(root/'teacher.progress.json', status)
        progress(**status)
    result = dict(format='sender_final_labels_complete_v1', complete=True,
        protocol=digest(root/'protocol.json'), protocol_semantic_sha256=semantic_hash(protocol),
        samples=records, windows=len(records), measured_final_renderings=8*len(records),
        measured_marginals=6*len(records), label_scope=sender.LABEL_SCOPE,
        no_UF_E_G_Rg_updates=True, current_sender_policy_used=False,
        scope='two measured states and three measured additions per window; not exhaustive')
    immutable(root/'labels.complete.json', result)
    return result


def verify_labels(root, protocol, samples, check=lambda: None):
    root = Path(root)
    result = read(root/'labels.complete.json')
    _validate_label_binding(protocol, result)
    if result['protocol'] != digest(root/'protocol.json'):
        raise ValueError('sender final-label protocol file changed')
    for row in protocol['rows']:
        check()
        if digest(root/'samples'/row['sample_id']/'complete.json') != result['samples'][row['sample_id']]:
            raise ValueError('sender final-label sample record changed')
        arm_batch(samples.get(row, generate=False), 'source', 'cpu')
    return result


def _validate_label_binding(protocol, value):
    count = len(protocol['rows'])
    if (value.get('format') != 'sender_final_labels_complete_v1' or value.get('complete') is not True
            or value.get('protocol_semantic_sha256') != semantic_hash(protocol)
            or set(value.get('samples', {})) != {r['sample_id'] for r in protocol['rows']}
            or value.get('windows') != count or value.get('measured_final_renderings') != 8*count
            or value.get('measured_marginals') != 6*count or value.get('label_scope') != sender.LABEL_SCOPE
            or value.get('current_sender_policy_used') is not False):
        raise ValueError('ALL fixed sender final labels must be complete before optimization')


def measured_regret(prediction, value, weight, packet_bytes):
    """Positive top1 or no-add among known candidates; density uses mean-byte units."""
    prediction, value, weight, costs = map(np.asarray, (prediction, value, weight, packet_bytes))
    if (any(v.shape != (16,) for v in (prediction,value,weight,costs))
            or not np.isfinite(prediction).all() or not np.isfinite(weight).all() or np.any(weight < 0)
            or costs.dtype.kind not in 'iu' or np.any(costs[weight > 0] <= 0)
            or not np.isfinite(value[weight > 0]).all() or (weight > 0).sum() != 3):
        raise ValueError('three measured candidates with finite predictions and actual byte costs required')
    known = np.where(weight > 0)[0].tolist()
    reference = float(costs[known].mean())
    true = {i: float(value[i])*reference/int(costs[i]) for i in known}
    predicted = {i: float(prediction[i])*reference/int(costs[i]) for i in known}
    positive = [i for i in known if predicted[i] > 0]
    chosen = min(positive, key=lambda i: (-predicted[i], i)) if positive else None
    positives_true = [i for i in known if true[i] > 0]
    best = min(positives_true, key=lambda i: (-true[i], i)) if positives_true else None
    picked = true[chosen] if chosen is not None else 0.
    reference_gain = true[best] if best is not None else 0.
    pairs = [(i,j) for n,i in enumerate(known) for j in known[n+1:] if true[i] != true[j]]
    concordant = sum(1. if (predicted[i]-predicted[j])*(true[i]-true[j]) > 0 else
                     .5 if predicted[i] == predicted[j] else 0. for i,j in pairs)
    return dict(regret=reference_gain-picked,
        density_regret_per_byte=(reference_gain-picked)/reference,
        selected_final_lpips_gain=float(value[chosen]) if chosen is not None else 0.,
        best_density_candidate_final_lpips_gain=float(value[best]) if best is not None else 0.,
        selected_normalized_density=picked, best_measured_normalized_density=reference_gain,
        pair_accuracy=concordant/len(pairs) if pairs else 1., ranked_pairs=len(pairs),
        no_add=float(chosen is None), harmful_add=float(chosen is not None and value[chosen] < 0),
        selected_region=chosen, best_measured_region=best, measured_candidates=3)


SCORE_KEYS = ('loss', 'regret', 'density_regret_per_byte', 'selected_final_lpips_gain',
    'best_density_candidate_final_lpips_gain', 'selected_normalized_density',
    'best_measured_normalized_density', 'pair_accuracy', 'ranked_pairs', 'no_add', 'harmful_add')


@torch.no_grad()
def validate(model, arm, rows, get, scale, device, check=lambda: None, ranking_weight=.1):
    model.eval()
    groups = {}
    for row in rows:
        check()
        batch = arm_batch(get(row), arm, device)
        with codec_precision():
            predicted = model(batch['inputs'])
            states = {}
            for index, name in enumerate(STATES):
                target = {key: value[index:index+1] if torch.is_tensor(value) else value
                          for key,value in batch['targets'].items()}
                loss, _ = sender.training_loss(predicted[index:index+1], target,
                    batch['packet_bytes'][index:index+1], gain_scale=scale, ranking_weight=ranking_weight)
                scores = measured_regret(predicted[index].cpu().numpy(), target['value'][0].cpu().numpy(),
                    target['weight'][0].cpu().numpy(), batch['packet_bytes'][index].cpu().numpy())
                states[name] = dict(loss=float(loss), **scores)
        groups.setdefault(row['dataset'], []).append(states)
    if not groups:
        raise ValueError('empty sender validation')
    by_dataset = {}
    for dataset, samples in groups.items():
        by_state = {name: {key: float(np.mean([s[name][key] for s in samples]))
                          for key in SCORE_KEYS} for name in STATES}
        by_dataset[dataset] = dict(samples=len(samples), by_state=by_state,
            **{key: float(np.mean([v[key] for v in by_state.values()])) for key in SCORE_KEYS})
    by_state = {name: {key: float(np.mean([v['by_state'][name][key] for v in by_dataset.values()]))
                      for key in SCORE_KEYS} for name in STATES}
    return dict(groups=by_dataset, by_state=by_state,
        **{key: float(np.mean([v[key] for v in by_state.values()])) for key in SCORE_KEYS},
        scope='dataset-equal and state-equal top1/no-add among three measured candidates; NOT full oracle or RD')


def run_training(output, protocol, get, *, scale_record, labels_binding,
                 device='cuda', check=lambda: None, stop_after=0, progress=lambda **kw: None):
    """Atomic paired source/zero_source updates, including optimizer and all RNG states."""
    _validate_label_binding(protocol, labels_binding)
    train, valid = split_rows(protocol)
    if (protocol.get('format') != FORMAT or tuple(protocol.get('arms', ())) != ARMS
            or type(protocol['epochs']) is not int or protocol['epochs'] <= 0):
        raise ValueError('invalid paired sender protocol')
    if (scale_record.get('train_samples') != [r['sample_id'] for r in train]
            or scale_record.get('measured_marginals') != 6*len(train)
            or scale_record.get('validation_used') is not False
            or scale_record.get('receiver_scale_reused') is not False
            or scale_record.get('label_scope') != sender.LABEL_SCOPE
            or type(scale_record.get('value')) not in (int,float)
            or not math.isfinite(scale_record['value']) or scale_record['value'] <= 0):
        raise ValueError('independent finite TRAIN-only final-marginal scale required')
    config = sender.Config(**protocol['teacher']['sender_config'])
    if config.zero_source:
        raise ValueError('compact teacher inputs must retain source; ablate only inside zero_source arm')
    architectures = {arm: asdict(replace(config, zero_source=(arm=='zero_source'))) for arm in ARMS}
    binding = dict(protocol=protocol, architectures=architectures,
                   labels=labels_binding, scale=scale_record, device_type=torch.device(device).type)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    immutable(output/'config.json', binding)
    if (output/'complete.json').exists():
        result = read(output/'complete.json')
        if result['config'] != digest(output/'config.json'):
            raise ValueError('completed sender config changed')
        verify_artifacts(output, result['artifacts'])
        return result
    with deterministic_training(protocol['seed'], device):
        return _run_training(output, binding, get, train, valid, device, check, stop_after, progress)


def _run_training(output, binding, get, train, valid, device, check, stop_after, progress):
    protocol, scale = binding['protocol'], float(binding['scale']['value'])
    models, optimizers = {}, {}
    initial = None
    for arm in ARMS:
        model = sender.SenderUtilityRouter(sender.Config(**binding['architectures'][arm])).to(device)
        if initial is None:
            initial = cpu_tree(model.state_dict())
        else:
            model.load_state_dict(initial, strict=True)
        models[arm] = model
        optimizers[arm] = torch.optim.AdamW(model.parameters(), lr=protocol['learning_rate'],
            weight_decay=protocol['weight_decay'])
    state = dict(epoch=0, cursor=0, updates=0, history=[], best={}, initial_validation=None,
                 epoch_losses={arm:0. for arm in ARMS})
    last = output/'resume.pt'
    if last.exists():
        loaded = torch.load(last, weights_only=True, map_location='cpu')
        if loaded['binding'] != binding:
            raise ValueError('sender resume binding changed')
        for arm in ARMS:
            models[arm].load_state_dict(loaded['models'][arm], strict=True)
            optimizers[arm].load_state_dict(loaded['optimizers'][arm])
        state = cpu_tree(loaded['state'])
        if (not 0 <= state['epoch'] <= protocol['epochs'] or not 0 <= state['cursor'] <= len(train)
                or state['updates'] != state['epoch']*len(train)+state['cursor']
                or len(state['history']) != state['epoch']):
            raise ValueError('invalid sender checkpoint cursor/history')
        restore_rng(loaded['rng'], device)

    def checkpoint():
        atomic_torch(last, dict(binding=binding, state=cpu_tree(state), rng=capture_rng(device),
            models={a:cpu_tree(m.state_dict()) for a,m in models.items()},
            optimizers={a:cpu_tree(o.state_dict()) for a,o in optimizers.items()}))

    def export(arm, weights, path, step, score, epoch):
        current = cpu_tree(models[arm].state_dict())
        models[arm].load_state_dict(weights, strict=True)
        model_binding = dict(protocol=protocol, architecture=binding['architectures'][arm], arm=arm,
            labels=binding['labels'], scale=binding['scale'])
        atomic_torch(path, sender.export_payload(models[arm], model_binding, step,
            dict(epoch=epoch, validation=score, scope='measured candidate ranking; not full-video RD')))
        models[arm].load_state_dict(current, strict=True)

    rank_weight = protocol['ranking_weight']
    if state['initial_validation'] is None:
        state['initial_validation'] = {arm: validate(models[arm], arm, valid, get, scale,
            device, check, rank_weight) for arm in ARMS}
        checkpoint()
    began = time.monotonic()
    for epoch in range(state['epoch'], protocol['epochs']):
        order = torch.randperm(len(train), generator=torch.Generator().manual_seed(protocol['seed']+epoch)).tolist()
        for position in range(state['cursor'], len(order)):
            check()
            row = train[order[position]]
            data = get(row)
            # No interrupt checkpoint between arms: replay this entire paired
            # update from the preceding atomic checkpoint if either arm fails.
            losses = {}
            for arm in ARMS:
                model, optimizer = models[arm], optimizers[arm]
                batch = arm_batch(data, arm, device)
                with codec_precision():
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    loss, _ = sender.training_loss(model(batch['inputs']), batch['targets'],
                        batch['packet_bytes'], gain_scale=scale, ranking_weight=rank_weight)
                    if not torch.isfinite(loss):
                        raise RuntimeError('nonfinite sender loss')
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                    if not torch.isfinite(norm):
                        raise RuntimeError('nonfinite sender gradient')
                    optimizer.step()
                losses[arm] = float(loss.detach())
            for arm in ARMS:
                state['epoch_losses'][arm] += losses[arm]
            state['cursor'] = position+1
            state['updates'] += 1
            checkpoint()
            status = dict(phase='smoke_sender_optimization' if protocol['smoke'] else 'formal_sender_optimization',
                epoch=epoch+1, epochs=protocol['epochs'], window=position+1, train_windows=len(train),
                updates_per_arm=state['updates'], sample_id=row['sample_id'], losses=losses,
                both_sender_optimizers_updated=True, all_final_labels_ready=True, checkpoint_loadable=True,
                seconds_this_process=time.monotonic()-began)
            save(output/'progress.json', status)
            progress(**status)
            print('SENDER_TRAIN', status, flush=True)
            if stop_after and state['updates'] >= stop_after:
                return None
        scores = {arm:validate(models[arm], arm, valid, get, scale, device, check, rank_weight) for arm in ARMS}
        for arm in ARMS:
            previous = state['best'].get(arm)
            score = scores[arm]
            if previous is None or (score['regret'],score['loss']) < (previous['score']['regret'],previous['score']['loss']):
                state['best'][arm] = dict(epoch=epoch+1, step=state['updates'], score=score,
                                        weights=cpu_tree(models[arm].state_dict()))
        state['history'].append(dict(epoch=epoch+1,
            train_loss={arm:state['epoch_losses'][arm]/len(train) for arm in ARMS}, validation=scores))
        state.update(epoch=epoch+1, cursor=0, epoch_losses={arm:0. for arm in ARMS})
        checkpoint()
        save(output/'history.json', state['history'])
        save(output/'initial_validation.json', state['initial_validation'])
        for arm in ARMS:
            best = state['best'][arm]
            export(arm, best['weights'], output/arm/'best.pt', best['step'], best['score'], best['epoch'])
        print(f'SENDER_EPOCH {epoch+1}/{protocol["epochs"]} best=' +
              str({arm:value['epoch'] for arm,value in state['best'].items()}), flush=True)
    # Authoritative final checkpoint may precede any JSON/model export. Repair
    # missing/stale exports unconditionally without another teacher/model step.
    save(output/'history.json', state['history'])
    save(output/'initial_validation.json', state['initial_validation'])
    for arm in ARMS:
        best = state['best'][arm]
        export(arm, models[arm].state_dict(), output/arm/'last.pt', state['updates'],
               state['history'][-1]['validation'][arm], protocol['epochs'])
        export(arm, best['weights'], output/arm/'best.pt', best['step'], best['score'], best['epoch'])
        loaded, _ = sender.load_model(output/arm/'best.pt')
        for key,value in loaded.state_dict().items():
            torch.testing.assert_close(value, best['weights'][key], rtol=0, atol=0)
    names = ['resume.pt','history.json','initial_validation.json'] + [
        f'{arm}/{name}.pt' for arm in ARMS for name in ('best','last')]
    result = dict(complete=True, config=digest(output/'config.json'), epochs=protocol['epochs'],
        updates_per_arm=state['updates'], parameters_per_arm=sum(p.numel() for p in models['source'].parameters()),
        selected={arm:dict(epoch=value['epoch'], step=value['step'], validation=value['score'])
                  for arm,value in state['best'].items()}, artifacts={name:digest(output/name) for name in names},
        role='source-aware sender final-marginal prediction', paired_arms=list(ARMS),
        source_ablation_complete=True, receiver_weights_shared=False, no_UF_E_G_Rg_updates=True,
        labels_scope=sender.LABEL_SCOPE, full_allocation_evaluation_pending=True,
        whole_video_RD_pending=True, semantic_supervision=False, transmitted_masks=False)
    save(output/'complete.json', result)
    return result


def verify_training(output, protocol, *, labels_binding=None):
    output = Path(output)
    binding = read(output/'config.json')
    if binding['protocol'] != protocol or (labels_binding is not None and binding['labels'] != labels_binding):
        raise ValueError('completed sender training inputs differ')
    result = read(output/'complete.json')
    if (not result.get('complete') or result['config'] != digest(output/'config.json')
            or result['epochs'] != protocol['epochs']):
        raise ValueError('sender completion binding is invalid')
    verify_artifacts(output, result['artifacts'])
    state = torch.load(output/'resume.pt', weights_only=True, map_location='cpu')
    if (state['binding'] != binding or state['state']['epoch'] != protocol['epochs']
            or state['state']['cursor'] != 0 or len(state['state']['history']) != protocol['epochs']
            or state['state']['updates'] != len(split_rows(protocol)[0])*protocol['epochs']
            or result['updates_per_arm'] != state['state']['updates']
            or tuple(result['paired_arms']) != ARMS
            or read(output/'history.json') != state['state']['history']
            or read(output/'initial_validation.json') != state['state']['initial_validation']):
        raise ValueError('sender final checkpoint/history/updates are inconsistent')
    for arm in ARMS:
        for kind in ('best','last'):
            model,payload = sender.load_model(output/arm/(kind+'.pt'))
            expected = state['state']['best'][arm]['weights'] if kind == 'best' else state['models'][arm]
            selected = state['state']['best'][arm]
            step = selected['step'] if kind == 'best' else state['state']['updates']
            epoch = selected['epoch'] if kind == 'best' else protocol['epochs']
            score = selected['score'] if kind == 'best' else state['state']['history'][-1]['validation'][arm]
            if (payload['binding']['protocol'] != protocol or payload['binding']['arm'] != arm
                    or payload['binding']['labels'] != binding['labels']
                    or payload['binding']['scale'] != binding['scale']
                    or payload['binding']['architecture'] != binding['architectures'][arm]
                    or asdict(model.config) != binding['architectures'][arm]
                    or payload['step'] != step or payload['selection']['epoch'] != epoch
                    or payload['selection']['validation'] != score):
                raise ValueError('sender exported payload binding changed')
            for key,value in model.state_dict().items():
                torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stop-after', type=int, default=0)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if not os.environ.get('TMUX') or not os.environ.get('ROUTERVC_SENDER_PARENT'):
        raise RuntimeError('supervised tmux sender worker required')
    protocol = read(args.root/'protocol.json')
    from demo.routervc_sender_data import Samples
    from demo.scalable_experiment import check_space
    stopped = False

    def stop(*_):
        nonlocal stopped
        stopped = True

    def check():
        if stopped:
            raise InterruptedError('resume from last final teacher state or paired sender update')
        check_space()
        for name,expected in protocol['code'].items():
            if digest(REPO/'demo'/name) != expected:
                raise ValueError('pinned sender dependency changed: '+name)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    check()
    samples = Samples(args.root, args.cache, digest(args.root/'protocol.json'), check)
    try:
        if args.verify_only:
            labels = verify_labels(args.root, protocol, samples, check)
            verify_training(args.output, protocol, labels_binding=labels)
            print('SENDER_TRAINING_VERIFIED_READ_ONLY', flush=True)
            return
        labels = (verify_labels(args.root, protocol, samples, check)
                  if (args.root/'labels.complete.json').exists()
                  else prepare_labels(args.root, protocol, samples, check))
        samples.release_generator()
        get = lambda row: samples.get(row, generate=False)
        scale = fit_scale(protocol, get, check)
        immutable(args.root/'train_scale.json', scale)
        run_training(args.output, protocol, get, scale_record=scale, labels_binding=labels,
                     device='cuda', check=check, stop_after=args.stop_after)
    except BaseException as error:
        save(args.output/'last_failure.json', dict(error=repr(error)))
        raise
    finally:
        samples.release_generator()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
