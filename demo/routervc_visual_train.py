"""Resumable visual-Router perceptual baseline, NOT semantic-protection training.

Consumes measured mixed-view four-state labels. Only decoded B/Y pixels enter
the network. Content supervision remains explicitly unknown until real labels
exist; untrained content heads must never drive a receiver policy.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import gc
import json
import math
import os
from pathlib import Path
import time

import torch

from demo import routervc_visual_router as visual
from demo.chunk_enhancement_codec import atomic_torch
from demo.scalable_codec import atomic_json, file_hash

FORMAT = 'routervc_visual_perceptual_train_v1'


def read(path):
    return json.loads(Path(path).read_text())


def immutable(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError(f'changed training configuration: {path}; use a new output')
    else:
        atomic_json(path, value)


def code_hashes():
    return {p.name: file_hash(p) for p in
            (Path(__file__), Path(visual.__file__))}


def teacher_entries(path):
    manifest = read(path)
    if not manifest.get('complete'):
        raise ValueError('measured teacher data must be complete before training')
    rows = manifest['samples']
    if not rows or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('empty/duplicate teacher examples')
    for row in rows:
        if row['router_split'] not in ('train', 'validation'):
            raise ValueError('evaluation samples must not enter the training ledger')
        for field, digest in (('path', 'sha256'), ('reconstruction_path', 'reconstruction_sha256')):
            if file_hash(Path(row[field])) != row[digest]:
                raise ValueError(f'changed measured teacher input: {row[field]}')
        labels = read(row['path'])
        for field in ('sample_id', 'dataset', 'sequence', 'router_split'):
            if labels.get(field) != row[field]:
                raise ValueError(f'label identity differs from manifest: {field}')
        if labels.get('source_role') == 'evaluation' or row.get('source_role') == 'evaluation':
            raise ValueError('evaluation-labelled source is not a training example')
    groups = {}
    for row in rows:
        key = (row['dataset'], row['sequence'])
        if key in groups and groups[key] != row['router_split']:
            raise ValueError('sequence appears in both Router train and validation')
        groups[key] = row['router_split']
    return rows


def prepare_cache(labels, cache, config, check=lambda: None):
    """Cache small letterboxed inputs on fast disk, not duplicate full videos."""
    labels, cache = Path(labels), Path(cache)
    rows = teacher_entries(labels)
    cache.mkdir(parents=True, exist_ok=True)
    # Global/local ablations share identical preprocessed tensors.
    shape = asdict(replace(config, use_global=True))
    binding = dict(format=FORMAT, labels_sha256=file_hash(labels), input_config=shape,
                   code=code_hashes(), dtype='float32 cache and receiver preprocessing',
                   semantic_supervision=False)
    immutable(cache/'request.json', binding)
    records = []
    for index, row in enumerate(rows):
        check()
        sid = row['sample_id']
        if not sid or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-' for c in sid):
            raise ValueError('unsafe sample identifier')
        path, done = cache/f'{sid}.pt', cache/f'{sid}.json'
        item_binding = dict(request=file_hash(cache/'request.json'), teacher=row)
        if done.exists():
            meta = read(done)
            if meta['binding'] != item_binding or file_hash(path) != meta['sha256']:
                raise ValueError('changed cached visual inputs')
        else:
            sample = visual.load_measured_sample(row['path'], row['reconstruction_path'])
            cases = [visual.build_training_case(**sample, region=i, with_e=e, config=config)
                     for i in range(16) for e in (False, True)]
            batch = visual.batch_examples(cases)
            if visual.supervision_metadata(batch['targets'])['semantic_supervision']:
                raise ValueError('this baseline runner must not silently introduce semantic labels')
            # Preserve the exact public receiver preprocessing; cache storage is
            # cheaper than introducing another train/inference rounding path.
            batch['inputs'] = {k: v.float() for k, v in batch['inputs'].items()}
            atomic_torch(path, batch)
            meta = dict(binding=item_binding, sha256=file_hash(path), cases=32)
            atomic_json(done, meta)
            del sample, cases, batch
        records.append(dict(**row, cache_path=str(path.resolve()), cache_sha256=meta['sha256']))
        print(f'VISUAL_INPUTS {index+1}/{len(rows)} {sid}', flush=True)
    immutable(cache/'complete.json', dict(complete=True, binding=binding, records=records))
    return records


def load_cases(records):
    """A few GB of compact tensors; no original/source video is loaded here."""
    parts = []
    for row in records:
        path = Path(row['cache_path'])
        if file_hash(path) != row['cache_sha256']:
            raise ValueError('changed cached examples')
        parts.append(torch.load(path, weights_only=True, map_location='cpu'))
    return visual.batch_examples(parts)


def select_batch(data, indices, device):
    return dict(inputs={k: v[indices].to(device=device, dtype=torch.float32) for k, v in data['inputs'].items()},
                targets={k: {f: v[indices].to(device=device, dtype=torch.float32) for f, v in item.items()}
                         for k, item in data['targets'].items()})


def train_scales(data):
    values, weights = (data['targets']['gains'][k] for k in ('value', 'weight'))
    return torch.tensor([values[..., i][weights[..., i] > 0].std(unbiased=False).clamp_min(.01)
                         if torch.any(weights[..., i] > 0) else 1. for i in range(6)])


@torch.no_grad()
def validation(model, data, scale, batch_size, device):
    model.eval()
    count = data['inputs']['coverage'].shape[0]
    error, weight = torch.zeros(6, device=device), torch.zeros(6, device=device)
    total, n = 0., 0
    for ids in torch.arange(count).split(batch_size):
        batch = select_batch(data, ids, device)
        output = model(batch['inputs'])
        loss, _ = visual.masked_training_loss(output, batch['targets'], gain_scale=scale)
        value, mask = (batch['targets']['gains'][k] for k in ('value', 'weight'))
        absolute = torch.where(mask > 0, (output['gains']-value).abs(), 0.)
        error += (absolute*mask).sum((0, 1))
        weight += mask.sum((0, 1))
        total += float(loss)*len(ids); n += len(ids)
    return dict(loss=total/n, gain_mae=(error/weight.clamp_min(1)).cpu().tolist())


def train_arm(train_data, valid_data, output, *, config, data_binding, epochs=120,
              batch_size=64, learning_rate=3e-4, device='cpu', seed=20261003,
              stop_after=0, check=lambda: None, progress=lambda **kw: None):
    """Epoch-atomic restart with per-epoch RNG; interrupted epoch is replayed."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    if epochs < 1 or batch_size < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('invalid training hyperparameters')
    if any(visual.supervision_metadata(data['targets'])['semantic_supervision']
           for data in (train_data, valid_data)):
        raise ValueError('perceptual baseline runner cannot consume semantic labels')
    binding = dict(format=FORMAT, model=asdict(config), data=data_binding, code=code_hashes(),
                   epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                   seed=seed, device=str(device), semantic_supervision=False,
                   purpose='global/local perceptual utility baseline; not a deployed content-protection Router')
    immutable(output/'config.json', binding)
    done = output/'complete.json'
    if done.exists():
        result = read(done)
        if result['config_sha256'] != file_hash(output/'config.json'):
            raise ValueError('completed model configuration changed')
        for name, digest in result['artifacts'].items():
            if file_hash(output/name) != digest:
                raise ValueError('completed training artifact changed')
        return result
    torch.manual_seed(seed)
    model = visual.VisualUtilityRouter(config).to(device)
    scale = train_scales(train_data).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    last = output/'resume.pt'; start, history = 0, []
    if last.exists():
        state = torch.load(last, weights_only=True, map_location=device)
        if state['binding'] != binding:
            raise ValueError('resume configuration changed')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        start, history = state['epoch'], state['history']
        torch.testing.assert_close(scale, state['scale'].to(device), rtol=0, atol=0)
    began = time.monotonic()
    count = train_data['inputs']['coverage'].shape[0]
    for epoch in range(start, epochs):
        check(); model.train()
        order = torch.randperm(count, generator=torch.Generator().manual_seed(seed+epoch))
        total = 0.
        for ids in order.split(batch_size):
            check()
            batch = select_batch(train_data, ids, device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = visual.masked_training_loss(model(batch['inputs']), batch['targets'], gain_scale=scale)
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite visual Router loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step(); total += float(loss.detach())*len(ids)
        measured = validation(model, valid_data, scale, batch_size, device)
        history.append(dict(epoch=epoch+1, train_loss=total/count, validation=measured))
        atomic_torch(last, dict(binding=binding, epoch=epoch+1, model=model.state_dict(),
                               optimizer=optimizer.state_dict(), scale=scale, history=history))
        atomic_json(output/'progress.json', history[-1])
        progress(phase='visual_router_training', arm='global_local' if config.use_global else 'local',
                 epoch=epoch+1, epochs=epochs, train_loss=total/count)
        print(f'VISUAL_TRAIN global={config.use_global} epoch={epoch+1}/{epochs} loss={total/count:.6f}', flush=True)
        if stop_after and epoch+1 == stop_after:
            return None
    # Keep the last epoch and report its paired validation, not a test-selected winner.
    payload = dict(format=visual.FORMAT, architecture=asdict(config), state_dict=model.state_dict(),
                   training_binding=binding, semantic_supervision=False,
                   content_heads_usable=False, metadata=model.metadata())
    atomic_torch(output/'model.pt', payload)
    result = dict(complete=True, config_sha256=file_hash(output/'config.json'), epochs=epochs,
                  parameters=model.metadata()['parameters'], validation=history[-1]['validation'],
                  wall_seconds_this_completion_attempt=time.monotonic()-began,
                  semantic_supervision=False, deployed_receiver=False,
                  artifacts={name: file_hash(output/name) for name in ('resume.pt', 'model.pt', 'progress.json')})
    atomic_json(done, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-hours', type=float, default=12.)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if not os.environ.get('TMUX'):
        raise RuntimeError('visual Router training requires tmux')
    from demo.chunk_enhancement_experiment import Run
    from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
    from contextlib import nullcontext
    args.command = 'visual_router'
    run = Run(args); run.thread.start()
    torch.set_num_threads(4)
    try:
        config = visual.VisualRouterConfig()
        records = prepare_cache(args.labels, args.cache, config, run.check)
        train_rows = [r for r in records if r['router_split'] == 'train']
        valid_rows = [r for r in records if r['router_split'] == 'validation']
        if args.smoke:
            # Deliberately same tiny data: implementation test, never a generalization score.
            train_rows = valid_rows = records
        if not train_rows or not valid_rows:
            raise ValueError('both existing training and validation groups are required')
        train_data, valid_data = load_cases(train_rows), load_cases(valid_rows)
        binding = dict(labels=file_hash(args.labels), cache=file_hash(args.cache/'complete.json'),
                       train_ids=[r['sample_id'] for r in train_rows],
                       valid_ids=[r['sample_id'] for r in valid_rows], smoke=args.smoke,
                       views='REDS resized full field; UVG existing crop', semantic_supervision=False)
        context = exclusive_native_evaluation(run) if args.device == 'cuda' else nullcontext()
        with context:
            if args.device == 'cuda':
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
                torch.backends.cuda.matmul.allow_tf32 = False
            for name, use_global in (('global_local', True), ('local', False)):
                train_arm(train_data, valid_data, args.output/name,
                          config=replace(config, use_global=use_global), data_binding=binding,
                          epochs=2 if args.smoke else args.epochs, batch_size=args.batch_size,
                          device=args.device, check=run.check, progress=run.update)
                gc.collect()
                if args.device == 'cuda':
                    torch.cuda.empty_cache()
        immutable(args.output/'complete.json', dict(complete=True, binding=binding,
                  arms={name: file_hash(args.output/name/'complete.json') for name in ('global_local', 'local')},
                  semantic_supervision=False, deployment_pending=True))
    except BaseException as error:
        atomic_json(args.output/'last_failure.json', dict(error=repr(error), progress=run.progress))
        raise
    finally:
        run.log_resources(); run.stop.set(); run.thread.join(timeout=3)


if __name__ == '__main__':
    main()
