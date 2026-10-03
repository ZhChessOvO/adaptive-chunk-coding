"""Supervised RouterVC encode/decode CLI, independent of historical experiments.

Encoding builds real UF/E packets and simulates only the shared CPU route; it
never executes the generator. Decoding starts a source-free fresh process.
Both commands require tmux and share the repository-wide single-GPU mutex.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from demo import routervc_format as fmt
from demo.routervc_encode import (DEFAULT_ENHANCEMENT, DEFAULT_ROUTER, load_input,
    prepare, select, subset_bank, compose_candidates, bank_info)
from demo.routervc_policy import grid_rois
from demo.routervc_decode import route
from demo.scalable_codec import atomic_bytes, atomic_json, file_hash
from demo.scalable_format import frame_hash, parse as parse_inner
from demo.chunk_enhancement_experiment import Run
from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
from demo.conditioned_generation_pipeline import execute

DEFAULT_ADAPTER = DEFAULT_ENHANCEMENT.parent/'adapter.pt'
CODE = ('routervc.py', 'routervc_encode.py', 'routervc_decode.py',
        'routervc_format.py', 'routervc_policy.py', 'run_routervc_codec.sh',
        'four_state_router.py', 'four_state_router_evaluate.py')


def read(path):
    return json.loads(Path(path).read_text())


def code_identity():
    return {name: file_hash(REPO/'demo'/name) for name in CODE}


def immutable(path, value):
    if path.exists():
        if read(path) != value:
            raise RuntimeError(f'changed configuration at {path}; choose a new output directory')
    else:
        atomic_json(path, value)


def verify_complete(root, record, binding):
    if record['binding'] != binding:
        raise RuntimeError('completed command inputs/configuration changed; choose a new output directory')
    for name, digest in record['artifacts'].items():
        if file_hash(root/name) != digest:
            raise RuntimeError(f'completed RouterVC artifact changed: {root/name}')
    return record


def validate_generation_geometry(shape, max_g):
    """Current RGB-first G profile does not silently resize or pad crops."""
    if type(max_g) is not int or not 0 <= max_g <= 16:
        raise ValueError('max-g must be an integer between 0 and 16')
    count, height, width, channels = shape
    if channels != 3:
        raise ValueError('expected RGB input')
    rois = grid_rois(height, width)
    if max_g:
        if count < 17:
            raise ValueError('RouterVC v1 G requires at least 17 frames')
        for x, y, w, h in rois:
            cw = min(width, x+w+64)-max(0, x-64)
            ch = min(height, y+h+64)-max(0, y-64)
            if min(w, h) <= 32 or cw % 16 or ch % 16:
                raise ValueError('RouterVC v1 G requires region cores >32 and clipped context size '
                                 'divisible by16; supported examples: 512x512, 384x256. '
                                 'No implicit resize; --max-g 0 permits E-only geometry.')
    return rois


def _budgets(args):
    if (args.e_budget is None) == (args.e_ratio is None):
        raise ValueError('specify exactly one of --e-budget and --e-ratio')
    if args.e_budget is not None and (type(args.e_budget) is not int or args.e_budget < 0):
        raise ValueError('e-budget must be a nonnegative integer')
    if args.e_ratio is not None and (not math.isfinite(args.e_ratio) or not 0 <= args.e_ratio <= 1):
        raise ValueError('e-ratio must be between zero and one')
    if not math.isfinite(args.boundary_lambda) or not 0 <= args.boundary_lambda <= 1:
        raise ValueError('boundary-lambda must be between zero and one')


def encode(args, run):
    began = time.monotonic()
    _budgets(args)
    run.update(phase='source_geometry_and_fingerprints')
    source, provenance = load_input(args.input, key=args.input_key, start=args.start, count=args.count)
    # Must happen BEFORE any codec/GPU model is initialized.
    rois = validate_generation_geometry(source.shape, args.max_g)
    source_shape = list(source.shape)
    del source
    config = fmt.make_config(args.router, args.adapter, max_g=args.max_g,
        boundary_lambda=args.boundary_lambda, seed=args.seed)
    fmt.validate(config)
    prepared = args.prepared_dir if args.prepared_dir is not None else args.output/'prepared'
    prepared = prepared.resolve()
    binding = dict(version=1, command='encode', input=provenance, prepared_dir=str(prepared),
        enhancement=file_hash(args.enhancement), router=file_hash(args.router), config=config,
        e_budget=args.e_budget, e_ratio=args.e_ratio, allocation=args.mode,
        code=code_identity())
    done = args.output/'encode.json'
    if done.exists():
        return verify_complete(args.output, read(done), binding)
    immutable(args.output/'encode.request.json', binding)
    preflight_seconds = time.monotonic()-began
    run.check()
    run.update(phase='prepare_real_UF_and_E_candidates')
    reused = (prepared/'complete.json').exists()
    phase = time.monotonic()
    preparation = prepare(args.input, prepared, enhancement=args.enhancement,
        input_key=args.input_key, start=args.start, count=args.count)
    prepare_call_seconds = time.monotonic()-phase
    bank = (prepared/'bank.acse').read_bytes()
    info = bank_info(bank)
    with np.load(prepared/'candidates.npz', allow_pickle=False) as cache:
        base, all_e = cache['base'].copy(), cache['all_E'].copy()
    if list(base.shape) != source_shape or frame_hash(all_e) != preparation['all_E_rgb_sha256']:
        raise RuntimeError('prepared reconstructions differ from verified sender manifest')
    budget = args.e_budget if args.e_budget is not None else int(sum(info['e_bytes'])*args.e_ratio)
    run.check()
    run.update(phase='single_backbone_E_allocation')
    plan = select(bank, base, all_e, args.router, budget, args.max_g, mode=args.mode)
    inner = subset_bank(bank, plan['selected_indices'])
    mixed = compose_candidates(base, all_e, plan['selected_indices'], rois)
    if frame_hash(mixed) != plan['mixed_rgb_sha256']:
        raise RuntimeError('sender E composition hash mismatch')
    run.update(phase='simulate_shared_receiver_policy_no_generation')
    phase = time.monotonic()
    expected_route = route(base, mixed, parse_inner(inner), config, args.router) if args.max_g else None
    policy_seconds = time.monotonic()-phase
    run.check()
    phase = time.monotonic()
    wire = fmt.wrap(inner, config)
    stream = args.output/'stream.rtvc'
    if stream.exists() and stream.read_bytes() != wire:
        raise RuntimeError('incomplete run already contains a different stream')
    if not stream.exists():
        atomic_bytes(stream, wire)
    stored_config, _, parsed, header_bytes = fmt.parse(stream.read_bytes())
    byte_counts = dict(base_native_bytes=len(parsed.base),
        ACSE_header_bytes=parsed.base_end-len(parsed.base),
        E_packet_bytes=sum(len(p.wire) for p in parsed.packets),
        RouterVC_shared_header_bytes=header_bytes, explicit_G_mask_bytes=0)
    actual_bytes = stream.stat().st_size
    if (sum(byte_counts.values()) != actual_bytes or stored_config != config
            or byte_counts['E_packet_bytes'] != plan['e_packet_bytes'] or plan['e_packet_bytes'] > budget):
        raise RuntimeError('serialized RouterVC byte accounting mismatch')
    record = dict(complete=True, binding=binding, stream=str(stream.resolve()), config=config,
        actual_on_disk_bytes=actual_bytes, byte_breakdown=byte_counts,
        bpp=8*actual_bytes/int(np.prod(source_shape[:3])), source_shape=source_shape,
        candidate_preparation=str((prepared/'complete.json').resolve()),
        candidate_preparation_sha256=file_hash(prepared/'complete.json'),
        candidate_cache_reused=reused, source_frames_used_by_router=False,
        generation_executed_at_sender=False, explicit_G_mask_bytes=0,
        expected_base_hash=frame_hash(base), expected_mixed_hash=frame_hash(mixed),
        expected_shared_route=expected_route,
        G_off_states=None if args.max_g else ['E' if i in plan['selected_indices'] else 'B' for i in range(16)],
        plan=plan, timing=dict(preflight_seconds=preflight_seconds,
            prepare_call_seconds=prepare_call_seconds,
            candidate_original_base_seconds=preparation['base_seconds'],
            candidate_original_encode_seconds=preparation['candidate_encode_seconds'],
            candidate_original_decode_seconds=preparation['candidate_decode_seconds'],
            prediction_seconds=plan['prediction_seconds'], allocation_seconds=plan['allocation_seconds'],
            shared_policy_simulation_seconds=policy_seconds, serialization_seconds=time.monotonic()-phase,
            wall_seconds_this_attempt=time.monotonic()-began,
            gpu_mutex_wait_seconds=getattr(run, 'gpu_wait_seconds', 0.)),
        budget_scope='E packet bytes only; base and constant shared header are separately charged',
        compute_scope='max-g is a region-call budget, not a millisecond bound; no G inference during encoding',
        artifacts={'stream.rtvc': file_hash(stream), 'encode.request.json': file_hash(args.output/'encode.request.json')})
    atomic_json(done, record)
    return record


def validate_decoded(output, stream, config):
    report = read(output/'decode.json')
    if report['stream_sha256'] != file_hash(stream) or report['total_bytes'] != stream.stat().st_size:
        raise RuntimeError('decoded stream identity/size mismatch')
    if (report['source_frames_read'] or not report['base_reference_unchanged']
            or not report['outside_generate_exact'] or report['config'] != config):
        raise RuntimeError('source-free decoder invariants failed')
    with np.load(output/'reconstruction.npz', allow_pickle=False) as cache:
        for field, key in [('reconstruction', 'output_hash'), ('base', 'base_hash'),
                           ('enhanced', 'generation_input_hash')]:
            if frame_hash(cache[field]) != report[key]:
                raise RuntimeError(f'decoded {field} artifact hash mismatch')
    return report


def decode(args, run):
    began = time.monotonic()
    config, _, inner, _ = fmt.parse(args.stream.read_bytes(), allow_incomplete_tail=args.allow_incomplete_tail)
    use_g = not args.disable_generation and config['max_g'] > 0 and config['blend'] > 0
    validate_generation_geometry((inner.meta['frame_count'], inner.meta['height'], inner.meta['width'], 3),
                                 config['max_g'] if use_g else 0)
    # Missing G/router assets are allowed for G-off. Missing E is allowed if no
    # E packet is received. Do not load/hash unused assets merely for bookkeeping.
    assets = {'enhancement': file_hash(args.enhancement) if inner.packets else None,
              'router': file_hash(args.router) if use_g else None,
              'adapter': file_hash(args.adapter) if use_g else None}
    binding = dict(version=1, command='decode', stream=file_hash(args.stream),
        disable_generation=args.disable_generation, allow_incomplete_tail=args.allow_incomplete_tail,
        used_assets=assets, config=config, code=code_identity())
    done = args.output/'decode.complete.json'
    if done.exists():
        return verify_complete(args.output, read(done), binding)
    immutable(args.output/'decode.request.json', binding)
    run.check()
    receiver = args.output/'fresh'
    receiver.mkdir(parents=True, exist_ok=True)
    if not (receiver/'decode.json').exists():
        argv = ['--stream', args.stream.resolve(), '--output', receiver.resolve(),
                '--router', args.router, '--adapter', args.adapter, '--enhancement', args.enhancement]
        if args.disable_generation:
            argv.append('--disable-generation')
        if args.allow_incomplete_tail:
            argv.append('--allow-incomplete-tail')
        # Parent holds the sole GPU mutex; this child does not acquire it again.
        execute(run, 'source_free_decode', 'routervc_decode.py', argv, distributed=True)
    report = validate_decoded(receiver, args.stream, config)
    result = dict(complete=True, binding=binding, report=report,
        receiver_output=str(receiver.resolve()), wall_seconds_this_attempt=time.monotonic()-began,
        gpu_mutex_wait_seconds=getattr(run, 'gpu_wait_seconds', 0.),
        artifacts={name: file_hash(args.output/name) for name in
                   ('decode.request.json', 'fresh/decode.json', 'fresh/reconstruction.npz')})
    atomic_json(done, result)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    enc = sub.add_parser('encode', help='source -> real stream; no generator inference')
    enc.add_argument('--input', type=Path, required=True)
    enc.add_argument('--input-key', default='source')
    enc.add_argument('--start', type=int, default=0)
    enc.add_argument('--count', type=int)
    enc.add_argument('--prepared-dir', type=Path, help='reuse a verified candidate bank for another budget')
    budget = enc.add_mutually_exclusive_group(required=True)
    budget.add_argument('--e-budget', type=int, help='maximum actual E packet bytes; header charged separately')
    budget.add_argument('--e-ratio', type=float, help='fraction of all-E packet bytes, in [0,1]')
    enc.add_argument('--max-g', type=int, default=4)
    enc.add_argument('--boundary-lambda', type=float, default=.004)
    enc.add_argument('--mode', choices=('prefix', 'independent'), default='prefix')
    enc.add_argument('--seed', type=int, default=20261003)
    dec = sub.add_parser('decode', help='fresh receiver; accepts no source input')
    dec.add_argument('--stream', type=Path, required=True)
    dec.add_argument('--disable-generation', action='store_true')
    dec.add_argument('--allow-incomplete-tail', action='store_true')
    for command in (enc, dec):
        command.add_argument('--output', type=Path, required=True)
        command.add_argument('--router', type=Path, default=DEFAULT_ROUTER)
        command.add_argument('--adapter', type=Path, default=DEFAULT_ADAPTER)
        command.add_argument('--enhancement', type=Path, default=DEFAULT_ENHANCEMENT)
        command.add_argument('--max-hours', type=float, default=24.)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if not os.environ.get('TMUX'):
        raise RuntimeError('RouterVC encode/decode are long GPU operations; run this CLI in tmux')
    if not math.isfinite(args.max_hours) or args.max_hours <= 0:
        raise ValueError('max-hours must be finite and positive')
    run = Run(args)
    run.thread.start()
    try:
        with exclusive_native_evaluation(run):
            result = encode(args, run) if args.command == 'encode' else decode(args, run)
            run.update(phase='complete', completed=1, total=1)
        print(json.dumps(dict(complete=True, command=args.command, output=str(args.output),
                             bytes=result.get('actual_on_disk_bytes', result.get('report', {}).get('total_bytes')))), flush=True)
        return result
    except BaseException as error:
        atomic_json(args.output/f'{args.command}.last_failure.json',
                    dict(error=repr(error), phase=run.progress))
        raise
    finally:
        run.log_resources()
        run.stop.set()
        run.thread.join(timeout=3)


if __name__ == '__main__':
    main()
