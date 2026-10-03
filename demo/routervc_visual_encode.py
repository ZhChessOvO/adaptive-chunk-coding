"""Source -> real UF/E stream with the isolated, perceptual visual Router.

Explicit --router is required: no automatic deployment of smoke/new weights.
Preparation is the unchanged UF/E implementation; the sender never executes G.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from demo import routervc_visual_format as fmt
from demo import routervc_visual_policy as policy
from demo.routervc import (_budgets, immutable, read, verify_complete,
                          validate_generation_geometry, DEFAULT_ADAPTER)
from demo.routervc_encode import (DEFAULT_ENHANCEMENT, _validate_candidates,
    load_input, prepare, subset_bank, compose_candidates, bank_info)
from demo.scalable_codec import atomic_bytes, atomic_json, file_hash
from demo.scalable_format import frame_hash, parse as parse_inner, sha256


def select(bank, base, all_e, router, extra_e_budget, max_g_calls, *, mode='prefix'):
    """Reuse real packet costs; isolated teacher utility is only a planning approximation."""
    began = time.monotonic()
    info = _validate_candidates(bank, base, all_e)
    utility, prediction = policy.predict_utility(bank, base, all_e, router)
    prediction_seconds = time.monotonic()-began
    began = time.monotonic()
    plan = policy.allocate(utility, info['e_bytes'], extra_e_budget, max_g_calls, mode=mode)
    indices = plan['selected_indices']
    inner = subset_bank(bank, indices)
    used = len(inner)-info['parsed'].base_end
    if used != sum(info['e_bytes'][i] for i in indices) or used > extra_e_budget:
        raise RuntimeError('serialized visual Router E budget mismatch')
    mixed = compose_candidates(base, all_e, indices, info['rois'])
    return dict(plan, e_packet_bytes=used, budget_e_packet_bytes=extra_e_budget,
        unused_e_budget_bytes=extra_e_budget-used, max_g_calls=max_g_calls,
        utility=utility.tolist(), prediction=prediction.tolist(),
        prediction_seconds=prediction_seconds, allocation_seconds=time.monotonic()-began,
        mixed_rgb_sha256=frame_hash(mixed), inner_sha256=sha256(inner), inner_bytes=len(inner),
        source_frames_used_by_router=False, generation_mask_transmitted=False,
        semantic_heads_used=False, allocation_is_mixed_video_oracle=False)


def encode(args, run):
    began = time.monotonic(); _budgets(args)
    run.update(phase='visual_sender_preflight')
    source, provenance = load_input(args.input, key=args.input_key, start=args.start, count=args.count)
    shape = list(source.shape); rois = validate_generation_geometry(shape, args.max_g)
    del source
    config = fmt.make_config(args.router, args.adapter, max_g=args.max_g,
                             boundary_lambda=args.boundary_lambda, seed=args.seed)
    prepared = (args.prepared_dir or args.output/'prepared').resolve()
    binding = dict(profile=fmt.PROFILE, input=provenance, prepared_dir=str(prepared),
        enhancement=file_hash(args.enhancement), config=config,
        e_budget=args.e_budget, e_ratio=args.e_ratio, mode=args.mode, code=fmt.code_identity())
    done = args.output/'encode.json'
    if done.exists():
        return verify_complete(args.output, read(done), binding)
    immutable(args.output/'encode.request.json', binding)
    # Reject old/incompatible/semantic-capable bundles before costly UF work.
    model = policy.load_model(args.router, expected_sha256=config['router'])
    run.check(); run.update(phase='unchanged_UF_E_preparation')
    reused = (prepared/'complete.json').exists(); phase = time.monotonic()
    record = prepare(args.input, prepared, enhancement=args.enhancement,
                     input_key=args.input_key, start=args.start, count=args.count)
    preparation_seconds = time.monotonic()-phase
    bank = (prepared/'bank.acse').read_bytes(); info = bank_info(bank)
    with np.load(prepared/'candidates.npz', allow_pickle=False) as cache:
        base, all_e = cache['base'].copy(), cache['all_E'].copy()
    if list(base.shape) != shape or frame_hash(all_e) != record['all_E_rgb_sha256']:
        raise ValueError('prepared candidate geometry/pixels differ')
    budget = args.e_budget if args.e_budget is not None else int(sum(info['e_bytes'])*args.e_ratio)
    run.check(); run.update(phase='visual_E_allocation_and_shared_CPU_route')
    plan = select(bank, base, all_e, model, budget, args.max_g, mode=args.mode)
    inner = subset_bank(bank, plan['selected_indices'])
    mixed = compose_candidates(base, all_e, plan['selected_indices'], rois)
    phase = time.monotonic()
    expected = policy.route(base, mixed, parse_inner(inner), config, model,
                            expected_policy=fmt.policy_identity()) if args.max_g else None
    shared_route_seconds = time.monotonic()-phase
    run.check(); wire = fmt.wrap(inner, config); stream = args.output/'stream.rtvc'
    if stream.exists() and stream.read_bytes() != wire:
        raise ValueError('partial encode contains a different stream')
    if not stream.exists():
        atomic_bytes(stream, wire)
    stored, _, parsed, header = fmt.parse(stream.read_bytes())
    counts = dict(base_native_bytes=len(parsed.base), ACSE_header_bytes=parsed.base_end-len(parsed.base),
        E_packet_bytes=sum(len(p.wire) for p in parsed.packets), RouterVC_shared_header_bytes=header,
        explicit_E_mask_bytes=0, explicit_G_mask_bytes=0, protection_mask_bytes=0)
    if stored != config or sum(counts.values()) != stream.stat().st_size or counts['E_packet_bytes'] != plan['e_packet_bytes']:
        raise RuntimeError('visual Router wire byte accounting differs')
    result = dict(complete=True, binding=binding, config=config, source_shape=shape,
        stream=str(stream.resolve()), actual_on_disk_bytes=stream.stat().st_size, byte_breakdown=counts,
        bpp=8*len(wire)/int(np.prod(shape[:3])), expected_shared_route=expected,
        expected_base_hash=frame_hash(base), expected_mixed_hash=frame_hash(mixed), plan=plan,
        candidate_cache_reused=reused, candidate_preparation=str((prepared/'complete.json').resolve()),
        candidate_preparation_sha256=file_hash(prepared/'complete.json'),
        timing=dict(prepare_call_seconds=preparation_seconds, shared_route_seconds=shared_route_seconds,
                    wall_seconds_this_attempt=time.monotonic()-began,
                    gpu_mutex_wait_seconds=getattr(run, 'gpu_wait_seconds', 0.)),
        generation_executed_at_sender=False, source_frames_used_by_router=False,
        semantic_heads_used=False, explicit_G_mask_bytes=0, automatically_promoted=False,
        budget_scope='E packet bytes only; native base and both headers charged separately',
        artifacts={name:file_hash(args.output/name) for name in ('stream.rtvc', 'encode.request.json')})
    atomic_json(done, result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--router', type=Path, required=True); p.add_argument('--prepared-dir', type=Path)
    p.add_argument('--input-key', default='source'); p.add_argument('--start', type=int, default=0)
    p.add_argument('--count', type=int)
    budgets = p.add_mutually_exclusive_group(required=True)
    budgets.add_argument('--e-budget', type=int); budgets.add_argument('--e-ratio', type=float)
    p.add_argument('--max-g', type=int, default=4); p.add_argument('--boundary-lambda', type=float, default=.004)
    p.add_argument('--mode', choices=('prefix','independent'), default='prefix')
    p.add_argument('--seed', type=int, default=20261003)
    p.add_argument('--enhancement', type=Path, default=DEFAULT_ENHANCEMENT)
    p.add_argument('--adapter', type=Path, default=DEFAULT_ADAPTER)
    p.add_argument('--max-hours', type=float, default=24.)
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args(argv); args.command = 'visual_encode'
    if args.worker:
        if not os.environ.get('TMUX') or not os.environ.get('ROUTERVC_VISUAL_PARENT'):
            raise RuntimeError('encode worker requires the supervising tmux parent')
        from demo.scalable_experiment import check_space
        args.output.mkdir(parents=True, exist_ok=True)
        # A bounded parent execute() owns timeout, heartbeat and the GPU mutex.
        run = SimpleNamespace(check=check_space,
            update=lambda **kw: print(json.dumps(kw), flush=True))
        return encode(args, run)
    return fmt.supervised(args, encode)


if __name__ == '__main__':
    main()
