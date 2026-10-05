"""Fresh asymmetric RouterVC receiver: stream + Rg, never source video or Rs.

G-off reads no Router/G assets; a no-E stream reads no enhancement assets.
The tmux parent owns the shared GPU mutex and starts a fresh child receiver.
"""
import argparse
import gc
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from demo import routervc_receiver_format as fmt
from demo import routervc_receiver_policy as policy
from demo import scalable_cooperation_format as cooperation
from demo.routervc import (DEFAULT_ADAPTER, immutable, read, verify_complete,
                          validate_decoded, validate_generation_geometry)
from demo.routervc_encode import DEFAULT_ENHANCEMENT
from demo.chunk_enhancement_codec import configure_torch, load_model, decode_enhancement
from demo.scalable_codec import BaseCodec, atomic_json, atomic_npz, file_hash
from demo.scalable_format import frame_hash
from demo.online_eg_decode import identities
from demo.internal_condition_decode import restore
from demo.conditioned_generation_pipeline import execute


def _receive(args):
    began = time.monotonic()
    data = args.stream.read_bytes()
    config, inner_bytes, inner, header = fmt.parse(data, allow_incomplete_tail=args.allow_incomplete_tail)
    use_g = not args.disable_generation and config['max_g'] > 0 and config['blend'] > 0
    shape = (inner.meta['frame_count'], inner.meta['height'], inner.meta['width'], 3)
    rois = validate_generation_geometry(shape, config['max_g'] if use_g else 0)
    coverage = policy.coverage_from_packets(inner, rois)
    if use_g and (args.router is None or file_hash(args.router) != config['receiver_router']):
        raise ValueError('receiver Router weights missing or different')
    configure_torch()
    torch.cuda.reset_peak_memory_stats()
    codec = BaseCodec(REPO/'checkpoints/cvpr2026_image.pth.tar', REPO/'checkpoints/cvpr2026_video_hts.pth.tar')
    if any(inner.meta[k] != v for k, v in codec.models.items()):
        raise ValueError('UF weights mismatch')
    if inner.packets:
        model = load_model(args.enhancement)
        enhanced, report, base = decode_enhancement(model, args.enhancement, codec, inner_bytes,
            allow_incomplete_tail=args.allow_incomplete_tail, return_base=True)
        del model
    else:
        base = codec.decode(inner.base, inner.meta['frame_count'])
        if tuple(base.shape) != shape or frame_hash(base) != inner.meta['base_rgb_sha256']:
            raise ValueError('UF base reconstruction mismatch')
        enhanced = base.copy()
        report = dict(base_bytes=len(inner.base), container_header_bytes=inner.base_end-len(inner.base),
            packet_bytes=0, incomplete_tail_bytes=inner.incomplete_tail_bytes, applied_packets=[],
            base_hash=frame_hash(base), non_enhanced_exact=True)
    del codec
    gc.collect()
    torch.cuda.empty_cache()
    codec_seconds = time.monotonic()-began
    phase = time.monotonic()
    if use_g:
        selection = policy.route(base, enhanced, inner, config, args.router,
                                 expected_policy=fmt.policy_identity())
    else:
        selection = dict(indices=[], rois=rois, coverage=coverage.tolist(), policy_skipped=True,
                         states=['E' if c > 0 else 'B' for c in coverage])
    policy_seconds = time.monotonic()-phase
    control = fmt.generation_control(config, selection['indices'], rois, len(base))
    cooperation.validate(control, inner)
    alpha = cooperation.weights(base.shape, control)
    if selection['indices']:
        hashes = identities(args.adapter)
        if any(config[k] != v for k, v in hashes.items()):
            raise ValueError('G weights/profile mismatch')
        generated, runtime = restore(enhanced, control, args.adapter, [])
        output = cooperation.combine(enhanced, generated, alpha)
    else:
        output, runtime = enhanced.copy(), None
    np.testing.assert_array_equal(output[alpha == 0], enhanced[alpha == 0])
    report.update(total_bytes=len(data), generation_control_bytes=header,
        stream_sha256=file_hash(args.stream), source_frames_read=False, pid=os.getpid(),
        base_reference_unchanged=True, outside_generate_exact=True,
        generation_executed=bool(selection['indices']), generation_assets_validated=bool(selection['indices']),
        receiver_router_used=use_g, sender_router_loaded=False, unreceived_candidates_read=False,
        explicit_E_mask_bytes=0, explicit_G_map_bytes=0, protection_mask_bytes=0,
        semantic_heads_used=False, profile=fmt.PROFILE, route=selection, config=config,
        generation_runtime=runtime, generation_input_hash=frame_hash(enhanced), output_hash=frame_hash(output),
        seconds=time.monotonic()-began, codec_seconds=codec_seconds, policy_seconds=policy_seconds,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    if sum(report[k] for k in ('base_bytes', 'container_header_bytes', 'packet_bytes',
                              'incomplete_tail_bytes', 'generation_control_bytes')) != len(data):
        raise RuntimeError('asymmetric receiver byte accounting mismatch')
    atomic_npz(args.output/'reconstruction.npz', reconstruction=output, base=base, enhanced=enhanced)
    return report


def receive(args):
    """Worker API; collect all memory segments despite per-ROI G resets."""
    maxima, reserved = [], []
    original_reset = torch.cuda.reset_peak_memory_stats
    def measured_reset(device=None):
        maxima.append(torch.cuda.max_memory_allocated(device))
        reserved.append(torch.cuda.max_memory_reserved(device))
        original_reset(device)
    with patch.object(torch.cuda, 'reset_peak_memory_stats', measured_reset):
        report = _receive(args)
    maxima.append(torch.cuda.max_memory_allocated())
    reserved.append(torch.cuda.max_memory_reserved())
    runtime = report['generation_runtime'] or {}
    report.update(last_call_cuda_peak_bytes=report['peak_cuda_allocated_bytes'],
        peak_cuda_allocated_bytes=max(maxima), peak_cuda_reserved_bytes=max(reserved),
        memory_scope='whole fresh worker; accumulated across per-call CUDA counter resets',
        generation_all_calls_peak_cuda_bytes=max(
            (w['runtime']['peak_cuda_allocated_bytes'] for w in runtime.get('windows', [])), default=0),
        memory_counter_segments=len(maxima))
    atomic_json(args.output/'decode.json', report)
    return report


def decode(args, run):
    config, _, inner, _ = fmt.parse(args.stream.read_bytes(), allow_incomplete_tail=args.allow_incomplete_tail)
    use_g = not args.disable_generation and config['max_g'] > 0 and config['blend'] > 0
    if use_g and args.router is None:
        raise ValueError('--router (Rg) is required unless generation is disabled')
    validate_generation_geometry((inner.meta['frame_count'], inner.meta['height'], inner.meta['width'], 3),
                                 config['max_g'] if use_g else 0)
    assets = dict(receiver_router=file_hash(args.router) if use_g else None,
                  adapter=file_hash(args.adapter) if use_g else None,
                  enhancement=file_hash(args.enhancement) if inner.packets else None)
    binding = dict(profile=fmt.PROFILE, stream=file_hash(args.stream), config=config, used_assets=assets,
        disable_generation=args.disable_generation, allow_incomplete_tail=args.allow_incomplete_tail,
        code=fmt.code_identity())
    done = args.output/'decode.complete.json'
    if done.exists():
        return verify_complete(args.output, read(done), binding)
    immutable(args.output/'decode.request.json', binding)
    run.check()
    fresh = args.output/'fresh'
    fresh.mkdir(exist_ok=True)
    if not (fresh/'decode.json').exists():
        argv = ['--worker', '--stream', args.stream.resolve(), '--output', fresh.resolve(),
                '--adapter', args.adapter, '--enhancement', args.enhancement]
        if args.router is not None:
            argv += ['--router', args.router]
        if args.disable_generation:
            argv += ['--disable-generation']
        if args.allow_incomplete_tail:
            argv += ['--allow-incomplete-tail']
        before = os.environ.get('ROUTERVC_RECEIVER_PARENT')
        os.environ['ROUTERVC_RECEIVER_PARENT'] = str(os.getpid())
        try:
            execute(run, 'receiver_fresh_decode', Path(__file__).name, argv, distributed=True)
        finally:
            if before is None:
                os.environ.pop('ROUTERVC_RECEIVER_PARENT', None)
            else:
                os.environ['ROUTERVC_RECEIVER_PARENT'] = before
    report = validate_decoded(fresh, args.stream, config)
    if (report.get('profile') != fmt.PROFILE or report.get('semantic_heads_used') is not False
            or any(report.get(k) != 0 for k in ('explicit_E_mask_bytes', 'explicit_G_map_bytes', 'protection_mask_bytes'))
            or report.get('receiver_router_used') != use_g
            or report.get('sender_router_loaded') is not False or report.get('unreceived_candidates_read') is not False
            or (not use_g and report['generation_executed'])):
        raise RuntimeError('saved receiver used a different asymmetric execution mode')
    result = dict(complete=True, binding=binding, report=report,
        artifacts={name: file_hash(args.output/name) for name in
                   ('decode.request.json', 'fresh/decode.json', 'fresh/reconstruction.npz')})
    atomic_json(done, result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stream', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--router', type=Path, help='receiver-only Rg checkpoint; no sender Rs is accepted')
    p.add_argument('--enhancement', type=Path, default=DEFAULT_ENHANCEMENT)
    p.add_argument('--adapter', type=Path, default=DEFAULT_ADAPTER)
    p.add_argument('--disable-generation', action='store_true')
    p.add_argument('--allow-incomplete-tail', action='store_true')
    p.add_argument('--max-hours', type=float, default=24.)
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    args.command = 'receiver_decode'
    if args.worker:
        if not os.environ.get('TMUX') or not os.environ.get('ROUTERVC_RECEIVER_PARENT'):
            raise RuntimeError('fresh worker requires the supervising tmux parent')
        try:
            return receive(args)
        finally:
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
    return fmt.supervised(args, decode)


if __name__ == '__main__':
    main()
