"""Fresh source-free visual Router receiver, with an asset-independent G-off path.

The CLI parent owns the GPU mutex and launches one fresh child. CPU contracts
alone are not GPU validation; no new model is automatically selected/deployed.
"""
import argparse
import gc
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from demo import routervc_visual_format as fmt
from demo import routervc_visual_policy as policy
from demo import scalable_cooperation_format as cooperation
from demo.routervc import (DEFAULT_ADAPTER, immutable, read, verify_complete,
                          validate_decoded, validate_generation_geometry)
from demo.routervc_encode import DEFAULT_ENHANCEMENT
from demo.routervc_decode import coverage_from_packets
from demo.chunk_enhancement_codec import configure_torch, load_model, decode_enhancement
from demo.scalable_codec import BaseCodec, atomic_json, atomic_npz, file_hash
from demo.scalable_format import frame_hash
from demo.online_eg_decode import identities
from demo.internal_condition_decode import restore
from demo.conditioned_generation_pipeline import execute


def receive(args):
    """Worker API: stream and shared weights only; no source/candidate input."""
    began = time.monotonic(); data = args.stream.read_bytes()
    config, inner_bytes, inner, header = fmt.parse(data, allow_incomplete_tail=args.allow_incomplete_tail)
    use_g = not args.disable_generation and config['max_g'] > 0 and config['blend'] > 0
    shape = (inner.meta['frame_count'], inner.meta['height'], inner.meta['width'], 3)
    rois = validate_generation_geometry(shape, config['max_g'] if use_g else 0)
    coverage = coverage_from_packets(inner, rois)
    if use_g and (args.router is None or file_hash(args.router) != config['router']):
        raise ValueError('visual Router weights missing or different')
    configure_torch(); torch.cuda.reset_peak_memory_stats()
    codec = BaseCodec(REPO/'checkpoints/cvpr2026_image.pth.tar', REPO/'checkpoints/cvpr2026_video_hts.pth.tar')
    if any(inner.meta[k] != v for k,v in codec.models.items()):
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
    del codec; gc.collect(); torch.cuda.empty_cache()
    codec_seconds = time.monotonic()-began; phase = time.monotonic()
    if use_g:
        selection = policy.route(base, enhanced, inner, config, args.router, expected_policy=fmt.policy_identity())
    else:
        selection = dict(indices=[], rois=rois, coverage=coverage.tolist(), policy_skipped=True,
                         states=['E' if c > 0 else 'B' for c in coverage])
    policy_seconds = time.monotonic()-phase
    control = fmt.generation_control(config, selection['indices'], rois, len(base))
    cooperation.validate(control, inner); alpha = cooperation.weights(base.shape, control)
    if selection['indices']:
        hashes = identities(args.adapter)
        if any(config[k] != v for k,v in hashes.items()):
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
        shared_router_used=use_g, explicit_E_mask_bytes=0, explicit_G_map_bytes=0, protection_mask_bytes=0,
        semantic_heads_used=False, profile=fmt.PROFILE, route=selection, config=config,
        generation_runtime=runtime, generation_input_hash=frame_hash(enhanced), output_hash=frame_hash(output),
        seconds=time.monotonic()-began, codec_seconds=codec_seconds, policy_seconds=policy_seconds,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    if sum(report[k] for k in ('base_bytes','container_header_bytes','packet_bytes',
                              'incomplete_tail_bytes','generation_control_bytes')) != len(data):
        raise RuntimeError('visual receiver byte accounting mismatch')
    atomic_npz(args.output/'reconstruction.npz', reconstruction=output, base=base, enhanced=enhanced)
    atomic_json(args.output/'decode.json', report)
    return report


def decode(args, run):
    config, _, inner, _ = fmt.parse(args.stream.read_bytes(), allow_incomplete_tail=args.allow_incomplete_tail)
    use_g = not args.disable_generation and config['max_g'] > 0 and config['blend'] > 0
    if use_g and args.router is None:
        raise ValueError('--router is required unless generation is disabled')
    validate_generation_geometry((inner.meta['frame_count'],inner.meta['height'],inner.meta['width'],3),
                                 config['max_g'] if use_g else 0)
    assets = dict(router=file_hash(args.router) if use_g else None,
                  adapter=file_hash(args.adapter) if use_g else None,
                  enhancement=file_hash(args.enhancement) if inner.packets else None)
    binding = dict(profile=fmt.PROFILE, stream=file_hash(args.stream), config=config, used_assets=assets,
        disable_generation=args.disable_generation, allow_incomplete_tail=args.allow_incomplete_tail,
        code=fmt.code_identity())
    done = args.output/'decode.complete.json'
    if done.exists():
        return verify_complete(args.output, read(done), binding)
    immutable(args.output/'decode.request.json', binding)
    fresh = args.output/'fresh'; fresh.mkdir(exist_ok=True)
    if not (fresh/'decode.json').exists():
        argv = ['--worker', '--stream', args.stream.resolve(), '--output', fresh.resolve(),
                '--adapter', args.adapter, '--enhancement', args.enhancement]
        if args.router is not None:
            argv += ['--router', args.router]
        if args.disable_generation:
            argv += ['--disable-generation']
        if args.allow_incomplete_tail:
            argv += ['--allow-incomplete-tail']
        before = os.environ.get('ROUTERVC_VISUAL_PARENT')
        os.environ['ROUTERVC_VISUAL_PARENT'] = str(os.getpid())
        try:
            execute(run, 'visual_fresh_decode', Path(__file__).name, argv, distributed=True)
        finally:
            if before is None:
                os.environ.pop('ROUTERVC_VISUAL_PARENT', None)
            else:
                os.environ['ROUTERVC_VISUAL_PARENT'] = before
    report = validate_decoded(fresh, args.stream, config)
    if (report.get('profile') != fmt.PROFILE or report.get('semantic_heads_used') is not False
            or report.get('explicit_G_map_bytes') != 0 or report['shared_router_used'] != use_g
            or (not use_g and report['generation_executed'])):
        raise RuntimeError('saved receiver used a different visual execution mode')
    result = dict(complete=True, binding=binding, report=report,
        artifacts={name:file_hash(args.output/name) for name in
                   ('decode.request.json','fresh/decode.json','fresh/reconstruction.npz')})
    atomic_json(done, result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stream', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--router', type=Path, help='required only when the shared G policy executes')
    p.add_argument('--enhancement', type=Path, default=DEFAULT_ENHANCEMENT)
    p.add_argument('--adapter', type=Path, default=DEFAULT_ADAPTER)
    p.add_argument('--disable-generation', action='store_true'); p.add_argument('--allow-incomplete-tail', action='store_true')
    p.add_argument('--max-hours', type=float, default=24.); p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args(argv); args.command = 'visual_decode'
    if args.worker:
        if not os.environ.get('TMUX') or not os.environ.get('ROUTERVC_VISUAL_PARENT'):
            raise RuntimeError('fresh worker requires the supervising tmux parent')
        try:
            return receive(args)
        finally:
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
    return fmt.supervised(args, decode)


if __name__ == '__main__':
    main()
