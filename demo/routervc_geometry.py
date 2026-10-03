"""Full public RouterVC CLI checks: 33-frame REDS and rectangular UVG Jockey.

This coordinator holds NO GPU mutex. Each public CLI subprocess acquires the
single GPU lock itself, so it can queue behind an existing experiment safely.
Source frames are used by encoding and this evaluator, never by the decoder.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from demo import routervc_format as fmt
from demo.routervc import code_identity, immutable, read
from demo.routervc_encode import bank_info
from demo.scalable_codec import atomic_json, atomic_npz, file_hash
from demo.scalable_format import frame_hash
from demo.scalable_experiment import load_source, quality
from demo.chunk_enhancement_experiment import Run
from demo.conditioned_generation_pipeline import execute

RUNS = Path('/root/autodl-fs/DCVC/runs')
DEFAULT = RUNS/'routervc_geometry_20261003'
PLAN = RUNS/'a800_scalable_mechanism_20260926/plan.json'
SAMPLES = ('mechanism-02-reds', 'mechanism-03-uvg')


def verify_artifacts(root, artifacts):
    for name, digest in artifacts.items():
        if file_hash(root/name) != digest:
            raise RuntimeError(f'changed geometry artifact: {root/name}')


def protocol(root, plan):
    rows = {r['sample_id']: r for r in read(plan)['mechanism_samples']}
    selected = [rows[name] for name in SAMPLES]
    expected = [(33, 256, 384), (17, 256, 384)]
    for row, geometry in zip(selected, expected):
        if (row['frame_count'], row['crop']['height'], row['crop']['width']) != geometry:
            raise RuntimeError('historical mechanism geometry changed')
    if selected[1]['sequence'] != 'Jockey':
        raise RuntimeError('expected the previously used Jockey diagnostic')
    value = dict(version=1, historical_plan=str(plan.resolve()), plan_sha256=file_hash(plan),
        selected_samples=selected, code=dict(code_identity(),
            geometry=file_hash(Path(__file__)), wrapper=file_hash(REPO/'demo/run_routervc_geometry.sh')),
        source_file_sha256={p: file_hash(Path(p)) for row in selected for p in row['source_files']},
        role='previously used REDS/UVG mechanism development; not independent evaluation',
        main=dict(e_ratio=.5, max_g=4, mode='prefix', boundary_lambda=.004),
        extra_prefix_ratios=[.25, .75], holds_parent_GPU_mutex=False)
    immutable(root/'protocol.json', value)
    return value


def prepare_source(root, sample, protocol_hash):
    path = root/'source.complete.json'
    if path.exists():
        result = read(path)
        if result['protocol_sha256'] != protocol_hash:
            raise RuntimeError('geometry source protocol changed')
        verify_artifacts(root, result['artifacts'])
        return result
    source = load_source(sample)
    atomic_npz(root/'source.npz', source=source)
    result = dict(sample_id=sample['sample_id'], protocol_sha256=protocol_hash,
        shape=list(source.shape), source_rgb_sha256=frame_hash(source),
        source_sample=sample, artifacts={'source.npz': file_hash(root/'source.npz')})
    atomic_json(path, result)
    return result


def encode_args(folder, ratio=.5):
    output = folder/('encode' if ratio == .5 else f'prefix_r{ratio:g}')
    args = ['encode', '--input', folder/'source.npz', '--output', output,
            '--e-ratio', ratio, '--max-g', 4, '--mode', 'prefix', '--boundary-lambda', .004]
    if ratio != .5:
        args += ['--prepared-dir', folder/'encode/prepared']
    return args, output


def validate(folder):
    sender = read(folder/'encode/encode.json')
    done = read(folder/'decode/decode.complete.json')
    decoded = read(folder/'decode/fresh/decode.json')
    if done['report'] != decoded:
        raise RuntimeError('decoder CLI report differs from fresh receiver')
    for root, record in ((folder/'encode', sender), (folder/'decode', done)):
        verify_artifacts(root, record['artifacts'])
    if (sender['expected_base_hash'] != decoded['base_hash']
            or sender['expected_mixed_hash'] != decoded['generation_input_hash']
            or sender['expected_shared_route'] != decoded['route']
            or decoded['total_bytes'] != sender['actual_on_disk_bytes']
            or decoded['source_frames_read'] or not decoded['base_reference_unchanged']
            or not decoded['outside_generate_exact']):
        raise RuntimeError('sender / independent receiver contract failed')
    bank = (folder/'encode/prepared/bank.acse').read_bytes()
    info = bank_info(bank)
    count = info['parsed'].meta['frame_count']
    expected_packets = 16*(1+(count-1)//8)
    if len(info['parsed'].packets) != expected_packets:
        raise RuntimeError('candidate bank does not cover every frame/cell')
    _, _, selected, _ = fmt.parse((folder/'encode/stream.rtvc').read_bytes())
    for index in sender['plan']['selected_indices']:
        packets = [p for p in selected.packets if p.meta['roi'] == info['rois'][index]]
        if [(p.meta['start'], p.meta['count']) for p in packets] != info['chunks']:
            raise RuntimeError('selected E region is not a full temporal bundle')
    with np.load(folder/'decode/fresh/reconstruction.npz', allow_pickle=False) as cache:
        for field, digest in [('base', decoded['base_hash']), ('enhanced', decoded['generation_input_hash']),
                              ('reconstruction', decoded['output_hash'])]:
            if frame_hash(cache[field]) != digest:
                raise RuntimeError('saved fresh pixels changed')
    return dict(sender=sender, decoded=decoded, candidate_packets=expected_packets,
                selected_packets=len(selected.packets), frame_count=count,
                complete_temporal_E_bundles=True, sender_receiver_route_equal=True,
                source_free_fresh_decode=True)


def snapshot(folder):
    names = ('encode/encode.json', 'encode/stream.rtvc', 'encode/prepared/complete.json',
        'encode/prepared/base.complete.json', 'decode/decode.complete.json',
        'decode/fresh/decode.json', 'decode/fresh/reconstruction.npz')
    return {name: dict(sha256=file_hash(folder/name), mtime_ns=(folder/name).stat().st_mtime_ns)
            for name in names}


def sample_run(run, sample, protocol_hash):
    sid = sample['sample_id']
    folder = run.root/sid
    folder.mkdir(parents=True, exist_ok=True)
    prepare_source(folder, sample, protocol_hash)
    result_path = folder/'result.json'
    if result_path.exists():
        old = read(result_path)
        if old['protocol_sha256'] != protocol_hash:
            raise RuntimeError('geometry result protocol mismatch')
        verify_artifacts(folder, old['artifacts'])
        validate(folder)
        return old
    enc_args, _ = encode_args(folder)
    dec_args = ['decode', '--stream', folder/'encode/stream.rtvc', '--output', folder/'decode']
    execute(run, f'{sid}_encode', 'routervc.py', enc_args)
    execute(run, f'{sid}_decode', 'routervc.py', dec_args)
    checks = validate(folder)
    resume_path = folder/'resume_audit.json'
    if not resume_path.exists():
        before = snapshot(folder)
        before_encode = read(folder/'encode/encode.json')['timing']
        before_decode = read(folder/'decode/fresh/decode.json')['seconds']
        execute(run, f'{sid}_encode_reentry', 'routervc.py', enc_args)
        execute(run, f'{sid}_decode_reentry', 'routervc.py', dec_args)
        if snapshot(folder) != before:
            raise RuntimeError('completed CLI reentry rewrote an original artifact or mtime')
        if (read(folder/'encode/encode.json')['timing'] != before_encode
                or read(folder/'decode/fresh/decode.json')['seconds'] != before_decode):
            raise RuntimeError('completed CLI reentry changed original elapsed times')
        atomic_json(resume_path, dict(complete=True, preserved=before,
            encode_timing=before_encode, decode_seconds=before_decode,
            no_artifact_or_mtime_change=True))
    prefix_check = None
    if sid == SAMPLES[0]:
        wires = {.5: (folder/'encode/stream.rtvc').read_bytes()}
        for ratio in (.25, .75):
            args, destination = encode_args(folder, ratio)
            execute(run, f'{sid}_prefix_{ratio:g}', 'routervc.py', args)
            wires[ratio] = (destination/'stream.rtvc').read_bytes()
            if not read(destination/'encode.json')['candidate_cache_reused']:
                raise RuntimeError('prefix check unexpectedly re-encoded candidate bank')
        if not wires[.5].startswith(wires[.25]) or not wires[.75].startswith(wires[.5]):
            raise RuntimeError('full RouterVC streams are not literal nested byte prefixes')
        prefix_check = dict(ratios=[.25, .5, .75], literal_byte_prefix=True,
                            total_bytes={str(k): len(v) for k, v in sorted(wires.items())},
                            additional_prefixes_decoded=False, candidate_bank_reused=True)
        atomic_json(folder/'prefix_audit.json', prefix_check)
    run.check()
    run.update(phase=f'{sid}_CPU_metrics_source_only_evaluator')
    with np.load(folder/'source.npz', allow_pickle=False) as cache:
        source = cache['source'].copy()
    from demo.stage_c_three_path_roi_probe import LPIPSAlex
    torch.set_num_threads(4)
    metric = LPIPSAlex(True)
    with np.load(folder/'decode/fresh/reconstruction.npz', allow_pickle=False) as cache:
        metrics = {key: quality(source, cache[field], metric) for key, field in
                   [('base', 'base'), ('E_only', 'enhanced'), ('RouterVC', 'reconstruction')]}
    artifacts = ('source.complete.json', 'source.npz', 'encode/encode.json', 'encode/stream.rtvc',
                 'decode/decode.complete.json', 'decode/fresh/decode.json',
                 'decode/fresh/reconstruction.npz', 'resume_audit.json')
    if prefix_check:
        artifacts += ('prefix_audit.json', 'prefix_r0.25/stream.rtvc', 'prefix_r0.75/stream.rtvc',
                      'prefix_r0.25/encode.json', 'prefix_r0.75/encode.json')
    result = dict(complete=True, protocol_sha256=protocol_hash, sample_id=sid,
        dataset=sample['dataset'], sequence=sample['sequence'], checks=checks,
        quality=metrics, prefix_check=prefix_check, role='existing mechanism development diagnostic',
        limitation='geometry/mechanism evidence; not a calibrated long-video Router quality claim',
        artifacts={name: file_hash(folder/name) for name in artifacts})
    atomic_json(result_path, result)
    return result


def main(args):
    if not os.environ.get('TMUX'):
        raise RuntimeError('run the geometry queue in tmux')
    args.command = 'geometry'
    run = Run(args)
    run.thread.start()
    began = time.monotonic()
    try:
        # Deliberately no exclusive_native_evaluation here: child CLI owns it.
        spec = protocol(run.root, args.plan)
        digest = file_hash(run.root/'protocol.json')
        results = []
        for sample in spec['selected_samples']:
            run.check()
            results.append(sample_run(run, sample, digest))
            run.update(phase='geometry_complete', completed=len(results), total=2)
        immutable(run.root/'summary.json', dict(complete=True, results=results,
            source_free_receivers=True, preserves_existing_pins=True, real_source_end_to_end=True))
        if not (run.root/'complete.json').exists():
            atomic_json(run.root/'complete.json', dict(complete=True, samples=2,
                elapsed_seconds=time.monotonic()-began, protocol_sha256=digest,
                summary_sha256=file_hash(run.root/'summary.json'),
                includes_wait_for_child_GPU_mutex=True))
    except BaseException as error:
        atomic_json(run.root/'last_failure.json', dict(error=repr(error), progress=run.progress))
        raise
    finally:
        run.log_resources()
        run.stop.set()
        run.thread.join(timeout=3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--plan', type=Path, default=PLAN)
    parser.add_argument('--max-hours', type=float, default=12.)
    main(parser.parse_args())
