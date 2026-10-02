"""Resumable, exclusive real-byte evaluation; no training or source-fed receiver."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from demo import scalable_cooperation_format as fmt
from demo.chunk_enhancement_experiment import Run, read
from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
from demo.conditioned_generation_pipeline import execute
from demo.online_eg_decode import identities
from demo.online_eg_eval_core import (DEFAULT, INITIAL, HISTORY, OLD, MODES, ARMS, CASES,
    CLIPS, CONTROLS, training_check, noise_pair)
from demo.patch_prefix_probe import load_frames, verify_artifacts
from demo.scalable_codec import atomic_bytes, atomic_json, file_hash
from demo.scalable_format import frame_hash, parse
from demo.scalable_experiment import load_source, quality, resources
from demo.scalable_cooperation_experiment import region_metrics
from demo.stage_c_three_path_roi_probe import LPIPSAlex

CODE = ('online_eg_eval_core.py', 'online_eg_evaluate.py', 'online_eg_decode.py',
        'online_eg_report.py', 'run_online_eg_evaluate.sh')


def jobs(first=False):
    values = []
    for case in CASES:
        for arm in ARMS:
            for kind in ('E', 'G'):
                values.append(dict(name=f'{arm}_{kind}_{case}', case=case, e_arm=arm,
                                   g_arm=arm if kind == 'G' else None, kind=kind))
    values += [dict(name='cross_oldE_newG', case='full', e_arm='fixed', g_arm='joint', kind='G'),
               dict(name='cross_newE_fixedG', case='full', e_arm='joint', g_arm='fixed', kind='G')]
    if first:
        values += [dict(name='repeat_full', case='full', e_arm='joint', g_arm='joint', kind='G'),
                   dict(name='repeat_none', case='none', e_arm='joint', g_arm='joint', kind='G'),
                   dict(name='G_off', case='full', e_arm='joint', g_arm='joint', kind='Goff'),
                   dict(name='initial_receiver', case='full', e_arm='fixed', g_arm='initial', kind='G')]
    return values


def grouped(results):
    expected = {(sid, j['name']) for i, sid in enumerate(CLIPS) for j in jobs(i == 0)}
    keys = [(r['sample_id'], r['name']) for r in results]
    if len(keys) != 92 or len(set(keys)) != 92 or set(keys) != expected:
        raise ValueError('missing or duplicated 92-point evaluation')
    return dict(zip(keys, results, strict=True))


def reuse_initial():
    summary = read(HISTORY/'summary.json')
    assert summary['complete']
    for f, digest in summary['protocol']['code'].items():
        assert file_hash(REPO/'demo'/f) == digest
    records = []
    for r in summary['results']:
        if r['candidate'] != 'rgb':
            continue
        folder = HISTORY/r['sample_id']/f"rgb_{r['mode']}"
        assert read(folder/'result.json') == r
        verify_artifacts(folder, r['artifacts'])
        d = read(folder/'decode.json')
        assert d == r['fresh_decode'] and not d['source_frames_read']
        assert d['assets']['lora'] == file_hash(INITIAL)
        assert d['total_bytes'] == r['bytes'] == (folder/'stream.acsg').stat().st_size
        assert frame_hash(load_frames(folder/'reconstruction.npz')) == d['output_hash']
        records.append(dict(sample_id=r['sample_id'], dataset=r['dataset'], case=r['mode'],
            bytes=r['bytes'], quality=r['quality'], roi_quality=r['roi_quality'],
            fresh_decode=d, output_path=str(folder/'reconstruction.npz'), reused=True))
    assert {(r['sample_id'], r['case']) for r in records} == {(s, m) for s in CLIPS for m in MODES}
    return records


def validate_point(folder, r, job, wire, expected, base_hash, profile):
    verify_artifacts(folder, r['artifacts'])
    assert r['job'] == job and (folder/r['stream_file']).read_bytes() == wire
    d = read(folder/'decode.json')
    assert d == r['fresh_decode'] and not d['source_frames_read']
    assert d['total_bytes'] == r['bytes'] == len(wire)
    assert d['base_hash'] == base_hash and d['non_enhanced_exact']
    pixels = load_frames(folder/'reconstruction.npz')
    assert frame_hash(pixels) == d['output_hash']
    total = sum(d[k] for k in ('base_bytes', 'container_header_bytes', 'packet_bytes', 'incomplete_tail_bytes'))
    if job['kind'] == 'E':
        np.testing.assert_array_equal(pixels, expected)
    else:
        control, _, _, overhead = fmt.parse(wire)
        assert d['generation_control_bytes'] == overhead
        total += overhead
        assert d['generation_input_hash'] == frame_hash(expected)
        assert d['base_reference_unchanged'] and d['outside_generate_exact']
        alpha = fmt.weights(pixels.shape, control)
        np.testing.assert_array_equal(pixels[alpha == 0], expected[alpha == 0])
        if job['kind'] == 'Goff':
            np.testing.assert_array_equal(pixels, expected)
            assert not d['generation_assets_validated'] and not d['generation_executed']
        else:
            assert d['assets'] == profile and d['generation_executed']
    assert total == len(wire)
    return pixels


def point(root, row, job, profiles, adapters, run, metric):
    sid = row['sample']['sample_id']
    prepared = root/'evaluation'/sid/'prepared'/job['e_arm']
    prep = read(prepared/'complete.json')
    verify_artifacts(prepared, prep['artifacts'])
    inner = (prepared/f"{job['case']}.acse").read_bytes()
    expected = load_frames(prepared/f"{job['case']}_expected.npz")
    assert frame_hash(expected) == prep['records'][job['case']]['output_hash']
    profile = profiles[job['g_arm']] if job['g_arm'] else None
    if job['kind'] == 'E':
        wire, filename = inner, 'stream.acse'
    else:
        assert file_hash(OLD/sid/'cooperate_l05.acsg') == row['points']['cooperate_l05']['stream_sha256']
        control, _, _, _ = fmt.parse((OLD/sid/'cooperate_l05.acsg').read_bytes())
        control.update(profile)
        control['strength'] = 1.
        wire, filename = fmt.wrap(inner, control), 'stream.acsg'
    folder = root/'evaluation'/sid/job['name']
    folder.mkdir(parents=True, exist_ok=True)
    complete = folder/'result.json'
    if complete.exists():
        r = read(complete)
        validate_point(folder, r, job, wire, expected, prep['base_hash'], profile)
        return r
    path = folder/filename
    if path.exists():
        assert path.read_bytes() == wire, 'incomplete point has changed configuration'
    atomic_bytes(path, wire)
    weights = root/job['e_arm']/'enhancement.pt'
    begin = time.monotonic()
    if job['kind'] == 'E':
        execute(run, f"{sid}_{job['name']}", 'chunk_enhancement_experiment.py',
                ['decode', '--stream', path, '--checkpoint', weights, '--output', folder])
    else:
        disabled = job['kind'] == 'Goff'
        execute(run, f"{sid}_{job['name']}", 'online_eg_decode.py',
            ['--stream', path, '--output', folder, '--enhancement', weights, '--adapter',
             '/nonexistent/generator.pt' if disabled else adapters[job['g_arm']],
             *(['--disable-generation'] if disabled else [])], distributed=True)
    elapsed = time.monotonic()-begin
    d = read(folder/'decode.json')
    pixels = load_frames(folder/'reconstruction.npz')
    source = load_source(row['sample'])
    assert frame_hash(source) == row['source_hash']
    per_region, roi = region_metrics(source, pixels, row['metric_regions'], metric())
    r = dict(sample_id=sid, dataset=row['sample']['dataset'], name=job['name'], job=job,
        bytes=len(wire), bpp=8*len(wire)/np.prod(source.shape[:3]),
        quality=quality(source, pixels, metric()), roi_quality=roi, per_region=per_region,
        fresh_decode=d, process_wall_seconds=elapsed, output_path=str(folder/'reconstruction.npz'),
        stream_file=filename, artifacts={p.name:file_hash(p) for p in
            (path, folder/'decode.json', folder/'reconstruction.npz')})
    validate_point(folder, r, job, wire, expected, prep['base_hash'], profile)
    atomic_json(complete, r)
    return r


def compare_checks(row, current, initial):
    sid = row['sample']['sample_id']
    reference = current['fixed_G_none']['fresh_decode']
    for name, r in current.items():
        if r['job']['kind'] == 'G':
            noise_pair(reference, r['fresh_decode'])
    for case in CASES:
        a, b = current[f'fixed_G_{case}'], current[f'joint_G_{case}']
        noise_pair(a['fresh_decode'], b['fresh_decode'], same_condition=case == 'none')
    for cross, own in [('cross_oldE_newG', 'fixed_G_full'), ('cross_newE_fixedG', 'joint_G_full')]:
        assert current[cross]['bytes'] == current[own]['bytes']
        noise_pair(current[cross]['fresh_decode'], current[own]['fresh_decode'], same_condition=True)
    for case in MODES:
        old = next(r for r in initial if r['sample_id'] == sid and r['case'] == case)
        now = current[f'fixed_G_{case}']
        assert now['bytes'] == old['bytes']
        noise_pair(now['fresh_decode'], old['fresh_decode'], same_condition=True)
    if 'repeat_full' in current:
        for test, ref in [('repeat_full', 'joint_G_full'), ('repeat_none', 'joint_G_none'), ('G_off', 'joint_E_full')]:
            np.testing.assert_array_equal(load_frames(Path(current[test]['output_path'])),
                                          load_frames(Path(current[ref]['output_path'])))
        old = next(r for r in initial if r['sample_id'] == sid and r['case'] == 'full')
        np.testing.assert_array_equal(load_frames(Path(current['initial_receiver']['output_path'])),
                                      load_frames(Path(old['output_path'])))
    folder = Path(current['fixed_E_full']['output_path']).parents[1]
    for arm in ARMS:
        for kind, ext in [('E', 'acse'), ('G', 'acsg')]:
            wires = [(folder/f'{arm}_{kind}_{m}'/f'stream.{ext}').read_bytes() for m in MODES]
            assert wires[2].startswith(wires[1]) and wires[1].startswith(wires[0])


def evaluate(root, run):
    training = training_check(root)
    initial = reuse_initial()
    adapters = {k:root/k/'adapter.pt' for k in ARMS}
    adapters['initial'] = INITIAL
    profiles = {k:identities(v) for k, v in adapters.items()}
    protocol = dict(code={f:file_hash(REPO/'demo'/f) for f in CODE},
        models={k:{n:file_hash(root/k/f'{n}.pt') for n in ('adapter','enhancement')} for k in ARMS},
        profiles=profiles, references=file_hash(OLD/'summary.json'), history=file_hash(HISTORY/'summary.json'),
        training_audit=file_hash(root/'training_audit.json'), cases=list(CASES),
        samples=[r['sample'] for r in read(OLD/'summary.json')['results']],
        roles='four reused REDS/UVG development clips; not independent generalization',
        design='40 E-only + 40 G + 8 full-q1 crossed E/G + 4 checks = 92 fresh decodes',
        rate='actual complete files; different E can change bytes; no same-q equal-rate claim',
        roi_scope='same first 17 frames and fixed regions; all frames for whole-frame quality',
        no_new_training=True)
    dest = root/'evaluation'
    dest.mkdir(exist_ok=True)
    path = dest/'protocol.json'
    if path.exists():
        assert read(path) == protocol, 'evaluation configuration changed'
    else:
        atomic_json(path, protocol)
    atomic_json(dest/'training_check.json', training)
    metric_instance = []
    def metric():
        if not metric_instance:
            metric_instance.append(LPIPSAlex(True))
        return metric_instance[0]
    results = []
    for index, row in enumerate(read(OLD/'summary.json')['results']):
        sid = row['sample']['sample_id']
        assert sid == list(CLIPS)[index]
        for arm in ARMS:
            run.check()
            prepared = dest/sid/'prepared'/arm
            if (prepared/'complete.json').exists():
                r = read(prepared/'complete.json')
                verify_artifacts(prepared, r['artifacts'])
                assert r['identities']['model'] == protocol['models'][arm]['enhancement']
                assert r['identities']['code'] == protocol['code']['online_eg_eval_core.py']
                assert r['identities']['source'] == row['source_hash']
            else:
                execute(run, f'{sid}_{arm}_encode', 'online_eg_eval_core.py',
                        ['--root', root, '--sample', sid, '--arm', arm])
        current = {}
        for job in jobs(index == 0):
            run.check()
            r = point(root, row, job, profiles, adapters, run, metric)
            results.append(r)
            current[job['name']] = r
            run.update(phase='evaluating', completed=len(results), total=92, sample=sid, point=job['name'])
        compare_checks(row, current, initial)
    grouped(results)
    summary = dict(complete=True, results=results, initial_reused=initial, protocol=protocol,
                   paired_diffusion_noise=True, literal_prefix=True, repeat_exact=True,
                   initial_receiver_exact=True, g_off_without_weights_exact=True)
    path = dest/'summary.json'
    if path.exists():
        saved = read(path)
        assert all(saved[k] == v for k, v in summary.items()), 'completed summary changed'
    else:
        atomic_json(path, dict(summary, elapsed_seconds=time.monotonic()-run.evaluation_started,
                              gpu_wait_seconds=run.gpu_wait_seconds, resources=resources()))


def main(args):
    run = Run(args)
    run.thread.start()
    try:
        with exclusive_native_evaluation(run):
            if args.command != 'report':
                assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                    '--format=csv,noheader'], text=True).strip(), 'GPU must be idle'
                evaluate(args.output, run)
            if args.command != 'evaluate':
                from demo.online_eg_report import report
                report(args.output)
            run.update(phase='complete')
            path = run.root/f'{args.command}.complete.json'
            if not path.exists():
                atomic_json(path, dict(complete=True, elapsed_seconds=time.monotonic()-run.evaluation_started,
                                       gpu_wait_seconds=run.gpu_wait_seconds))
    except BaseException as e:
        atomic_json(run.root/'evaluation.last_failure.json', dict(error=repr(e), progress=run.progress))
        raise
    finally:
        run.log_resources()
        run.stop.set()
        run.thread.join(timeout=3)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['run','evaluate','report'])
    p.add_argument('--output', type=Path, default=DEFAULT)
    p.add_argument('--max-hours', type=float, default=8.)
    args = p.parse_args()
    if not os.environ.get('TMUX'):
        p.error('Run inside tmux')
    main(args)
