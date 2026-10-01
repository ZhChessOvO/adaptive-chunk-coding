"""Paired real-stream evaluation after interface/LoRA coadaptation.

The 58 new decodes use immutable E packets and the pinned internal receiver.
No-E equivalence is with each model's OWN LoRA, not a different trained arm.
"""
import time

import numpy as np

from demo.chunk_enhancement_experiment import read
from demo.joint_condition_pipeline import audit
from demo.internal_condition_train import REPO
from demo.internal_condition_decode import identities
from demo.internal_condition_pipeline import variant, decode_point, assert_noise, OLD, MODES
from demo.feature_condition_report import CLIPS
from demo.patch_prefix_probe import load_frames, verify_artifacts
from demo.scalable_codec import atomic_json, file_hash
from demo.scalable_format import frame_hash
from demo.scalable_experiment import load_source, quality, resources
from demo.scalable_cooperation_experiment import region_metrics
from demo.stage_c_three_path_roi_probe import LPIPSAlex

ARMS = ('rgb', 'internal', 'zero')
HISTORY = OLD.parent / 'a800_internal_condition_20260930'


def cases(first=False):
    result = [(k, m) for m in MODES for k in ARMS]
    result += [(k, 'full') for k in ('off', 'without', 'shuffled')]
    result += [(k, 'none') for k in ('off', 'zero_off')]
    if first:
        result += [('repeat', 'full'), ('G_off', 'full')]
    return result


def exact_reference(candidate, mode):
    if candidate == 'repeat':
        return 'internal', 'full'
    if mode == 'none' and candidate in ('off', 'zero_off'):
        return ('internal' if candidate == 'off' else 'zero'), 'none'
    return None


def grouped(points):
    keys = [(r['sample_id'], r['mode'], r['candidate']) for r in points]
    expected = {(sid, m, k) for i, sid in enumerate(CLIPS) for k, m in cases(i == 0)}
    if len(keys) != 58 or len(keys) != len(expected) or set(keys) != expected:
        raise ValueError('incomplete or duplicated 58-point joint evaluation')
    lookup = dict(zip(keys, points))
    for sid in CLIPS:
        for mode in MODES:
            assert len({r['bytes'] for r in points if r['sample_id'] == sid and r['mode'] == mode}) == 1
    return lookup


def evaluate(output, run):
    for f, h in read(output/'queue_protocol.json')['code'].items():
        assert file_hash(REPO/'demo'/f) == h, f'Training source changed: {f}'
    training = audit(output)
    root = output/'evaluation'
    root.mkdir(exist_ok=True)
    refs = read(OLD/'summary.json')['results']
    assert [r['sample']['sample_id'] for r in refs] == list(CLIPS)
    adapters = {k:output/k/'adapter.pt' for k in ARMS}
    for name, source, mode in [('off','internal','off'), ('without','internal','zero'),
                               ('shuffled','internal','shuffle'), ('zero_off','zero','off')]:
        adapters[name] = variant(adapters[source], root/'models'/f'{name}.pt', mode)
    protocol = dict(code={f:file_hash(REPO/'demo'/f) for f in
        ('joint_condition_evaluate.py','joint_condition_evaluation_queue.py','run_joint_condition_evaluate.sh')},
        profiles={k:identities(v) for k,v in adapters.items()},
        references=file_hash(OLD/'summary.json'), training_audit=file_hash(output/'training_audit.json'),
        historical_summary=file_hash(HISTORY/'evaluation/summary.json'),
        roles='four reused development clips; not independent generalization',
        points='36 main + 12 full same-weight ablations + 8 own-LoRA no-E + repeat/G-off = 58',
        historical='12 initial RGB points reused, hash checked, NOT counted as new decodes')
    path = root/'protocol.json'
    if path.exists():
        assert read(path) == protocol, 'Pinned evaluation changed'
    else:
        atomic_json(path, protocol)
    results, metric = [], None
    for index, row in enumerate(refs):
        sid = row['sample']['sample_id']
        source = load_source(row['sample'])
        assert frame_hash(source) == row['source_hash']
        reports, paths = {}, {}
        for candidate, mode in cases(index == 0):
            run.check()
            key = 'internal' if candidate in ('repeat','G_off') else candidate
            dest, d = decode_point(root, row, candidate, mode, adapters[key], run, candidate == 'G_off')
            reports[candidate,mode], paths[candidate,mode] = d, dest
            pixels = load_frames(dest/'reconstruction.npz')
            if candidate != 'G_off':
                assert_noise(reports['rgb',mode], d)
            reference = exact_reference(candidate, mode)
            if reference:
                np.testing.assert_array_equal(pixels, load_frames(paths[reference]/'reconstruction.npz'))
            result_path = dest/'result.json'
            if result_path.exists():
                result = read(result_path)
                verify_artifacts(dest, result['artifacts'])
                assert result['fresh_decode'] == d
                assert (result['sample_id'],result['candidate'],result['mode']) == (sid,candidate,mode)
                assert result['direct'] == row['points'][MODES[mode][1]]
            else:
                if metric is None:
                    metric = LPIPSAlex(True)
                per_region, roi = region_metrics(source, pixels, row['metric_regions'], metric)
                result = dict(sample_id=sid, dataset=row['sample']['dataset'], candidate=candidate, mode=mode,
                    bytes=d['total_bytes'], quality=quality(source,pixels,metric),
                    per_region=per_region, roi_quality=roi, fresh_decode=d,
                    direct=row['points'][MODES[mode][1]],
                    artifacts={p.name:file_hash(p) for p in (dest/'stream.acsg',dest/'decode.json',dest/'reconstruction.npz')})
                atomic_json(result_path, result)
            results.append(result)
            run.update(phase='evaluation', completed=len(results), total=58)
    grouped(results)
    summary = dict(complete=True, results=results, training=training, protocol=protocol,
        paired_noise=True, no_E_own_LoRA_exact=True, repeat_exact=True, G_off_without_weights_exact=True)
    path = root/'summary.json'
    if path.exists():
        saved = read(path)
        for k,v in summary.items():
            assert saved[k] == v, f'Completed summary changed: {k}'
    else:
        atomic_json(path, dict(summary, elapsed_seconds=time.monotonic()-run.started, resources=resources()))
