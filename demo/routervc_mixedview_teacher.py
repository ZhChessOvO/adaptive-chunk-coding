"""Mixed-view four-state teacher: REDS whole views + verified existing UVG crops.

No model updates, semantic annotations, deployment G mask, or additive mixed
video oracle is claimed. Diagnostic ACSG control bytes are recorded separately
from the eventual RTVC receiver-derived policy. Run the queue inside tmux.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
RUNS = Path('/root/autodl-fs/DCVC/runs')
OLD = RUNS/'a800_four_state_20261002'
MODELS = RUNS/'a800_online_eg_20261002/joint'
ENHANCEMENT, ADAPTER = MODELS/'enhancement.pt', MODELS/'adapter.pt'
SCHEMA = 'routervc-mixedview-four-state-v1'
STATES = ('B', 'E', 'G', 'EG')


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def immutable(path, value):
    if Path(path).exists():
        if read(path) != value:
            raise ValueError(f'changed teacher configuration: {path}; use a new output')
    else:
        save(path, value)


def verify(root, artifacts):
    for name, expected in artifacts.items():
        if digest(Path(root)/name) != expected:
            raise ValueError(f'changed teacher artifact: {Path(root)/name}')


def grid(height, width):
    if min(height, width) < 64 or height % 4 or width % 4:
        raise ValueError('teacher requires a regular 4x4 grid without implicit padding')
    return [[x, y, width//4, height//4]
            for y in range(0, height, height//4) for x in range(0, width, width//4)]


def control(profile, sample_id, index, rois):
    if len(rois) != 16 or index not in range(16):
        raise ValueError('invalid teacher region')
    seed = int(hashlib.sha256(f'four-state-v1/{sample_id}/{index}'.encode()).hexdigest()[:12], 16)
    return dict(profile, seed=seed, strength=1., window=17, stride=8, context=64,
                feather=16, processing_scale=1, blend=1., protect=[],
                generate=[[0, 17, *rois[index]]])


def crop(frames, roi):
    x, y, width, height = roi
    return frames[:, y:y+height, x:x+width].copy()


def isolate(base, enhanced, roi):
    out = base.copy(); x, y, width, height = roi
    out[:, y:y+height, x:x+width] = enhanced[:, y:y+height, x:x+width]
    return out


def aggregate_diagnostic_bytes(base, e_bytes, states, shared_g, roi_control):
    """Diagnostic ACSG costs only; do not multiply the shared base/G header."""
    if len(e_bytes) != len(states) or any(s not in STATES for s in states):
        raise ValueError('invalid four-state cost layout')
    count = sum(s in ('G', 'EG') for s in states)
    return base + sum(n for n, s in zip(e_bytes, states) if s in ('E', 'EG')) + (
        shared_g + count*roi_control if count else 0)


def selected_views(data, smoke):
    completed = read(Path(data)/'complete.json')
    if not completed.get('complete'):
        raise ValueError('mixed-view data preparation is incomplete')
    values = []
    for entry in completed['samples']:
        path = Path(entry['view_json'])
        if digest(path) != entry['view_sha256']:
            raise ValueError('prepared mixed-view record changed')
        value = read(path)
        if not value.get('complete') or value['sample']['sample_id'] != entry['sample_id']:
            raise ValueError('prepared sample identity/completion mismatch')
        values.append(dict(view_json=str(path.resolve()), view_sha256=digest(path), **value))
    reds = [v for v in values if v['sample']['dataset'] == 'REDS']
    uvg = [v for v in values if v['sample']['dataset'] == 'UVG']
    if smoke:
        if min(len(reds), len(uvg)) < 2:
            raise ValueError('smoke needs two REDS whole views and two UVG crops')
        values = [reds[0], uvg[0], reds[1], uvg[1]]
    else:
        if (len(reds), len(uvg)) != (90, 30):
            raise ValueError('formal pilot120 requires exactly 90 REDS + 30 UVG views')
        if Counter(v['sample']['router_split'] for v in values) != dict(train=96, validation=24):
            raise ValueError('formal teacher must preserve the old 96/24 grouped Router split')
        values = [v for i in range(30) for v in (*reds[3*i:3*i+3], uvg[i])]
    for value in values:
        sample = value['sample']
        expected = [1024, 576] if sample['dataset'] == 'REDS' else [512, 512]
        if (sample['frame_count'] != 17 or sample['transform']['coded_size'] != expected
                or not sample.get('historical_teacher_sample_id')
                or sample['router_split'] not in ('train', 'validation')):
            raise ValueError('unsupported mixed-view sample geometry/provenance')
        if sample['dataset'] == 'REDS':
            folder = Path(value['view_json']).parent/'fullview'
            if digest(folder/'complete.json') != value['fullview_complete_sha256']:
                raise ValueError('REDS fullview preparation record changed')
            verify(folder, read(folder/'complete.json')['artifacts'])
        else:
            for name, expected_hash in value['source_hashes']['files'].items():
                if digest(name) != expected_hash:
                    raise ValueError('UVG crop source pixels changed')
    if len({v['sample']['sample_id'] for v in values}) != len(values):
        raise ValueError('duplicate teacher sample IDs')
    groups = {}
    for value in values:
        s = value['sample']; key = (s['dataset'], s['sequence'])
        if groups.setdefault(key, s['router_split']) != s['router_split']:
            raise ValueError('Router source sequence occurs in both train and validation')
    return values


def protocol(root, data, smoke):
    from demo.online_eg_decode import identities
    from demo.four_state_core import CODE as OLD_CODE
    from demo.chunk_enhancement_experiment import CODE_FILES
    views = selected_views(data, smoke)
    profile = identities(ADAPTER)
    old = read(OLD/'protocol.json')
    if (old['profile'] != profile or old['enhancement'] != digest(ENHANCEMENT)
            or old['adapter'] != digest(ADAPTER)):
        raise ValueError('UVG teacher reuse requires the exact historical E/G models and profile')
    code = {name:digest(REPO/name) for name in CODE_FILES}
    for name in (*OLD_CODE, 'routervc_mixedview_teacher.py',
                 'routervc_mixedview_teacher_receive.py', 'run_routervc_mixedview_teacher.sh',
                 'routervc_encode.py', 'routervc_policy.py', 'scalable_codec.py',
                 'scalable_experiment.py', 'scalable_cooperation_format.py',
                 'online_eg_decode.py', 'online_eg_eval_core.py', 'internal_condition_decode.py',
                 'internal_condition_model.py', 'feature_interface_model.py',
                 'conditioned_generation_pipeline.py', 'chunk_enhancement_evaluate.py'):
        code['demo/'+name] = digest(REPO/'demo'/name)
    value = dict(schema=SCHEMA, smoke=smoke, data=str(Path(data).resolve()),
                 data_sha256=digest(Path(data)/'complete.json'), views=views,
                 enhancement=digest(ENHANCEMENT), adapter=digest(ADAPTER), profile=profile,
                 code=code, historical_protocol=digest(OLD/'protocol.json'),
                 historical_labels=digest(OLD/'labels.json'), states=list(STATES),
                 content_annotation_status='unknown', teacher_objective='pure LPIPS baseline',
                 receiver_inputs='decoded B, decoded candidate Y, actual E coverage only',
                 semantics='isolated per-region counterfactuals, not an additive mixed-video oracle',
                 bytes='base/shared diagnostic control charged once; E packet bundles per region',
                 deployment='RTVC shared receiver policy; no per-region G/protection mask added',
                 no_model_updates=True, independent_system_test=False)
    immutable(root/'protocol.json', value)
    return value


def costs(bank, settings, index):
    from demo.routervc_encode import subset_bank
    from demo import scalable_cooperation_format as fmt
    base, enhancement = subset_bank(bank, []), subset_bank(bank, [index])
    generated, cooperative = fmt.wrap(base, settings), fmt.wrap(enhancement, settings)
    shared = len(generated)-len(base)-fmt.REGION.size
    return dict(base_container_bytes=len(base), e_packet_bytes=len(enhancement)-len(base),
                g_region_bytes=fmt.REGION.size, g_shared_bytes=shared,
                individual_stream_bytes=dict(B=len(base), E=len(enhancement),
                                             G=len(generated), EG=len(cooperative)))


def prepare_new(root):
    import numpy as np
    from demo.routervc_encode import load_input, prepare
    from demo.scalable_codec import atomic_npz
    from demo.scalable_format import frame_hash
    from demo.scalable_experiment import check_space
    p = read(root/'protocol.json'); entries = []
    for view in p['views']:
        sample = view['sample']; sid = sample['sample_id']
        if sample['dataset'] != 'REDS':
            continue
        check_space(); folder = root/'encoded'/sid; folder.mkdir(parents=True, exist_ok=True)
        binding = dict(protocol=digest(root/'protocol.json'), view=view['view_sha256'])
        done = folder/'complete.json'
        if done.exists():
            record = read(done)
            if record['binding'] != binding:
                raise ValueError('changed fullview sender binding')
            verify(folder, record['artifacts'])
        else:
            source, _ = load_input(Path(view['frames_dir']))
            if source.shape != (17, 576, 1024, 3):
                raise ValueError('unexpected prepared REDS shape')
            cache = folder/'source.npz'
            if not cache.exists():
                atomic_npz(cache, source=source)
            with np.load(cache, allow_pickle=False) as existing:
                np.testing.assert_array_equal(existing['source'], source)
            started = time.monotonic()
            prepared = prepare(Path(view['frames_dir']), folder/'prepared', enhancement=ENHANCEMENT)
            bank = (folder/'prepared/bank.acse').read_bytes()
            rois = grid(source.shape[1], source.shape[2])
            if prepared['rois'] != rois:
                raise ValueError('prepared bank and teacher grid differ')
            record = dict(complete=True, sample_id=sid, binding=binding, shape=list(source.shape),
                          rois=rois, source_path=str(cache.resolve()), source_sha256=digest(cache),
                          source_rgb_sha256=frame_hash(source),
                          bank_path=str((folder/'prepared/bank.acse').resolve()),
                          bank_sha256=digest(folder/'prepared/bank.acse'),
                          expected_base_hash=prepared['base_rgb_sha256'],
                          expected_E_hash=prepared['all_E_rgb_sha256'],
                          per_region_cost=[costs(bank, control(p['profile'], sid, i, rois), i)
                                           for i in range(16)],
                          sender_prepare_seconds=time.monotonic()-started,
                          candidate_preparation=prepared,
                          artifacts={name:digest(folder/name) for name in
                                     ('source.npz', 'prepared/complete.json', 'prepared/base.acse',
                                      'prepared/bank.acse', 'prepared/candidates.npz')})
            save(done, record)
        # Receiver index intentionally omits source paths and sender RGB caches.
        entries.append({k:record[k] for k in ('sample_id', 'shape', 'rois', 'bank_path',
                                             'bank_sha256', 'expected_base_hash', 'expected_E_hash')}
                       | dict(encoded_manifest=digest(done)))
        print(f'PREPARE {sid} {len(entries)} new whole views', flush=True)
    immutable(root/'receiver_index.json', dict(complete=True, profile=p['profile'],
              enhancement=p['enhancement'], adapter=p['adapter'], samples=entries))


def _base_label(view, source_path, received_path, shape):
    s = view['sample']
    return dict(sample_id=s['sample_id'], dataset=s['dataset'], sequence=s['sequence'],
                router_split=s['router_split'], view_kind=s['view_kind'], shape=list(shape),
                historical_teacher_sample_id=s['historical_teacher_sample_id'],
                source_path=str(Path(source_path).resolve()), source_sha256=digest(source_path),
                received_path=str(Path(received_path).resolve()), received_sha256=digest(received_path),
                content_annotation_status='unknown', semantic_labels_available=False,
                receiver_uses_source=False, independent_system_test=False,
                sample=s,
                quality_scope='17-frame isolated native ROI after feather; neighboring E absent')


def reuse_uvg(root, view):
    import numpy as np
    from demo.four_state_core import rows
    from demo.routervc_encode import load_input
    from demo.scalable_format import frame_hash
    sample = view['sample']; sid = sample['historical_teacher_sample_id']
    old_entry = next(r for r in read(OLD/'labels.json')['samples'] if r['sample_id'] == sid)
    label_path = Path(old_entry['path'])
    if digest(label_path) != old_entry['sha256']:
        raise ValueError('historical UVG label hash changed')
    old_label = read(label_path)
    row = next(r for r in rows(False) if r['sample_id'] == sid)
    source_path = Path(row['pair_path'])
    if digest(source_path) != row['pair_hash'] or old_label['dependencies']['pair'] != row['pair_hash']:
        raise ValueError('historical UVG source cache changed')
    source, _ = load_input(Path(view['frames_dir']))
    with np.load(source_path, allow_pickle=False) as original:
        np.testing.assert_array_equal(source, original['source'])
    received = OLD/'received'/sid
    er = read(received/'E.json'); verify(received, er['artifacts'])
    received_complete = read(received/'complete.json')
    if received_complete['e_report'] != digest(received/'E.json') or er['source_frames_read']:
        raise ValueError('historical UVG receiver provenance changed')
    if (digest(received/'complete.json') != old_label['dependencies']['received']
            or digest(OLD/'encoded'/sid/'complete.json') != old_label['dependencies']['encoded']):
        raise ValueError('historical UVG teacher dependencies changed')
    encoded = read(OLD/'encoded'/sid/'complete.json'); verify(OLD/'encoded'/sid, encoded['artifacts'])
    regions = []
    for i, region in enumerate(old_label['regions']):
        folder = received/f'cell_{i:02d}'; record = read(folder/'result.json')
        verify(folder, record['artifacts'])
        if digest(folder/'result.json') != received_complete['region_results'][i]:
            raise ValueError('historical UVG cell record changed')
        if record['control'] != control(read(root/'protocol.json')['profile'], sid, i, grid(512, 512)):
            raise ValueError('historical UVG generation profile differs')
        regions.append({k:region[k] for k in ('region', 'roi', 'quality', 'lpips_gain',
                                              'interaction_gain', 'costs', 'g_seconds')})
    result = _base_label(view, source_path, received/'received_E.npz', source.shape)
    result.update(regions=regions, reused_historical_UVG=True,
                  source_rgb_sha256=frame_hash(source),
                  dependencies=dict(protocol=digest(root/'protocol.json'), view=view['view_sha256'],
                                    historical_label=str(label_path.resolve()),
                                    historical_label_sha256=old_entry['sha256'],
                                    historical_received=digest(received/'complete.json'),
                                    historical_encoded=digest(OLD/'encoded'/sid/'complete.json')),
                  timing_scope='reused historical region runtimes; no new UVG inference')
    return result


def labels(root):
    import numpy as np
    import torch
    from demo.scalable_experiment import quality, check_space
    from demo.stage_c_three_path_roi_probe import LPIPSAlex
    torch.set_num_threads(4)
    p = read(root/'protocol.json'); entries = []; metric = None
    for view in p['views']:
        check_space(); s = view['sample']; sid = s['sample_id']
        path = root/'labels'/f'{sid}.json'
        if path.exists():
            result = read(path)
            if (result['dependencies']['protocol'] != digest(root/'protocol.json')
                    or result['dependencies']['view'] != view['view_sha256']):
                raise ValueError('completed label input binding changed')
            for key in ('source', 'received'):
                if digest(Path(result[key+'_path'])) != result[key+'_sha256']:
                    raise ValueError('completed label pixel cache changed')
        elif s['dataset'] == 'UVG':
            result = reuse_uvg(root, view); save(path, result)
        else:
            began = time.monotonic(); encoded = read(root/'encoded'/sid/'complete.json')
            received = root/'received'/sid; er = read(received/'E.json')
            verify(received, er['artifacts'])
            source_path, received_path = Path(encoded['source_path']), received/'received_E.npz'
            if digest(source_path) != encoded['source_sha256']:
                raise ValueError('label source changed')
            with np.load(source_path, allow_pickle=False) as cache:
                source = cache['source'].copy()
            with np.load(received_path, allow_pickle=False) as cache:
                base, enhanced = cache['base'].copy(), cache['enhanced'].copy()
            if metric is None:
                metric = LPIPSAlex(True)
            regions = []
            for i, roi in enumerate(encoded['rois']):
                folder = received/f'cell_{i:02d}'; cell = read(folder/'result.json')
                verify(folder, cell['artifacts'])
                with np.load(folder/'outputs.npz', allow_pickle=False) as outputs:
                    variants = dict(B=crop(base, roi), E=crop(enhanced, roi),
                                    G=outputs['G'].copy(), EG=outputs['EG'].copy())
                scores = {state:quality(crop(source, roi), pixels, metric)
                          for state, pixels in variants.items()}
                gain = {state:scores['B']['lpips_alex']-scores[state]['lpips_alex'] for state in STATES}
                regions.append(dict(region=i, roi=roi, quality=scores, lpips_gain=gain,
                    interaction_gain=gain['EG']-gain['E']-gain['G'], costs=encoded['per_region_cost'][i],
                    g_seconds={state:cell['reports'][state]['runtime']['seconds_model_load_excluded']
                               for state in ('G', 'EG')}))
            result = _base_label(view, source_path, received_path, source.shape)
            result.update(regions=regions, reused_historical_UVG=False,
                dependencies=dict(protocol=digest(root/'protocol.json'), view=view['view_sha256'],
                                  encoded=digest(root/'encoded'/sid/'complete.json'),
                                  received=digest(received/'complete.json')),
                e_bank_decode_seconds=er['seconds'], e_timing_scope='entire candidate bank; not per-region estimate',
                seconds=time.monotonic()-began)
            save(path, result)
        entries.append({k:result[k] for k in ('sample_id', 'dataset', 'sequence', 'router_split', 'view_kind')}
                       | dict(path=str(path.resolve()), sha256=digest(path),
                              reconstruction_path=result['received_path'],
                              reconstruction_sha256=result['received_sha256']))
        print(f'LABEL {sid} {len(entries)}/{len(p["views"])}', flush=True)
    immutable(root/'labels.json', dict(complete=True, schema=SCHEMA, samples=entries,
              region_count=16*len(entries), states=list(STATES), protocol=digest(root/'protocol.json'),
              datasets=dict(Counter(r['dataset'] for r in entries)),
              router_splits=dict(Counter(r['router_split'] for r in entries)),
              semantic_labels_available=False, independent_system_test=False))


def snapshot(root):
    return {str(p.relative_to(root)):dict(sha256=digest(p), mtime_ns=p.stat().st_mtime_ns)
            for p in sorted((root/'received').rglob('*.json'))}


def replay(run):
    from demo.conditioned_generation_pipeline import execute
    before = snapshot(run.root)
    execute(run, 'teacher_readonly_replay', 'routervc_mixedview_teacher_receive.py',
            ['--root', run.root, '--verify-only'])
    if snapshot(run.root) != before:
        raise ValueError('completed receiver replay changed records or timings')
    immutable(run.root/'replay_audit.json', dict(complete=True, inference_forbidden=True,
                                               no_metric_recalculation=True, preserved=before))


def fresh_checks(run):
    import numpy as np
    from demo.conditioned_generation_pipeline import execute
    from demo.online_eg_eval_core import noise_pair
    checked_paths = []
    for row in read(run.root/'receiver_index.json')['samples']:
        for index in (0, 5):
            folder = run.root/'received'/row['sample_id']/f'cell_{index:02d}'
            record = read(folder/'result.json')
            for state in ('G', 'EG'):
                out = run.root/'fresh_checks'/row['sample_id']/f'cell_{index:02d}_{state}'
                done = out/'checked.json'
                if done.exists():
                    verify(out, read(done)['artifacts'])
                else:
                    out.mkdir(parents=True, exist_ok=True)
                    execute(run, f'fresh_{row["sample_id"]}_{index}_{state}', 'online_eg_decode.py',
                        ['--stream', folder/f'{state}.acsg', '--output', out,
                         '--enhancement', ENHANCEMENT, '--adapter', ADAPTER], distributed=True)
                    report = read(out/'decode.json'); expected = record['reports'][state]
                    if (report['source_frames_read'] or not report['outside_generate_exact']
                            or report['output_hash'] != expected['output_hash']
                            or report['generation_input_hash'] != expected['generation_input_hash']
                            or report['total_bytes'] != expected['total_bytes']):
                        raise ValueError('fresh process and persistent teacher differ')
                    with np.load(out/'reconstruction.npz', allow_pickle=False) as cache:
                        actual = crop(cache['reconstruction'], row['rois'][index])
                    with np.load(folder/'outputs.npz', allow_pickle=False) as cache:
                        np.testing.assert_array_equal(actual, cache[state])
                    noise_pair(report, dict(generation_runtime=expected['runtime']), same_condition=True)
                    save(done, dict(complete=True, fresh_receiver_exact=True,
                         teacher_cell_sha256=digest(folder/'result.json'),
                         artifacts={n:digest(out/n) for n in ('reconstruction.npz', 'decode.json')}))
                if read(done)['teacher_cell_sha256'] != digest(folder/'result.json'):
                    raise ValueError('fresh comparison teacher changed')
                checked_paths.append(str(done.resolve()))
    immutable(run.root/'fresh_audit.json', dict(complete=True, fresh_receiver_exact=True,
              boundary_and_interior=True, checks=checked_paths,
              uvG_scope='existing crops reused with original fresh-decode evidence; no repeat inference'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('smoke', 'run', 'verify', 'prepare', 'labels'))
    parser.add_argument('--data', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke-root', type=Path)
    parser.add_argument('--max-hours', type=float, default=24.)
    args = parser.parse_args(argv)
    if not math.isfinite(args.max_hours) or args.max_hours <= 0:
        raise ValueError('max-hours must be finite and positive')
    if args.command in ('prepare', 'labels'):
        if not os.environ.get('TMUX') or not os.environ.get('ROUTERVC_MIXEDVIEW_PARENT'):
            raise RuntimeError('worker must be spawned by the tmux teacher queue')
        return prepare_new(args.output) if args.command == 'prepare' else labels(args.output)
    if not os.environ.get('TMUX'):
        raise RuntimeError('run mixed-view teacher in tmux')
    if args.data is None:
        raise ValueError('--data prepared mixedview directory is required')
    from demo.chunk_enhancement_experiment import Run
    from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
    from demo.conditioned_generation_pipeline import execute
    run = Run(args); run.thread.start()
    try:
        old = read(run.root/'protocol.json') if (run.root/'protocol.json').exists() else None
        is_smoke = old['smoke'] if args.command == 'verify' and old else args.command == 'smoke'
        p = protocol(run.root, args.data, is_smoke)
        if args.command == 'run':
            smoke = args.smoke_root or args.output.with_name(args.output.name+'_smoke')
            if (not read(smoke/'complete.json')['complete']
                    or not read(smoke/'fresh_audit.json')['fresh_receiver_exact']
                    or not read(smoke/'replay_audit.json')['inference_forbidden']):
                raise ValueError('formal teacher requires completed fresh/resume smoke')
            tested = read(smoke/'protocol.json')
            if any(tested[k] != p[k] for k in ('code', 'enhancement', 'adapter', 'profile')):
                raise ValueError('formal teacher code/models differ from smoke')
        with exclusive_native_evaluation(run):
            os.environ['ROUTERVC_MIXEDVIEW_PARENT'] = str(os.getpid())
            if args.command == 'verify':
                replay(run)
                ledger = read(run.root/'labels.json')
                for entry in ledger['samples']:
                    if digest(Path(entry['path'])) != entry['sha256']:
                        raise ValueError('saved teacher label changed')
                if read(run.root/'complete.json')['labels'] != digest(run.root/'labels.json'):
                    raise ValueError('teacher completion label hash differs')
                return
            execute(run, 'prepare_fullviews', 'routervc_mixedview_teacher.py',
                    ['prepare', '--output', run.root])
            if is_smoke and not (run.root/'partial_resume.json').exists():
                execute(run, 'receive_partial', 'routervc_mixedview_teacher_receive.py',
                        ['--root', run.root, '--stop-after', 2], distributed=True)
                save(run.root/'partial_resume.json', dict(preserved=snapshot(run.root)))
            execute(run, 'receive_fullviews', 'routervc_mixedview_teacher_receive.py',
                    ['--root', run.root], distributed=True)
            if is_smoke:
                for path, expected in read(run.root/'partial_resume.json')['preserved'].items():
                    actual = run.root/path
                    if expected != dict(sha256=digest(actual), mtime_ns=actual.stat().st_mtime_ns):
                        raise ValueError('resuming receiver rewrote completed region/timing')
                fresh_checks(run)
            replay(run)
            execute(run, 'teacher_labels_CPU', 'routervc_mixedview_teacher.py',
                    ['labels', '--output', run.root])
            done = run.root/'complete.json'
            if not done.exists():
                save(done, dict(complete=True, stage=SCHEMA, no_model_updates=True,
                     protocol=digest(run.root/'protocol.json'), labels=digest(run.root/'labels.json'),
                     elapsed_seconds_this_completion_attempt=time.monotonic()-run.started,
                     no_router_training=True, semantic_labels_available=False))
            run.update(phase='complete', completed=len(p['views']), total=len(p['views']))
    except BaseException as error:
        save(run.root/'last_failure.json', dict(error=repr(error), progress=run.progress)); raise
    finally:
        run.log_resources(); run.stop.set(); run.thread.join(timeout=3)


if __name__ == '__main__':
    main()
