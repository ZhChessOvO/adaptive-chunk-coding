"""Fixed-E 13-window receiver-only diagnostic; never train/replace sender Rs.

Freeze the 03.20 q2 shared-global/local E packet selections and generator
controls. Reuse 39 authenticated historical points; fresh-decode 78 Rg points
and eight repeat/G-off checks. REDS full views and UVG crops remain separate.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from demo.routervc_fullview_probe import read, digest, save, immutable, verify_artifacts
from demo.routervc_receiver_train import ROOT
from demo.routervc_light_teacher import ENHANCEMENT, ADAPTER
from demo.routervc_visual_evaluate import PhaseDeadline, source_shape

OLD = Path('/root/autodl-fs/DCVC/runs/routervc_light_router_20261004/evaluation')
REVISION = Path('/root/autodl-fs/DCVC/runs/routervc_revision_20261003')
RATIOS = (0., .25, .5)
ARMS = ('core', 'halo')
METRICS = ('lpips_alex', 'psnr_db', 'temporal_delta_mae')
CODE = ('routervc_receiver_evaluate.py', 'routervc_receiver_report.py',
        'run_routervc_receiver_evaluate.sh', 'routervc_light_evaluate.py',
        'routervc_light_decode.py', 'routervc_qstep_probe.py', 'routervc_visual_evaluate.py')


def point_plan():
    return [dict(name=f'{arm}_e{ratio:g}_g8', arm=arm, ratio=ratio, max_g=8,
                 reused=arm == 'shared') for arm in ('shared', *ARMS) for ratio in RATIOS]


def completed_models(training):
    """Reject partial/smoke models; choose each arm's TRAIN-validation best."""
    from demo.routervc_receiver_router import load_model
    training = Path(training)
    if not (training/'complete.json').exists():
        return None
    top = read(training/'complete.json')
    root = training/'formal/router'
    done, config = read(root/'complete.json'), read(root/'config.json')
    protocol = config['protocol']
    if (not top.get('complete') or top['router'] != digest(root/'complete.json')
            or top['protocol'] != digest(training/'formal/protocol.json')
            or top['labels'] != digest(training/'formal/labels.complete.json')
            or not done.get('complete') or done['config'] != digest(root/'config.json')
            or protocol != read(training/'formal/protocol.json') or protocol['smoke'] is not False
            or done['epochs'] != 120 or done['updates_per_arm'] != 11520
            or done['sender_training_complete'] is not False
            or done['semantic_supervision'] is not False or set(done['selected']) != set(ARMS)):
        raise ValueError('receiver evaluation requires completed formal paired 120-epoch training')
    verify_artifacts(root, done['artifacts'])
    models = {}
    for arm in ARMS:
        path = root/arm/'best.pt'
        _, payload = load_model(path, expected_sha256=done['artifacts'][f'{arm}/best.pt'])
        if (payload['binding'] != config or payload['arm'] != arm
                or payload['epoch'] != done['selected'][arm]['epoch']
                or payload['selection'] != done['selected'][arm]['validation']):
            raise ValueError('receiver best model differs from training validation selection')
        models[arm] = dict(path=str(path), sha256=digest(path), epoch=payload['epoch'],
                           validation=payload['selection'])
    return dict(arms=models, training_complete=digest(training/'complete.json'),
                router_complete=digest(root/'complete.json'), no_automatic_model_promotion=True)


def code_hashes():
    from demo import routervc_receiver_format as fmt
    return dict(fmt.code_identity(), **{n: digest(REPO/'demo'/n) for n in CODE})


def make_protocol(training, models):
    from demo import routervc_receiver_format as fmt
    from demo.routervc_light_evaluate import check_point
    old = read(OLD/'protocol.json')
    verify_artifacts(OLD, read(OLD/'complete.json')['artifacts'])
    for name, expected in old['code'].items():
        if digest(REPO/'demo'/name) != expected:
            raise ValueError('historical q2 evaluator source changed: '+name)
    if digest(ENHANCEMENT) != read(OLD.parent/'teacher/protocol.json')['enhancement']:
        raise ValueError('frozen enhancement model changed')
    references = {}
    if len(old['sources']) != 13 or old['qstep'] != 2.:
        raise ValueError('expected the fixed 13-window q2 diagnostic')
    for entry in old['sources']:
        sample = entry['sample']; sid = sample['sample_id']
        references[sid] = {}
        for ratio in RATIOS:
            folder = OLD/'samples'/sid/f'new_global_local_e{ratio:g}_g8'
            result = check_point(folder, old, sample)
            if result['reused'] or result['qstep'] != 2.:
                raise ValueError('reference must be the measured q2 trained shared Router')
            references[sid][f'e{ratio:g}'] = dict(folder=str(folder),
                result_sha256=digest(folder/'result.json'), encode_sha256=digest(folder/'encode.json'),
                stream_sha256=digest(folder/'stream.rtvc'))
    old_config = old['configs']['new_global_local']
    if old_config['seed'] != 20261003 or old_config['max_g'] != 8 or old_config['boundary_lambda'] != 0:
        raise ValueError('historical seed/G-budget/control changed')
    configs = {arm: fmt.make_config(Path(models['arms'][arm]['path']), ADAPTER,
               max_g=8, boundary_lambda=0., seed=old_config['seed']) for arm in ARMS}
    for config in configs.values():
        if any(config[key] != old_config[key] for key in (*fmt.CONTROL_KEYS, *fmt.legacy.cooperation.HASHES)):
            raise ValueError('only receiver weights/policy may change, not generator controls')
    return dict(schema='routervc-receiver-fixed-E-evaluation-v1', training=str(training),
        models=models, configs=configs, references=references, sources=old['sources'],
        inputs=old['inputs'], points=point_plan(), qstep=2., code=code_hashes(),
        old_protocol_sha256=digest(OLD/'protocol.json'), old_complete_sha256=digest(OLD/'complete.json'),
        enhancement=dict(path=str(ENHANCEMENT), sha256=digest(ENHANCEMENT)),
        adapter=dict(path=str(ADAPTER), sha256=digest(ADAPTER)),
        E_selection='unchanged 03.20 new_global_local literal ACSE2 bytes; no reallocation',
        budget='old realized selections at q2 E-bank byte caps 0/0.25/0.5; actual complete bytes charged',
        noise_scope='same seed and ordinal-based rule; different routes are NOT universally pixelwise noise-paired',
        no_sender_training=True, freeze_UF_E_G=True, semantic_supervision=False,
        independent_system_test=False, scope='13 historically used windows: 6 REDS resized full views and 7 UVG crops',
        expected_fresh_points=78, expected_historical_points=39, expected_repeat_Goff_checks=8)


def reference(protocol, sid, ratio, *, verify=True):
    from demo.routervc_light_evaluate import check_point
    item = protocol['references'][sid][f'e{ratio:g}']; folder = Path(item['folder'])
    for name, field in (('result.json','result_sha256'), ('encode.json','encode_sha256'),
                        ('stream.rtvc','stream_sha256')):
        if digest(folder/name) != item[field]:
            raise ValueError('fixed historical E point changed: '+name)
    result = read(folder/'result.json')
    if verify:
        sample = next(v['sample'] for v in protocol['sources'] if v['sample']['sample_id'] == sid)
        check_point(folder, read(OLD/'protocol.json'), sample)
    return folder, result, read(folder/'encode.json')


def rewrap_fixed_E(old_wire, config):
    from demo import routervc_visual_format as old_fmt
    from demo import routervc_receiver_format as fmt
    old_config, inner, parsed, _ = old_fmt.parse(old_wire)
    if any(config[key] != old_config[key] for key in (*fmt.CONTROL_KEYS, *fmt.legacy.cooperation.HASHES)):
        raise ValueError('fixed-E comparison changed a generator control')
    if any(p.meta['qstep'] != 2. for p in parsed.packets):
        raise ValueError('fixed E requires q2 packets')
    wire = fmt.wrap(inner, config)
    _, actual, _, header = fmt.parse(wire)
    if actual != inner or len(wire) != header+len(inner):
        raise ValueError('receiver wrapper altered E/base bytes or byte accounting')
    return wire, header, hashlib.sha256(inner).hexdigest()


def encode_point(root, protocol, sid, point):
    """CPU-only rewrap/re-predict from actual old received Y; never encode E."""
    import numpy as np
    from demo import routervc_receiver_format as fmt
    from demo.routervc_receiver_policy import route
    from demo.scalable_codec import atomic_bytes
    folder = Path(root)/'samples'/sid/point['name']
    folder.mkdir(parents=True, exist_ok=True)
    old_folder, old, old_encoded = reference(protocol, sid, point['ratio'])
    binding = dict(protocol=digest(Path(root)/'protocol.json'), reference=digest(old_folder/'result.json'), point=point)
    done = folder/'encode.json'
    if done.exists():
        saved = read(done)
        if saved['binding'] != binding:
            raise ValueError('receiver encode binding changed')
        verify_artifacts(folder, saved['artifacts'])
        return saved
    wire, header, inner_hash = rewrap_fixed_E((old_folder/'stream.rtvc').read_bytes(),
                                             protocol['configs'][point['arm']])
    config, _, inner, _ = fmt.parse(wire)
    with np.load(old_folder/'fresh/reconstruction.npz', allow_pickle=False) as data:
        expected = route(data['base'], data['enhanced'], inner, config,
            protocol['models']['arms'][point['arm']]['path'], expected_policy=fmt.policy_identity())
    stream = folder/'stream.rvrc'
    if stream.exists() and stream.read_bytes() != wire:
        raise ValueError('partially prepared receiver wire changed')
    if not stream.exists():
        atomic_bytes(stream, wire)
    saved = dict(complete=True, binding=binding, expected_route=expected,
        expected_base_hash=old['decode']['base_hash'], expected_mixed_hash=old['decode']['generation_input_hash'],
        fixed_E_indices=old_encoded['plan']['selected_indices'], old_plan=old_encoded['plan'],
        unchanged_inner_sha256=inner_hash, actual_header_bytes=header,
        actual_stream_bytes=stream.stat().st_size, explicit_masks_bytes=0,
        artifacts={'stream.rvrc':digest(stream)})
    save(done, saved)
    return saved


def noise_overlap(old, new):
    """Only identical ordinal + processing geometry has exactly paired noise."""
    if (old.get('route', {}).get('indices') == new.get('route', {}).get('indices')
            and old.get('output_hash') != new.get('output_hash')):
        raise ValueError('identical G decisions on fixed Y must reproduce historical pixels')
    left, right = old.get('generation_runtime'), new.get('generation_runtime')
    if not left or not right:
        return dict(paired_windows=0, scope='same seed rule; changed route is not universally paired')
    def index(report, runtime):
        route = report['route']
        return {(w['region'], tuple(route['rois'][route['indices'][w['region']]]),
                 w['start'], tuple(w['crop'])): o for w, o in
                zip(runtime['windows'], runtime['condition_windows'], strict=True)}
    a, b = index(old, left), index(new, right)
    keys = a.keys() & b.keys()
    for key in keys:
        for field in ('before_vae','after_vae','before_diffusion','diffusion_noise','conditions'):
            if a[key][field] != b[key][field]:
                raise ValueError('same ordinal/geometry/received Y changed noise or condition')
    return dict(paired_windows=len(keys), old_windows=len(a), new_windows=len(b),
        scope='exact only for common ordinal/core-ROI/start/processing-crop; no universal cross-route pairing claim')


def validate_new(folder, protocol, sample):
    import numpy as np
    from demo import routervc_receiver_format as fmt
    from demo.scalable_format import frame_hash
    from demo import scalable_cooperation_format as cooperation
    folder = Path(folder)
    encoded = read(folder/'encode.json'); report = read(folder/'fresh/decode.json')
    stream = folder/'stream.rvrc'
    config, _, inner, header = fmt.parse(stream.read_bytes())
    if (report['source_frames_read'] or not report['outside_generate_exact']
            or not report['base_reference_unchanged'] or report['sender_router_loaded']
            or report['unreceived_candidates_read'] or not report['receiver_router_used']
            or report['semantic_heads_used'] or report['profile'] != fmt.PROFILE
            or report['stream_sha256'] != digest(stream) or report['config'] != config
            or report['route'] != encoded['expected_route']
            or report['base_hash'] != encoded['expected_base_hash']
            or report['generation_input_hash'] != encoded['expected_mixed_hash']
            or report['generation_control_bytes'] != header
            or any(report[k] != 0 for k in ('explicit_E_mask_bytes','explicit_G_map_bytes','protection_mask_bytes'))):
        raise ValueError('fresh receiver source/model/route/byte contract failed')
    size = stream.stat().st_size
    if report['total_bytes'] != size or sum(report[k] for k in (
            'base_bytes','container_header_bytes','packet_bytes','incomplete_tail_bytes','generation_control_bytes')) != size:
        raise ValueError('actual receiver byte ledger mismatch')
    if report['peak_cuda_allocated_bytes'] < report['generation_all_calls_peak_cuda_bytes']:
        raise ValueError('whole-worker peak cannot be less than all-call G peak')
    with np.load(folder/'fresh/reconstruction.npz', allow_pickle=False) as data:
        for key, field in (('base','base_hash'),('enhanced','generation_input_hash'),('reconstruction','output_hash')):
            if data[key].shape != source_shape(sample) or frame_hash(data[key]) != report[field]:
                raise ValueError('fresh receiver saved pixels differ')
        control = fmt.generation_control(config, report['route']['indices'], report['route']['rois'], len(data['base']))
        mask = cooperation.weights(data['base'].shape, control)
        np.testing.assert_array_equal(data['reconstruction'][mask == 0], data['enhanced'][mask == 0])
    return report


def check_point(folder, protocol, sample):
    folder = Path(folder); result = read(folder/'result.json')
    verify_artifacts(folder, result['artifacts'])
    if result['protocol'] != digest(folder.parents[2]/'protocol.json'):
        raise ValueError('receiver point protocol changed')
    ref, old, _ = reference(protocol, sample['sample_id'], result['point']['ratio'])
    if result['reference_sha256'] != digest(ref/'result.json'):
        raise ValueError('receiver reference changed')
    if result['reused']:
        for key in ('bytes','bpp','quality','decode','same_wire_G_off_quality'):
            if result[key] != old[key]:
                raise ValueError('historical receiver metric changed')
    else:
        if validate_new(folder, protocol, sample) != result['decode']:
            raise ValueError('saved receiver measurement changed')
        from demo import routervc_visual_format as old_fmt
        from demo import routervc_receiver_format as fmt
        if fmt.parse((folder/'stream.rvrc').read_bytes())[1] != old_fmt.parse((ref/'stream.rtvc').read_bytes())[1]:
            raise ValueError('fixed-E inner bytes changed')
        if (result['same_wire_G_off_quality'] != old['same_wire_G_off_quality']
                or result['decode']['generation_input_hash'] != old['decode']['generation_input_hash']):
            raise ValueError('G-off reuse requires identical actual Y')
        if noise_overlap(old['decode'], result['decode']) != result['noise_pairing']:
            raise ValueError('noise overlap evidence changed')
    return result


def recovery_checks(run, protocol, sample, arm, *, verify_only=False):
    import numpy as np
    from demo.conditioned_generation_pipeline import execute
    from demo.online_eg_eval_core import noise_pair
    sid = sample['sample_id']; parent = run.root/'samples'/sid/f'{arm}_e0.5_g8'
    formal = read(parent/'fresh/decode.json'); paths = []
    for mode in ('repeat','G_off'):
        folder = run.root/'checks'/sid/arm/mode; done = folder/'checked.json'
        if done.exists():
            saved = read(done); verify_artifacts(folder, saved['artifacts'])
            if saved['reference'] != digest(parent/'result.json'):
                raise ValueError('receiver recovery reference changed')
        else:
            if verify_only:
                raise ValueError('read-only verification cannot start receiver checks')
            folder.mkdir(parents=True, exist_ok=True)
            if not (folder/'decode.json').exists():
                argv = ['--worker','--stream',parent/'stream.rvrc','--output',folder,
                    '--enhancement',ENHANCEMENT,'--adapter',ADAPTER if mode == 'repeat' else folder/'absent_G.pt',
                    '--router',protocol['models']['arms'][arm]['path'] if mode == 'repeat' else folder/'absent_Rg.pt']
                if mode == 'G_off': argv.append('--disable-generation')
                execute(run, f'{sid}_{arm}_{mode}', 'routervc_receiver_decode.py', argv, distributed=True)
            report = read(folder/'decode.json')
            for key in ('stream_sha256','total_bytes','base_hash','generation_input_hash'):
                if report[key] != formal[key]:
                    raise ValueError('receiver repeat/G-off input changed')
            if report['source_frames_read'] or not report['outside_generate_exact'] or report['sender_router_loaded']:
                raise ValueError('recovery check used unavailable source or sender')
            if mode == 'repeat':
                if report['output_hash'] != formal['output_hash'] or report['route'] != formal['route']:
                    raise ValueError('fresh receiver repeat changed pixels or route')
                if formal['generation_runtime'] is not None:
                    noise_pair(report, formal, same_condition=True)
            elif (report['generation_executed'] or report['receiver_router_used']
                  or report['generation_assets_validated'] or report['output_hash'] != formal['generation_input_hash']):
                raise ValueError('G-off used Rg/G models or modified received Y')
            with np.load(folder/'reconstruction.npz', allow_pickle=False) as a, \
                    np.load(parent/'fresh/reconstruction.npz', allow_pickle=False) as b:
                np.testing.assert_array_equal(a['reconstruction'], b['reconstruction' if mode == 'repeat' else 'enhanced'])
            save(done, dict(complete=True, reference=digest(parent/'result.json'), mode=mode,
                artifacts={n:digest(folder/n) for n in ('decode.json','reconstruction.npz')}))
        paths.append(str(done.relative_to(run.root)))
    return paths


def evaluate(run, protocol, *, verify_only=False):
    import numpy as np
    import torch
    from demo.conditioned_generation_pipeline import execute
    from demo.scalable_experiment import quality
    from demo.stage_c_three_path_roi_probe import LPIPSAlex
    completed = (run.root/'complete.json').exists()
    if verify_only and not completed:
        raise ValueError('read-only verification requires complete evaluation')
    if completed:
        verify_artifacts(run.root, read(run.root/'complete.json')['artifacts'])
    records, checks, prefixes = [], [], []
    metric = None
    for entry in protocol['sources']:
        sample = entry['sample']; sid = sample['sample_id']; source = None
        for point in protocol['points']:
            run.check(); folder = run.root/'samples'/sid/point['name']
            if (folder/'result.json').exists():
                result = check_point(folder, protocol, sample)
            else:
                if completed: raise ValueError('completed replay cannot infer, encode or score')
                folder.mkdir(parents=True, exist_ok=True)
                ref, old, _ = reference(protocol, sid, point['ratio'])
                common = dict(complete=True, point=point, sample_id=sid, dataset=sample['dataset'],
                    protocol=digest(run.root/'protocol.json'), reference=str(ref), reference_sha256=digest(ref/'result.json'),
                    same_wire_G_off_quality=old['same_wire_G_off_quality'], reused=point['reused'])
                if point['reused']:
                    result = dict(common, **{key:old[key] for key in ('bytes','bpp','quality','decode')}, artifacts={},
                        timing_scope='authenticated historical q2 shared Router, not measured again')
                else:
                    encode_point(run.root, protocol, sid, point)
                    fresh = folder/'fresh'; fresh.mkdir(exist_ok=True)
                    if not (fresh/'decode.json').exists():
                        execute(run, f'{sid}_{point["name"]}', 'routervc_receiver_decode.py',
                            ['--worker','--stream',folder/'stream.rvrc','--output',fresh,
                             '--enhancement',ENHANCEMENT,'--adapter',ADAPTER,
                             '--router',protocol['models']['arms'][point['arm']]['path']], distributed=True)
                    report = validate_new(folder, protocol, sample)
                    if metric is None:
                        torch.set_num_threads(4); metric = LPIPSAlex(True)
                    if source is None:
                        item = protocol['inputs'][sid]
                        if digest(item['source_path']) != item['source_sha256']:
                            raise ValueError('fixed diagnostic source changed')
                        with np.load(item['source_path'], allow_pickle=False) as data:
                            source = data['source'].copy()
                    run.update(phase='CPU_quality', sample=sid, point=point['name'])
                    with np.load(fresh/'reconstruction.npz', allow_pickle=False) as data:
                        scores = quality(source, data['reconstruction'], metric)
                    size = (folder/'stream.rvrc').stat().st_size
                    result = dict(common, bytes=size, bpp=8*size/math.prod(source.shape[:3]),
                        quality=scores, decode=report, noise_pairing=noise_overlap(old['decode'], report),
                        G_off_scope='same received-Y saved-pixel quality; new stream bytes still charged',
                        artifacts={n:digest(folder/n) for n in ('stream.rvrc','encode.json','fresh/decode.json','fresh/reconstruction.npz')})
                save(folder/'result.json', result)
                check_point(folder, protocol, sample)
            records.append(result)
            run.update(phase='fixed_E_receiver_evaluation', completed=len(records), total=117)
        for arm in ARMS:
            paths = [run.root/'samples'/sid/f'{arm}_e{r:g}_g8/stream.rvrc' for r in RATIOS]
            for left, right in zip(paths, paths[1:]):
                if not right.read_bytes().startswith(left.read_bytes()):
                    raise ValueError('fixed receiver wrapper lost literal E prefix')
                prefixes.append(dict(sample_id=sid, arm=arm, low=str(left.relative_to(run.root)),
                                     high=str(right.relative_to(run.root))))
        if sid == next(v['sample']['sample_id'] for v in protocol['sources'] if v['sample']['dataset'] == sample['dataset']):
            for arm in ARMS:
                checks += recovery_checks(run, protocol, sample, arm, verify_only=completed)
    if not completed:
        if len(records) != 117 or len(checks) != 8 or len(prefixes) != 52:
            raise ValueError('fixed receiver diagnostic coverage differs')
        summary = dict(complete=True, points=117, fresh_points=78, reused_points=39,
            repeat_Goff_checks=checks, literal_prefix_pairs=prefixes, independent_system_test=False,
            rows=[{k:r[k] for k in ('sample_id','dataset','point','bytes','bpp','quality','same_wire_G_off_quality','reused')}
                  for r in records], sender_training_complete=False)
        save(run.root/'summary.json', summary)
        names = ['protocol.json','summary.json'] + checks + [
            f'samples/{r["sample_id"]}/{r["point"]["name"]}/result.json' for r in records]
        save(run.root/'complete.json', dict(complete=True, artifacts={n:digest(run.root/n) for n in names}))
    print('RECEIVER_EVALUATION_VERIFIED' if completed else 'RECEIVER_EVALUATION_COMPLETE', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=ROOT/'evaluation')
    parser.add_argument('--wait', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--max-hours', type=float, default=12.)
    parser.add_argument('--wait-hours', type=float, default=36.)
    args = parser.parse_args(argv)
    if not os.environ.get('TMUX'):
        raise RuntimeError('receiver evaluation requires tmux')
    if args.verify_only and not (args.output/'complete.json').exists():
        raise ValueError('read-only verification requires an already complete receiver evaluation')
    if any(not math.isfinite(x) or x <= 0 for x in (args.max_hours, args.wait_hours)):
        raise ValueError('positive finite wall-clock limits required')
    from demo.chunk_enhancement_experiment import Run
    from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
    run = Run(SimpleNamespace(output=args.output, command='receiver_evaluation', max_hours=args.max_hours+args.wait_hours))
    run.thread.start(); deadline = PhaseDeadline(run.check); run.check = deadline.check
    try:
        deadline.begin('wait_for_receiver_training', args.wait_hours*3600)
        while (models := completed_models(args.training)) is None:
            if not args.wait or args.verify_only:
                raise ValueError('formal receiver training not complete; --wait keeps GPU mutex free')
            run.check(); run.update(phase='waiting_for_receiver_training', holds_GPU_mutex=False); run.stop.wait(30)
        deadline.begin('receiver_evaluation', args.max_hours*3600)
        protocol = make_protocol(args.training, models)
        immutable(run.root/'protocol.json', protocol)
        os.environ['ROUTERVC_RECEIVER_PARENT'] = str(os.getpid())
        with nullcontext() if (run.root/'complete.json').exists() or args.verify_only else exclusive_native_evaluation(run):
            evaluate(run, protocol, verify_only=args.verify_only)
    except BaseException as error:
        save(run.root/'last_failure.json', dict(error=repr(error), progress=run.progress))
        raise
    finally:
        run.log_resources(); run.stop.set(); run.thread.join(timeout=3); run.lock.close()


if __name__ == '__main__':
    main()
