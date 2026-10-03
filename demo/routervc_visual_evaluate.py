"""Bounded, resumable 13-window visual-Router evaluation after formal training.

Six REDS resized whole views and seven existing UVG evaluation crops remain
separate reporting groups. These historically exposed inputs are not an unseen
benchmark. A fresh/repeat/G-off smoke precedes the 169-point queue. No training,
model promotion, semantic-protection claim, downloads or quality gates occur.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from demo.routervc_fullview_probe import read, digest, save, immutable, verify_artifacts, byte_ledger

REVISION = Path('/root/autodl-fs/DCVC/runs/routervc_revision_20261003')
WEIGHTS = Path('/root/autodl-fs/DCVC/runs/a800_online_eg_20261002/joint')
ARMS = ('global_local', 'local')
REDS_SEQUENCES = ('000', '005', '010', '015', '020', '025')
UVG_SEQUENCES = ('Beauty', 'Bosphorus', 'HoneyBee', 'Jockey', 'ReadySetGo', 'ShakeNDry', 'YachtRide')
SCHEMA = 'routervc-visual-bounded-evaluation-v1'


def selected_samples(manifest):
    """Fixed approved sample list; training/validation rows cannot enter."""
    samples = manifest['samples']
    wanted = [f'reds-val-{s}-f000-n17-fullview' for s in REDS_SEQUENCES]
    wanted += [f'uvg-{s.lower()}-f000-historical-evaluation-crop' for s in UVG_SEQUENCES]
    rows = []
    for sid in wanted:
        found = [r for r in samples if r['sample_id'] == sid]
        if len(found) != 1:
            raise ValueError(f'missing/duplicate bounded evaluation sample: {sid}')
        sample = found[0]
        is_reds = sample['dataset'] == 'REDS'
        size = [1024, 576] if is_reds else [512, 512]
        if (sample.get('router_split') != 'evaluation' or sample['frame_count'] != 17
                or sample['original_frame_start'] != 0
                or sample['view_kind'] != ('resized_full_frame' if is_reds else 'existing_spatial_crop')
                or sample['whole_frame'] is not is_reds
                or sample['transform']['coded_size'] != size
                or sample['transform']['valid_rect'] != [0, 0, *size]
                or any(sample['transform']['padding'])):
            raise ValueError('evaluation role/view geometry differs from the approved bounded profile')
        rows.append(sample)
    return rows


def point_plan():
    return [dict(name=f'{arm}_e{ratio:g}_g{cap}', kind='router', arm=arm, ratio=ratio, max_g=cap)
            for arm in ARMS for cap in (4, 8) for ratio in (.25, .5)] + [
        dict(name=f'uf_qp{qp}', kind='uf', qp=qp) for qp in (8, 16, 24, 32)] + [
        dict(name='wholeframe_g_one_roi', kind='full_g')]


def completed_models(root):
    """None means still training. A present but invalid completion is an error."""
    import torch
    from demo.routervc_visual_policy import load_model
    root = Path(root)
    if not (root / 'complete.json').exists():
        return None
    done = read(root / 'complete.json')
    if (done.get('complete') is not True or done.get('semantic_supervision') is not False
            or done.get('binding', {}).get('smoke') is not False or set(done.get('arms', {})) != set(ARMS)):
        raise ValueError('evaluation requires BOTH completed formal, non-semantic, non-smoke Router arms')
    records, architectures = {}, []
    for arm in ARMS:
        folder = root / arm
        if digest(folder / 'complete.json') != done['arms'][arm]:
            raise ValueError('formal arm completion hash changed')
        complete, config = read(folder / 'complete.json'), read(folder / 'config.json')
        if (complete.get('complete') is not True or complete.get('semantic_supervision') is not False
                or complete.get('config_sha256') != digest(folder / 'config.json')
                or config.get('semantic_supervision') is not False
                or config.get('data') != done['binding'] or config['data'].get('smoke') is not False
                or complete.get('epochs') != config.get('epochs')
                or config['model']['use_global'] is not (arm == 'global_local')):
            raise ValueError('formal Router arm configuration/smoke identity differs')
        verify_artifacts(folder, complete['artifacts'])
        if 'model.pt' not in complete['artifacts']:
            raise ValueError('formal completion lacks the final model artifact')
        model = load_model(folder / 'model.pt', expected_sha256=complete['artifacts']['model.pt'])
        payload = torch.load(folder/'model.pt', weights_only=True, map_location='cpu')
        if payload['training_binding'] != config or vars(model.config) != config['model']:
            raise ValueError('model internal training binding differs from its formal configuration')
        architectures.append({k:v for k,v in vars(model.config).items() if k != 'use_global'})
        records[arm] = dict(path=str((folder / 'model.pt').resolve()), sha256=complete['artifacts']['model.pt'],
                            complete_sha256=digest(folder / 'complete.json'), config_sha256=digest(folder / 'config.json'))
    if architectures[0] != architectures[1]:
        raise ValueError('formal global/local arms are not equal-capacity paired architectures')
    return dict(root_complete_sha256=digest(root / 'complete.json'), arms=records, semantic_supervision=False)


class PhaseDeadline:
    """Two per-invocation wall-clock caps, including subprocess supervision."""
    def __init__(self, base_check=lambda: None, now=time.monotonic):
        self.base_check, self.now, self.deadline, self.phase = base_check, now, None, None

    def begin(self, phase, seconds):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('phase duration must be finite and positive')
        self.phase, self.deadline = phase, self.now() + seconds

    def check(self):
        self.base_check()
        if self.deadline is None or self.now() >= self.deadline:
            raise InterruptedError(f'{self.phase} wall-clock limit reached; verified point checkpoints are resumable')


def wait_for_models(run, root, *, inspect=completed_models):
    # Deliberately no GPU mutex here. A corrupted complete marker is not a wait.
    while True:
        run.check()
        value = inspect(root)
        if value is not None:
            run.check()  # Inspection/loading itself must not bypass the wait cap.
            return value
        run.update(phase='waiting_for_formal_visual_Routers', holds_GPU_mutex=False)
        run.stop.wait(30)


def source_shape(sample):
    width, height = sample['transform']['coded_size']
    return (17, height, width, 3)


def code_hashes():
    from demo import routervc_visual_format as fmt
    from demo.routervc_fullview_probe import _code_hashes
    code = _code_hashes()
    code.update({'demo/'+k:v for k,v in fmt.code_identity().items()})
    for name in ('routervc_visual_evaluate.py', 'run_routervc_visual_evaluate.sh',
                 'routervc_mixedview_data.py', 'routervc_fullview_data.py',
                 'routervc_fullview_probe_report.py'):
        code['demo/'+name] = digest(REPO / 'demo' / name)
    return code


def setup(run, args, models):
    from demo import routervc_visual_format as fmt
    from demo import routervc_fullview_data as full
    samples = selected_samples(read(args.manifest))
    sources = []
    for sample in samples:
        run.check()
        sources.append(dict(sample=sample, original_file_hashes=full.source_hash(sample)))
    code = code_hashes()
    configs = {f'{arm}_g{cap}': fmt.make_config(models['arms'][arm]['path'], WEIGHTS/'adapter.pt',
                   max_g=cap, boundary_lambda=0., seed=20261003) for arm in ARMS for cap in (4, 8)}
    protocol = dict(schema=SCHEMA, manifest_sha256=digest(args.manifest), sources=sources,
        models=models, configs=configs, points=point_plan(), code=code,
        enhancement=dict(path=str(WEIGHTS/'enhancement.pt'), sha256=digest(WEIGHTS/'enhancement.pt')),
        adapter=dict(path=str(WEIGHTS/'adapter.pt'), sha256=digest(WEIGHTS/'adapter.pt')),
        parent_holds_GPU_mutex=True, workers_reacquire_mutex=False,
        wait_hours=args.wait_hours, inference_hours=args.max_hours,
        deadline_scope='per invocation: wait cap without GPU mutex; evaluation cap includes lock wait, preparation, inference and CPU metrics',
        reporting_groups=['REDS_fullview', 'UVG_crop'], independent_system_test=False,
        source_scope='REDS resized full field of view; UVG existing 512px spatial crops; historical exposure recorded per sample',
        semantic_supervision=False, semantic_fidelity_measured=False, no_model_promotion=True,
        native_UF_rate='native stream.bin bytes; helper JSON excluded', fixed_frame_index=8)
    immutable(run.root/'protocol.json', protocol)
    return protocol


def prepare_source(run, entry, *, verify_only=False):
    import numpy as np
    from demo.routervc_mixedview_data import load_rgb
    from demo.routervc_fullview_data import source_hash
    from demo.scalable_codec import atomic_npz
    from demo.scalable_format import frame_hash
    sample = entry['sample']; folder = run.root/'samples'/sample['sample_id']
    if not verify_only:
        folder.mkdir(parents=True, exist_ok=True)
    binding = dict(protocol_sha256=digest(run.root/'protocol.json'), source=entry)
    done = folder/'source.complete.json'
    if source_hash(sample) != entry['original_file_hashes']:
        raise ValueError('evaluation source files changed')
    if done.exists():
        saved = read(done)
        if saved['binding'] != binding:
            raise ValueError('source cache binding changed')
        verify_artifacts(folder, saved['artifacts'])
    else:
        if verify_only:
            raise ValueError('missing completed source cache; read-only verification may not prepare it')
        pixels = load_rgb(sample)
        if pixels.dtype != np.uint8 or pixels.shape != source_shape(sample):
            raise ValueError('prepared evaluation view differs from manifest geometry')
        atomic_npz(folder/'source.npz', source=pixels)
        save(done, dict(complete=True, binding=binding, rgb_sha256=frame_hash(pixels),
                        artifacts={'source.npz':digest(folder/'source.npz')}))
    return folder


def receiver_command(folder, point, protocol):
    if point['kind'] in ('router', 'off', 'repeat'):
        off = point['kind'] == 'off'
        argv = ['--worker', '--stream', folder/'stream.rtvc', '--output', folder/'fresh',
                '--enhancement', protocol['enhancement']['path'],
                '--adapter', '/missing/unused-G.pt' if off else protocol['adapter']['path']]
        if off:
            argv += ['--disable-generation']
        else:
            argv += ['--router', protocol['models']['arms'][point['arm']]['path']]
        return 'routervc_visual_decode.py', argv, not off
    return 'routervc_baselines.py', ['decode-uf' if point['kind'] == 'uf' else 'decode-g',
                '--output', folder, '--adapter', protocol['adapter']['path']], point['kind'] == 'full_g'


def prepare_point(run, sample_folder, point, protocol):
    from demo.conditioned_generation_pipeline import execute
    from demo.scalable_codec import atomic_bytes
    from demo import routervc_visual_format as fmt
    from demo import scalable_cooperation_format as cooperation
    from demo.routervc_encode import subset_bank
    folder = sample_folder/point['name']; folder.mkdir(exist_ok=True)
    immutable(folder/'job.json', dict(point=point, protocol_sha256=digest(run.root/'protocol.json'),
                                      source_sha256=digest(sample_folder/'source.npz')))
    prefix = sample_folder.name+'_'+point['name']
    if point['kind'] == 'router':
        execute(run, prefix+'_encode', 'routervc_visual_encode.py',
            ['--worker', '--input', sample_folder/'source.npz', '--output', folder,
             '--prepared-dir', sample_folder/'prepared', '--router', protocol['models']['arms'][point['arm']]['path'],
             '--adapter', protocol['adapter']['path'], '--enhancement', protocol['enhancement']['path'],
             '--e-ratio', point['ratio'], '--max-g', point['max_g'], '--boundary-lambda', 0.,
             '--mode', 'prefix', '--seed', 20261003])
    elif point['kind'] == 'uf':
        if (folder/'encode.json').exists():
            old = read(folder/'encode.json'); verify_artifacts(folder, old['artifacts'])
            if old['source_hash'] != digest(sample_folder/'source.npz') or old['qp'] != point['qp']:
                raise ValueError('UF encode source/QP differs')
        else:
            extra = ['--reuse-base', sample_folder/'prepared/base.acse'] if point['qp'] == 8 else []
            execute(run, prefix+'_encode', 'routervc_baselines.py', ['encode-uf', '--output', folder,
                '--source', sample_folder/'source.npz', '--source-hash', digest(sample_folder/'source.npz'),
                '--qp', point['qp'], *extra])
    else:
        if point['kind'] in ('off', 'repeat'):
            wire = (sample_folder/point['reference']/'stream.rtvc').read_bytes()
            target = folder/'stream.rtvc'
        else:
            meta = read(sample_folder/'prepared/complete.json')['binding']['source']['source_shape']
            control = fmt.generation_control(protocol['configs']['global_local_g8'], [], [], meta[0])
            control['generate'] = [[0, meta[0], 0, 0, meta[2], meta[1]]]
            wire = cooperation.wrap(subset_bank((sample_folder/'prepared/bank.acse').read_bytes(), []), control)
            target = folder/'stream.acsg'
        if target.exists() and target.read_bytes() != wire:
            raise ValueError('saved diagnostic/baseline stream differs')
        if not target.exists():
            atomic_bytes(target, wire)
    return folder


def validate_receiver(sample_folder, point, sample, protocol):
    import numpy as np
    from demo import routervc_visual_format as fmt
    from demo.scalable_format import frame_hash
    from demo.routervc_baselines import validate_uf_metadata
    folder = sample_folder/point['name']; decoded = read(folder/'fresh/decode.json')
    filename = 'stream.bin' if point['kind'] == 'uf' else 'stream.acsg' if point['kind'] == 'full_g' else 'stream.rtvc'
    stream = folder/filename
    if (decoded['source_frames_read'] or not decoded['base_reference_unchanged']
            or not decoded['outside_generate_exact'] or decoded['stream_sha256'] != digest(stream)):
        raise ValueError('fresh receiver identity/source-free invariant failed')
    ledger = byte_ledger(point['kind'], folder, decoded)
    with np.load(folder/'fresh/reconstruction.npz', allow_pickle=False) as cache:
        for field, key in [('reconstruction','output_hash'),('base','base_hash'),('enhanced','generation_input_hash')]:
            if cache[field].shape != source_shape(sample) or frame_hash(cache[field]) != decoded[key]:
                raise ValueError('fresh receiver pixel geometry/hash differs')
        if point['kind'] == 'off':
            np.testing.assert_array_equal(cache['reconstruction'], cache['enhanced'])
        if point['kind'] in ('uf', 'full_g'):
            np.testing.assert_array_equal(cache['base'], cache['enhanced'])
    if point['kind'] in ('router', 'repeat'):
        reference = point.get('reference', point['name'])
        encoded = read(sample_folder/reference/'encode.json')
        verify_artifacts(sample_folder/reference, encoded['artifacts'])
        config, _, _, _ = fmt.parse(stream.read_bytes())
        expected_config = protocol['configs'][f"{point['arm']}_g{point['max_g']}"]
        if (config != expected_config or decoded['config'] != config
                or decoded['route'] != encoded['expected_shared_route']
                or decoded['generation_input_hash'] != encoded['expected_mixed_hash']
                or decoded['base_hash'] != encoded['expected_base_hash']):
            raise ValueError('sender/receiver actual mixed-Y route or pixels differ')
    elif point['kind'] == 'uf':
        metadata = read(folder/'transmitted_meta.json')
        validate_uf_metadata(metadata, stream.read_bytes())
        if (metadata['qp'] != point['qp'] or metadata['base_rgb_sha256'] != decoded['base_hash']
                or digest(folder/'transmitted_meta.json') != decoded['metadata_sha256']):
            raise ValueError('native UF metadata differs')
    elif point['kind'] == 'full_g':
        from demo import scalable_cooperation_format as cooperation
        control, _, parsed, _ = cooperation.parse(stream.read_bytes())
        if decoded['actual_G_roi_calls'] != 1 or decoded['geometry'] != 'one_full_frame_ROI':
            raise ValueError('full-frame baseline must be exactly one G ROI')
        shape = source_shape(sample)
        if (parsed.packets or control['generate'] != [[0, shape[0], 0, 0, shape[2], shape[1]]]
                or decoded['config'] != control):
            raise ValueError('full-frame G control differs from its actual empty-E stream')
    if point['kind'] in ('off', 'repeat'):
        original = sample_folder/point['reference']
        reference = read(original/'fresh/decode.json')
        config, _, _, _ = fmt.parse(stream.read_bytes())
        if decoded['config'] != config:
            raise ValueError('smoke receiver configuration differs from actual stream')
        if stream.read_bytes() != (original/'stream.rtvc').read_bytes() or decoded['generation_input_hash'] != reference['generation_input_hash']:
            raise ValueError('smoke control is not the same actual stream/received pixels')
        if point['kind'] == 'off' and (decoded['generation_executed'] or decoded['generation_assets_validated']):
            raise ValueError('G-off used generation assets')
        if point['kind'] == 'repeat' and decoded['output_hash'] != reference['output_hash']:
            raise ValueError('repeated fresh decoder pixels differ')
    if point['kind'] != 'uf' or point['qp'] == 8:
        if decoded['base_hash'] != read(sample_folder/'prepared/complete.json')['base_rgb_sha256']:
            raise ValueError('shared QP8 base differs across methods')
    return decoded, ledger


def validate_result(root, sample_folder, point, sample, protocol):
    result = read(sample_folder/point['name']/'result.json')
    if (not result['complete'] or result['point'] != point
            or result.get('sample_id') != sample['sample_id']
            or result['protocol_sha256'] != digest(root/'protocol.json')):
        raise ValueError('completed evaluation point binding changed')
    verify_artifacts(sample_folder/point['name'], result['artifacts'])
    decoded, ledger = validate_receiver(sample_folder, point, sample, protocol)
    if result['decode'] != decoded or result['byte_ledger'] != ledger:
        raise ValueError('completed evaluation point receiver/bytes changed')
    if (result['bytes'] != ledger['bytes'] or not math.isclose(
            result['bpp'], 8*ledger['bytes']/math.prod(source_shape(sample)[:3]), rel_tol=1e-12)):
        raise ValueError('completed point rate differs from real bytes/valid geometry')
    return result


def evaluate_point(run, sample_folder, point, sample, protocol, metric):
    import numpy as np
    from PIL import Image
    from demo.conditioned_generation_pipeline import execute
    from demo.scalable_experiment import quality
    folder = sample_folder/point['name']
    if (folder/'result.json').exists():
        return validate_result(run.root, sample_folder, point, sample, protocol), metric
    began = time.monotonic(); run.check()
    folder = prepare_point(run, sample_folder, point, protocol)
    if not (folder/'fresh/decode.json').exists():
        (folder/'fresh').mkdir(exist_ok=True)
        script, argv, distributed = receiver_command(folder, point, protocol)
        execute(run, sample_folder.name+'_'+point['name']+'_fresh', script, argv, distributed=distributed)
    decoded, ledger = validate_receiver(sample_folder, point, sample, protocol)
    measurements = None
    if point['kind'] not in ('off', 'repeat'):
        if metric is None:
            import torch
            from demo.stage_c_three_path_roi_probe import LPIPSAlex
            torch.set_num_threads(4); metric = LPIPSAlex(True)
        run.check(); run.update(phase='CPU_metrics', sample=sample['sample_id'], point=point['name'])
        with np.load(sample_folder/'source.npz', allow_pickle=False) as cache:
            source = cache['source'].copy()
        with np.load(folder/'fresh/reconstruction.npz', allow_pickle=False) as cache:
            output = cache['reconstruction'].copy()
        measurements = quality(source, output, metric)
        temporary = folder/'fixed_frame.png.tmp'
        Image.fromarray(output[8]).save(temporary, format='PNG'); os.replace(temporary, folder/'fixed_frame.png')
    run.check()
    names = ['job.json', 'fresh/decode.json', 'fresh/reconstruction.npz']
    if measurements is not None:
        names.append('fixed_frame.png')
    names += ['stream.bin', 'transmitted_meta.json', 'encode.json'] if point['kind'] == 'uf' else [
        'stream.acsg' if point['kind'] == 'full_g' else 'stream.rtvc']
    if point['kind'] == 'router':
        names += ['encode.json', 'encode.request.json']
    result = dict(complete=True, point=point, sample_id=sample['sample_id'],
        protocol_sha256=digest(run.root/'protocol.json'), byte_ledger=ledger, bytes=ledger['bytes'],
        bpp=8*ledger['bytes']/math.prod(source_shape(sample)[:3]), quality=measurements, decode=decoded,
        elapsed_seconds_this_completion_attempt=time.monotonic()-began,
        artifacts={name:digest(folder/name) for name in names})
    save(folder/'result.json', result)
    return validate_result(run.root, sample_folder, point, sample, protocol), metric


def smoke(run, protocol, metric):
    entry = protocol['sources'][0]; sample = entry['sample']; folder = prepare_source(run, entry)
    point = point_plan()[0]; done = run.root/'smoke.complete.json'
    first_result, metric = evaluate_point(run, folder, point, sample, protocol, metric)
    controls = [dict(point, name='smoke_repeat', kind='repeat', reference=point['name']),
                dict(point, name='smoke_G_off', kind='off', reference=point['name'])]
    for control in controls:
        _, metric = evaluate_point(run, folder, control, sample, protocol, metric)
    value = dict(complete=True, protocol_sha256=digest(run.root/'protocol.json'),
        sample_id=sample['sample_id'], actual_mixed_route_equal=True, repeat_pixels_exact=True,
        same_wire_G_off_exact=True, formal_model_only=True, quality_threshold=None,
        generation_executed=first_result['decode']['generation_executed'],
        actual_G_calls=len(first_result['decode']['route']['indices']),
        received_E_coverage=first_result['decode']['route']['coverage'],
        generation_scope='If zero G cells were selected, this smoke covers routing/repeat/G-off but not the new receiver G branch.',
        artifacts={str((folder/p['name']/'result.json').relative_to(run.root)):
                   digest(folder/p['name']/'result.json') for p in [point, *controls]})
    immutable(done, value)
    return metric


def check_prefixes(folder):
    for arm in ARMS:
        for cap in (4, 8):
            low = (folder/f'{arm}_e0.25_g{cap}/stream.rtvc').read_bytes()
            high = (folder/f'{arm}_e0.5_g{cap}/stream.rtvc').read_bytes()
            if not high.startswith(low):
                raise ValueError('same-arm/G-cap E budgets are not literal byte prefixes')


def finish(run, protocol, records):
    """A compact CSV/JSON and one fixed comparison per sample, not a new atlas."""
    import numpy as np
    from PIL import Image, ImageDraw
    from demo.routervc_fullview_probe_report import export_display, atomic_text
    pairs = {(r['sample_id'], r['point']['name']) for r in records}
    expected = {(v['sample']['sample_id'], p['name']) for v in protocol['sources'] for p in point_plan()}
    if len(records) != 169 or pairs != expected:
        raise ValueError('formal summary requires all 13 samples x 13 points')
    rows, images = [], []
    by_id = {v['sample']['sample_id']:v['sample'] for v in protocol['sources']}
    for result in records:
        sample = by_id[result['sample_id']]; point = result['point']; decoded = result['decode']
        group = 'REDS_fullview' if sample['dataset'] == 'REDS' else 'UVG_crop'
        rows.append(dict(sample_id=sample['sample_id'], group=group, sequence=sample['sequence'],
            history=sample['history'], component_training_sequence=sample.get('component_training_sequence', False),
            point=point['name'], bytes=result['bytes'], bpp=result['bpp'], **result['quality'],
            receiver_seconds=decoded['seconds'], policy_seconds=decoded.get('policy_seconds'),
            peak_cuda_bytes=decoded.get('peak_cuda_allocated_bytes'),
            result_path=f"samples/{sample['sample_id']}/{point['name']}/result.json"))
    report = run.root/'report'; report.mkdir(exist_ok=True)
    for sample in by_id.values():
        folder = run.root/'samples'/sample['sample_id']; check_prefixes(folder)
        path = report/(sample['sample_id']+'.jpg')
        display_path = report/(sample['sample_id']+'.display.json')
        if path.exists() and display_path.exists():
            if digest(path) != read(display_path)['sha256']:
                raise ValueError('display-only image changed')
        else:
            with np.load(folder/'source.npz', allow_pickle=False) as cache:
                source = Image.fromarray(cache['source'][8])
            names = ['uf_qp8', 'uf_qp32', 'global_local_e0.5_g8', 'local_e0.5_g8', 'wholeframe_g_one_roi']
            panels = [('Source', source)] + [(name, Image.open(folder/name/'fixed_frame.png')) for name in names]
            width, height = source.size
            canvas = Image.new('RGB', (width*3, (height+28)*2), 'white'); draw = ImageDraw.Draw(canvas)
            for i, (name, picture) in enumerate(panels):
                x, y = (i%3)*width, (i//3)*(height+28)
                draw.text((x+7, y+6), name, fill='black'); canvas.paste(picture, (x, y+28))
                if name != 'Source':
                    picture.close()
            temporary = report/(sample['sample_id']+'.png.tmp')
            canvas.save(temporary, format='PNG')
            display = export_display(temporary, path)
            # This is a transient composition, not any measured PNG/reconstruction.
            temporary.unlink()
            display.pop('source_path'); display.pop('source_sha256')
            immutable(display_path, display)
        images.append(dict(sample_id=sample['sample_id'], path=str(path.relative_to(run.root)),
                           sha256=digest(path), display_only=True, metrics_input=False,
                           original_fixed_pngs='samples/'+sample['sample_id']+'/*/fixed_frame.png'))
    groups = {}
    for group in ('REDS_fullview', 'UVG_crop'):
        groups[group] = {}
        for point in point_plan():
            values = [r for r in rows if r['group'] == group and r['point'] == point['name']]
            groups[group][point['name']] = dict(windows=len(values),
                **{key:sum(r[key] for r in values)/len(values) for key in
                   ('bpp', 'lpips_alex', 'psnr_db', 'temporal_delta_mae', 'receiver_seconds')})
    table = io.StringIO(newline=''); writer = csv.DictWriter(table, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    if (report/'points.csv').exists():
        if (report/'points.csv').read_text() != table.getvalue().replace('\r\n', '\n'):
            raise ValueError('saved report CSV differs from measured points')
    else:
        atomic_text(report/'points.csv', table.getvalue())
    summary = dict(complete=True, schema=SCHEMA, protocol_sha256=digest(run.root/'protocol.json'),
        samples=13, points=169, rows=rows, group_means=groups, fixed_visuals=images,
        historical_exposure=True, independent_system_test=False, semantic_fidelity_measured=False,
        no_model_promotion=True, native_UF_sidecar_excluded=True,
        averaging='equal window means, reported separately by REDS full-view and UVG crop',
        timing_scope='recorded fresh receiver costs include loading; not a controlled latency benchmark',
        extra_smoke_decodes=2, smoke_sha256=digest(run.root/'smoke.complete.json'),
        artifact_hashes={r['result_path']:digest(run.root/r['result_path']) for r in rows})
    immutable(run.root/'summary.json', summary)
    assets = [run.root/'summary.json', run.root/'smoke.complete.json', report/'points.csv']
    assets += [report/p.name for p in sorted(report.glob('*.jpg'))]
    assets += list(sorted(report.glob('*.display.json')))
    immutable(run.root/'complete.json', dict(complete=True, protocol_sha256=digest(run.root/'protocol.json'),
        artifacts={str(p.relative_to(run.root)):digest(p) for p in assets}, points=169,
        semantic_supervision=False, no_model_promotion=True))
    return summary


def evaluate(run, protocol, *, verify_only=False):
    metric, records = None, []
    if verify_only:
        complete = read(run.root/'complete.json'); verify_artifacts(run.root, complete['artifacts'])
        if complete['protocol_sha256'] != digest(run.root/'protocol.json'):
            raise ValueError('completed evaluation protocol changed')
        summary = read(run.root/'summary.json'); verify_artifacts(run.root, summary['artifact_hashes'])
        done = read(run.root/'smoke.complete.json')
        verify_artifacts(run.root, done['artifacts'])
        entry = protocol['sources'][0]; sample = entry['sample']; folder = run.root/'samples'/sample['sample_id']
        first = point_plan()[0]
        for name, kind in [('smoke_repeat','repeat'), ('smoke_G_off','off')]:
            validate_result(run.root, folder, dict(first, name=name, kind=kind, reference=first['name']), sample, protocol)
    else:
        metric = smoke(run, protocol, metric)
    for entry in protocol['sources']:
        sample = entry['sample']; folder = prepare_source(run, entry, verify_only=verify_only)
        for point in point_plan():
            run.check()
            if verify_only:
                result = validate_result(run.root, folder, point, sample, protocol)
            else:
                result, metric = evaluate_point(run, folder, point, sample, protocol, metric)
            records.append(result)
            run.update(phase='point_complete', sample=sample['sample_id'], point=point['name'],
                       completed=len(records), total=169)
        check_prefixes(folder)
    if verify_only:
        return summary
    return finish(run, protocol, records)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=REVISION/'mixedview_manifest.json')
    parser.add_argument('--models', type=Path, default=REVISION/'visual_router')
    parser.add_argument('--output', type=Path, default=REVISION/'visual_evaluation')
    parser.add_argument('--wait-hours', type=float, default=48.)
    parser.add_argument('--max-hours', type=float, default=12.)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args(argv)
    if not os.environ.get('TMUX'):
        raise RuntimeError('visual evaluation/wait requires tmux')
    if (not math.isfinite(args.wait_hours) or not 0 < args.wait_hours <= 48
            or not math.isfinite(args.max_hours) or not 0 < args.max_hours <= 12):
        raise ValueError('wait-hours must be in (0,48], evaluation max-hours in (0,12]')
    # Fail early on unavailable/wrong split data; never launch large downloads.
    selected_samples(read(args.manifest))
    verified_models = None
    if args.verify_only:
        # Verification may record diagnostics, but must not create its inputs or
        # wait for training to turn an incomplete run into a verifiable one.
        for name in ('request.json', 'protocol.json', 'complete.json'):
            if not (args.output/name).is_file():
                raise ValueError(f'verify-only requires existing {name}')
        verified_models = completed_models(args.models)
        if verified_models is None:
            raise ValueError('verify-only requires completed formal models; it will not wait')
    from demo.chunk_enhancement_experiment import Run
    from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
    run = Run(SimpleNamespace(output=args.output, command='visual_evaluate',
                              max_hours=args.wait_hours+args.max_hours))
    deadline = PhaseDeadline(run.check); run.check = deadline.check
    deadline.begin('waiting_for_formal_models', args.wait_hours*3600)
    run.thread.start()
    previous_parent = os.environ.get('ROUTERVC_VISUAL_PARENT')
    try:
        request = dict(models_root=str(args.models.resolve()), manifest_sha256=digest(args.manifest),
                       code=code_hashes(), samples=13, points=169, extra_smoke_decodes=2)
        immutable(run.root/'request.json', request)
        models = verified_models if args.verify_only else wait_for_models(run, args.models)
        if request['code'] != code_hashes() or request['manifest_sha256'] != digest(args.manifest):
            raise ValueError('evaluation code/manifest changed while waiting for formal training')
        deadline.begin('bounded_evaluation', args.max_hours*3600)
        protocol = setup(run, args, models)
        if args.verify_only or (run.root/'complete.json').exists():
            result = evaluate(run, protocol, verify_only=True)
        else:
            with exclusive_native_evaluation(run):
                os.environ['ROUTERVC_VISUAL_PARENT'] = str(os.getpid())
                result = evaluate(run, protocol)
        print(json.dumps(dict(complete=result['complete'], samples=result['samples'], points=result['points'])), flush=True)
    except BaseException as error:
        save(run.root/'last_failure.json', dict(error=repr(error), phase=deadline.phase, progress=run.progress))
        raise
    finally:
        if previous_parent is None:
            os.environ.pop('ROUTERVC_VISUAL_PARENT', None)
        else:
            os.environ['ROUTERVC_VISUAL_PARENT'] = previous_parent
        run.log_resources(); run.stop.set(); run.thread.join(timeout=3)


if __name__ == '__main__':
    main()
