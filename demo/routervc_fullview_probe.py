"""One 1024x576 full-view calibration, not native-resolution/test-set evidence.

Ten fresh receivers: four existing Router configurations, same-wire G-off,
four native UF QPs and one-ROI whole-frame G. The parent alone holds the global
GPU mutex. Existing pinned workers are reused without entering their queues.
Completed points are checked without inference, metrics or timing rewrites.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
RUNS = Path('/root/autodl-fs/DCVC/runs')
WEIGHTS = RUNS/'a800_online_eg_20261002/joint'
ROUTERS = RUNS/'a800_four_state_router_20261002'
SHAPE = (17, 576, 1024, 3)
SCHEMA = 'routervc-fullview-probe-v1'


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def immutable(path, value):
    if Path(path).exists():
        if read(path) != value:
            raise ValueError(f'changed bound inputs/configuration: {path}; use a new output')
    else:
        save(path, value)


def verify_artifacts(root, artifacts):
    for name, expected in artifacts.items():
        if digest(Path(root)/name) != expected:
            raise ValueError(f'changed fullview artifact: {Path(root)/name}')


def validate_source_preparation(input_path, source_record):
    """A PNG preparation certificate must describe the actual input directory."""
    input_path, source_record = Path(input_path), Path(source_record)
    if not input_path.is_dir():
        raise ValueError('source preparation records currently require a PNG input directory, not NPZ')
    record = read(source_record)
    if (not record.get('complete') or not isinstance(record.get('frames_dir'), str)
            or Path(record['frames_dir']).resolve() != input_path.resolve()
            or record.get('frame_count') != SHAPE[0]):
        raise ValueError('source preparation frames_dir/count do not describe the supplied input')
    artifacts = record.get('artifacts', {})
    recorded_pngs = {str((source_record.parent/name).resolve()): value
                     for name, value in artifacts.items() if Path(name).suffix.lower() == '.png'}
    actual_pngs = {str(p.resolve()): digest(p) for p in sorted(input_path.glob('*.png'))}
    if len(actual_pngs) != SHAPE[0] or recorded_pngs != actual_pngs:
        raise ValueError('source preparation PNG artifact set does not match the exact input frames')
    verify_artifacts(source_record.parent, artifacts)
    sample = record.get('binding', {}).get('sample', {})
    transform = sample.get('transform', {})
    if (not transform or record.get('transform') != transform
            or transform.get('valid_rect') != [0, 0, SHAPE[2], SHAPE[1]]
            or transform.get('coded_size') != [SHAPE[2], SHAPE[1]]):
        raise ValueError('source preparation must describe the full valid 1024x576 rectangle without padding')
    verified = sample.get('whole_frame') is True and transform.get('crop') is None
    return dict(path=str(source_record.resolve()), sha256=digest(source_record), record=record), verified


def validate_code_compatibility(old_code, new_code):
    self_name = 'demo/routervc_fullview_probe.py'
    if self_name not in old_code or self_name not in new_code:
        raise ValueError('runner snapshot identity missing from dependency set')
    old = {k:v for k,v in old_code.items() if k != self_name}
    new = {k:v for k,v in new_code.items() if k != self_name}
    if old != new:
        raise ValueError('non-runner dependency key set or content changed')


def copy_file(src, dst):
    """Link immutable binary/media payloads; keep every JSON copy independent."""
    src, dst = Path(src), Path(dst)
    source_hash = digest(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if digest(dst) != source_hash:
            raise ValueError(f'recovery destination differs: {dst}')
        return
    if src.suffix.lower() in {'.npz', '.bin', '.acse', '.acsg', '.rtvc', '.png'}:
        try:
            os.link(src, dst)
        except OSError:
            pass  # Cross-filesystem/unsupported hard links use an atomic copy.
        else:
            if digest(dst) != source_hash:
                raise ValueError(f'immutable source changed while linking: {src}')
            return
    temporary = dst.with_name(dst.name+'.import.tmp')
    shutil.copy2(src, temporary)
    if digest(temporary) != source_hash:
        raise ValueError(f'immutable source changed while copying: {src}')
    os.replace(temporary, dst)


def validate_prepared_binding(prepared, base, protocol, source_file_hash):
    """Pure metadata validation; historical source path may name the first run."""
    binding = prepared.get('binding', {})
    source = binding.get('source', {})
    if (source.get('kind') != 'npz' or source.get('key') != 'source'
            or source.get('start') != 0 or source.get('requested_count') is not None
            or not isinstance(source.get('path'), str) or not Path(source['path']).is_absolute()
            or source.get('file_sha256') != source_file_hash
            or source.get('source_shape') != protocol['source_shape']
            or source.get('source_rgb_sha256') != protocol['source_rgb_sha256']):
        raise ValueError('prepared source hash/shape/selection differs from protocol source')
    expected = dict(enhancement=protocol['weights']['enhancement']['sha256'],
                    model_i=protocol['weights']['uf_i']['sha256'],
                    model_p=protocol['weights']['uf_p']['sha256'])
    if any(binding.get(k) != v for k, v in expected.items()):
        raise ValueError('prepared model identities differ from protocol weights')
    encoder_code = protocol['code'].get('demo/routervc_encode.py')
    if (not isinstance(encoder_code, str) or not encoder_code
            or binding.get('version') != 1 or binding.get('base_qp') != 8
            or binding.get('qstep') != 1. or binding.get('padded_frames') != 0
            or binding.get('source_code') != encoder_code):
        raise ValueError('prepared codec/source-code profile differs from fixed protocol')
    if base.get('binding') != binding:
        raise ValueError('base.complete binding differs from prepared candidate binding')
    if (prepared.get('padding') != dict(temporal_frames=0, spatial_pixels=0,
                                       input_shape=protocol['source_shape'])
            or base.get('base_rgb_sha256') != prepared.get('base_rgb_sha256')
            or base.get('seconds') != prepared.get('base_seconds')):
        raise ValueError('prepared base geometry/pixels/original timing disagree')
    if (set(prepared.get('artifacts', {})) != {'base.acse', 'bank.acse', 'candidates.npz'}
            or set(base.get('artifacts', {})) != {'base.acse'}
            or base['artifacts']['base.acse'] != prepared['artifacts']['base.acse']):
        raise ValueError('prepared/base artifact sets disagree')


def validate_prepared(root, protocol):
    """Read-only cache audit, including actual source pixels and native base bytes."""
    import numpy as np
    from demo.scalable_format import frame_hash, parse
    from demo.routervc_encode import bank_info
    root = Path(root)
    source_record = read(root/'source.complete.json')
    if (source_record['protocol_sha256'] != digest(root/'protocol.json')
            or set(source_record['artifacts']) != {'source.npz'}):
        raise ValueError('source cache completion is bound to another protocol')
    verify_artifacts(root, source_record['artifacts'])
    prepared = read(root/'prepared/complete.json')
    base = read(root/'prepared/base.complete.json')
    validate_prepared_binding(prepared, base, protocol, digest(root/'source.npz'))
    verify_artifacts(root/'prepared', base['artifacts'])
    verify_artifacts(root/'prepared', prepared['artifacts'])
    with np.load(root/'source.npz', allow_pickle=False) as cache:
        validate_shape(cache['source'].shape)
        if cache['source'].dtype != np.uint8 or frame_hash(cache['source']) != protocol['source_rgb_sha256']:
            raise ValueError('actual prepared source pixels differ from protocol')
    native = parse((root/'prepared/base.acse').read_bytes())
    bank_bytes = (root/'prepared/bank.acse').read_bytes()
    info = bank_info(bank_bytes)
    bank = info['parsed']
    if (info['rois'] != prepared['rois'] or info['e_bytes'] != prepared['e_packet_bytes']
            or [list(c) for c in info['chunks']] != prepared['actual_chunks']
            or len(bank_bytes) != prepared['candidate_bank_bytes']
            or bank.base_end != prepared['base_container_bytes']
            or len(bank.base) != prepared['base_native_bytes']):
        raise ValueError('prepared packet geometry/real byte accounting changed')
    expected_meta = dict(frame_count=SHAPE[0], height=SHAPE[1], width=SHAPE[2], base_qp=8,
                         model_i_sha256=protocol['weights']['uf_i']['sha256'],
                         model_p_sha256=protocol['weights']['uf_p']['sha256'],
                         base_rgb_sha256=prepared['base_rgb_sha256'])
    if (native.packets or native.base != bank.base or
            any(p.meta.get(k) != v for p in (native, bank) for k, v in expected_meta.items())
            or bank.meta.get('enhancement_model_sha256') != protocol['weights']['enhancement']['sha256']):
        raise ValueError('prepared bank/native base metadata or payload differs')
    with np.load(root/'prepared/candidates.npz', allow_pickle=False) as cache:
        for key, expected_hash in [('base', prepared['base_rgb_sha256']),
                                   ('all_E', prepared['all_E_rgb_sha256'])]:
            validate_shape(cache[key].shape)
            if cache[key].dtype != np.uint8 or frame_hash(cache[key]) != expected_hash:
                raise ValueError('prepared candidate pixels differ from recorded entropy decode')
    return prepared


def validate_shape(shape):
    if tuple(shape) != SHAPE:
        raise ValueError(f'this bounded probe requires explicit prepared RGB shape {SHAPE}; '
                         'it does not crop, resize, pad or truncate input')


def point_plan():
    routes = [dict(name=f'{arm}_e{ratio:g}_g8', kind='router', arm=arm, ratio=ratio)
              for arm in ('context', 'local') for ratio in (.25, .5)]
    return routes + [dict(name='context_e0.5_g_off', kind='off',
                         reference='context_e0.5_g8', arm='context', ratio=.5)] + [
        dict(name=f'uf_qp{qp}', kind='uf', qp=qp) for qp in (8, 16, 24, 32)
    ] + [dict(name='wholeframe_g_one_roi', kind='full_g')]


def byte_ledger(kind, folder, report):
    """UF sidecar belongs to experiment verification, not native codec rate."""
    folder = Path(folder)
    if kind == 'uf':
        native = (folder/'stream.bin').stat().st_size
        sidecar = (folder/'transmitted_meta.json').stat().st_size
        if (report['native_bytes'] != native or report['metadata_bytes'] != sidecar
                or report['total_bytes'] != native + sidecar):
            raise ValueError('native UF helper accounting differs from actual files')
        return dict(bytes=native, native_bytes=native, audit_sidecar_bytes=sidecar,
                    helper_total_bytes=native+sidecar,
                    rate_scope='native stream.bin only; includes its SPS/NAL framing; audit JSON excluded')
    stream = folder/('stream.acsg' if kind == 'full_g' else 'stream.rtvc')
    actual = stream.stat().st_size
    if report['total_bytes'] != actual:
        raise ValueError('RouterVC/G actual on-disk bytes differ from receiver')
    native = report.get('base_bytes', report.get('native_bytes'))
    return dict(bytes=actual, native_bytes=native, audit_sidecar_bytes=0,
                rate_scope='complete stream including every container/control/header/E byte')


def _code_hashes():
    from demo.routervc import CODE
    from demo.routervc_baselines import CODE as BASELINE_CODE
    from demo.chunk_enhancement_experiment import CODE_FILES
    paths = {f'demo/{name}' for name in (*CODE, *BASELINE_CODE)} | set(CODE_FILES)
    paths |= {'demo/routervc_fullview_probe.py', 'demo/run_routervc_fullview_probe.sh',
              'demo/conditioned_generation_pipeline.py', 'demo/scalable_experiment.py',
              'demo/chunk_enhancement_evaluate.py', 'demo/internal_condition_model.py',
              'demo/feature_interface_model.py'}
    return {name: digest(REPO/name) for name in sorted(paths)}


def setup(run, args):
    import numpy as np
    from demo.routervc_encode import load_input
    from demo.routervc import validate_generation_geometry
    from demo.scalable_codec import atomic_npz
    from demo.scalable_format import frame_hash
    from demo import routervc_format as fmt
    source, provenance = load_input(args.input)
    validate_shape(source.shape)
    validate_generation_geometry(source.shape, 8)
    weights = dict(enhancement=WEIGHTS/'enhancement.pt', adapter=WEIGHTS/'adapter.pt',
                   context=ROUTERS/'context/model.pt', local=ROUTERS/'local/model.pt',
                   uf_i=REPO/'checkpoints/cvpr2026_image.pth.tar',
                   uf_p=REPO/'checkpoints/cvpr2026_video_hts.pth.tar')
    configs = {arm: fmt.make_config(weights[arm], weights['adapter'], max_g=8,
                                    boundary_lambda=0., seed=20261003)
               for arm in ('context', 'local')}
    source_record = args.source_record
    if source_record is None and args.input.is_dir() and (args.input.parent/'complete.json').exists():
        source_record = args.input.parent/'complete.json'
    preparation = None
    whole_view_verified = False
    if source_record is not None:
        preparation, whole_view_verified = validate_source_preparation(args.input, source_record)
    protocol = dict(schema=SCHEMA, source=provenance, source_shape=list(source.shape),
                    source_rgb_sha256=frame_hash(source), source_preparation=preparation,
                    source_scope='prepared 1024x576; not native resolution; no transform in probe',
                    whole_view_provenance_verified=whole_view_verified,
                    role='single-window mechanism/calibration; not representative dataset evaluation',
                    weights={k: dict(path=str(p.resolve()), sha256=digest(p)) for k, p in weights.items()},
                    configs=configs, code=_code_hashes(), points=point_plan(),
                    parent_holds_global_GPU_mutex=True, children_reacquire_mutex=False,
                    router_boundary_lambda=0., router_allocation='prefix', no_model_promotion=True,
                    native_UF_rate='stream.bin bytes; old helper JSON is verification-only',
                    source_free_receivers=True, fixed_visual_frame=8)
    immutable(run.root/'protocol.json', protocol)
    prepared = run.root/'source.complete.json'
    if prepared.exists():
        record = read(prepared)
        if record['protocol_sha256'] != digest(run.root/'protocol.json'):
            raise ValueError('cached source protocol changed')
        verify_artifacts(run.root, record['artifacts'])
    else:
        atomic_npz(run.root/'source.npz', source=source)
        save(prepared, dict(protocol_sha256=digest(run.root/'protocol.json'),
                            artifacts={'source.npz': digest(run.root/'source.npz')}))
    with np.load(run.root/'source.npz', allow_pickle=False) as cache:
        if frame_hash(cache['source']) != protocol['source_rgb_sha256']:
            raise ValueError('source cache pixels differ from prepared input')
    return protocol


def encode_arguments(root, point):
    return ['encode', '--input', root/'source.npz', '--output', root/point['name'],
            '--prepared-dir', root/'prepared', '--router', ROUTERS/point['arm']/'model.pt',
            '--adapter', WEIGHTS/'adapter.pt', '--enhancement', WEIGHTS/'enhancement.pt',
            '--e-ratio', point['ratio'], '--max-g', 8, '--boundary-lambda', 0,
            '--mode', 'prefix', '--seed', 20261003]


def worker_encode(argv):
    """Private child entry: parent owns the mutex and controls the deadline."""
    if not os.environ.get('TMUX') or not os.environ.get('ROUTERVC_FULLVIEW_PARENT'):
        raise RuntimeError('worker-encode must be spawned by the supervised tmux parent')
    from demo.routervc import encode, parser as codec_parser
    from demo.scalable_experiment import check_space
    args = codec_parser().parse_args(argv)
    run = SimpleNamespace(root=args.output, gpu_wait_seconds=0., check=check_space,
                          update=lambda **kw: print(json.dumps(kw), flush=True))
    encode(args, run)  # function, deliberately NOT routervc.main() / lock-owning CLI


def prepare_point(run, point, protocol):
    from demo.conditioned_generation_pipeline import execute
    from demo.scalable_codec import atomic_bytes
    from demo import routervc_format as fmt
    from demo import scalable_cooperation_format as cooperation
    folder = run.root/point['name']
    folder.mkdir(exist_ok=True)
    immutable(folder/'job.json', dict(point=point, protocol_sha256=digest(run.root/'protocol.json')))
    kind = point['kind']
    if kind == 'router':
        # Re-entry also validates the unchanged bank; completed encode is no-inference.
        execute(run, point['name']+'_encode', 'routervc_fullview_probe.py',
                ['worker-encode', *encode_arguments(run.root, point)])
    elif kind == 'off':
        wire = (run.root/point['reference']/'stream.rtvc').read_bytes()
        path = folder/'stream.rtvc'
        if path.exists() and path.read_bytes() != wire:
            raise ValueError('same-wire G-off ablation changed')
        if not path.exists():
            atomic_bytes(path, wire)
    elif kind == 'uf':
        done = folder/'encode.json'
        if done.exists():
            old = read(done)
            verify_artifacts(folder, old['artifacts'])
            if old['source_hash'] != digest(run.root/'source.npz') or old['qp'] != point['qp']:
                raise ValueError('UF encoded source/QP changed')
        else:
            extra = ['--reuse-base', run.root/'prepared/base.acse'] if point['qp'] == 8 else []
            execute(run, point['name']+'_encode', 'routervc_baselines.py',
                    ['encode-uf', '--output', folder, '--source', run.root/'source.npz',
                     '--source-hash', digest(run.root/'source.npz'), '--qp', point['qp'], *extra])
    else:
        config = protocol['configs']['context']
        control = fmt.generation_control(config, [], [], SHAPE[0])
        control['generate'] = [[0, SHAPE[0], 0, 0, SHAPE[2], SHAPE[1]]]
        from demo.routervc_encode import subset_bank
        # Public prepare retains legacy base.acse; G cooperation requires the
        # compact ACSE2 prefix from the candidate bank, even when E is empty.
        wire = cooperation.wrap(subset_bank((run.root/'prepared/bank.acse').read_bytes(), []), control)
        path = folder/'stream.acsg'
        if path.exists() and path.read_bytes() != wire:
            raise ValueError('full-frame G baseline wire changed')
        if not path.exists():
            atomic_bytes(path, wire)


def receiver_command(root, point):
    folder = root/point['name']
    if point['kind'] in ('router', 'off'):
        off = point['kind'] == 'off'
        argv = ['--stream', folder/'stream.rtvc', '--output', folder/'fresh',
                '--enhancement', WEIGHTS/'enhancement.pt',
                '--adapter', Path('/missing/fullview-generator.pt') if off else WEIGHTS/'adapter.pt',
                '--router', Path('/missing/fullview-router.pt') if off else ROUTERS/point['arm']/'model.pt']
        if off:
            argv += ['--disable-generation']
        return 'routervc_decode.py', argv, not off
    return 'routervc_baselines.py', ['decode-uf' if point['kind'] == 'uf' else 'decode-g',
                                    '--output', folder, '--adapter', WEIGHTS/'adapter.pt'], point['kind'] != 'uf'


def validate_receiver(root, point):
    import numpy as np
    from demo.scalable_format import frame_hash
    from demo import routervc_format as fmt
    folder = root/point['name']
    d = read(folder/'fresh/decode.json')
    stream = folder/('stream.bin' if point['kind'] == 'uf' else
                     'stream.acsg' if point['kind'] == 'full_g' else 'stream.rtvc')
    if (d['source_frames_read'] or not d['base_reference_unchanged']
            or not d['outside_generate_exact'] or d['stream_sha256'] != digest(stream)):
        raise ValueError('fresh source-free receiver invariants failed')
    ledger = byte_ledger(point['kind'], folder, d)
    with np.load(folder/'fresh/reconstruction.npz', allow_pickle=False) as cache:
        for name, field in [('reconstruction', 'output_hash'), ('base', 'base_hash'),
                            ('enhanced', 'generation_input_hash')]:
            validate_shape(cache[name].shape)
            if frame_hash(cache[name]) != d[field]:
                raise ValueError(f'fresh pixels changed: {name}')
        if point['kind'] == 'off':
            np.testing.assert_array_equal(cache['reconstruction'], cache['enhanced'])
        if point['kind'] in ('uf', 'full_g'):
            np.testing.assert_array_equal(cache['base'], cache['enhanced'])
    if point['kind'] == 'router':
        encoded = read(folder/'encode.json')
        verify_artifacts(folder, encoded['artifacts'])
        config, _, _, _ = fmt.parse(stream.read_bytes())
        if (d['config'] != config or d['route'] != encoded['expected_shared_route']
                or d['generation_input_hash'] != encoded['expected_mixed_hash']
                or d['base_hash'] != encoded['expected_base_hash']):
            raise ValueError('sender and fresh receiver route/pixels differ')
    elif point['kind'] == 'off':
        reference = read(root/point['reference']/'fresh/decode.json')
        if (d['generation_input_hash'] != reference['generation_input_hash']
                or d['generation_executed'] or d.get('generation_assets_validated')
                or stream.read_bytes() != (root/point['reference']/'stream.rtvc').read_bytes()):
            raise ValueError('model-free same-wire G-off failed')
    elif point['kind'] == 'uf':
        from demo.routervc_baselines import validate_uf_metadata
        meta = read(folder/'transmitted_meta.json')
        validate_uf_metadata(meta, stream.read_bytes())
        if (meta['qp'] != point['qp'] or meta['base_rgb_sha256'] != d['base_hash']
                or digest(folder/'transmitted_meta.json') != d['metadata_sha256']):
            raise ValueError('UF verification metadata changed')
    elif d['actual_G_roi_calls'] != 1 or d['geometry'] != 'one_full_frame_ROI':
        raise ValueError('whole-frame baseline is not exactly one ROI')
    if point['kind'] != 'uf' or point['qp'] == 8:
        if d['base_hash'] != read(root/'prepared/complete.json')['base_rgb_sha256']:
            raise ValueError('shared native QP8 reconstruction differs')
    return d, ledger


def validate_result(root, point, protocol_hash):
    folder = root/point['name']
    result = read(folder/'result.json')
    if result['point'] != point or result['protocol_sha256'] != protocol_hash:
        raise ValueError('completed point protocol changed')
    verify_artifacts(folder, result['artifacts'])
    decoded, ledger = validate_receiver(root, point)
    if result['decode'] != decoded or result['byte_ledger'] != ledger:
        raise ValueError('completed point bytes/receiver record changed')
    return result


def validate_summary(root, records, protocol_hash):
    summary = read(root/'summary.json')
    complete = read(root/'complete.json')
    if (summary['protocol_sha256'] != protocol_hash or summary['records'] != records
            or complete['protocol_sha256'] != protocol_hash
            or complete['summary_sha256'] != digest(root/'summary.json')
            or summary['fixed_frame_sha256'] != digest(root/'fixed_frame_comparison.png')):
        raise ValueError('completed summary/visual/protocol changed')
    verify_artifacts(root, summary['artifact_hashes'])
    prepared = validate_prepared(root, read(root/'protocol.json'))
    if summary['candidate_preparation'] != prepared:
        raise ValueError('completed candidate preparation record changed')


def import_completed(previous, run, protocol):
    """Reuse verified completed points across a runner-only audit/fix revision.

    Original results/timings remain untouched. New records explicitly cite the
    old result hash; only missing points run again. No baseline metric is copied
    into a different source, model, seed, shape or operating-point configuration.
    """
    previous = Path(previous).resolve()
    if previous == run.root.resolve():
        raise ValueError('recovery must use a new output directory')
    old = read(previous/'protocol.json')
    for key in ('source', 'source_shape', 'source_rgb_sha256', 'source_preparation', 'weights', 'configs', 'points'):
        if old[key] != protocol[key]:
            raise ValueError(f'previous probe differs in {key}')
    validate_code_compatibility(old['code'], protocol['code'])
    snapshot = previous/'runner_before_baseline_fix.py'
    if digest(snapshot) != old['code']['demo/routervc_fullview_probe.py']:
        raise ValueError('missing exact previous runner snapshot')
    receipt = dict(previous=str(previous), previous_protocol=digest(previous/'protocol.json'),
                   current_protocol=digest(run.root/'protocol.json'), previous_runner=digest(snapshot),
                   purpose='reuse measured points after runner-only recovery/audit hardening')
    immutable(run.root/'import.json', receipt)
    prepared = validate_prepared(previous, old)
    copy_file(previous/'source.npz', run.root/'source.npz')
    immutable(run.root/'source.complete.json', dict(protocol_sha256=receipt['current_protocol'],
              artifacts={'source.npz':digest(run.root/'source.npz')}))
    for name in (*prepared['artifacts'], 'complete.json', 'base.complete.json'):
        copy_file(previous/'prepared'/name, run.root/'prepared'/name)
    for point in point_plan():
        folder = previous/point['name']
        if not (folder/'result.json').exists():
            continue
        original = validate_result(previous, point, receipt['previous_protocol'])
        target = run.root/point['name']
        for name in original['artifacts']:
            if name != 'job.json':
                copy_file(folder/name, target/name)
        immutable(target/'job.json', dict(point=point, protocol_sha256=receipt['current_protocol']))
        result = dict(original, protocol_sha256=receipt['current_protocol'],
                      reused_from=dict(path=str(folder/'result.json'), sha256=digest(folder/'result.json'),
                                       fresh_decode_and_metric_times_preserved=True))
        result['artifacts'] = dict(original['artifacts'], **{'job.json': digest(target/'job.json')})
        immutable(target/'result.json', result)
    validate_prepared(run.root, protocol)
    if (previous/'complete.json').exists():
        originals = [validate_result(previous, point, receipt['previous_protocol']) for point in point_plan()]
        validate_summary(previous, originals, receipt['previous_protocol'])
        copy_file(previous/'fixed_frame_comparison.png', run.root/'fixed_frame_comparison.png')


def evaluate_point(run, point, protocol, metric):
    import numpy as np
    from PIL import Image
    from demo.conditioned_generation_pipeline import execute
    from demo.scalable_experiment import quality
    folder = run.root/point['name']
    protocol_hash = digest(run.root/'protocol.json')
    if (folder/'result.json').exists():
        return validate_result(run.root, point, protocol_hash), metric
    began = time.monotonic()
    prepare_point(run, point, protocol)
    if not (folder/'fresh/decode.json').exists():
        (folder/'fresh').mkdir(exist_ok=True)
        script, argv, distributed = receiver_command(run.root, point)
        execute(run, point['name']+'_fresh_decode', script, argv, distributed=distributed)
    decoded, ledger = validate_receiver(run.root, point)
    if metric is None:
        import torch
        from demo.stage_c_three_path_roi_probe import LPIPSAlex
        torch.set_num_threads(4)
        metric = LPIPSAlex(True)  # CPU; no persistent parent CUDA context
    with np.load(run.root/'source.npz', allow_pickle=False) as cache:
        source = cache['source'].copy()
    with np.load(folder/'fresh/reconstruction.npz', allow_pickle=False) as cache:
        output = cache['reconstruction'].copy()
    if source.shape != output.shape:
        raise ValueError('metric source/output geometry differs')
    run.check(); run.update(phase=point['name']+'_CPU_metrics')
    measurements = quality(source, output, metric)
    temporary = folder/'fixed_frame.png.tmp'
    Image.fromarray(output[8]).save(temporary, format='PNG')
    os.replace(temporary, folder/'fixed_frame.png')
    names = ['job.json', 'fresh/decode.json', 'fresh/reconstruction.npz', 'fixed_frame.png']
    names += ['stream.bin', 'transmitted_meta.json', 'encode.json'] if point['kind'] == 'uf' else [
        'stream.acsg' if point['kind'] == 'full_g' else 'stream.rtvc']
    if point['kind'] == 'router':
        names += ['encode.json', 'encode.request.json']
    result = dict(complete=True, point=point, protocol_sha256=protocol_hash,
                  byte_ledger=ledger, bytes=ledger['bytes'],
                  bpp=8*ledger['bytes']/(SHAPE[0]*SHAPE[1]*SHAPE[2]),
                  quality=measurements, decode=decoded,
                  elapsed_seconds_this_completion_attempt=time.monotonic()-began,
                  decode_seconds=decoded['seconds'],
                  artifacts={name: digest(folder/name) for name in names})
    save(folder/'result.json', result)
    return validate_result(run.root, point, protocol_hash), metric


def finish(run, protocol, records):
    from PIL import Image, ImageDraw
    import numpy as np
    for arm in ('context', 'local'):
        lo = (run.root/f'{arm}_e0.25_g8/stream.rtvc').read_bytes()
        hi = (run.root/f'{arm}_e0.5_g8/stream.rtvc').read_bytes()
        if not hi.startswith(lo):
            raise ValueError('two E budgets are not literal same-policy prefixes')
    figure = run.root/'fixed_frame_comparison.png'
    if not figure.exists():
        with np.load(run.root/'source.npz', allow_pickle=False) as cache:
            panels = [('Source: prepared full view', Image.fromarray(cache['source'][8]))]
        for record in records:
            name = record['point']['name']
            panels.append((f"{name}: {record['bpp']:.5f} bpp | LPIPS {record['quality']['lpips_alex']:.4f}",
                           Image.open(run.root/name/'fixed_frame.png')))
        canvas = Image.new('RGB', (3*SHAPE[2], 4*(SHAPE[1]+28)), 'white')
        draw = ImageDraw.Draw(canvas)
        for i, (label, panel) in enumerate(panels):
            x, y = (i % 3)*SHAPE[2], (i // 3)*(SHAPE[1]+28)
            draw.text((x+8, y+6), label, fill='black'); canvas.paste(panel, (x, y+28))
        temporary = figure.with_name(figure.name+'.tmp')
        canvas.save(temporary, format='PNG'); os.replace(temporary, figure)
    summary = dict(complete=True, schema=SCHEMA, protocol_sha256=digest(run.root/'protocol.json'),
                   points=10, shape=list(SHAPE), records=records, literal_E_prefixes=True,
                   role=protocol['role'], source_scope=protocol['source_scope'],
                   no_model_promotion=True, native_uf_sidecar_excluded_from_rate=True,
                   artifact_hashes={str((run.root/p['name']/'result.json').relative_to(run.root)):
                                    digest(run.root/p['name']/'result.json') for p in point_plan()},
                   fixed_frame_sha256=digest(figure),
                   candidate_preparation=read(run.root/'prepared/complete.json'))
    immutable(run.root/'summary.json', summary)
    done = run.root/'complete.json'
    if not done.exists():
        save(done, dict(complete=True, protocol_sha256=digest(run.root/'protocol.json'),
                        summary_sha256=digest(run.root/'summary.json'),
                        wall_seconds_this_completion_attempt=time.monotonic()-run.started,
                        GPU_mutex_wait_seconds=getattr(run, 'gpu_wait_seconds', 0.)))
    elif read(done)['summary_sha256'] != digest(run.root/'summary.json'):
        raise ValueError('completion summary changed')


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ['worker-encode']:
        return worker_encode(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-record', type=Path)
    parser.add_argument('--max-hours', type=float, default=6.)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--reuse-previous-probe', type=Path)
    parser.add_argument('--import-only', action='store_true',
                        help='require all ten saved points and finalize on CPU; never run inference/metrics')
    args = parser.parse_args(argv)
    if not os.environ.get('TMUX'):
        raise RuntimeError('run fullview probe and audits in tmux')
    if not math.isfinite(args.max_hours) or args.max_hours <= 0:
        raise ValueError('max-hours must be finite and positive')
    if args.import_only and (args.reuse_previous_probe is None or args.verify_only):
        raise ValueError('--import-only needs --reuse-previous-probe and cannot combine with --verify-only')
    from demo.chunk_enhancement_experiment import Run
    from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
    args.command = 'fullview_probe'
    run = Run(args); run.thread.start()
    try:
        protocol = setup(run, args)
        if args.reuse_previous_probe is not None:
            import_completed(args.reuse_previous_probe, run, protocol)
        if args.import_only:
            records = [validate_result(run.root, p, digest(run.root/'protocol.json')) for p in point_plan()]
            finish(run, protocol, records)
            validate_summary(run.root, records, digest(run.root/'protocol.json'))
            print(json.dumps(dict(imported=True, points=len(records), inference_calls=0,
                                  metric_recalculations=0, GPU_mutex_needed=False)), flush=True)
            return
        if args.verify_only:
            records = [validate_result(run.root, p, digest(run.root/'protocol.json')) for p in point_plan()]
            validate_summary(run.root, records, digest(run.root/'protocol.json'))
            print(json.dumps(dict(verified=True, points=len(records))), flush=True)
            return
        with exclusive_native_evaluation(run):
            os.environ['ROUTERVC_FULLVIEW_PARENT'] = str(os.getpid())
            records, metric = [], None
            for point in point_plan():
                run.check()
                record, metric = evaluate_point(run, point, protocol, metric)
                records.append(record)
                run.update(phase='point_complete', completed=len(records), total=10)
            finish(run, protocol, records)
    except BaseException as error:
        save(run.root/'last_failure.json', dict(error=repr(error), progress=run.progress))
        raise
    finally:
        run.log_resources(); run.stop.set(); run.thread.join(timeout=3)


if __name__ == '__main__':
    main()
