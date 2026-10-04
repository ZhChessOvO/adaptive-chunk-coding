"""Fixed-weight qE=2 probe with unchanged qE=1 E-region selections.

The original global/local Router is not retrained or promoted. Only the global
arm is used as a controlled reference here. The receiver still reroutes G from
the actually received RGB. Different qE files are different RD points; within
each qE, the original E25/E50 selection order remains a literal byte prefix.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from demo.routervc_fullview_probe import read, digest, save, immutable, verify_artifacts

ORIGINAL = Path('/root/autodl-fs/DCVC/runs/routervc_revision_20261003/visual_evaluation_recovered')
OUTPUT = Path('/root/autodl-fs/DCVC/runs/routervc_efficiency_20261004/q2_probe')
POINTS = ('global_local_e0.25_g8', 'global_local_e0.5_g8')
QSTEP = 2.


def geometry_key(packet):
    m = packet.meta
    return m['start'], m['count'], tuple(m['roi'])


def select_same_packets(bank, reference):
    from demo.scalable_format import parse
    parsed = parse(bank)
    by_key = {geometry_key(p): p for p in parsed.packets}
    if len(by_key) != len(parsed.packets):
        raise ValueError('duplicate q2 packet geometry')
    if parsed.base != reference.base:
        raise ValueError('immutable UF bottom stream changed')
    chosen = [by_key[geometry_key(p)] for p in reference.packets]
    if any(p.meta['qstep'] != QSTEP for p in chosen):
        raise ValueError('new E packet is not qE=2')
    return bank[:parsed.base_end]+b''.join(p.wire for p in chosen)


def prepare_worker(args):
    import numpy as np
    import torch
    from demo import routervc_visual_format as fmt
    from demo.routervc_visual_policy import route
    from demo.chunk_enhancement_codec import configure_torch, decode_features, encode_enhancement, load_model
    from demo.chunk_enhancement_experiment import codec
    from demo.routervc_encode import compose_candidates
    from demo.scalable_codec import atomic_bytes
    from demo.scalable_format import frame_hash, parse

    began = time.monotonic()
    protocol = read(args.output/'protocol.json')
    original = args.root/'samples'/args.sample_id
    target = args.output/'samples'/args.sample_id
    target.mkdir(parents=True, exist_ok=True)
    before = read(original/'prepared/complete.json')
    source_path = original/'source.npz'
    if digest(source_path) != before['binding']['source']['file_sha256']:
        raise ValueError('q probe source changed')
    with np.load(source_path, allow_pickle=False) as data:
        source = data['source'].copy()
    if frame_hash(source) != before['binding']['source']['source_rgb_sha256']:
        raise ValueError('q probe source pixel identity changed')
    configure_torch()
    codec_model = codec()
    base_data = (original/'prepared/base.acse').read_bytes()
    if digest(original/'prepared/base.acse') != before['artifacts']['base.acse']:
        raise ValueError('original base changed')
    base, chunks = decode_features(codec_model, base_data)
    if frame_hash(base) != before['base_rgb_sha256']:
        raise ValueError('UF bottom reconstruction changed')
    weight = Path(protocol['original']['enhancement']['path'])
    model = load_model(weight).requires_grad_(False)
    with torch.inference_mode():
        prefix, wires, all_e, details = encode_enhancement(model, weight,
            base_data, source, base, chunks, before['rois'], QSTEP, compact=True)
    bank = prefix+b''.join(wires)
    atomic_bytes(target/'bank_q2.acse', bank)
    for point in POINTS:
        folder = target/point
        folder.mkdir(exist_ok=True)
        reference = original/point/'stream.rtvc'
        config, _, old_inner, _ = fmt.parse(reference.read_bytes())
        inner = select_same_packets(bank, old_inner)
        stream = fmt.wrap(inner, config)
        chosen = sorted({before['rois'].index(p.meta['roi']) for p in old_inner.packets})
        mixed = compose_candidates(base, all_e, chosen, before['rois'])
        expected = route(base, mixed, parse(inner), config,
            Path(protocol['original']['models']['arms']['global_local']['path']),
            expected_policy=fmt.policy_identity())
        atomic_bytes(folder/'stream.rtvc', stream)
        save(folder/'encode.json', dict(qstep=QSTEP,
            original_selection=point, reference_sha256=digest(reference),
            stream_sha256=digest(folder/'stream.rtvc'), E_indices=chosen,
            expected_base_hash=frame_hash(base), expected_mixed_hash=frame_hash(mixed),
            expected_shared_route=expected, all_headers_charged=True,
            explicit_E_mask_bytes=0, explicit_G_map_bytes=0,
            selection_scope='same E packet locations/order as old q1; not reoptimized at same total bytes'))
    low, high = [(target/p/'stream.rtvc').read_bytes() for p in POINTS]
    if not high.startswith(low):
        raise ValueError('new q2 E25/E50 selections lost their byte prefix')
    names = ['bank_q2.acse']+[f'{p}/{n}' for p in POINTS for n in ('stream.rtvc','encode.json')]
    save(target/'prepare.complete.json', dict(complete=True, protocol_sha256=digest(args.output/'protocol.json'),
        source_sha256=digest(source_path), literal_prefix_exact=True, qstep=QSTEP,
        seconds=time.monotonic()-began, packet_details=details,
        artifacts={n:digest(target/n) for n in names}))
    print(f'Q2_PREPARED {args.sample_id}', flush=True)


def make_protocol(root):
    from demo.routervc_visual_evaluate import code_hashes
    old = read(root/'protocol.json')
    if not read(root/'complete.json')['complete'] or old['code'] != code_hashes():
        raise ValueError('old evaluation incomplete or original code changed')
    for item in (old['enhancement'], old['adapter'], old['models']['arms']['global_local']):
        # Formal Router descriptors and E/G descriptors both carry sha256.
        if digest(item['path']) != item['sha256']:
            raise ValueError('frozen weight changed')
    files = ['complete.json','protocol.json','summary.json']
    for entry in old['sources']:
        sid = entry['sample']['sample_id']
        files += [f'samples/{sid}/{name}' for name in
                  ('source.npz','prepared/complete.json','prepared/base.acse')]
        files += [f'samples/{sid}/{point}/{name}' for point in POINTS
                  for name in ('stream.rtvc','result.json')]
    return dict(schema='routervc-q2-fixed-selection-v1', qstep=QSTEP, original=old,
        source_root=str(root.resolve()), source_artifacts={n:digest(root/n) for n in files},
        probe_code={n:digest(REPO/n) for n in
                    ('demo/routervc_qstep_probe.py','demo/run_routervc_qstep_probe.sh')},
        points=26, samples=13, no_model_promotion=True, new_training=False,
        selected_arm='global_local as fixed reference only; other arm retained',
        interpretation='E25/E50 name old q1 packet selections, not percentages of the new q2 bank')


def validate_prepared(root, sid):
    folder = root/'samples'/sid
    saved = read(folder/'prepare.complete.json')
    if (not saved['complete'] or not saved['literal_prefix_exact'] or saved['qstep'] != QSTEP
            or saved['protocol_sha256'] != digest(root/'protocol.json')):
        raise ValueError('q2 preparation binding mismatch')
    verify_artifacts(folder, saved['artifacts'])


def validate_decode(folder, protocol, sample):
    import numpy as np
    from demo.scalable_format import frame_hash
    from demo.routervc_visual_evaluate import source_shape
    encoded = read(folder/'encode.json')
    report = read(folder/'fresh/decode.json')
    stream = folder/'stream.rtvc'
    if (report['source_frames_read'] or not report['outside_generate_exact']
            or not report['base_reference_unchanged']
            or report['stream_sha256'] != digest(stream)
            or report['route'] != encoded['expected_shared_route']
            or report['generation_input_hash'] != encoded['expected_mixed_hash']
            or report['base_hash'] != encoded['expected_base_hash']
            or report['semantic_heads_used'] is not False):
        raise ValueError('q2 fresh decode mismatch')
    if sum(report[k] for k in ('base_bytes','container_header_bytes','packet_bytes',
                               'incomplete_tail_bytes','generation_control_bytes')) != stream.stat().st_size:
        raise ValueError('q2 real-file byte ledger mismatch')
    with np.load(folder/'fresh/reconstruction.npz', allow_pickle=False) as data:
        for key, field in [('base','base_hash'),('enhanced','generation_input_hash'),('reconstruction','output_hash')]:
            if data[key].shape != source_shape(sample) or frame_hash(data[key]) != report[field]:
                raise ValueError('q2 saved reconstruction changed')
    return report


def evaluate(run, args, protocol):
    import numpy as np
    import torch
    from PIL import Image
    from demo.conditioned_generation_pipeline import execute
    from demo.scalable_experiment import quality
    from demo.stage_c_three_path_roi_probe import LPIPSAlex
    from demo.routervc_visual_evaluate import source_shape
    metric = None
    rows = []
    complete = (run.root/'complete.json').exists()
    if complete:
        verify_artifacts(run.root, read(run.root/'complete.json')['artifacts'])
    for entry in protocol['original']['sources']:
        sample = entry['sample']; sid = sample['sample_id']
        sample_folder = run.root/'samples'/sid
        if not (sample_folder/'prepare.complete.json').exists():
            if complete: raise ValueError('missing completed q2 preparation')
            execute(run, f'prepare_{sid}', Path(__file__).name,
                ['prepare', '--root', args.root, '--output', run.root, '--sample-id', sid])
        validate_prepared(run.root, sid)
        for point in POINTS:
            run.check(); folder = sample_folder/point
            if (folder/'result.json').exists():
                result = read(folder/'result.json')
                if result['protocol_sha256'] != digest(run.root/'protocol.json'):
                    raise ValueError('q2 point protocol changed')
                verify_artifacts(folder, result['artifacts'])
                if validate_decode(folder, protocol, sample) != result['decode']:
                    raise ValueError('saved q2 decoder report changed')
            else:
                if complete: raise ValueError('read-only replay cannot infer or score')
                if not (folder/'fresh/decode.json').exists():
                    (folder/'fresh').mkdir(exist_ok=True)
                    execute(run, f'fresh_{sid}_{point}', 'routervc_visual_decode.py',
                        ['--worker','--stream',folder/'stream.rtvc','--output',folder/'fresh',
                         '--router',protocol['original']['models']['arms']['global_local']['path'],
                         '--enhancement',protocol['original']['enhancement']['path'],
                         '--adapter',protocol['original']['adapter']['path']], distributed=True)
                decoded = validate_decode(folder, protocol, sample)
                run.update(phase='CPU_q2_metrics', sample=sid, point=point)
                if metric is None:
                    torch.set_num_threads(4); metric = LPIPSAlex(True)
                with np.load(args.root/'samples'/sid/'source.npz', allow_pickle=False) as data:
                    source = data['source'].copy()
                with np.load(folder/'fresh/reconstruction.npz', allow_pickle=False) as data:
                    output, enhanced = data['reconstruction'].copy(), data['enhanced'].copy()
                score, e_score = quality(source, output, metric), quality(source, enhanced, metric)
                image_path = folder/'fixed_frame.png'
                Image.fromarray(output[8]).save(image_path.with_suffix('.tmp'), format='PNG')
                os.replace(image_path.with_suffix('.tmp'), image_path)
                size = (folder/'stream.rtvc').stat().st_size
                names = ['stream.rtvc','encode.json','fresh/decode.json','fresh/reconstruction.npz','fixed_frame.png']
                result = dict(complete=True, sample_id=sid, dataset=sample['dataset'], point=point,
                    qstep=QSTEP, protocol_sha256=digest(run.root/'protocol.json'),
                    bytes=size, bpp=8*size/math.prod(source_shape(sample)[:3]), quality=score,
                    same_wire_G_off_quality=e_score, decode=decoded,
                    artifacts={n:digest(folder/n) for n in names})
                save(folder/'result.json', result)
            rows.append(result)
            run.update(completed=len(rows), total=26)
    if not complete:
        verify_artifacts(args.root, protocol['source_artifacts'])
        means = {}
        for dataset in ('REDS','UVG'):
            means[dataset] = {}
            for point in POINTS:
                group = [r for r in rows if r['dataset'] == dataset and r['point'] == point]
                means[dataset][point] = dict(windows=len(group),
                    bpp=sum(r['bpp'] for r in group)/len(group),
                    **{k:sum(r['quality'][k] for r in group)/len(group) for k in group[0]['quality']},
                    receiver_seconds=sum(r['decode']['seconds'] for r in group)/len(group),
                    mean_peak_cuda_bytes=sum(r['decode']['peak_cuda_allocated_bytes'] for r in group)/len(group))
        save(run.root/'summary.json', dict(complete=True, points=26, qstep=QSTEP,
            literal_q2_prefix_pairs=13, no_mask_bytes=True, no_model_promotion=True,
            old_outputs_unchanged=True, group_means=means,
            rows=[{k:r[k] for k in ('sample_id','dataset','point','qstep','bytes','bpp','quality','same_wire_G_off_quality')} for r in rows]))
        names = ['protocol.json','summary.json']
        names += [f"samples/{r['sample_id']}/{r['point']}/result.json" for r in rows]
        names += [f"samples/{e['sample']['sample_id']}/prepare.complete.json" for e in protocol['original']['sources']]
        save(run.root/'complete.json', dict(complete=True, artifacts={n:digest(run.root/n) for n in names}))
    print('Q2_PROBE_VERIFIED_READ_ONLY' if complete else 'Q2_PROBE_COMPLETE', flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('run','prepare'))
    p.add_argument('--root', type=Path, default=ORIGINAL)
    p.add_argument('--output', type=Path, default=OUTPUT)
    p.add_argument('--sample-id')
    p.add_argument('--max-hours', type=float, default=8.)
    args = p.parse_args(argv)
    if args.command == 'prepare': return prepare_worker(args)
    if not os.environ.get('TMUX'): raise RuntimeError('q2 probe requires tmux')
    if args.output.resolve().is_relative_to(args.root.resolve()):
        raise ValueError('q2 output must be separate from the old evaluation')
    from demo.chunk_enhancement_experiment import Run
    from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
    run = Run(SimpleNamespace(output=args.output, command='q2_probe', max_hours=args.max_hours))
    run.thread.start()
    old_parent = os.environ.get('ROUTERVC_VISUAL_PARENT')
    try:
        protocol = make_protocol(args.root)
        immutable(run.root/'protocol.json', protocol)
        os.environ['ROUTERVC_VISUAL_PARENT'] = str(os.getpid())
        with nullcontext() if (run.root/'complete.json').exists() else exclusive_native_evaluation(run):
            evaluate(run, args, protocol)
    except BaseException as error:
        save(run.root/'last_failure.json',dict(error=repr(error),progress=run.progress)); raise
    finally:
        if old_parent is None: os.environ.pop('ROUTERVC_VISUAL_PARENT',None)
        else: os.environ['ROUTERVC_VISUAL_PARENT'] = old_parent
        run.log_resources(); run.stop.set(); run.thread.join(timeout=3); run.lock.close()


if __name__ == '__main__': main()
