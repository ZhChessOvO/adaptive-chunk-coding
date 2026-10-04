"""Measured qE=2 E/EG teachers, with authenticated unchanged B/G reuse.

No UF/E/G weight updates. Preserves the existing 120-window mixed-view split.
Source is opened only by encoding/scoring, never by the receive worker.
"""
from __future__ import annotations
import argparse
from collections import Counter
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))
from demo.routervc_fullview_probe import read, digest, save, immutable, verify_artifacts
from demo.routervc_mixedview_teacher import ENHANCEMENT, ADAPTER, OLD, crop
from demo.routervc_light_packets import QSTEP, bank_info, subset_bank

HISTORY = Path('/root/autodl-fs/DCVC/runs/routervc_revision_20261003/mixedview_teacher')
OUTPUT = Path('/root/autodl-fs/DCVC/runs/routervc_light_router_20261004')
CODE = ('routervc_light_packets.py', 'routervc_light_teacher.py', 'routervc_light_receive.py',
        'routervc_light_queue.py', 'run_routervc_light_router.sh')


def protocol(smoke):
    from demo.routervc_visual_train import teacher_entries
    from demo.online_eg_decode import identities
    previous = read(HISTORY/'protocol.json')
    if not read(HISTORY/'complete.json')['complete']:
        raise ValueError('original teacher incomplete')
    for name, expected in previous['code'].items():
        if digest(REPO/name) != expected: raise ValueError('historical teacher source changed: '+name)
    profile = identities(ADAPTER)
    if (previous['profile'] != profile or previous['enhancement'] != digest(ENHANCEMENT)
            or previous['adapter'] != digest(ADAPTER)):
        raise ValueError('B/G reuse requires exactly the same component models')
    entries = teacher_entries(HISTORY/'labels.json')
    if smoke:
        entries = [next(e for e in entries if e['dataset'] == d) for d in ('REDS', 'UVG')]
    elif (Counter(e['dataset'] for e in entries) != dict(REDS=90, UVG=30)
          or Counter(e['router_split'] for e in entries) != dict(train=96, validation=24)):
        raise ValueError('preserve the existing mixed-view sample split')
    rows = []
    for entry in entries:
        label = read(entry['path']); sid = entry['sample_id']
        received = Path(label['received_path']).parent
        bank = (HISTORY/'encoded'/sid/'prepared/bank.acse' if entry['dataset'] == 'REDS'
                else OLD/'encoded'/label['historical_teacher_sample_id']/'packets.acse')
        er = read(received/'E.json'); verify_artifacts(received, er['artifacts'])
        complete = read(received/'complete.json')
        if complete['e_report'] != digest(received/'E.json') or er['source_frames_read']:
            raise ValueError('old receiver evidence changed')
        cells = []
        for i, expected in enumerate(complete['region_results']):
            cell = received/f'cell_{i:02d}/result.json'
            if digest(cell) != expected: raise ValueError('old cell record changed')
            cells.append(dict(path=str(cell), sha256=expected))
        rows.append(dict(entry=entry, source_path=label['source_path'],
            source_sha256=label['source_sha256'], bank_path=str(bank), bank_sha256=digest(bank),
            old_received=str(received), old_receive_sha256=digest(received/'complete.json'),
            cells=cells, sample_id=sid, shape=label['shape'], rois=[r['roi'] for r in label['regions']]))
    return dict(schema='routervc-light-teacher-v1', qstep=QSTEP, smoke=smoke, samples=rows,
        previous_labels=digest(HISTORY/'labels.json'), previous_protocol=digest(HISTORY/'protocol.json'),
        profile=profile, enhancement=digest(ENHANCEMENT), adapter=digest(ADAPTER),
        code={n:digest(REPO/'demo'/n) for n in CODE},
        training_code={n:digest(REPO/'demo'/n) for n in ('routervc_visual_train.py','routervc_visual_router.py')},
        no_UF_E_G_updates=True, semantic_supervision=False,
        reuse='B/G unchanged pixels, controls and measured quality; E/EG remeasured at qE=2',
        independent_system_test=False)


def prepare(root):
    import numpy as np
    import torch
    from demo.chunk_enhancement_codec import configure_torch, decode_features, encode_enhancement, load_model
    from demo.chunk_enhancement_experiment import codec
    from demo.scalable_codec import atomic_bytes
    from demo.scalable_format import parse, frame_hash
    from demo.scalable_experiment import check_space
    p = read(root/'protocol.json'); configure_torch(); native = model = None; entries = []
    for row in p['samples']:
        check_space(); sid = row['sample_id']; dest = root/'encoded'/sid
        dest.mkdir(parents=True, exist_ok=True); done = dest/'complete.json'
        binding = dict(protocol=digest(root/'protocol.json'), sample=row)
        if done.exists():
            record = read(done)
            if record['binding'] != binding: raise ValueError('changed q2 encode binding')
            verify_artifacts(dest, record['artifacts'])
        else:
            began = time.monotonic()
            if digest(row['source_path']) != row['source_sha256'] or digest(row['bank_path']) != row['bank_sha256']:
                raise ValueError('source or immutable base changed')
            with np.load(row['source_path'], allow_pickle=False) as data: source = data['source'].copy()
            old_bank = Path(row['bank_path']).read_bytes(); old = parse(old_bank)
            if native is None: native, model = codec(), load_model(ENHANCEMENT).requires_grad_(False)
            base, chunks = decode_features(native, old_bank[:old.base_end])
            with np.load(row['entry']['reconstruction_path'], allow_pickle=False) as data:
                np.testing.assert_array_equal(base, data['base'])
            if list(source.shape) != row['shape']: raise ValueError('source geometry changed')
            with torch.inference_mode():
                prefix, wires, enhanced, detail = encode_enhancement(model, ENHANCEMENT,
                    old_bank[:old.base_end], source, base, chunks, row['rois'], QSTEP, compact=True)
            bank = prefix+b''.join(wires); info = bank_info(bank)
            if info['parsed'].base != old.base: raise ValueError('UF bytes changed')
            atomic_bytes(dest/'bank.acse', bank)
            record = dict(complete=True, binding=binding, bank_sha256=digest(dest/'bank.acse'),
                expected_base_hash=frame_hash(base), expected_E_hash=frame_hash(enhanced),
                source_rgb_sha256=frame_hash(source), seconds=time.monotonic()-began,
                e_packet_bytes=info['e_bytes'], packet_details=detail,
                artifacts={'bank.acse':digest(dest/'bank.acse')})
            save(done, record)
            del chunks, source, base, enhanced
        # Source-free receiver index, including only old G evidence and stream paths.
        entries.append(dict(sample_id=sid, bank_path=str((dest/'bank.acse').resolve()),
            bank_sha256=record['bank_sha256'], expected_base_hash=record['expected_base_hash'],
            expected_E_hash=record['expected_E_hash'], shape=row['shape'], rois=row['rois'],
            cells=row['cells'], encoded_manifest=digest(done)))
        print(f'LIGHT_ENCODE {len(entries)}/{len(p["samples"])} {sid}', flush=True)
    immutable(root/'receiver_index.json', dict(complete=True, samples=entries, profile=p['profile'],
              enhancement=p['enhancement'], adapter=p['adapter']))


def make_labels(root):
    import numpy as np
    import torch
    from demo.scalable_experiment import quality, check_space
    from demo.stage_c_three_path_roi_probe import LPIPSAlex
    torch.set_num_threads(4); metric = None; entries = []; p = read(root/'protocol.json')
    for row in p['samples']:
        check_space(); sid = row['sample_id']; path = root/'labels'/f'{sid}.json'
        dest = root/'received'/sid
        binding = dict(protocol=digest(root/'protocol.json'), old_label=row['entry']['sha256'],
                       encoded=digest(root/'encoded'/sid/'complete.json'), received=digest(dest/'complete.json'))
        if path.exists():
            result = read(path)
            if result['dependencies'] != binding: raise ValueError('changed label dependencies')
            if digest(result['received_path']) != result['received_sha256']: raise ValueError('changed teacher RGB')
        else:
            if digest(row['entry']['path']) != row['entry']['sha256']: raise ValueError('old labels changed')
            if digest(row['source_path']) != row['source_sha256']: raise ValueError('label source changed')
            old = read(row['entry']['path'])
            with np.load(row['source_path'], allow_pickle=False) as data: source = data['source'].copy()
            with np.load(dest/'received_E.npz', allow_pickle=False) as data:
                base, enhanced = data['base'].copy(), data['enhanced'].copy()
            if metric is None: metric = LPIPSAlex(True)
            regions = []; bank = (root/'encoded'/sid/'bank.acse').read_bytes()
            info = bank_info(bank)
            for i, roi in enumerate(row['rois']):
                folder = dest/f'cell_{i:02d}'; cell = read(folder/'result.json'); verify_artifacts(folder, cell['artifacts'])
                with np.load(folder/'outputs.npz', allow_pickle=False) as data: eg = data['EG'].copy()
                # Reuse B/G only: their source, decoded B, model, seed and controls are authenticated.
                scores = {s:old['regions'][i]['quality'][s] for s in ('B','G')}
                scores.update(E=quality(crop(source, roi), crop(enhanced, roi), metric),
                              EG=quality(crop(source, roi), eg, metric))
                gains = {s:scores['B']['lpips_alex']-scores[s]['lpips_alex'] for s in ('B','E','G','EG')}
                costs = dict(old['regions'][i]['costs'])
                costs['e_packet_bytes'] = info['e_bytes'][i]
                b = info['parsed'].base_end; settings = cell['control']
                from demo import scalable_cooperation_format as fmt
                costs['base_container_bytes'] = b
                costs['individual_stream_bytes'] = dict(B=b, E=b+info['e_bytes'][i],
                    G=len(fmt.wrap(subset_bank(bank, []), settings)),
                    EG=len(fmt.wrap(subset_bank(bank, [i]), settings)))
                regions.append(dict(region=i, roi=roi, quality=scores, lpips_gain=gains,
                    interaction_gain=gains['EG']-gains['E']-gains['G'], costs=costs,
                    g_seconds=dict(G=old['regions'][i]['g_seconds']['G'],
                        EG=cell['report']['runtime']['seconds_model_load_excluded'])))
            result = {k:v for k,v in old.items() if k not in ('regions','dependencies','received_path',
                'received_sha256','seconds','e_bank_decode_seconds','e_timing_scope','timing_scope')}
            result.update(qstep=QSTEP, regions=regions, dependencies=binding,
                received_path=str((dest/'received_E.npz').resolve()), received_sha256=digest(dest/'received_E.npz'),
                reuse_scope='B/G only; E/EG newly measured', reused_historical_UVG=False,
                semantic_labels_available=False, content_annotation_status='unknown')
            save(path, result)
        entries.append({k:result[k] for k in ('sample_id','dataset','sequence','router_split','view_kind')} |
            dict(path=str(path.resolve()),sha256=digest(path),reconstruction_path=result['received_path'],
                 reconstruction_sha256=result['received_sha256']))
        print(f'LIGHT_LABELS {len(entries)}/{len(p["samples"])} {sid}', flush=True)
    immutable(root/'labels.json', dict(complete=True, samples=entries, qstep=QSTEP,
        protocol=digest(root/'protocol.json'), semantic_labels_available=False))


def fresh_checks(run):
    import numpy as np
    from demo.conditioned_generation_pipeline import execute
    from demo.online_eg_eval_core import noise_pair
    records = []
    for row in read(run.root/'receiver_index.json')['samples']:
        for i in (0, 5):
            cell_dir = run.root/'received'/row['sample_id']/f'cell_{i:02d}'
            cell = read(cell_dir/'result.json')
            # G reuse is independently checked for one boundary cell per dataset.
            for state in (('G', 'EG') if i == 0 else ('EG',)):
                old_cell = Path(row['cells'][i]['path'])
                stream = old_cell.parent/'G.acsg' if state == 'G' else cell_dir/'EG.acsg'
                expected = read(old_cell)['reports']['G'] if state == 'G' else cell['report']
                out = run.root/'fresh_checks'/row['sample_id']/f'{i}_{state}'; out.mkdir(parents=True,exist_ok=True)
                if not (out/'checked.json').exists():
                    if not (out/'decode.json').exists():
                        execute(run,f'fresh_{row["sample_id"]}_{i}_{state}','online_eg_decode.py',
                            ['--stream',stream,'--output',out,'--enhancement',ENHANCEMENT,'--adapter',ADAPTER],distributed=True)
                    report = read(out/'decode.json')
                    if (report['source_frames_read'] or not report['outside_generate_exact']
                            or report['output_hash'] != expected['output_hash']
                            or report['generation_input_hash'] != expected['generation_input_hash']
                            or report['total_bytes'] != expected['total_bytes']):
                        raise ValueError('fresh teacher or unchanged G differs')
                    noise_pair(report,dict(generation_runtime=expected['runtime']),same_condition=True)
                    with np.load(out/'reconstruction.npz',allow_pickle=False) as data:
                        actual = crop(data['reconstruction'],row['rois'][i])
                    pixels = old_cell.parent/'outputs.npz' if state == 'G' else cell_dir/'outputs.npz'
                    with np.load(pixels,allow_pickle=False) as data: np.testing.assert_array_equal(actual,data[state])
                    save(out/'checked.json',dict(complete=True,stream_sha256=digest(stream),
                        artifacts={n:digest(out/n) for n in ('decode.json','reconstruction.npz')}))
                check = read(out/'checked.json'); verify_artifacts(out,check['artifacts'])
                if check['stream_sha256'] != digest(stream): raise ValueError('fresh stream changed')
                records.append(str((out/'checked.json').resolve()))
    immutable(run.root/'fresh_audit.json',dict(complete=True,checks=records,B_G_reuse_checked=True))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('smoke','run','verify','prepare','labels'))
    p.add_argument('--output',type=Path,required=True); p.add_argument('--smoke-root',type=Path)
    p.add_argument('--max-hours',type=float,default=24.)
    args = p.parse_args(argv)
    if not os.environ.get('TMUX') or not math.isfinite(args.max_hours) or args.max_hours <= 0:
        raise ValueError('requires tmux and positive finite max-hours')
    if args.command in ('prepare','labels'):
        if not os.environ.get('ROUTERVC_LIGHT_PARENT'): raise RuntimeError('supervised worker required')
        return prepare(args.output) if args.command == 'prepare' else make_labels(args.output)
    from demo.chunk_enhancement_experiment import Run
    from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
    from demo.conditioned_generation_pipeline import execute
    from contextlib import nullcontext
    run = Run(SimpleNamespace(output=args.output,command='light_teacher',max_hours=args.max_hours)); run.thread.start()
    try:
        smoke = read(run.root/'protocol.json')['smoke'] if args.command == 'verify' else args.command == 'smoke'
        prot = protocol(smoke); immutable(run.root/'protocol.json',prot)
        if args.command == 'run':
            tested = read(args.smoke_root/'protocol.json')
            if (not read(args.smoke_root/'complete.json')['complete']
                    or not read(args.smoke_root/'fresh_audit.json')['complete']
                    or any(tested[k] != prot[k] for k in ('code','training_code','profile','enhancement','adapter'))):
                raise ValueError('formal queue requires matching complete smoke')
        os.environ['ROUTERVC_LIGHT_PARENT'] = str(os.getpid())
        completed = (run.root/'complete.json').exists()
        if args.command == 'verify' and not completed: raise ValueError('teacher not complete')
        with nullcontext() if completed else exclusive_native_evaluation(run):
            if not completed:
                execute(run,'prepare_q2','routervc_light_teacher.py',['prepare','--output',run.root])
                if smoke and not (run.root/'partial_resume.json').exists():
                    execute(run,'partial_q2','routervc_light_receive.py',['--root',run.root,'--stop-after',2],distributed=True)
                    save(run.root/'partial_resume.json',dict(preserved=snapshot(run.root)))
                execute(run,'receive_q2','routervc_light_receive.py',['--root',run.root],distributed=True)
                if smoke:
                    for path,value in read(run.root/'partial_resume.json')['preserved'].items():
                        if snapshot_item(run.root/path) != value: raise ValueError('resuming rewrote completed evidence')
                    fresh_checks(run)
                execute(run,'q2_labels','routervc_light_teacher.py',['labels','--output',run.root])
            before = snapshot(run.root)
            execute(run,'readonly_receive','routervc_light_receive.py',['--root',run.root,'--verify-only'])
            if snapshot(run.root) != before: raise ValueError('read-only receive changed evidence')
            from demo.routervc_visual_train import teacher_entries
            teacher_entries(run.root/'labels.json')
            record = dict(complete=True,protocol=digest(run.root/'protocol.json'),
                labels=digest(run.root/'labels.json'),samples=len(prot['samples']),qstep=QSTEP,
                replay_no_inference=True,semantic_supervision=False)
            immutable(run.root/'complete.json',record)
            run.update(phase='complete',completed=len(prot['samples']),total=len(prot['samples']))
    except BaseException as error:
        save(run.root/'last_failure.json',dict(error=repr(error),progress=run.progress)); raise
    finally:
        run.log_resources(); run.stop.set(); run.thread.join(timeout=3); run.lock.close()


def snapshot_item(path):
    return dict(sha256=digest(path),mtime_ns=path.stat().st_mtime_ns)


def snapshot(root):
    return {str(p.relative_to(root)):snapshot_item(p) for p in sorted((root/'received').rglob('*.json'))}


if __name__ == '__main__': main()
