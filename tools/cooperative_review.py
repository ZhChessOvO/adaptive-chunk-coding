"""Frozen P2 checkpoint review: equal packets/profile/fusion, fresh old/new R_g.

No optimization or automatic sender training. All formal steps require tmux.
The converted old model has zero selection-state weights; policy equivalence
is checked on EVERY evaluated B/Y, not assumed from initialization alone.
"""
import argparse
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from demo.chunk_enhancement_codec import configure_torch, atomic_torch
from demo.chunk_enhancement_experiment import Run
from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
from demo.routervc_fullview_probe import read, save, digest, immutable, verify_artifacts
from demo.scalable_codec import atomic_bytes
from demo.scalable_format import frame_hash
from routervc.cooperation import data, receiver, stream
from routervc.latent import routing
from tools.fusion_review import execute
from tools.fusion_finish import memory_snapshot
from tools.latent_boundary_report import EVALUATION, ROOT as FUSION

ROOT = Path('/root/autodl-fs/DCVC/runs/routervc_cooperation_review_20261011')
RATIOS = (0., .25, .5, 1.)
ARMS = ('old_policy', 'cooperative')
CODE = ('tools/cooperative_review.py', 'tools/cooperative_review_worker.py',
        'tools/cooperative_review_report.py', 'tools/test_cooperative_review.py',
        'tools/run_cooperative_review.sh')


def source_rgb(row):
    if 'source_path' in row:
        if digest(Path(row['source_path'])) != row['source_sha256']:
            raise ValueError('source changed')
        with np.load(row['source_path'], allow_pickle=False) as z:
            return z['source'].copy()
    from PIL import Image
    frames = []
    for path, expected in zip(row['files'], row['hashes'], strict=True):
        if digest(Path(path)) != expected: raise ValueError('source frame changed')
        with Image.open(path) as im: frames.append(np.array(im.convert('RGB')))
    result = np.stack(frames)
    if list(result.shape) != row['shape']: raise ValueError('source shape changed')
    return result


def same_receipt(report, path, expected, *, disabled=False):
    if (not report['complete'] or report['stream'] != digest(path)
            or report['actual_bytes'] != path.stat().st_size
            or report['header_bytes'] != stream.HEADER_BYTES or report['mask_bytes'] != 0
            or report['source_frames_read'] or report['sender_router_loaded']
            or report['generation_disabled'] != disabled):
        raise ValueError('fresh receipt byte/profile/access failure')
    for name in ('base_hash', 'enhanced_hash'):
        if report[name] != expected[name]: raise ValueError('B/E reconstruction changed')
    if disabled and (report['generated'] or report['selected'] is not None
                     or report['output_hash'] != report['enhanced_hash']):
        raise ValueError('G-off is not model-free Y')


def fresh(run, path, checkpoint, output, disabled=False):
    output.mkdir(parents=True, exist_ok=True)
    immutable(output/'request.json', dict(stream=digest(path), disabled=disabled,
        checkpoint=None if disabled else digest(checkpoint),
        worker=digest(data.REPO/'tools/cooperative_review_worker.py'), profile=stream.identity()))
    if not (output/'complete.json').exists():
        options = ['decode', '--stream', path, '--output', output]
        if disabled: options += ['--disable-generation']
        else: options += ['--receiver', checkpoint]
        execute(run, 'fresh_worker', 'tools.cooperative_review_worker', options, True)
    report = read(output/'complete.json'); verify_artifacts(output, report['artifacts'])
    return report


def prepare(root):
    original = data.protocol()
    if original != read(data.ROOT/'protocol.json'): raise ValueError('P2 training protocol drift')
    complete = read(data.ROOT/'complete.json'); verify_artifacts(data.ROOT, complete['artifacts'])
    trained = read(data.ROOT/'router/complete.json')
    verify_artifacts(data.ROOT/'router', trained['artifacts'])
    if trained['updates'] != 5760: raise ValueError('P2 not complete')
    best = data.ROOT/'router/best.pt'
    _, payload = receiver.load(best, digest(best))
    if payload['epoch'] != trained['best']['epoch']: raise ValueError('best selection changed')
    initial = Path(original['initial_path'])
    old_path = root/'old_policy.pt'
    binding = dict(original_path=str(initial), original_sha256=original['initial_sha256'],
        converted_without_training=True, reason='same 121-byte profile and multiband; zero state weights',
        initialized_with=receiver.identity())
    if old_path.exists():
        _, old_payload = receiver.load(old_path, digest(old_path))
        if old_payload['binding'] != binding: raise ValueError('baseline changed')
    else:
        model = receiver.initialize(initial, original['initial_sha256'])
        atomic_torch(old_path, receiver.payload(model, binding, 0, None))
    models = dict(old_policy=old_path, cooperative=best)
    previous = read(EVALUATION/'summary.json'); jobs = []
    for row in previous['scope']['rows']:
        for cap in RATIOS:
            point = EVALUATION/'samples'/row['sample_id']/f'source_e{round(cap*100):03d}'
            jobs.append(dict(cohort='diagnostic', row=row, cap=cap, wire=str(point/'stream.rvlrg'),
                             receipt=str(point/'receive/complete.json')))
    for row in original['rows']:
        if row['router_split'] == 'validation':
            jobs.append(dict(cohort='grouped_validation', row=row, cap=row['E_cap'],
                             wire=row['stream'], receipt=str(Path(row['capture'])/'complete.json')))
    if len(jobs) != 64: raise ValueError('expected 52 diagnostics + 12 internal validation states')
    for job in jobs:
        job.update(wire_sha256=digest(Path(job['wire'])), receipt_sha256=digest(Path(job['receipt'])))
    protocol = dict(format='cooperative_paired_review_v1', jobs=jobs,
        models={k:dict(path=str(v), sha256=digest(v)) for k,v in models.items()},
        code={n:digest(data.REPO/n) for n in CODE}, profile=stream.identity(),
        training_complete=digest(data.ROOT/'router/complete.json'), best_epoch=payload['epoch'],
        previous_summary=digest(EVALUATION/'summary.json'), max_g=8, masks_sent=False,
        budgets='fractions of candidate E bytes; same actual B/E per pair',
        role='13 reused diagnostics, 12 checkpoint-selection validation windows; NOT new independent tests',
        freeze=['R_s', 'UF', 'width3 E', 'G', 'multiband'], auto_sender_training=False)
    immutable(root/'protocol.json', protocol)
    history = read(data.ROOT/'router/history.json')
    save(root/'training_audit.json', dict(updates=trained['updates'], best=trained['best'],
        initial=history['initial'], last=history['epochs'][-1],
        label_regret_is_not_picture_quality=True, checkpoint=digest(best)))
    # Verify literal prefixes independently for both profiles before evaluation.
    prefix_checks = 0
    for arm, model in models.items():
        previous_wire = {}
        for job in jobs:
            inner, config, _ = routing.parse(Path(job['wire']).read_bytes())
            if config['max_g'] != 8 or config['boundary'] != 0: raise ValueError('old policy scope differs')
            blob = stream.wrap(inner, digest(model), config['assets_sha256'], max_g=8, seed=config['seed'])
            if len(blob) != Path(job['wire']).stat().st_size: raise ValueError('header length changed')
            folder = location(root, job)/arm; folder.mkdir(parents=True, exist_ok=True)
            path = folder/'stream.rvlcoop'
            if path.exists() and path.read_bytes() != blob: raise ValueError('stream changed on resume')
            if not path.exists(): atomic_bytes(path, blob)
            key = (job['cohort'], job['row']['sample_id'])
            if key in previous_wire:
                if not blob.startswith(previous_wire[key]): raise ValueError('literal prefix lost')
                prefix_checks += 1
            previous_wire[key] = blob
    save(root/'prefix_audit.json', dict(complete=True, literal_prefix_pairs=prefix_checks,
        same_inner_bytes=True, same_total_bytes=True, outer_header_bytes=121, extra_mask_bytes=0))
    return protocol, models


def location(root, job):
    return root/job['cohort']/job['row']['sample_id']/f'e{round(job["cap"]*100):03d}'


def evaluate_pair(root, job, models, old_model, metric, run):
    from demo.scalable_experiment import quality
    from routervc.fusion.blend import geometry
    from routervc.fusion.boundaries import CATEGORIES, edges, measure, aggregate
    folder = location(root, job); point_file = folder/'pair.json'
    if (point_file).exists():
        point = read(point_file); verify_artifacts(folder, point['artifacts']); return point
    if digest(Path(job['wire'])) != job['wire_sha256'] or digest(Path(job['receipt'])) != job['receipt_sha256']:
        raise ValueError('previous point binding changed')
    inner, config, _ = routing.parse(Path(job['wire']).read_bytes())
    expected = read(job['receipt']); reports = {}
    for arm in ARMS:
        run.check(); run.update(phase='fresh_pair', sample=job['row']['sample_id'],
            cohort=job['cohort'], cap=job['cap'], arm=arm)
        path = folder/arm/'stream.rvlcoop'
        reports[arm] = fresh(run, path, models[arm], folder/arm/'receive')
        same_receipt(reports[arm], path, expected)
    source = source_rgb(job['row']); scores = {}; parity = None
    for arm in ARMS:
        report = reports[arm]
        with np.load(folder/arm/'receive/pixels.npz', allow_pickle=False) as z:
            base, received, output = z['base'], z['enhanced'], z['reconstruction']
        if any(frame_hash(pixels) != report[key] for pixels,key in
               ((base,'base_hash'),(received,'enhanced_hash'),(output,'output_hash'))):
            raise ValueError('decoded arrays differ from receipt')
        if source.shape != output.shape: raise ValueError('wrong source shape')
        support, _, _ = geometry(output.shape, report['generated'])
        np.testing.assert_array_equal(output[:, ~support], received[:, ~support])
        if arm == 'old_policy':
            old = routing.route(base, received, inner, config, old_model)
            if sorted(old['indices']) != sorted(report['generated']):
                raise ValueError('converted baseline does not reproduce original policy')
            if sorted(old['indices']) != sorted(expected['generated']):
                raise ValueError('original policy no longer reproduces prior receipt')
            parity = dict(indices=old['indices'], converted_indices=report['generated'], exact_set=True)
            if job['cohort'] == 'diagnostic' and job['cap'] == .5:
                prior = FUSION/'p1_controls/samples'/job['row']['sample_id']/'controls'
                verify_artifacts(prior, read(prior/'complete.json')['artifacts'])
                with np.load(prior/'controls.npz') as z:
                    np.testing.assert_array_equal(output, z['multiband'])
        boundaries = edges(source.shape, report['detail']['received_regions'], report['generated'])
        run.update(phase='whole_picture_scoring', arm=arm)
        scores[arm] = dict(quality=quality(source, output, metric),
            boundaries={cat:aggregate([measure(source, output, e) for e in boundaries if e['category']==cat])
                        for cat in CATEGORIES},
            selected=report['generated'], G_calls=len(report['generated']),
            bpp=report['bpp'], actual_bytes=report['actual_bytes'],
            fresh_seconds=report['seconds'], peak_GiB=report['peak_cuda_allocated_bytes']/2**30,
            reserved_GiB=report['peak_cuda_reserved_bytes']/2**30,
            output_hash=report['output_hash'])
    if scores['old_policy']['actual_bytes'] != scores['cooperative']['actual_bytes']:
        raise ValueError('paired rate differs')
    names = [f'{a}/{n}' for a in ARMS for n in ('stream.rvlcoop', 'receive/complete.json', 'receive/pixels.npz')]
    result = dict(complete=True, cohort=job['cohort'], sample_id=job['row']['sample_id'],
        dataset=job['row']['dataset'], cap=job['cap'], old_policy_parity=parity, scores=scores,
        artifacts={n:digest(folder/n) for n in names})
    save(point_file, result); return result


def smoke(root, protocol, models, old_model, metric, run):
    marker = root/'smoke.json'
    if marker.exists():
        receipt=read(marker); verify_artifacts(root, receipt['artifacts'])
        if receipt['protocol'] != digest(root/'protocol.json'): raise ValueError('smoke profile drift')
        return
    names = []
    for dataset in ('REDS', 'UVG'):
        job = next(j for j in protocol['jobs'] if j['cohort']=='diagnostic'
                   and j['row']['dataset']==dataset and j['cap']==.5)
        evaluate_pair(root, job, models, old_model, metric, run)
        folder = location(root, job); original = read(folder/'cooperative/receive/complete.json')
        for mode in ('repeat', 'Goff'):
            run.update(phase='smoke_'+mode, sample=job['row']['sample_id'])
            dest = folder/'cooperative'/mode; path=folder/'cooperative/stream.rvlcoop'
            result = fresh(run, path, models['cooperative'], dest, mode=='Goff')
            same_receipt(result, path, original, disabled=mode=='Goff')
            expected = original['enhanced_hash' if mode=='Goff' else 'output_hash']
            if result['output_hash'] != expected: raise ValueError('repeat/Goff failed')
            names.append(str((dest/'complete.json').relative_to(root)))
        names.append(str((folder/'pair.json').relative_to(root)))
    save(marker, dict(complete=True, protocol=digest(root/'protocol.json'), fresh_processes=8,
        max_g=8, old_policy_and_multiband_pixels_exact=True, repeats_exact=True,
        Goff_model_free=True, artifacts={n:digest(root/n) for n in names}))


class ReviewRun(Run):
    def log_resources(self):
        super().log_resources()
        with (self.root/'memory.jsonl').open('a') as log:
            log.write(json.dumps(dict(unix_seconds=time.time(), **memory_snapshot()))+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('all','smoke','verify'), nargs='?', default='all')
    p.add_argument('--output', type=Path, default=ROOT)
    args=p.parse_args()
    if not os.environ.get('TMUX'): raise RuntimeError('tmux required')
    configure_torch()
    run=ReviewRun(SimpleNamespace(output=args.output, command='review', max_hours=48));run.thread.start()
    try:
        protocol, models=prepare(args.output)
        if args.command=='verify' or (args.output/'complete.json').exists():
            done=read(args.output/'complete.json');verify_artifacts(args.output, done['artifacts'])
            for job in protocol['jobs']:
                folder=location(args.output,job);verify_artifacts(folder,read(folder/'pair.json')['artifacts'])
            run.update(phase='complete_verified');return
        from demo.routervc_receiver_router import load_model
        old_model,_=load_model(data.protocol()['initial_path'],expected_sha256=data.protocol()['initial_sha256'])
        from demo.stage_c_three_path_roi_probe import LPIPSAlex
        with exclusive_native_evaluation(run):
            metric=LPIPSAlex(True)
            smoke(args.output, protocol, models, old_model, metric, run)
            if args.command=='smoke':return
            run.update(phase='formal_paired_evaluation_started', total=128, completed=4)
            points=[]
            for job in protocol['jobs']:
                points.append(evaluate_pair(args.output, job, models, old_model, metric, run))
                run.update(phase='formal_pair_complete', completed=len(points)*2, total=128,
                           sample=job['row']['sample_id'], cap=job['cap'])
            from tools.cooperative_review_report import report
            run.update(phase='automatic_report');artifacts=report(args.output, protocol, points, run.check)
            save(args.output/'complete.json', dict(complete=True, paired_states=64,
                fresh_points=128, additional_repeat_Goff_checks=4, sender_training_started=False,
                next='Review paired pictures and metrics before choosing sender adaptation',
                artifacts={n:digest(args.output/n) for n in
                    ['protocol.json','training_audit.json','prefix_audit.json','smoke.json',*artifacts,
                     *[str((location(args.output,j)/'pair.json').relative_to(args.output)) for j in protocol['jobs']]]}))
            run.update(phase='paired_evaluation_complete', completed=128)
    except BaseException as error:
        save(args.output/'last_failure.json',dict(error=repr(error),progress=run.progress));raise
    finally:
        run.log_resources();run.stop.set();run.thread.join(timeout=3);run.lock.close()


if __name__=='__main__':main()
