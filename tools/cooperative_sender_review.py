"""P2b sender-only actual-stream review, with fixed P2 R_g/G/multiband.

No optimization or promotion. Reuse only authenticated identical streams;
all newly decoded streams run in a source/sender-free worker process.
"""
import argparse
from pathlib import Path
import os
from types import SimpleNamespace

import numpy as np

from demo.chunk_enhancement_codec import configure_torch
from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
from demo.routervc_fullview_probe import read, save, digest, immutable, verify_artifacts
from demo.scalable_codec import atomic_bytes
from demo.scalable_format import frame_hash
from routervc.cooperation import sender_data as data, sender, stream
from routervc.latent import routing, router_data
from tools.cooperative_sender import SenderRun, execute, fresh
from tools.cooperative_review import source_rgb, same_receipt
from tools.latent_boundary_report import EVALUATION

ROOT = Path('/root/autodl-fs/DCVC/runs/routervc_cooperative_sender_review_20261011')
ARMS = ('old_sender', 'source', 'zero_source')
RATIOS = (0., .25, .5, 1.)
CODE = ('tools/cooperative_sender_review.py', 'tools/cooperative_sender_review_report.py',
        'tools/test_cooperative_sender_review.py', 'tools/run_cooperative_sender_review.sh',
        'tools/cooperative_review.py', 'tools/fusion_finish.py', 'tools/plot_latent_diagnostics.py')


def location(root, job):
    return root/job['cohort']/job['row']['sample_id']/f'e{round(job["cap"]*100):03d}'


def verified_file(path, expected):
    if digest(Path(path)) != expected:
        raise ValueError('bound input changed: '+str(path))


def prepare(root):
    training = data.ROOT/'formal'
    completed = read(training/'complete.json'); verify_artifacts(training, completed['artifacts'])
    trained = read(training/'router/complete.json')
    verify_artifacts(training/'router', trained['artifacts'])
    if trained['epochs'] != 120 or trained['updates_per_arm'] != 5760:
        raise ValueError('paired sender training incomplete')
    original = read(training/'protocol.json')
    for name, expected in original['code'].items(): verified_file(data.REPO/name, expected)
    models = {arm:dict(path=str(training/'router'/arm/'best.pt'),
        sha256=digest(training/'router'/arm/'best.pt'), epoch=trained['selected'][arm]['epoch'])
        for arm in ARMS[1:]}
    for model in models.values(): sender.load(Path(model['path']))
    models['old_sender'] = original['initialization']
    prior = read(EVALUATION/'summary.json')
    old_review = read(data.REVIEW/'protocol.json')
    if (not read(data.REVIEW/'complete.json')['complete']
            or old_review['models']['cooperative']['sha256'] != original['teacher']['receiver']['sha256']
            or old_review['profile'] != stream.identity()):
        raise ValueError('old sender / P2 receiver reference is not complete or same profile')
    jobs = []
    for row in prior['scope']['rows']:
        bankdir = router_data.ROOT/'receiver_review/samples'/row['sample_id']
        encoded = read(bankdir/'encoded.json')
        if encoded['binding']['row'] != row: raise ValueError('diagnostic bank/source differs')
        plan = EVALUATION/'samples'/row['sample_id']/'source_planning'
        for cap in RATIOS:
            jobs.append(dict(cohort='diagnostic', row=row, cap=cap,
                bank=str(bankdir/'bank.rvlp'), encoded=str(bankdir/'encoded.json'), old_plan=str(plan)))
    for row in original['rows']:
        if row['router_split'] == 'validation':
            jobs.append(dict(cohort='grouped_validation', row=row, cap=row['sender_parent_cap'],
                bank=row['bank_path'], encoded=row['encoded_path'], old_plan=str(Path(row['order_path']).parent)))
    if len(jobs) != 64: raise ValueError('expected 52 diagnostic + 12 internal-validation groups')
    for job in jobs:
        job.update(bank_sha256=digest(job['bank']), encoded_sha256=digest(job['encoded']),
                   old_plan_sha256=digest(Path(job['old_plan'])/'complete.json'))
        plan = Path(job['old_plan']); verify_artifacts(plan, read(plan/'complete.json')['artifacts'])
        order = read(plan/'order.json')
        if (order['binding']['bank_sha256'] != job['bank_sha256']
                or order['binding']['sender_sha256'] != models['old_sender']['sha256']):
            raise ValueError('old source-sender trajectory differs')
    # Index only P2 R_g + multiband receipts, never old R_g/feather pictures.
    references = []
    for old_job in old_review['jobs']:
        folder = location(data.REVIEW, old_job)/'cooperative'
        pair = read(folder.parent/'pair.json')
        for name in ('stream.rvlcoop', 'receive/complete.json'):
            verified_file(folder/name, pair['artifacts']['cooperative/'+name])
        references.append(dict(wire=str(folder/'stream.rvlcoop'),
            wire_sha256=digest(folder/'stream.rvlcoop'), receive=str(folder/'receive'),
            receipt_sha256=digest(folder/'receive/complete.json')))
    protocol = dict(format='cooperative_sender_review_v1', jobs=jobs, models=models,
        teacher=original['teacher'], receiver_profile=stream.identity(),
        code={n:digest(data.REPO/n) for n in sorted(set(CODE)|set(original['code']))},
        training_complete=digest(training/'complete.json'),
        previous_summary=digest(EVALUATION/'summary.json'), references=references,
        scope='13 reused diagnostics x four E-byte caps; 12 checkpoint-selection windows at assigned E25/E50',
        actual_bpp_required=True, same_cap_is_not_same_rate=True, masks_sent=False,
        freeze=['UF', 'width3', 'P2_R_g_epoch108', 'G', 'multiband'],
        no_optimization=True, no_automatic_promotion=True)
    immutable(root/'protocol.json', protocol)
    return protocol


def check_protocol(root):
    protocol = read(root/'protocol.json')
    for name, expected in protocol['code'].items(): verified_file(data.REPO/name, expected)
    for model in protocol['models'].values(): verified_file(model['path'], model['sha256'])
    verified_file(data.ROOT/'formal/complete.json', protocol['training_complete'])
    verified_file(EVALUATION/'summary.json', protocol['previous_summary'])
    if protocol['receiver_profile'] != stream.identity(): raise ValueError('receiver profile changed')
    return protocol


def plan_job(root, protocol, job, run):
    from demo.routervc_sender_router import prefix_under_budget
    sid = job['row']['sample_id']; folder = root/'planning'/job['cohort']/sid
    verified_file(job['bank'], job['bank_sha256']); bank = Path(job['bank']).read_bytes()
    verified_file(job['encoded'], job['encoded_sha256'])
    # Rebuild the old prefix from its authenticated original trajectory, not new predictions.
    plan = Path(job['old_plan']); verified_file(plan/'complete.json', job['old_plan_sha256'])
    old = read(plan/'order.json'); costs = np.asarray(routing.bundle_bytes(bank), np.int64)
    teacher = protocol['teacher']
    for cap in RATIOS:
        chosen = prefix_under_budget(old['order'], costs, int(costs.sum()*cap))
        raw = stream.wrap(routing.subset(bank, chosen['indices']), teacher['receiver']['sha256'],
                          teacher['G_assets_hash'], teacher['max_g'], teacher['seed'])
        path = folder/'old_sender'/f'e{round(cap*100):03d}.rvlcoop'
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_bytes() != raw: raise ValueError('old prefix changed')
        if not path.exists(): atomic_bytes(path, raw)
    for arm in ARMS[1:]:
        dest = folder/arm
        if not (dest/'complete.json').exists():
            run.update(phase='conditional_sender_planning', sample=sid, arm=arm)
            execute(run, 'plan_'+sid+'_'+arm, ['plan', '--output', root,
                '--sample-id', sid, '--cohort', job['cohort'], '--arm', arm],
                module='tools.cooperative_sender_review')
        verify_artifacts(dest, read(dest/'complete.json')['artifacts'])
    for arm in ARMS:
        previous = None
        for cap in RATIOS:
            raw = (folder/arm/f'e{round(cap*100):03d}.rvlcoop').read_bytes()
            if previous is not None and not raw.startswith(previous): raise ValueError('literal prefix lost')
            previous = raw
    return folder


def verify_point(point):
    for path, expected in point['artifacts'].items(): verified_file(path, expected)
    receive = Path(point['receive']); report = read(receive/'complete.json')
    verify_artifacts(receive, report['artifacts'])
    same_receipt(report, Path(point['wire']), report)
    if report['actual_bytes'] != point['actual_bytes']: raise ValueError('point bytes changed')
    return report


def evaluate(root, protocol, job, plans, arm, metric, run, cache, force_fresh=False):
    from demo.scalable_experiment import quality
    from routervc.fusion.blend import geometry
    folder = location(root, job)/arm; folder.mkdir(parents=True, exist_ok=True)
    point_file = folder/'point.json'
    if point_file.exists():
        point = read(point_file); verify_point(point); cache[point['stream_sha256']] = point['receive']; return point
    path = plans/arm/f'e{round(job["cap"]*100):03d}.rvlcoop'; wire_sha = digest(path)
    reused = not force_fresh and wire_sha in cache
    dest = Path(cache[wire_sha]) if reused else folder/'receive'
    run.update(phase='reuse_identical_stream' if reused else 'fresh_sender_evaluation',
               sample=job['row']['sample_id'], cohort=job['cohort'], cap=job['cap'], arm=arm)
    if reused:
        receipt = read(dest/'complete.json'); verify_artifacts(dest, receipt['artifacts'])
    else:
        receipt = fresh(run, path, dest, protocol)
    same_receipt(receipt, path, receipt)
    inner, config = stream.parse(path.read_bytes())
    if (config['receiver_sha256'] != protocol['teacher']['receiver']['sha256']
            or config['assets_sha256'] != protocol['teacher']['G_assets_hash']
            or config['seed'] != protocol['teacher']['seed'] or config['max_g'] != 8):
        raise ValueError('comparison must use the same frozen receiving system')
    encoded = read(job['encoded'])
    if (receipt['base_hash'] != encoded['base_hash']
            or receipt['detail']['base_reference_hashes'] != encoded['base_reference_hashes']):
        raise ValueError('B reference changed')
    source = source_rgb(job['row'])
    with np.load(dest/'pixels.npz', allow_pickle=False) as z:
        base, received, output = z['base'], z['enhanced'], z['reconstruction']
    for pixels, name in ((base, 'base_hash'), (received, 'enhanced_hash'), (output, 'output_hash')):
        if frame_hash(pixels) != receipt[name]: raise ValueError('pixels disagree with receipt')
    if output.shape != source.shape: raise ValueError('wrong source view')
    support, _, _ = geometry(output.shape, receipt['generated'])
    np.testing.assert_array_equal(output[:, ~support], received[:, ~support])
    if abs(receipt['bpp']-path.stat().st_size*8/np.prod(source.shape[:3])) > 1e-12:
        raise ValueError('bpp must count the complete file')
    run.update(phase='whole_picture_scoring', sample=job['row']['sample_id'], arm=arm)
    point = dict(complete=True, cohort=job['cohort'], sample_id=job['row']['sample_id'],
        dataset=job['row']['dataset'], cap=job['cap'], arm=arm,
        wire=str(path), stream_sha256=wire_sha, receive=str(dest), reused_identical_stream=reused,
        actual_bytes=receipt['actual_bytes'], bpp=receipt['bpp'], quality=quality(source, output, metric),
        received_quality=quality(source, received, metric),
        G_calls=len(receipt['generated']), G_regions=receipt['generated'],
        E_regions=receipt['detail']['received_regions'],
        fresh_seconds=receipt['seconds'], peak_GiB=receipt['peak_cuda_allocated_bytes']/2**30,
        reserved_GiB=receipt['peak_cuda_reserved_bytes']/2**30,
        output_hash=receipt['output_hash'],
        artifacts={str(p):digest(p) for p in (path, dest/'complete.json', dest/'pixels.npz')})
    save(point_file, point); cache[wire_sha] = str(dest); return point


def smoke(root, protocol, run, metric, cache):
    path = root/'smoke.json'
    if path.exists():
        done = read(path); verify_artifacts(root, done['artifacts'])
        if done['protocol'] != digest(root/'protocol.json'): raise ValueError('smoke protocol changed')
        for name in done['point_files']: verify_point(read(root/name))
        return
    names = []; point_files = []
    for dataset in ('REDS', 'UVG'):
        job = next(j for j in protocol['jobs'] if j['cohort']=='diagnostic'
                   and j['row']['dataset']==dataset and j['cap']==.5)
        plans = plan_job(root, protocol, job, run)
        for arm in ARMS[1:]:
            point = evaluate(root, protocol, job, plans, arm, metric, run, cache, force_fresh=True)
            name = str((location(root, job)/arm/'point.json').relative_to(root))
            names.append(name); point_files.append(name)
        point = read(location(root, job)/'source/point.json'); original = verify_point(point)
        for variant in ('repeat', 'Goff'):
            dest = location(root, job)/'source'/variant
            run.update(phase='startup_'+variant, sample=job['row']['sample_id'])
            result = fresh(run, point['wire'], dest, protocol, disabled=variant=='Goff')
            same_receipt(result, Path(point['wire']), original, disabled=variant=='Goff')
            if result['output_hash'] != original['enhanced_hash' if variant=='Goff' else 'output_hash']:
                raise ValueError('deterministic repeat / G-off differs')
            names += [str((dest/n).relative_to(root)) for n in ('complete.json', 'pixels.npz')]
    save(path, dict(complete=True, protocol=digest(root/'protocol.json'), fresh_processes=8,
        two_final_sender_arms=True, repeats_exact=True, Goff_model_free=True,
        sourcefree_senderfree=True, point_files=point_files,
        artifacts={n:digest(root/n) for n in names}))


def verify(root):
    protocol = check_protocol(root); done = read(root/'complete.json')
    verify_artifacts(root, done['artifacts'])
    for job in protocol['jobs']:
        verified_file(job['bank'], job['bank_sha256'])
        verified_file(job['encoded'], job['encoded_sha256'])
        plans = root/'planning'/job['cohort']/job['row']['sample_id']
        for arm in ARMS:
            if arm != 'old_sender': verify_artifacts(plans/arm, read(plans/arm/'complete.json')['artifacts'])
            previous = None
            for cap in RATIOS:
                raw = (plans/arm/f'e{round(cap*100):03d}.rvlcoop').read_bytes()
                if previous is not None and not raw.startswith(previous): raise ValueError('literal prefix lost')
                previous = raw
        for arm in ARMS: verify_point(read(location(root, job)/arm/'point.json'))
    print('COOPERATIVE_SENDER_REVIEW_VERIFIED', root, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('all', 'smoke', 'verify', 'training-report', 'plan'))
    parser.add_argument('--output', type=Path, default=ROOT)
    parser.add_argument('--sample-id'); parser.add_argument('--cohort'); parser.add_argument('--arm', choices=ARMS[1:])
    args = parser.parse_args()
    if not os.environ.get('TMUX'): raise RuntimeError('tmux required')
    if args.command == 'training-report':
        from tools.cooperative_sender_review_report import training_report
        args.output.mkdir(parents=True, exist_ok=True); training_report(args.output); return
    if args.command == 'verify': verify(args.output); return
    configure_torch()
    runroot = args.output/('planning_worker/'+str(args.sample_id)+'/'+str(args.arm) if args.command=='plan' else 'queue')
    run = SenderRun(SimpleNamespace(output=runroot, command='sender_review', max_hours=48)); run.thread.start()
    # Bound the entire review, not just a nested planning-worker directory.
    try:
        if args.command == 'plan':
            protocol = check_protocol(args.output)
            job = next(j for j in protocol['jobs'] if j['row']['sample_id']==args.sample_id and j['cohort']==args.cohort)
            verified_file(job['bank'], job['bank_sha256'])
            sender.write_prefixes(args.output/'planning'/args.cohort/args.sample_id/args.arm,
                source_rgb(job['row']), Path(job['bank']).read_bytes(),
                Path(protocol['models'][args.arm]['path']), check=run.check, progress=run.update)
            return
        protocol = prepare(args.output); check_protocol(args.output)
        if (args.output/'complete.json').exists(): verify(args.output); return
        cache = {}
        for ref in protocol['references']:
            verified_file(ref['wire'], ref['wire_sha256'])
            verified_file(Path(ref['receive'])/'complete.json', ref['receipt_sha256'])
            cache[ref['wire_sha256']] = ref['receive']
        # Reuse successfully saved new points when resuming, with full hash checks.
        for cohort in ('diagnostic', 'grouped_validation'):
            for path in (args.output/cohort).glob('*/*/*/point.json'):
                point = read(path); verify_point(point); cache[point['stream_sha256']] = point['receive']
        from demo.stage_c_three_path_roi_probe import LPIPSAlex
        from tools.cooperative_sender_review_report import training_report, report
        training_report(args.output)
        with exclusive_native_evaluation(run):
            metric = LPIPSAlex(True)
            smoke(args.output, protocol, run, metric, cache)
            if args.command == 'smoke': return
            save(args.output/'formal_started.json', dict(started=True, protocol=digest(args.output/'protocol.json'),
                startup_checks=digest(args.output/'smoke.json'), logical_points=192, optimization=False))
            run.update(phase='formal_sender_evaluation_started', total=192, completed=4)
            points = []
            for job in protocol['jobs']:
                plans = plan_job(args.output, protocol, job, run)
                for arm in ARMS:
                    points.append(evaluate(args.output, protocol, job, plans, arm, metric, run, cache))
                    run.update(phase='formal_point_complete', total=192, completed=len(points))
            run.update(phase='automatic_figures')
            names = report(args.output, protocol, points, run.check)
            names += [str((location(args.output,j)/a/'point.json').relative_to(args.output))
                      for j in protocol['jobs'] for a in ARMS]
            names += ['protocol.json', 'smoke.json', 'formal_started.json', 'training_audit.json', 'training_diagnostic.png']
            save(args.output/'complete.json', dict(complete=True, logical_points=192,
                newly_decoded_points=sum(not p['reused_identical_stream'] for p in points),
                additional_repeat_Goff_checks=4, optimization=False, automatic_promotion=False,
                artifacts={n:digest(args.output/n) for n in names}))
            verify(args.output); run.update(phase='complete_wait_for_picture_review', completed=192)
    except BaseException as error:
        save(run.root/'last_failure.json', dict(error=repr(error), progress=run.progress)); raise
    finally:
        run.log_resources(); run.stop.set(); run.thread.join(timeout=3); run.lock.close()


if __name__ == '__main__': main()
