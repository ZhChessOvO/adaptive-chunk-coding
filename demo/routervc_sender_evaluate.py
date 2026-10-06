"""Diagnostic RD for asymmetric source / zero-source / old fixed-E senders.

All three use the selected R_g and its EXACT training G policy (including seed).
Old E selections are retained but freshly decoded: historical receiver scores
used a different seed. Candidate entropy packets are reused, not re-encoded;
one source-free full-E decode per window authenticates candidate RGB. This is
not a source-to-bitstream encoding-time measurement or an independent benchmark.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import math
import os
from pathlib import Path
from types import SimpleNamespace

from demo.routervc_fullview_probe import read, digest, save, immutable, verify_artifacts
from demo import routervc_receiver_format as fmt
from demo import routervc_receiver_evaluate as receiver_eval
from demo import routervc_sender_encode as encoder
from demo import routervc_sender_train as training
from demo.routervc_light_packets import bank_info, subset_bank
from demo.routervc_visual_evaluate import PhaseDeadline

ROOT = training.ROOT
RATIOS = (0., .25, .5)
ARMS = ('fixed_old_E', 'source', 'zero_source')
CODE = ('routervc_sender_evaluate.py', 'run_routervc_sender_evaluate.sh',
        'routervc_receiver_evaluate.py', 'routervc_visual_evaluate.py')


def point_plan():
    return [dict(name=f'{arm}_e{ratio:g}_g8', arm=arm, ratio=ratio, max_g=8)
            for arm in ARMS for ratio in RATIOS]


def code_hashes():
    names = set(training.CODE) | set(fmt.CODE) | set(CODE)
    return {name: digest(training.REPO/'demo'/name) for name in sorted(names)}


def completed_models(root):
    root = Path(root)
    if not (root/'complete.json').exists():
        return None
    top = read(root/'complete.json')
    protocol = read(root/'formal/protocol.json')
    if (not top.get('complete') or protocol['smoke'] is not False
            or protocol['epochs'] != 120
            or top['protocol'] != digest(root/'formal/protocol.json')
            or top['router'] != digest(root/'formal/router/complete.json')
            or top['labels'] != digest(root/'formal/labels.complete.json')):
        raise ValueError('completed formal sender training required')
    model_root = root/'formal/router'
    result = training.verify_training(model_root, protocol)
    if result['updates_per_arm'] != 11520:
        raise ValueError('sender paired update count differs')
    config = protocol['teacher']['wire_config']
    models = {}
    for arm in training.ARMS:
        path = model_root/arm/'best.pt'
        context = encoder.load_sender(path, config, expected_sha256=digest(path),
            completion_record=model_root/'complete.json',
            expected_completion_sha256=digest(model_root/'complete.json'))
        if context.binding['arm'] != arm or not context.binding['formal_weights']:
            raise ValueError('sender arm is not a completed formal checkpoint')
        models[arm] = dict(path=str(path), sha256=digest(path),
            selection=result['selected'][arm])
    return dict(arms=models, completion=str(model_root/'complete.json'),
        completion_sha256=digest(model_root/'complete.json'),
        top_sha256=digest(root/'complete.json'))


def make_protocol(root, models):
    teacher = read(Path(root)/'formal/protocol.json')['teacher']
    rg_root = Path(teacher['receiver']['complete_path']).parents[2]
    old_root = rg_root/'evaluation'
    old = read(old_root/'protocol.json')
    verify_artifacts(old_root, read(old_root/'complete.json')['artifacts'])
    if old['code'] != receiver_eval.code_hashes():
        raise ValueError('completed receiver evaluator source changed')
    for item in (teacher['receiver'], teacher['enhancement'], teacher['adapter']):
        if digest(item['path']) != item['sha256']:
            raise ValueError('frozen model changed')
    for name, expected in read(Path(root)/'formal/protocol.json')['code'].items():
        if digest(training.REPO/'demo'/name) != expected:
            raise ValueError('pinned sender training code changed: '+name)
    if (len(old['sources']) != 13 or old['qstep'] != 2.
            or teacher['receiver']['sha256'] != teacher['wire_config']['receiver_router']):
        raise ValueError('diagnostic data or fixed receiver binding differs')
    references = old['references']
    for entry in old['sources']:
        sid = entry['sample']['sample_id']
        for ratio in RATIOS:
            receiver_eval.reference(old, sid, ratio)
    return dict(schema='routervc-asymmetric-sender-evaluation-v1', training=str(root),
        models=models, receiver=teacher['receiver'], config=teacher['wire_config'],
        enhancement=teacher['enhancement'], adapter=teacher['adapter'],
        sources=old['sources'], inputs=old['inputs'], references=references,
        prior_protocol=str(old_root/'protocol.json'), prior_protocol_sha256=digest(old_root/'protocol.json'),
        prior_complete=str(old_root/'complete.json'), prior_complete_sha256=digest(old_root/'complete.json'),
        points=point_plan(), code=code_hashes(), qstep=2.,
        budget='same actual E-bank byte caps 0/25/50 percent; realized bytes can differ',
        control='old E selections, freshly decoded with the SAME trained R_g/G config as both new senders',
        noise_scope='same seed/rule in all three arms; changed routes are not universally pixelwise noise-paired',
        candidate_scope='old authentic entropy packet reuse plus fresh full-E source-free decoding; NOT fresh encoding timing',
        expected_points=117, expected_candidate_decodes=13, expected_checks=8,
        expected_prefix_pairs=78, masks_transmitted=False, freeze_UF_E_G_Rg=True,
        independent_system_test=False, on_policy_adaptation=False, semantic_supervision=False)


def decode(run, protocol, stream, output, name, *, off=False):
    from demo.conditioned_generation_pipeline import execute
    output.mkdir(parents=True, exist_ok=True)
    if (output/'decode.json').exists():
        return
    argv = ['--worker', '--stream', stream, '--output', output,
        '--enhancement', protocol['enhancement']['path'],
        '--adapter', output/'absent_G.pt' if off else protocol['adapter']['path'],
        '--router', output/'absent_Rg.pt' if off else protocol['receiver']['path']]
    if off:
        argv.append('--disable-generation')
    execute(run, name, 'routervc_receiver_decode.py', argv, distributed=True)


def prepare_candidates(run, protocol, sample, *, verify_only=False):
    """Independently decode full E, then authenticate it through the sender API."""
    import numpy as np
    from demo.scalable_codec import atomic_bytes
    from demo.scalable_format import frame_hash
    sid = sample['sample_id']; item = protocol['inputs'][sid]
    for path, key in (('source_path', 'source_sha256'), ('bank_path', 'bank_sha256')):
        if digest(item[path]) != item[key]:
            raise ValueError('fixed evaluation source/bank changed')
    bank = Path(item['bank_path']).read_bytes()
    folder = run.root/'samples'/sid/'candidates'
    done = folder/'E.json'
    if not done.exists():
        if verify_only:
            raise ValueError('read-only evaluation cannot prepare candidate pixels')
        folder.mkdir(parents=True, exist_ok=True)
        wire = fmt.wrap(bank, protocol['config'])
        if (folder/'stream.rvrc').exists() and (folder/'stream.rvrc').read_bytes() != wire:
            raise ValueError('partial full-E candidate stream changed')
        if not (folder/'stream.rvrc').exists():
            atomic_bytes(folder/'stream.rvrc', wire)
        decode(run, protocol, folder/'stream.rvrc', folder/'fresh', sid+'_full_E', off=True)
        report = read(folder/'fresh/decode.json')
        if (report['stream_sha256'] != digest(folder/'stream.rvrc')
                or report['config'] != protocol['config']
                or report['total_bytes'] != len(wire)
                or report['source_frames_read'] or report['sender_router_loaded']
                or report['unreceived_candidates_read'] or report['receiver_router_used']
                or report['generation_executed'] or report['generation_assets_validated']
                or not report['base_reference_unchanged']
                or report['output_hash'] != report['generation_input_hash']):
            raise ValueError('full-E source-free candidate decode failed')
        link = folder/'received_E.npz'
        if not link.exists():
            link.symlink_to('fresh/reconstruction.npz')
        if link.resolve() != (folder/'fresh/reconstruction.npz').resolve():
            raise ValueError('candidate RGB link points elsewhere')
        with np.load(link, allow_pickle=False) as arrays:
            if (frame_hash(arrays['base']) != report['base_hash']
                    or frame_hash(arrays['enhanced']) != report['generation_input_hash']):
                raise ValueError('candidate RGB differs from fresh entropy decode')
            np.testing.assert_array_equal(arrays['enhanced'], arrays['reconstruction'])
        save(done, dict(complete=True, protocol=digest(run.root/'protocol.json'),
            source_frames_read=False, sender_candidate_pixels_read=False, base_reference_unchanged=True,
            bank_sha256=item['bank_sha256'], base_hash=report['base_hash'],
            enhanced_hash=report['generation_input_hash'],
            candidate_encoding_reused=True, independent_entropy_decode=True,
            artifacts={n:digest(folder/n) for n in ('received_E.npz','stream.rvrc','fresh/decode.json')}))
    record = read(done)
    if record['protocol'] != digest(run.root/'protocol.json'):
        raise ValueError('candidate preparation protocol changed')
    verify_artifacts(folder, record['artifacts'])
    with np.load(item['source_path'], allow_pickle=False) as data:
        source = data['source'].copy()
    with np.load(folder/'received_E.npz', allow_pickle=False) as data:
        base, all_e = data['base'].copy(), data['enhanced'].copy()
    return encoder.authenticate_candidates(bank, source, base, all_e,
        receive_record=done, expected_receive_sha256=digest(done),
        expected_bank_sha256=item['bank_sha256'], expected_source_rgb_sha256=frame_hash(source))


def load_order(root, protocol, sid, arm, candidates):
    """Persist one order before truncating at any of the evaluation budgets."""
    folder = root/'samples'/sid/'plans'; folder.mkdir(parents=True, exist_ok=True)
    path = folder/(arm+'.json')
    model = protocol['models']['arms'][arm]
    if path.exists():
        plan = read(path)
        if plan['sender']['sha256'] != model['sha256'] or plan['config'] != protocol['config']:
            raise ValueError('saved sender order/model policy changed')
        # Also authenticate the plan hash/candidate identity before reuse.
        encoder.encode_prefix(candidates, plan, 0)
        return plan
    context = encoder.load_sender(model['path'], protocol['config'], expected_sha256=model['sha256'],
        completion_record=protocol['models']['completion'],
        expected_completion_sha256=protocol['models']['completion_sha256'])
    plan = encoder.conditional_order(candidates, context)
    immutable(path, plan)
    return plan


def encode_point(root, protocol, sample, point, candidates):
    from demo.scalable_codec import atomic_bytes
    from demo.scalable_format import frame_hash
    from demo.routervc_encode import compose_candidates
    from demo.routervc_receiver_policy import route
    sid = sample['sample_id']; folder = root/'samples'/sid/point['name']
    folder.mkdir(parents=True, exist_ok=True)
    done = folder/'encode.json'
    binding = dict(protocol=digest(root/'protocol.json'), point=point, candidates=candidates.binding)
    if done.exists():
        value = read(done)
        if value['binding'] != binding:
            raise ValueError('sender encoding binding changed')
        verify_artifacts(folder, value['artifacts'])
        return value
    info = bank_info(candidates.bank); budget = int(sum(info['e_bytes'])*point['ratio'])
    if point['arm'] == 'fixed_old_E':
        old_protocol = read(protocol['prior_protocol'])
        ref, _, old_encoded = receiver_eval.reference(old_protocol, sid, point['ratio'])
        indices = old_encoded['plan']['selected_indices']
        # Confirm selection against the literal historical packet payload.
        from demo import routervc_visual_format as old_fmt
        inner = subset_bank(candidates.bank, indices)
        if inner != old_fmt.parse((ref/'stream.rtvc').read_bytes())[1]:
            raise ValueError('fixed-old-E comparison changed entropy packet bytes')
        wire = fmt.wrap(inner, protocol['config'])
        ledger = dict(indices=indices, packet_bytes=sum(info['e_bytes'][i] for i in indices),
            total_bytes=len(wire), e_budget=budget, candidate_encode_measured_here=False,
            candidate_encode_seconds=None, allocation_seconds=None,
            reference=str(ref), reference_sha256=digest(ref/'result.json'))
    else:
        plan = load_order(root, protocol, sid, point['arm'], candidates)
        wire, ledger = encoder.encode_prefix(candidates, plan, budget)
        indices = ledger['indices']
    config, _, inner, _ = fmt.parse(wire)
    if ledger['packet_bytes'] > budget:
        raise ValueError('sender exceeded actual E byte budget')
    received = compose_candidates(candidates.base, candidates.all_e, indices, info['rois'])
    expected = route(candidates.base, received, inner, config, protocol['receiver']['path'],
                     expected_policy=fmt.policy_identity())
    stream = folder/'stream.rvrc'
    if stream.exists() and stream.read_bytes() != wire:
        raise ValueError('partial sender stream changed')
    if not stream.exists():
        atomic_bytes(stream, wire)
    value = dict(complete=True, binding=binding, point=point, ledger=ledger,
        expected_route=expected, expected_base_hash=frame_hash(candidates.base),
        expected_mixed_hash=frame_hash(received), artifacts={'stream.rvrc':digest(stream)})
    save(done, value)
    return value


def check_point(folder, protocol, sample):
    result = read(folder/'result.json'); encoded = read(folder/'encode.json')
    if (not result.get('complete') or result['protocol'] != digest(folder.parents[2]/'protocol.json')
            or result['point'] not in protocol['points']
            or encoded['point'] != result['point']
            or encoded['binding']['protocol'] != result['protocol']):
        raise ValueError('completed sender point binding changed')
    verify_artifacts(folder, result['artifacts'])
    report = receiver_eval.validate_new(folder, protocol, sample)
    size = (folder/'stream.rvrc').stat().st_size
    if (report != result['decode'] or report['config'] != protocol['config']
            or result['bytes'] != size or encoded['ledger']['total_bytes'] != size
            or encoded['ledger']['packet_bytes'] != report['packet_bytes']
            or not math.isclose(result['bpp'], 8*size/math.prod(receiver_eval.source_shape(sample)[:3]), rel_tol=1e-12)):
        raise ValueError('sender fresh measurement or complete-stream bytes changed')
    return result


def verify_sample(root, protocol, sample):
    """Completed replay still checks external source, bank, and candidate pixels."""
    sid = sample['sample_id']; item = protocol['inputs'][sid]
    for path, sha in (('source_path','source_sha256'),('bank_path','bank_sha256')):
        if digest(item[path]) != item[sha]:
            raise ValueError('completed sender evaluation input changed')
    folder = root/'samples'/sid/'candidates'
    record = read(folder/'E.json')
    if record['protocol'] != digest(root/'protocol.json'):
        raise ValueError('completed candidate protocol changed')
    verify_artifacts(folder, record['artifacts'])


def evaluate(run, protocol, *, verify_only=False):
    import numpy as np
    import torch
    from demo.scalable_experiment import quality
    from demo.stage_c_three_path_roi_probe import LPIPSAlex
    complete = (run.root/'complete.json').exists()
    if verify_only and not complete:
        raise ValueError('read-only verification needs a completed evaluation')
    if complete:
        verify_artifacts(run.root, read(run.root/'complete.json')['artifacts'])
    torch.set_num_threads(4)
    records, checks, prefixes, metric = [], [], [], None
    checked_datasets = set()
    for entry in protocol['sources']:
        sample = entry['sample']; sid = sample['sample_id']
        if complete:
            verify_sample(run.root, protocol, sample)
        candidate = None
        for point in protocol['points']:
            run.check(); folder = run.root/'samples'/sid/point['name']
            if (folder/'result.json').exists():
                result = check_point(folder, protocol, sample)
            else:
                if complete:
                    raise ValueError('completed replay cannot allocate, infer or score')
                if candidate is None:
                    candidate = prepare_candidates(run, protocol, sample)
                encode_point(run.root, protocol, sample, point, candidate)
                decode(run, protocol, folder/'stream.rvrc', folder/'fresh', sid+'_'+point['name'])
                report = receiver_eval.validate_new(folder, protocol, sample)
                if metric is None:
                    metric = LPIPSAlex(True)
                run.update(phase='CPU_quality', sample=sid, point=point['name'])
                with np.load(folder/'fresh/reconstruction.npz', allow_pickle=False) as data:
                    scores = quality(candidate.source, data['reconstruction'], metric)
                    g_off = quality(candidate.source, data['enhanced'], metric)
                size = (folder/'stream.rvrc').stat().st_size
                result = dict(complete=True, protocol=digest(run.root/'protocol.json'),
                    sample_id=sid, dataset=sample['dataset'], point=point, bytes=size,
                    bpp=8*size/math.prod(candidate.source.shape[:3]), quality=scores,
                    same_wire_G_off_quality=g_off, decode=report, reused=False,
                    timing_scope='fresh receiver worker incl loading; sender is candidate-cache allocation only',
                    artifacts={n:digest(folder/n) for n in ('encode.json','stream.rvrc',
                        'fresh/decode.json','fresh/reconstruction.npz')})
                save(folder/'result.json', result)
                check_point(folder, protocol, sample)
            records.append(result)
            run.update(phase='sender_RD_evaluation', completed=len(records), total=protocol['expected_points'])
        for arm in ARMS:
            paths=[run.root/'samples'/sid/f'{arm}_e{r:g}_g8/stream.rvrc' for r in RATIOS]
            for left, right in zip(paths, paths[1:]):
                if not right.read_bytes().startswith(left.read_bytes()):
                    raise ValueError('sender budgets are not literal stream prefixes')
                prefixes.append(dict(sample_id=sid, arm=arm,
                    low=str(left.relative_to(run.root)), high=str(right.relative_to(run.root))))
        # E0 must be bit-identical and yield identical RGB across sender arms.
        zero=[run.root/'samples'/sid/f'{arm}_e0_g8' for arm in ARMS]
        if (len({digest(p/'stream.rvrc') for p in zero}) != 1
                or len({read(p/'fresh/decode.json')['output_hash'] for p in zero}) != 1):
            raise ValueError('E0 output unexpectedly depends on sender model')
        if sample['dataset'] not in checked_datasets:
            checked_datasets.add(sample['dataset'])
            receiver_binding=dict(models=dict(arms={arm:protocol['receiver'] for arm in training.ARMS}))
            for arm in training.ARMS:
                checks += receiver_eval.recovery_checks(run, receiver_binding, sample, arm, verify_only=complete)
        del candidate
    if not complete:
        if (len(records),len(checks),len(prefixes)) != (117,8,78):
            raise ValueError('sender diagnostic coverage differs')
        names=['protocol.json','summary.json']+checks+[
            f'samples/{r["sample_id"]}/{r["point"]["name"]}/result.json' for r in records]
        for entry in protocol['sources']:
            sid=entry['sample']['sample_id']
            names += [f'samples/{sid}/candidates/E.json'] + [f'samples/{sid}/plans/{a}.json' for a in training.ARMS]
        save(run.root/'summary.json',dict(complete=True,points=len(records),fresh_points=len(records),
            candidate_decodes=13,checks=checks,literal_prefix_pairs=prefixes,rows=records,
            independent_system_test=False,on_policy_adaptation=False,semantic_supervision=False))
        save(run.root/'complete.json',dict(complete=True,artifacts={n:digest(run.root/n) for n in names}))
    print('SENDER_EVALUATION_VERIFIED' if complete else 'SENDER_EVALUATION_COMPLETE',flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training',type=Path,default=ROOT)
    p.add_argument('--output',type=Path,default=ROOT/'evaluation')
    p.add_argument('--wait',action='store_true')
    p.add_argument('--verify-only',action='store_true')
    p.add_argument('--max-hours',type=float,default=12.)
    p.add_argument('--wait-hours',type=float,default=72.)
    args=p.parse_args(argv)
    if not os.environ.get('TMUX'):
        raise RuntimeError('sender evaluation requires tmux')
    if any(not math.isfinite(v) or v<=0 for v in (args.max_hours,args.wait_hours)):
        raise ValueError('positive finite deadlines required')
    if args.verify_only and not (args.output/'complete.json').exists():
        raise ValueError('read-only evaluation requires completion')
    from demo.chunk_enhancement_experiment import Run
    from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
    run=Run(SimpleNamespace(output=args.output,command='sender_evaluation',max_hours=args.max_hours+args.wait_hours))
    run.thread.start();deadline=PhaseDeadline(run.check);run.check=deadline.check
    try:
        deadline.begin('waiting_for_sender',args.wait_hours*3600)
        while (models:=completed_models(args.training)) is None:
            if not args.wait or args.verify_only:
                raise ValueError('formal sender training incomplete')
            run.check();run.update(phase='waiting_for_sender',holds_GPU_mutex=False);run.stop.wait(30)
        deadline.begin('sender_evaluation',args.max_hours*3600)
        protocol=make_protocol(args.training,models)
        immutable(run.root/'protocol.json',protocol)
        os.environ['ROUTERVC_RECEIVER_PARENT']=str(os.getpid())
        with nullcontext() if (run.root/'complete.json').exists() else exclusive_native_evaluation(run):
            evaluate(run,protocol,verify_only=args.verify_only)
    except BaseException as error:
        save(run.root/'last_failure.json',dict(error=repr(error),progress=run.progress))
        raise
    finally:
        run.log_resources();run.stop.set();run.thread.join(timeout=3);run.lock.close()


if __name__ == '__main__':
    main()
