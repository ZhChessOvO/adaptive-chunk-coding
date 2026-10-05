"""Sparse, measured FINAL marginal labels for the source-aware sender R_s.

For two fixed received sets per window, render their parents and three distinct
one-packet-bundle additions: eight full receiver results, six measured labels.
Every new state reruns the fixed R_g and frozen G on actual Y. Old single-ROI
scores cannot replace these final LPIPS measurements (G list/seed ordinals may
change). Unknown candidates are masked, not labelled zero or harmless.
"""
from collections import OrderedDict
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from demo import routervc_sender_router as sender
from demo import routervc_receiver_format as fmt
from demo import routervc_receiver_policy as policy
from demo.routervc_receiver_data import make_rows as receiver_rows
from demo.routervc_fullview_probe import read, digest, save, verify_artifacts
from demo.routervc_light_teacher import ENHANCEMENT, ADAPTER
from demo.routervc_light_packets import bank_info, subset_bank, predict_utility
from demo.routervc_encode import compose_candidates
from demo.routervc_visual_policy import allocate
from demo.scalable_codec import atomic_bytes, atomic_npz
from demo.chunk_enhancement_codec import atomic_torch
from demo.scalable_format import frame_hash

OLD_ROUTER = Path('/root/autodl-fs/DCVC/runs/routervc_light_router_20261004/router/global_local/model.pt')
STATE_COUNTS = (2, 4, 8, 12)
LABEL_SCOPE = sender.LABEL_SCOPE


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _seed(value):
    return int(hashlib.sha256(value.encode()).hexdigest()[:16], 16)


def assign_plans(rows):
    """Balance state counts and random/old-prefix sources within dataset/split."""
    counters, result = {}, []
    for row in sorted(rows, key=lambda r: (r['dataset'], r['router_split'], r['sequence'], r['sample_id'])):
        key = (row['dataset'], row['router_split'])
        index = counters.get(key, 0)
        counters[key] = index + 1
        result.append(dict(row, sender_plan=dict(count=STATE_COUNTS[(index//2) % 4],
            kind='random' if index % 2 == 0 else 'old_prefix',
            seed=_seed('sender-initial-v1/'+row['sample_id']))))
    return result


def make_rows(smoke=False):
    rows = assign_plans(receiver_rows(False))
    if smoke:
        rows = [next(r for r in rows if r['dataset'] == dataset and r['router_split'] == 'train'
                     and r['sender_plan']['count'] == count and r['sender_plan']['kind'] == kind)
                for dataset, count, kind in (('REDS', 2, 'random'), ('UVG', 12, 'old_prefix'))]
    return rows


def make_settings(receiver_checkpoint, expected_sha256, *, smoke,
                  receiver_complete=None, seed=20261005):
    """Bind an EXPLICIT R_g; never guess or auto-promote core/halo checkpoints."""
    from demo.routervc_receiver_router import load_model
    if type(smoke) is not bool or type(expected_sha256) is not str or len(expected_sha256) != 64:
        raise ValueError('explicit receiver checkpoint hash and smoke flag required')
    path = Path(receiver_checkpoint).resolve()
    model, payload = load_model(path, expected_sha256=expected_sha256)
    receiver = dict(path=str(path), sha256=expected_sha256, arm=payload['arm'], epoch=payload['epoch'],
                    smoke_weights=payload['binding'].get('protocol', {}).get('smoke', True))
    del model
    if not smoke:
        if receiver_complete is None:
            raise ValueError('formal sender labels require explicit completed R_g training record')
        done = Path(receiver_complete).resolve()
        result = read(done)
        try:
            relative = str(path.relative_to(done.parent))
        except ValueError as error:
            raise ValueError('selected R_g does not belong to the specified completed training') from error
        if (not result.get('complete') or receiver['smoke_weights'] is not False
                or payload['binding'].get('protocol', {}).get('format') != 'routervc_receiver_training_v1'
                or result.get('role') != 'receiver-only conditional G'
                or result.get('artifacts', {}).get(relative) != expected_sha256):
            raise ValueError('formal R_g is incomplete, smoke-only or unauthenticated')
        verify_artifacts(done.parent, result['artifacts'])
        receiver.update(complete_path=str(done), complete_sha256=digest(done))
    config = fmt.make_config(path, ADAPTER, max_g=8, boundary_lambda=0., seed=seed)
    return dict(format='routervc_sender_final_teacher_v1', receiver=receiver, wire_config=config,
        prior_router=dict(path=str(OLD_ROUTER), sha256=digest(OLD_ROUTER)),
        enhancement=dict(path=str(ENHANCEMENT), sha256=digest(ENHANCEMENT)),
        adapter=dict(path=str(ADAPTER), sha256=digest(ADAPTER)),
        sender_config=asdict(sender.Config()), profile=fmt.PROFILE,
        receiver_code=fmt.code_identity(), sender_input_code=sender.code_identity(),
        label_scope=LABEL_SCOPE, objective='whole_frame_lpips_alex', auxiliary_objective_weights=[0., 0.],
        max_g=8, boundary_lambda=0., candidate_count=3, states_per_window=2,
        final_renderings_per_window=8, current_sender_policy_used=False,
        scoring_precision='configure_torch FP32 via codec_precision; no G precision mutation',
        masks_transmitted=False, source_only_in_sender_inputs_and_offline_metrics=True)


def candidates(selected, old_order, packet_bytes, sample_id):
    """Old high density proposal, geometry/price diversity, then uniform random."""
    selected, order = list(selected), list(old_order)
    costs = np.asarray(packet_bytes)
    if (sorted(order) != list(range(16)) or len(set(selected)) != len(selected)
            or any(type(i) is not int or not 0 <= i < 16 for i in selected)
            or costs.shape != (16,) or costs.dtype.kind not in 'iu' or np.any(costs <= 0)):
        raise ValueError('invalid complete prior ordering or actual packet costs')
    available = [i for i in order if i not in selected]
    if len(available) < 3:
        raise ValueError('three distinct unreceived candidates are required')
    first = available[0]
    log_cost = np.log(costs.astype(float))
    span = max(float(log_cost.max()-log_cost.min()), 1e-12)
    def diversity(i):
        distance = (abs(i//4-first//4)+abs(i%4-first%4))/6.
        price = abs(float(log_cost[i]-log_cost[first]))/span
        return distance + price, -i
    second = max(available[1:], key=diversity)
    remaining = [i for i in available if i not in (first, second)]
    rng = np.random.default_rng(_seed('sender-candidate-v1/'+sample_id+'/'+str(sorted(selected))))
    third = remaining[int(rng.integers(len(remaining)))]
    return [first, second, third]


def complete_proposal_order(utility, packet_bytes, old_prefix):
    """Keep the real old prefix, rank the rest ONLY as candidate proposals.

    The deployed legacy allocator stops at nonpositive gain. Continuing its
    same marginal/byte arithmetic here does not extend that deployed prefix.
    """
    from demo.routervc_encode import _table_score
    utility, costs, chosen = np.asarray(utility, dtype=np.float64), np.asarray(packet_bytes), list(old_prefix)
    if (utility.shape != (16,4) or not np.isfinite(utility).all() or costs.shape != (16,)
            or costs.dtype.kind not in 'iu' or np.any(costs <= 0)
            or len(set(chosen)) != len(chosen)
            or any(type(i) is not int or not 0 <= i < 16 for i in chosen)):
        raise ValueError('invalid old table, positive packet costs or literal prefix')
    score,_ = _table_score(utility,chosen,8)
    remaining = set(range(16))-set(chosen)
    while remaining:
        options=[]
        for region in sorted(remaining):
            value,_ = _table_score(utility,chosen+[region],8)
            options.append(((value-score)/int(costs[region]),-region,region,value))
        _,_,region,score = max(options)
        chosen.append(region)
        remaining.remove(region)
    return chosen


def plan_states(row, old_order, packet_bytes, proposal_order=None):
    plan = row['sender_plan']
    if plan['count'] not in STATE_COUNTS or plan['kind'] not in ('random', 'old_prefix'):
        raise ValueError('invalid fixed sender teacher state')
    old_order = list(old_order)
    if (len(set(old_order)) != len(old_order)
            or any(type(i) is not int or not 0 <= i < 16 for i in old_order)):
        raise ValueError('old prefix must contain unique complete region bundles')
    proposals = old_order if proposal_order is None else list(proposal_order)
    if sorted(proposals) != list(range(16)) or proposals[:len(old_order)] != old_order:
        raise ValueError('complete proposal order must preserve, not fake, the old prefix')
    fallback = plan['kind'] == 'old_prefix' and len(old_order) < plan['count']
    actual_kind = 'random_fallback' if fallback else plan['kind']
    order = (old_order if actual_kind == 'old_prefix'
             else np.random.default_rng(plan['seed']).permutation(16).tolist())
    selections = [[], order[:plan['count']]]
    return [dict(selected=selected, kind='empty' if not selected else actual_kind,
                 requested_kind='empty' if not selected else plan['kind'],
                 requested_count=0 if not selected else plan['count'], actual_count=len(selected),
                 original_old_prefix=old_order, proposal_order=proposals,
                 fallback_reason='old_positive_gain_prefix_shorter_than_requested' if selected and fallback else None,
                 candidate_roles=['old_marginal_per_byte_proposal','geometry_price_diversity','uniform_random'],
                 candidates=candidates(selected, proposals, packet_bytes, row['sample_id']))
            for selected in selections]


def measured_targets(plans, records, packet_bytes):
    """Six measured final LPIPS differences; all other entries remain unknown."""
    values, weights = torch.zeros(2, 16), torch.zeros(2, 16)
    diagnostics = []
    if len(plans) != 2 or len(records) != 2:
        raise ValueError('two sender states required')
    for state, (plan, measured) in enumerate(zip(plans, records)):
        parent = measured['parent']
        for region in plan['candidates']:
            child = measured['children'][str(region)]
            additional = child['total_bytes']-parent['total_bytes']
            if additional != int(packet_bytes[region]) or additional <= 0:
                raise ValueError('candidate actual byte increment differs from its bundle')
            value = parent['quality']['lpips_alex']-child['quality']['lpips_alex']
            if not np.isfinite(value):
                raise ValueError('final LPIPS marginal must be measured and finite')
            values[state, region], weights[state, region] = value, 1.
            diagnostics.append(dict(state=state, region=region, delta_lpips=float(value),
                incremental_bytes=additional, delta_lpips_per_byte=float(value/additional),
                Goff_parent=parent['Goff_quality'], Goff_child=child['Goff_quality'],
                parent_G=parent['route']['indices'], child_G=child['route']['indices'],
                G_selection_changed=parent['route']['indices'] != child['route']['indices']))
    return dict(value=values, weight=weights, label_scope=LABEL_SCOPE), diagnostics


def measure_state(dest, bank, base, all_e, selected, source, config, router, *, binding,
                  generate, score, route=policy.route, retain_pixels=False, check=lambda: None):
    """One atomic full final receive/render/score. G and R_g never get source X."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    check()
    if (not all(isinstance(v, np.ndarray) and v.dtype == np.uint8 for v in (source, base, all_e))
            or base.ndim != 4 or base.shape[-1] != 3 or source.shape != base.shape or all_e.shape != base.shape):
        raise ValueError('paired uint8 source/base/candidate videos required before rendering')
    info = bank_info(bank)
    selected = list(selected)
    received = compose_candidates(base, all_e, selected, info['rois'])
    raw = subset_bank(bank, selected)
    wire = fmt.wrap(raw, config)
    parsed_config, _, inner, header = fmt.parse(wire)
    input_binding = dict(binding=binding, selected=selected, received_hash=frame_hash(received),
        bank_sha256=hashlib.sha256(bank).hexdigest(), config=parsed_config,
        receiver_policy=fmt.policy_identity(), retain_pixels=bool(retain_pixels),
        score_source_hash=frame_hash(source), metrics='whole-frame LPIPS-Alex, PSNR, temporal_delta_mae')
    done = dest/'result.json'
    if done.exists():
        result = read(done)
        if result['binding'] != input_binding or result['render_key'] != _hash(result['render_identity']):
            raise ValueError('saved final rendering binding changed')
        verify_artifacts(dest, result['artifacts'])
        if (result['total_bytes'] != len(wire)
                or result['stream_sha256'] != hashlib.sha256(wire).hexdigest()):
            raise ValueError('saved final rendering byte identity changed')
        return result
    atomic_bytes(dest/'stream.rvrc', wire)
    actual_bytes = (dest/'stream.rvrc').stat().st_size
    if actual_bytes != len(wire) or digest(dest/'stream.rvrc') != hashlib.sha256(wire).hexdigest():
        raise ValueError('final teacher stream differs from actual written bytes')
    began = time.monotonic()
    # Every new S AND every child S+j calls R_g again on its own actual Y.
    selection = route(base, received, inner, parsed_config, router, expected_policy=fmt.policy_identity())
    control = fmt.generation_control(parsed_config, selection['indices'], info['rois'], len(base))
    render_identity = dict(received_hash=frame_hash(received), base_hash=frame_hash(base),
        wire_config=parsed_config, ordered_G_indices=selection['indices'], control=control,
        receiver_code=fmt.code_identity(),
        noise_scheme='seed + 65536 * ordered_G_ordinal + temporal_window_ordinal')
    if selection['indices']:
        with torch.no_grad():
            output, runtime = generate(received, control)
    else:
        output, runtime = received.copy(), None
    from demo import scalable_cooperation_format as cooperation
    alpha = cooperation.weights(base.shape, control)
    if output.dtype != np.uint8 or output.shape != received.shape:
        raise ValueError('G returned invalid final RGB')
    np.testing.assert_array_equal(output[alpha == 0], received[alpha == 0])
    # Ground truth is used only here, after source-free routing and generation.
    measured, direct = score(source, output), score(source, received)
    if any(not np.isfinite(item.get('lpips_alex', np.nan)) for item in (measured, direct)):
        raise ValueError('whole-frame LPIPS scoring failed')
    names = ['stream.rvrc']
    if retain_pixels:
        atomic_npz(dest/'reconstruction.npz', reconstruction=output, enhanced=received)
        names.append('reconstruction.npz')
    result = dict(complete=True, binding=input_binding, render_key=_hash(render_identity),
        render_identity=render_identity, route=selection, quality=measured, Goff_quality=direct,
        generation_runtime=runtime, generation_input_hash=frame_hash(received), output_hash=frame_hash(output),
        base_hash=frame_hash(base), total_bytes=actual_bytes, header_bytes=header,
        E_packet_bytes=sum(info['e_bytes'][i] for i in selected), stream_sha256=hashlib.sha256(wire).hexdigest(),
        source_frames_used_by_receiver=False, source_frames_used_by_generator=False,
        source_only_for_offline_scores=True, masks_transmitted=False, explicit_mask_bytes=0,
        final_metric_scope='whole supplied frame; not sum of local G scores',
        pixels_retained=bool(retain_pixels), generation_executed=bool(selection['indices']),
        seconds=time.monotonic()-began, artifacts={name:digest(dest/name) for name in names})
    save(done, result)
    return result


class Samples:
    def __init__(self, root, cache, protocol, check=lambda: None, max_cached=2):
        if type(max_cached) is not int or not 1 <= max_cached <= 2:
            raise ValueError('sender compact LRU is limited to one or two windows')
        self.root, self.cache, self.check = Path(root), Path(cache), check
        if isinstance(protocol, str):
            if digest(self.root/'protocol.json') != protocol:
                raise ValueError('sender protocol hash mismatch')
            self.protocol = read(self.root/'protocol.json')
        else:
            self.protocol = protocol
        # Equivalent dictionary/file callers use the same semantic binding.
        self.protocol_hash = _hash(self.protocol)
        self.settings = self.protocol['teacher']
        if (self.settings['label_scope'] != LABEL_SCOPE or self.settings['max_g'] != 8
                or self.settings['wire_config']['max_g'] != 8):
            raise ValueError('initial final-marginal teacher requires fixed G8')
        self.max_cached, self.memory, self.verified = max_cached, OrderedDict(), {}
        self.generator = self.metric = self.prior = self.receiver = None
        self.peak = 0

    def get(self, row, *, generate=True):
        sid = row['sample_id']
        if sid in self.memory:
            self.memory.move_to_end(sid)
            return self.memory[sid]
        path, done = self.cache/f'{sid}.pt', self.root/'samples'/sid/'complete.json'
        binding = dict(protocol=self.protocol_hash, sample=row, teacher=self.settings)
        if not done.exists():
            if not generate:
                raise ValueError('read-only sender replay cannot generate missing labels')
            self.prepare(row, path, done, binding)
        result = read(done)
        if result['binding'] != binding:
            raise ValueError('sender compact cache binding changed')
        signature = (path.stat().st_size, path.stat().st_mtime_ns, digest(done))
        if self.verified.get(sid) != signature:
            if digest(path) != result['cache_sha256']:
                raise ValueError('sender compact tensors changed')
            verify_artifacts(done.parent, result['artifacts'])
            for name, expected in result['reused_artifacts'].items():
                if digest(name) != expected:
                    raise ValueError('source/candidate teacher artifact changed')
            self.verified[sid] = signature
        value = torch.load(path, weights_only=True, map_location='cpu')
        value.update(sample_id=sid, dataset=row['dataset'])
        self.memory[sid] = value
        while len(self.memory) > self.max_cached:
            self.memory.popitem(last=False)
        return value

    def _generate(self, pixels, control):
        from demo.four_state_receive import PersistentRGB
        if self.generator is None:
            self.generator = PersistentRGB()
        output, runtime = self.generator(pixels, control)
        self.peak = max(self.peak, torch.cuda.max_memory_allocated(),
            *(w['runtime']['peak_cuda_allocated_bytes'] for w in runtime['windows']))
        return output, runtime

    def _score(self, source, output):
        from demo.stage_c_three_path_roi_probe import LPIPSAlex
        from demo.scalable_experiment import quality
        from demo.four_state_receive import codec_precision
        if self.metric is None:
            self.metric = LPIPSAlex(True)
        with codec_precision():
            return quality(source, output, self.metric)

    def prepare(self, row, path, done, binding):
        self.check()
        reused = {}
        for name, expected in (('path','sha256'), ('source_path','source_sha256'),
                              ('reconstruction_path','reconstruction_sha256'),
                              ('bank_path','bank_sha256'), ('receive_record','receive_record_sha256')):
            if digest(row[name]) != row[expected]:
                raise ValueError('changed source/candidate binding: '+name)
            reused[row[name]] = row[expected]
        for name in ('receiver', 'prior_router', 'enhancement', 'adapter'):
            asset = self.settings[name]
            if digest(asset['path']) != asset['sha256']:
                raise ValueError('frozen sender teacher asset changed: '+name)
            reused[asset['path']] = asset['sha256']
        receiver = self.settings['receiver']
        if not self.protocol.get('smoke', False) and (receiver.get('smoke_weights') is not False
                or not receiver.get('complete_path')):
            raise ValueError('formal sender teacher cannot use smoke or incomplete R_g weights')
        if 'complete_path' in receiver:
            if digest(receiver['complete_path']) != receiver['complete_sha256']:
                raise ValueError('explicit receiver completion record changed')
            reused[receiver['complete_path']] = receiver['complete_sha256']
        if (self.settings['receiver_code'] != fmt.code_identity()
                or self.settings['sender_input_code'] != sender.code_identity()):
            raise ValueError('frozen sender teacher code changed')
        if (Path(self.settings['adapter']['path']).resolve() != Path(ADAPTER).resolve()
                or Path(self.settings['enhancement']['path']).resolve() != Path(ENHANCEMENT).resolve()):
            raise ValueError('this fixed teacher only supports the completed joint E/G assets')
        bank = Path(row['bank_path']).read_bytes()
        info = bank_info(bank)
        with np.load(row['reconstruction_path'], allow_pickle=False) as data:
            base, all_e = data['base'].copy(), data['enhanced'].copy()
        received_record = read(row['receive_record'])
        if (frame_hash(base) != received_record['base_hash']
                or frame_hash(all_e) != received_record['enhanced_hash']):
            raise ValueError('actual source-free candidate RGB changed')
        with np.load(row['source_path'], allow_pickle=False) as data:
            source = data['source'].copy()
        if self.prior is None:
            from demo.routervc_visual_policy import load_model
            self.prior = load_model(self.settings['prior_router']['path'],
                expected_sha256=self.settings['prior_router']['sha256'])
        if self.receiver is None:
            from demo.routervc_receiver_router import load_model
            self.receiver, _ = load_model(receiver['path'], expected_sha256=receiver['sha256'])
        utility, _ = predict_utility(bank, base, all_e, self.prior)
        old_plan = allocate(utility, np.asarray(info['e_bytes']), sum(info['e_bytes']), 8, mode='prefix')
        proposal_order = complete_proposal_order(utility,info['e_bytes'],old_plan['prefix_order'])
        plans = plan_states(row, old_plan['prefix_order'], info['e_bytes'], proposal_order)
        config = self.settings['wire_config']
        if config['receiver_router'] != receiver['sha256'] or config['policy'] != fmt.policy_identity():
            raise ValueError('explicit receiver differs from frozen wire profile')
        retained = self.protocol.get('smoke', False) or row['sample_id'] in self.protocol.get('retain_pixels_samples', [])
        batches, records, artifacts = [], [], {}
        for state, plan in enumerate(plans):
            self.check()
            folder = done.parent/f'state{state}'
            sets = [('parent', plan['selected'])] + [(f'add_{i:02d}', plan['selected']+[i]) for i in plan['candidates']]
            measured = {}
            for name, selected in sets:
                result = measure_state(folder/name, bank, base, all_e, selected, source, config, self.receiver,
                    binding=dict(sample=binding, state=state, candidate=name), generate=self._generate,
                    score=self._score, retain_pixels=retained, check=self.check)
                measured[name] = result
                for artifact, expected in result['artifacts'].items():
                    artifacts[str((folder/name/artifact).relative_to(done.parent))] = expected
                artifacts[str((folder/name/'result.json').relative_to(done.parent))] = digest(folder/name/'result.json')
                print(f'SENDER_FINAL_LABEL {row["sample_id"]} state={state} candidate={name}', flush=True)
            records.append(dict(parent=measured['parent'], children={str(i):measured[f'add_{i:02d}'] for i in plan['candidates']}))
            coverage = np.zeros(16, np.float32)
            coverage[plan['selected']] = 1
            received = compose_candidates(base, all_e, plan['selected'], info['rois'])
            batches.append(sender.build_inputs(source, base, received, all_e, coverage,
                np.asarray(info['e_bytes'], dtype=np.int64), 8,
                config=sender.Config(**self.settings['sender_config'])))
        targets, diagnostics = measured_targets(plans, records, info['e_bytes'])
        value = dict(inputs={key:torch.cat([v[key] for v in batches]) for key in sender.INPUT_KEYS},
            targets=targets, plans=plans, diagnostics=diagnostics,
            render_records=[[str((done.parent/f'state{s}'/n/'result.json').resolve())
                for n in ['parent', *(f'add_{i:02d}' for i in p['candidates'])]] for s,p in enumerate(plans)],
            packet_bytes=torch.tensor(info['e_bytes'], dtype=torch.int64).expand(2,16).clone())
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_torch(path, value)
        save(done, dict(complete=True, binding=binding, cache_path=str(path), cache_sha256=digest(path),
            artifacts=artifacts, reused_artifacts=reused, measured_final_renderings=8, measured_marginals=6,
            current_sender_policy_used=False, old_proposal=dict(prefix_order=old_plan['prefix_order'],
                complete_proposal_order=proposal_order,
                prior_model_sha256=self.settings['prior_router']['sha256'], proposals_only=True),
            label_scope=LABEL_SCOPE, masks_transmitted=False,
            scope='two sampled states, three candidates each; NOT exhaustive 16-region allocation oracle'))
        print(f'SENDER_SAMPLE_READY {row["sample_id"]}', flush=True)

    def release_generator(self):
        self.generator = self.metric = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def audit_prior_plans(output):
    """CPU-only plan validation on authenticated real B/Y compact inputs.

    Reconstruct the old four-state prediction table from its 32 cached isolated
    receiver views. No GT quality labels, native codec, diffusion or GPU runs.
    """
    from demo.routervc_visual_policy import load_model
    rows = make_rows(False)
    chosen=[]
    for dataset in ('REDS','UVG'):
        for split, count in (('train',8),('validation',4)):
            chosen.extend([r for r in rows if r['dataset']==dataset and r['router_split']==split][:count])
    model = load_model(OLD_ROUTER)
    records=[]
    for row in chosen:
        if digest(row['cache_path']) != row['cache_sha256']:
            raise ValueError('authenticated old compact inputs changed')
        cache=torch.load(row['cache_path'],weights_only=True,map_location='cpu')
        with torch.no_grad():
            prediction=model(cache['inputs'])['gains'][:,0].numpy()
        if prediction.shape != (32,6):
            raise ValueError('expected measured old isolated B/E input pairs')
        direct=prediction[1::2,0]
        utility=np.stack((np.zeros_like(direct),direct,prediction[::2,1],direct+prediction[1::2,1]),1)
        bank=Path(row['bank_path']).read_bytes()
        if hashlib.sha256(bank).hexdigest()!=row['bank_sha256']:
            raise ValueError('old candidate bank changed')
        costs=bank_info(bank)['e_bytes']
        old=allocate(utility,np.asarray(costs),sum(costs),8,mode='prefix')['prefix_order']
        proposal=complete_proposal_order(utility,costs,old)
        plans=plan_states(row,old,costs,proposal)
        if plans[1]['actual_count']!=row['sender_plan']['count']:
            raise ValueError('requested density was lost')
        records.append(dict(sample_id=row['sample_id'],dataset=row['dataset'],split=row['router_split'],
            old_positive_prefix_length=len(old),plans=plans,packet_bytes=costs))
        print('SENDER_REAL_PLAN_AUDIT',row['sample_id'],len(old),plans[1]['kind'],flush=True)
    result=dict(complete=True,samples=len(records),source='authenticated old compact decoded-B/Y inputs',
        source_quality_labels_used=False,GPU_used=False,receiver_or_generator_rendered=False,
        prior_model_sha256=digest(OLD_ROUTER),data_code_sha256=digest(Path(__file__)),records=records,
        random_fallbacks=sum(r['plans'][1]['kind']=='random_fallback' for r in records))
    save(output,result)
    return result


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(description='CPU-only sender plan audit; does not generate teacher labels')
    parser.add_argument('--plan-audit',type=Path,required=True)
    args=parser.parse_args()
    import os
    if not os.environ.get('TMUX'):
        raise RuntimeError('run real multi-window audit in tmux')
    audit_prior_plans(args.plan_audit)
