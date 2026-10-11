"""P2b final-picture E marginals on authenticated frozen R_s trajectories.

The small pilot reuses packet banks and source views, never stale G pixels or
old marginal labels. Every changed received Y reruns P2 R_g and frozen G/F.
"""
from collections import Counter, OrderedDict
from dataclasses import asdict
import gc
import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from demo import routervc_sender_router as network, routervc_sender_train as old_fit
from demo.routervc_fullview_probe import read, save, digest, immutable, verify_artifacts
from demo.scalable_codec import atomic_bytes, atomic_npz
from demo.scalable_format import frame_hash
from routervc.latent import routing, router_data, sender_data as previous_sender
from routervc.cooperation import data as receiver_data, receiver, stream

REPO = Path(__file__).resolve().parents[2]
ROOT = Path('/root/autodl-fs/DCVC/runs/routervc_cooperative_sender_20261011')
RECEIVER = receiver_data.ROOT/'router/best.pt'
RECEIVER_SHA = '2cf9c14de85bb86fe9f976bfc80a3ebe52901f6b56b7f9aacf0c44558ef3335a'
REVIEW = Path('/root/autodl-fs/DCVC/runs/routervc_cooperation_review_20261011')
FORMAT = 'cooperative_sender_p2b_v1'
CODE = (*previous_sender.CODE, *receiver_data.CODE,
        'routervc/cooperation/sender_data.py', 'routervc/cooperation/sender.py',
        'routervc/cooperation/sender_training.py', 'tools/cooperative_sender.py',
        'tools/cooperative_sender_worker.py', 'tools/cooperative_sender_decode.py', 'tools/run_cooperative_sender.sh',
        'tools/test_cooperative_sender.py', 'tools/cooperative_worker.py',
        'tools/latent_router_worker.py', 'demo/chunk_enhancement_experiment.py',
        'demo/chunk_enhancement_evaluate.py', 'tools/storage_guard.py')


def assign_caps(rows):
    """Alternate real byte caps within each dataset/split, before reading scores."""
    counts, result = Counter(), []
    for row in sorted(rows, key=lambda r:(r['dataset'], r['router_split'], r['sample_id'])):
        group = row['dataset'], row['router_split']
        result.append(dict(row, sender_parent_cap=(.25, .5)[counts[group] % 2]))
        counts[group] += 1
    return result


def make_protocol(smoke=False):
    previous = read(receiver_data.PREVIOUS/'protocol.json')
    completed = read(receiver_data.PREVIOUS/'complete.json')
    if not completed['complete'] or completed['protocol_sha256'] != digest(receiver_data.PREVIOUS/'protocol.json'):
        raise ValueError('completed actual sender trajectories required')
    selected = read(receiver_data.ROOT/'router/complete.json')
    review = read(REVIEW/'complete.json')
    if (not selected['complete'] or selected['best']['epoch'] != 108
            or selected['artifacts']['best.pt'] != RECEIVER_SHA
            or digest(RECEIVER) != RECEIVER_SHA or not review['complete']
            or review['paired_states'] != 64):
        raise ValueError('reviewed preselected P2 epoch108 required')
    init = Path(previous['sender'])
    sender_done = read(previous_sender.ROOT/'formal/router/complete.json')
    if (not sender_done['complete'] or digest(init) != previous['sender_sha256']
            or sender_done['artifacts']['source/best.pt'] != digest(init)):
        raise ValueError('completed source-visible R_s warm start required')
    initial, payload = network.load_model(init, expected_sha256=previous['sender_sha256'])
    if initial.config.zero_source or payload['step'] <= 0 or payload['binding']['protocol']['smoke']:
        raise ValueError('warm start must be a trained formal source sender')
    config = asdict(initial.config); del initial
    rows = assign_caps(previous['rows'])
    if (Counter(r['router_split'] for r in rows) != dict(train=48, validation=12)
            or Counter(r['dataset'] for r in rows) != dict(REDS=45, UVG=15)):
        raise ValueError('preserve 60-window grouped pilot')
    groups = {}
    for row in rows:
        key = row['dataset'], row['sequence']
        if groups.setdefault(key, row['router_split']) != row['router_split']:
            raise ValueError('sequence crosses split')
        plan = receiver_data.PREVIOUS/'samples'/row['sample_id']/'planning'
        done = read(plan/'complete.json'); verify_artifacts(plan, done['artifacts'])
        order = read(plan/'order.json')
        if (order['binding']['sender_sha256'] != previous['sender_sha256']
                or order['binding']['bank_sha256'] != row['bank_sha256']):
            raise ValueError('frozen source sender trajectory differs')
        row.update(order_path=str(plan/'order.json'), order_sha256=digest(plan/'order.json'),
                   planning_complete=str(plan/'complete.json'), planning_sha256=digest(plan/'complete.json'))
    if smoke:
        rows = [next(r for r in rows if r['dataset'] == d and r['router_split'] == 'train'
                     and r['sender_parent_cap'] == cap) for d,cap in (('REDS', .25), ('UVG', .5))]
    p2 = read(receiver_data.ROOT/'protocol.json')
    from routervc.latent.generation import asset_hash
    from routervc.latent.packet_codec import packet_hash
    return dict(format=FORMAT, revision=FORMAT, smoke=smoke, rows=rows,
        code={n:digest(REPO/n) for n in sorted(set(CODE))},
        receiver_profile=stream.identity(), packet_profile=packet_hash(),
        initialization=dict(path=str(init), sha256=previous['sender_sha256'],
            kind='same_trained_source_sender_for_both_arms', optimizer_reset=True,
            completed_path=str(previous_sender.ROOT/'formal/router/complete.json'),
            completed_sha256=digest(previous_sender.ROOT/'formal/router/complete.json')),
        teacher=dict(sender_config=config, receiver=dict(path=str(RECEIVER), sha256=RECEIVER_SHA),
            G_assets=p2['G_assets'], G_assets_hash=asset_hash(p2['G_assets']), max_g=8, seed=routing.SEED),
        receiver_review=dict(path=str(REVIEW/'complete.json'), sha256=digest(REVIEW/'complete.json')),
        trajectory_protocol_sha256=digest(receiver_data.PREVIOUS/'protocol.json'),
        arms=['source', 'zero_source'], epochs=2 if smoke else 120, seed=20261011,
        learning_rate=1e-4, weight_decay=1e-4, ranking_weight=.1,
        label_scope=network.LABEL_SCOPE, parents='empty + actual frozen R_s E25/E50 byte prefix',
        candidates='signed R_s gain/byte; source-Y MSE/byte; deterministic spatial-price diversity',
        measured_renderings_per_window=8, measured_marginals_per_window=6,
        current_sender_policy_used=True, teacher_frozen_during_fit=True,
        source_only_in_encoder_and_offline_scoring=True, masks_sent=False,
        freeze=['UF', 'E_width3', 'P2_R_g', 'G', 'multiband_F'],
        labels='LPIPS(parent final multiband)-LPIPS(child final multiband); negative retained',
        cache='lossless FP32 compressed CPU inputs, LRU two; no formal full RGB or G halos',
        output_limit_bytes=8*1024**3,
        scope='bounded development pilot; sparse candidate validation, not independent test or full oracle',
        evaluation_pending=True, automatic_promotion=False)


def trajectory_parent(order, cap, costs):
    if order['bundle_bytes'] != list(map(int, costs)):
        raise ValueError('trajectory actual packet costs differ')
    parent = network.prefix_under_budget(order['order'], np.asarray(costs, np.int64), int(sum(costs)*cap))['indices']
    if not parent or len(parent) > 13:
        raise ValueError('bounded partial parent needs one to thirteen real received regions')
    return parent


def proposals(selected, order, source, received, costs):
    """Old predictions select proposals only; all supervision is newly measured."""
    step = next((s for s in order['steps'] if s['selected'] == list(selected)), None)
    if step is None or step['received_hash'] != frame_hash(received):
        raise ValueError('parent differs from actual frozen sender trajectory')
    costs = np.asarray(costs, np.int64); gains = np.asarray(step['gains'], dtype=float)
    if gains.shape != (16,) or not np.isfinite(gains).all():
        raise ValueError('invalid frozen sender proposal gains')
    remaining = [i for i in range(16) if i not in selected]
    if len(remaining) < 3: raise ValueError('three distinct additions required')
    first = min(remaining, key=lambda i:(-gains[i]/int(costs[i]), i))
    error = []
    for x, y, w, h in network.grid_rois(*source.shape[1:3]):
        d = source[:, y:y+h, x:x+w].astype(np.float32)-received[:, y:y+h, x:x+w]
        error.append(float(np.mean(d*d)))
    second = min((i for i in remaining if i != first), key=lambda i:(-error[i]/int(costs[i]), i))
    prices = np.log(costs.astype(float)); span = max(float(np.ptp(prices)), 1e-12)
    def diverse(i):
        value = min((abs(i//4-j//4)+abs(i%4-j%4))/6.
                    + abs(prices[i]-prices[j])/span for j in (first, second))
        return value, -i
    third = max((i for i in remaining if i not in (first, second)), key=diverse)
    return dict(selected=list(selected), candidates=[first, second, third],
        kind='empty' if not selected else 'actual_frozen_sender_prefix',
        candidate_roles=['old_sender_signed_gain_per_byte', 'source_Y_error_per_byte', 'spatial_price_diversity'],
        sender_would_stop=not bool(step['ranked']), parent_received_hash=frame_hash(received))


class Samples:
    def __init__(self, root, cache, protocol, check=lambda:None, progress=lambda **kw:None):
        self.root, self.cache, self.protocol = Path(root), Path(cache), protocol
        self.check, self.progress = check, progress
        self.memory = OrderedDict(); self.model = self.generator = self.metric = None
        self.measured = 0; self.stop_after_renderings = 0

    def _measure(self, folder, bank, selected, source, encoded):
        teacher = self.protocol['teacher']; folder.mkdir(parents=True, exist_ok=True)
        inner = routing.subset(bank, selected)
        wire = stream.wrap(inner, teacher['receiver']['sha256'], teacher['G_assets_hash'],
                           max_g=teacher['max_g'], seed=teacher['seed'])
        binding = dict(protocol=digest(self.root/'protocol.json'), selected=selected,
            source_hash=frame_hash(source), bank_sha256=hashlib.sha256(bank).hexdigest())
        if (folder/'result.json').exists():
            result = read(folder/'result.json')
            if (result['binding'] != binding or result['stream_sha256'] != hashlib.sha256(wire).hexdigest()
                    or result['total_bytes'] != len(wire)):
                raise ValueError('saved final teacher state differs')
            verify_artifacts(folder, result['artifacts']); return result
        self.check(); began = time.monotonic(); atomic_bytes(folder/'stream.rvlcoop', wire)
        base, received, detail = router_data.decode(inner)
        if (frame_hash(base) != encoded['base_hash']
                or detail['base_reference_hashes'] != encoded['base_reference_hashes']):
            raise ValueError('child E packets changed B reference')
        if self.model is None:
            self.model, _ = receiver.load(teacher['receiver']['path'], teacher['receiver']['sha256'])
        route = receiver.select(self.model, base, received, routing.coverage(routing.packets.parse(inner)), teacher['max_g'])
        patches, reports = [], []
        if route['indices'] and self.generator is None:
            from routervc.latent.generation import assets
            if assets() != teacher['G_assets']: raise ValueError('frozen G assets changed')
            from routervc.fusion.capture import CaptureRGB
            self.generator = CaptureRGB()
        from routervc.cooperation.render import render
        with torch.no_grad():
            for region in route['indices']:
                self.check(); self.progress(phase='sender_final_G', state=str(folder), region=region)
                _, raw, report = self.generator(received,
                    routing.control(received.shape, teacher['G_assets'], region, teacher['seed']))
                patches.append(dict(region=region, pixels=raw, crop=report['crop'], core=report['core']))
                reports.append(report)
            output = render(received, patches, route['indices']) if route['indices'] else received.copy()
        from demo.four_state_receive import codec_precision
        from demo.scalable_experiment import quality
        if self.metric is None:
            from demo.stage_c_three_path_roi_probe import LPIPSAlex
            self.metric = LPIPSAlex(True)
        with torch.no_grad(), codec_precision():
            final = quality(source, output, self.metric); direct = quality(source, received, self.metric)
        if not all((k == 'psnr_db' and v is None) or (v is not None and np.isfinite(v))
                   for q in (final, direct) for k,v in q.items()):
            raise ValueError('nonfinite whole-picture teacher metric')
        result = dict(complete=True, binding=binding, quality=final, Goff_quality=direct,
            route=route, base_hash=frame_hash(base), received_hash=frame_hash(received),
            output_hash=frame_hash(output), total_bytes=len(wire), header_bytes=stream.HEADER_BYTES,
            stream_sha256=digest(folder/'stream.rvlcoop'), runtime=reports,
            seconds=time.monotonic()-began, source_read_by_receiver=False, mask_bytes=0,
            actual_latent_mixture=True, multiband_final=True, stale_G_cache_reused=False,
            artifacts={'stream.rvlcoop':digest(folder/'stream.rvlcoop')})
        save(folder/'result.json', result); self.measured += 1
        self.progress(phase='sender_final_teacher', measured_this_process=self.measured, state=str(folder))
        if self.stop_after_renderings and self.measured >= self.stop_after_renderings:
            raise InterruptedError('intentional final-teacher restart check')
        return result

    def prepare(self, row):
        folder = self.root/'samples'/row['sample_id']; folder.mkdir(parents=True, exist_ok=True)
        if (folder/'complete.json').exists(): return self.verify(row)
        self.check()
        for key in ('bank', 'encoded', 'order'):
            if digest(row[key+'_path']) != row[key+'_sha256']: raise ValueError(key+' input changed')
        source = router_data.source(row); bank = Path(row['bank_path']).read_bytes()
        encoded = read(row['encoded_path']); order = read(row['order_path'])
        base, full, _ = router_data.decode(bank)
        if frame_hash(base) != encoded['base_hash'] or frame_hash(full) != encoded['full_hash']:
            raise ValueError('authenticated full bank endpoint changed')
        costs = routing.bundle_bytes(bank); partial = trajectory_parent(order, row['sender_parent_cap'], costs)
        _, mixed, _ = router_data.decode(routing.subset(bank, partial))
        plans = [proposals(s, order, source, y, costs) for s,y in (([], base), (partial, mixed))]
        inputs, records = [], []
        for index, (plan, y) in enumerate(zip(plans, (base, mixed))):
            cov = routing.coverage(routing.packets.parse(routing.subset(bank, plan['selected'])))
            inputs.append(network.build_inputs(source, base, y, full, cov,
                np.asarray(costs, np.int64), self.protocol['teacher']['max_g'],
                config=network.Config(**self.protocol['teacher']['sender_config'])))
            parent = self._measure(folder/f's{index}/parent', bank, plan['selected'], source, encoded)
            if parent['received_hash'] != frame_hash(y): raise ValueError('teacher and sender Y differ')
            children = {str(i):self._measure(folder/f's{index}/add{i}', bank, [*plan['selected'], i], source, encoded)
                        for i in plan['candidates']}
            records.append(dict(parent=parent, children=children))
        from demo.routervc_sender_data import measured_targets
        targets, diagnostics = measured_targets(plans, records, np.asarray(costs, np.int64))
        packed = {f'input_{k}':torch.cat([v[k] for v in inputs]).numpy() for k in inputs[0]}
        packed.update(value=targets['value'].numpy(), weight=targets['weight'].numpy())
        cache = self.cache/(row['sample_id']+'.npz'); cache.parent.mkdir(parents=True, exist_ok=True)
        atomic_npz(cache, **packed); save(folder/'plans.json', dict(plans=plans, diagnostics=diagnostics, packet_bytes=costs))
        names = ['plans.json', *(f's{s}/{n}/result.json' for s,p in enumerate(plans)
                  for n in ['parent', *(f'add{i}' for i in p['candidates'])])]
        done = dict(complete=True, binding=dict(protocol=digest(self.root/'protocol.json'), row=row),
            cache_path=str(cache), cache_sha256=digest(cache), measured_final_renderings=8,
            measured_marginals=6, label_scope=network.LABEL_SCOPE, current_sender_policy_used=True,
            old_labels_reused=False, artifacts={n:digest(folder/n) for n in names})
        save(folder/'complete.json', done); return done

    def verify(self, row):
        folder = self.root/'samples'/row['sample_id']; done = read(folder/'complete.json')
        if (done['binding'] != dict(protocol=digest(self.root/'protocol.json'), row=row)
                or digest(done['cache_path']) != done['cache_sha256'] or not done['current_sender_policy_used']):
            raise ValueError('new teacher binding or cache differs')
        for key in ('bank', 'encoded', 'order'):
            if digest(row[key+'_path']) != row[key+'_sha256']: raise ValueError(key+' input changed')
        verify_artifacts(folder, done['artifacts'])
        return done

    def get(self, row, generate=False):
        sid = row['sample_id']
        if sid not in self.memory:
            done = self.prepare(row) if generate else self.verify(row)
            with np.load(done['cache_path'], allow_pickle=False) as z:
                inputs = {k[6:]:torch.from_numpy(z[k].copy()) for k in z.files if k.startswith('input_')}
                targets = {k:torch.from_numpy(z[k].copy()) for k in ('value', 'weight')}
            targets['label_scope'] = network.LABEL_SCOPE
            self.memory[sid] = dict(inputs=inputs, targets=targets,
                plans=read(self.root/'samples'/sid/'plans.json')['plans'], packet_bytes=inputs['packet_bytes'].squeeze(-1))
        self.memory.move_to_end(sid)
        while len(self.memory) > 2: self.memory.popitem(last=False)
        return self.memory[sid]

    def release(self):
        self.memory.clear(); self.model = self.generator = self.metric = None; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()


def prepare_labels(root, protocol, samples, check=lambda:None, progress=lambda **kw:None):
    root = Path(root); records = {}
    for index, row in enumerate(protocol['rows']):
        check(); samples.get(row, generate=True)
        path = root/'samples'/row['sample_id']/'complete.json'; records[row['sample_id']] = digest(path)
        status = dict(phase='sender_final_teacher', completed_windows=index+1, windows=len(protocol['rows']),
            measured_final_renderings=8*(index+1), measured_marginals=6*(index+1), optimizer_started=False)
        save(root/'teacher.progress.json', status); progress(**status)
    result = dict(format='cooperative_sender_labels_v1', complete=True,
        protocol_sha256=old_fit.semantic_hash(protocol), samples=records, windows=len(records),
        measured_final_renderings=8*len(records), measured_marginals=6*len(records),
        current_sender_policy_used=True, no_UF_E_G_Rg_F_updates=True, label_scope=network.LABEL_SCOPE)
    immutable(root/'labels.complete.json', result); return result


def verify_labels(root, protocol, samples, check=lambda:None):
    root = Path(root); labels = read(root/'labels.complete.json')
    if (labels['protocol_sha256'] != old_fit.semantic_hash(protocol)
            or labels.get('format') != 'cooperative_sender_labels_v1' or not labels.get('complete')
            or labels.get('current_sender_policy_used') is not True
            or labels.get('measured_marginals') != 6*len(protocol['rows'])
            or labels.get('measured_final_renderings') != 8*len(protocol['rows'])):
        raise ValueError('teacher protocol or completion changed')
    for row in protocol['rows']:
        check(); folder = root/'samples'/row['sample_id']; samples.verify(row)
        if labels['samples'][row['sample_id']] != digest(folder/'complete.json'): raise ValueError('teacher receipt changed')
        batch = samples.get(row); old_fit.arm_batch(batch, 'source', 'cpu')
        bank = Path(row['bank_path']).read_bytes()
        for state, plan in enumerate(batch['plans']):
            parent = read(folder/f's{state}/parent/result.json')
            for name, selected in [('parent', plan['selected']),
                    *((f'add{i}', [*plan['selected'], i]) for i in plan['candidates'])]:
                dest = folder/f's{state}'/name; result = read(dest/'result.json')
                verify_artifacts(dest, result['artifacts']); wire = (dest/'stream.rvlcoop').read_bytes()
                inner, config = stream.parse(wire)
                if (inner != routing.subset(bank, selected) or result['stream_sha256'] != digest(dest/'stream.rvlcoop')
                        or result['total_bytes'] != len(wire) or result['mask_bytes'] != 0
                        or config['receiver_sha256'] != protocol['teacher']['receiver']['sha256']
                        or config['assets_sha256'] != protocol['teacher']['G_assets_hash']
                        or result['binding']['protocol'] != digest(root/'protocol.json')):
                    raise ValueError('teacher actual stream binding differs')
                if name != 'parent':
                    region = int(name[3:]); gain = parent['quality']['lpips_alex']-result['quality']['lpips_alex']
                    if (np.float32(gain) != batch['targets']['value'][state, region].item()
                            or result['total_bytes']-parent['total_bytes'] != int(batch['packet_bytes'][state, region])):
                        raise ValueError('cached signed final marginal differs')
    return labels
