"""Measured G-only supervision on six actual received enhancement densities.

E0 and E4/E8 reuse unchanged measured teachers. E2/E12/E16 run frozen G
on the actual mixed Y, never synthesize targets from isolated gains. All
patterns are nested region counts, not byte budgets. Source is offline scoring
only. Bounded compact-tensor caching avoids retaining every video in RAM.
"""
from collections import OrderedDict
import gc
import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from demo import routervc_visual_router as visual
from demo import routervc_mixed_router as mixed
from demo.routervc_fullview_probe import read, digest, save, verify_artifacts
from demo.routervc_light_teacher import ENHANCEMENT, ADAPTER
from demo.routervc_light_packets import bank_info, subset_bank
from demo.routervc_encode import compose_candidates
from demo.scalable_codec import atomic_bytes, atomic_npz
from demo.chunk_enhancement_codec import atomic_torch
from demo.scalable_format import frame_hash

OLD = Path('/root/autodl-fs/DCVC/runs/routervc_mixed_router_20261004')
VIEW_NAMES = ('e0', 'e2', 'e4', 'e8', 'e12', 'e16')
COUNTS = (0, 2, 4, 8, 12, 16)
NEW_COUNTS = (2, 12, 16)


def selection(sample_id, count):
    if count not in COUNTS:
        raise ValueError('unsupported enhancement region count')
    seed = int(hashlib.sha256(('mixed-v1/'+sample_id).encode()).hexdigest()[:16], 16)
    return np.random.default_rng(seed).permutation(16)[:count].tolist()


def make_rows(smoke=False):
    protocol_path = OLD/'formal/protocol.json'
    previous = read(protocol_path)
    complete = read(OLD/'complete.json')
    if not complete['complete'] or complete['protocol'] != digest(protocol_path):
        raise ValueError('old mixed training is incomplete or changed')
    for name, expected in previous['code'].items():
        if digest(Path(__file__).parent/name) != expected:
            raise ValueError('historical source changed: '+name)
    labels_path = OLD/'formal/labels.complete.json'
    if complete['labels'] != digest(labels_path):
        raise ValueError('historical label manifest changed')
    labels = read(labels_path)
    rows = previous['rows']
    if smoke:
        rows = [next(r for r in rows if r['dataset'] == dataset and r['router_split'] == 'train')
                for dataset in ('REDS', 'UVG')]
    result = []
    for row in rows:
        done = OLD/'formal/samples'/row['sample_id']/'complete.json'
        if digest(done) != labels['samples'][row['sample_id']]:
            raise ValueError('old mixed sample record changed')
        old = read(done)
        result.append(dict(row, mixed_record=str(done), mixed_record_sha256=digest(done),
            mixed_cache_path=old['cache_path'], mixed_cache_sha256=old['cache_sha256']))
    return result


class Samples:
    def __init__(self, root, cache, protocol, check=lambda: None, max_cached=8):
        if type(max_cached) is not int or max_cached < 1:
            raise ValueError('positive bounded cache size required')
        self.root, self.cache, self.protocol, self.check = Path(root), Path(cache), protocol, check
        self.generator = self.metric = None
        self.memory, self.verified = OrderedDict(), {}
        self.max_cached, self.peak = max_cached, 0

    def get(self, row, *, generate=True):
        sid = row['sample_id']
        if sid in self.memory:
            self.memory.move_to_end(sid)
            return self.memory[sid]
        path, done = self.cache/f'{sid}.pt', self.root/'samples'/sid/'complete.json'
        binding = dict(protocol=self.protocol, sample=row)
        if not done.exists():
            if not generate:
                raise ValueError('read-only verification cannot generate labels')
            self.prepare(row, path, done, binding)
        record = read(done)
        if record['binding'] != binding:
            raise ValueError('receiver sample binding changed')
        signature = (path.stat().st_size, path.stat().st_mtime_ns, digest(done))
        if self.verified.get(sid) != signature:
            if digest(path) != record['cache_sha256']:
                raise ValueError('receiver compact tensors changed')
            verify_artifacts(done.parent, record['artifacts'])
            for p, expected in record['reused_artifacts'].items():
                if digest(p) != expected:
                    raise ValueError('reused teacher changed: '+p)
            self.verified[sid] = signature
        value = torch.load(path, weights_only=True, map_location='cpu')
        value.update(dataset=row['dataset'], sample_id=sid)
        self.memory[sid] = value
        while len(self.memory) > self.max_cached:
            self.memory.popitem(last=False)
        return value

    def prepare(self, row, path, done, binding):
        from demo.four_state_receive import PersistentRGB
        from demo import scalable_cooperation_format as fmt
        from demo.scalable_experiment import quality
        from demo.stage_c_three_path_roi_probe import LPIPSAlex
        from demo.routervc_mixedview_teacher import crop
        from demo.online_eg_eval_core import noise_pair
        from demo.routervc_receiver_router import g_only_targets
        self.check()
        dest = done.parent
        dest.mkdir(parents=True, exist_ok=True)
        reused = {}
        for name, expected in (('path','sha256'), ('source_path','source_sha256'),
                ('reconstruction_path','reconstruction_sha256'), ('bank_path','bank_sha256'),
                ('receive_record','receive_record_sha256'), ('mixed_record','mixed_record_sha256'),
                ('mixed_cache_path','mixed_cache_sha256')):
            if digest(row[name]) != row[expected]:
                raise ValueError('changed source/teacher binding: '+name)
            reused[row[name]] = row[expected]
        label = read(row['path'])
        bank = Path(row['bank_path']).read_bytes()
        info = bank_info(bank)
        with np.load(row['reconstruction_path'], allow_pickle=False) as data:
            base, enhanced = data['base'].copy(), data['enhanced'].copy()
        er = read(row['receive_record'])
        if frame_hash(base) != er['base_hash'] or frame_hash(enhanced) != er['enhanced_hash']:
            raise ValueError('entropy decoded receiver pixels mismatch')
        with np.load(row['source_path'], allow_pickle=False) as data:
            source = data['source'].copy()
        original = torch.load(row['mixed_cache_path'], weights_only=True, map_location='cpu')
        rois = info['rois']
        qualities = [r['quality'] for r in label['regions']]
        views, halos, selections, artifacts = [], [], [], {}
        for count in COUNTS:
            selected = selection(row['sample_id'], count)
            selections.append(selected)
            coverage = np.zeros(16, np.float32)
            coverage[selected] = 1
            if count in (4, 8):
                index = (4, 8).index(count)
                if selected != original['selections'][index]:
                    raise ValueError('reused selection differs')
                views.append(dict(inputs={k:v[index:index+1] for k,v in original['inputs'].items()},
                    targets=g_only_targets({k:{f:v[index:index+1] for f,v in fields.items()}
                                            for k,fields in original['targets'].items()})))
                halos.append(original['halo_pairs'][index:index+1])
                continue
            received = compose_candidates(base, enhanced, selected, rois)
            if count == 0:
                # G|B measurements and preprocessing/control are unchanged.
                targets = g_only_targets(visual.build_view_targets(qualities, 0))
            else:
                scores = []
                for i, roi in enumerate(rois):
                    self.check()
                    folder = dest/f'e{count}/cell_{i:02d}'
                    folder.mkdir(parents=True, exist_ok=True)
                    cell_path = folder/'result.json'
                    reference = row['cell_records'][i]
                    if digest(reference['path']) != reference['sha256']:
                        raise ValueError('old frozen G control changed')
                    reused[reference['path']] = reference['sha256']
                    old = read(reference['path'])
                    settings = old['control']
                    cell_binding = dict(sample=binding, e_regions=count, selected=selected, region=i,
                        control=settings, condition_hash=frame_hash(received))
                    if cell_path.exists():
                        result = read(cell_path)
                        if result['binding'] != cell_binding:
                            raise ValueError('receiver cell binding changed')
                        verify_artifacts(folder, result['artifacts'])
                    else:
                        if self.generator is None:
                            self.generator = PersistentRGB()
                        wire = fmt.wrap(subset_bank(bank, selected), settings)
                        atomic_bytes(folder/'teacher.acsg', wire)
                        began = time.monotonic()
                        with torch.no_grad():
                            output, runtime = self.generator(received, settings)
                        self.peak = max(self.peak, torch.cuda.max_memory_allocated(),
                            *(w['runtime']['peak_cuda_allocated_bytes'] for w in runtime['windows']))
                        mask = fmt.weights(base.shape, settings)
                        np.testing.assert_array_equal(output[mask == 0], received[mask == 0])
                        noise_pair(dict(generation_runtime=old['report']['runtime']),
                                   dict(generation_runtime=runtime))
                        if self.metric is None:
                            self.metric = LPIPSAlex(True)
                        measured = quality(crop(source, roi), crop(output, roi), self.metric)
                        atomic_npz(folder/'generated.npz', generated=crop(output, roi))
                        result = dict(binding=cell_binding, quality=measured,
                            output_hash=frame_hash(output), generation_input_hash=frame_hash(received),
                            outside_generate_exact=True, generator_reads_source=False,
                            source_used_for_offline_scoring=True, runtime=runtime,
                            seconds=time.monotonic()-began, total_bytes=len(wire),
                            e_packet_bytes=sum(info['e_bytes'][k] for k in selected),
                            teacher_controls_charged=True, deployment_G_mask=False,
                            peak_cuda_allocated_bytes=self.peak,
                            artifacts={n:digest(folder/n) for n in ('teacher.acsg','generated.npz')})
                        save(cell_path, result)
                        del output, mask
                    scores.append(result['quality'])
                    for name, expected in result['artifacts'].items():
                        artifacts[str((folder/name).relative_to(dest))] = expected
                    artifacts[str(cell_path.relative_to(dest))] = digest(cell_path)
                    print(f'RECEIVER_LABEL {row["sample_id"]} e={count} cell={i+1}/16', flush=True)
                targets = g_only_targets(mixed.mixed_targets(qualities, selected, scores))
            inputs = mixed.build_inputs(base, received, coverage)
            halos.append(mixed.build_inputs(base, received, coverage, halo=64)['local_pairs'])
            views.append(dict(inputs=inputs, targets=targets))
        value = dict(inputs={k:torch.cat([v['inputs'][k] for v in views]) for k in visual.INPUT_KEYS},
            targets={k:torch.cat([v['targets'][k] for v in views]) for k in ('value','weight')},
            halo_pairs=torch.cat(halos), selections=selections, view_names=list(VIEW_NAMES))
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_torch(path, value)
        save(done, dict(complete=True, binding=binding, cache_path=str(path), cache_sha256=digest(path),
            artifacts=artifacts, reused_artifacts=reused, new_measured_G_cells=48, reused_G_cells=48,
            scope='local conditional G ranking; six fixed region counts, not whole-video RD'))
        print(f'RECEIVER_SAMPLE_READY {row["sample_id"]}', flush=True)

    def release_generator(self):
        self.generator = self.metric = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
