"""Lazily measure frozen-G responses to actual multi-E reconstructions.

Labels may be prepared during epoch one; a sample enters optimization only after
its measured labels and compact tensors are atomically complete. This is a fixed
offline teacher, not on-policy learning. Receiver inputs never include GT.
"""
import gc
from pathlib import Path
import time

import numpy as np
import torch

from demo import routervc_visual_router as visual
from demo import routervc_mixed_router as mixed
from demo.routervc_fullview_probe import read, digest, save, immutable, verify_artifacts
from demo.routervc_light_teacher import ENHANCEMENT, ADAPTER
from demo.routervc_light_packets import bank_info, subset_bank
from demo.routervc_encode import compose_candidates
from demo.scalable_codec import atomic_bytes, atomic_npz
from demo.chunk_enhancement_codec import atomic_torch
from demo.scalable_format import frame_hash

OLD = Path('/root/autodl-fs/DCVC/runs/routervc_light_router_20261004')
OLD_CACHE = Path('/root/autodl-tmp/DCVC/cache/routervc_light_20261004/formal')


def manifest(smoke=False):
    from demo.routervc_visual_train import teacher_entries
    entries = teacher_entries(OLD/'teacher/labels.json')
    previous = read(OLD/'teacher/protocol.json')
    for key, values in (('code', previous['code']), ('training_code', previous['training_code'])):
        for name, expected in values.items():
            if digest(Path(__file__).parent/name) != expected:
                raise ValueError(f'historical {key} source changed: {name}')
    if not read(OLD/'teacher/complete.json')['complete']:
        raise ValueError('q2 teacher incomplete')
    cache = read(OLD_CACHE/'complete.json')
    if cache['binding']['labels_sha256'] != digest(OLD/'teacher/labels.json'):
        raise ValueError('old compact cache differs from q2 labels')
    caches = {r['sample_id']:r for r in cache['records']}
    if smoke:
        entries = [next(e for e in entries if e['dataset'] == d and e['router_split'] == 'train')
                   for d in ('REDS', 'UVG')]
    rows = []
    for e in entries:
        label = read(e['path']); sid = e['sample_id']
        bank = OLD/'teacher/encoded'/sid/'bank.acse'
        er = OLD/'teacher/received'/sid/'E.json'
        record = read(er); verify_artifacts(er.parent, record['artifacts'])
        if (record['source_frames_read'] or record['sender_candidate_pixels_read']
                or record['bank_sha256'] != digest(bank)):
            raise ValueError('q2 receive provenance failed')
        rows.append(dict(**e, bank_path=str(bank), bank_sha256=digest(bank),
            cache_path=caches[sid]['cache_path'], cache_sha256=caches[sid]['cache_sha256'],
            source_path=label['source_path'], source_sha256=label['source_sha256'],
            receive_record=str(er), receive_record_sha256=digest(er),
            cell_records=[dict(path=str(er.parent/f'cell_{i:02d}/result.json'),
                sha256=digest(er.parent/f'cell_{i:02d}/result.json')) for i in range(16)]))
    return rows


class Samples:
    def __init__(self, root, cache, protocol, check=lambda: None):
        self.root, self.cache, self.protocol, self.check = Path(root), Path(cache), protocol, check
        self.generator = self.metric = None
        self.memory = {}
        self.peak = 0

    def get(self, row, *, generate=True):
        sid = row['sample_id']
        if sid in self.memory:
            return self.memory[sid]
        path = self.cache/f'{sid}.pt'; done = self.root/'samples'/sid/'complete.json'
        binding = dict(protocol=self.protocol, sample=row)
        if done.exists():
            record = read(done)
            if record['binding'] != binding or digest(path) != record['cache_sha256']:
                raise ValueError('mixed cache binding changed')
            verify_artifacts(done.parent, record['artifacts'])
        else:
            if not generate:
                raise ValueError('read-only replay cannot generate missing labels')
            self.prepare(row, path, done, binding)
        if digest(row['cache_path']) != row['cache_sha256']:
            raise ValueError('isolated control cache changed')
        value = torch.load(path, weights_only=True, map_location='cpu')
        original = torch.load(row['cache_path'], weights_only=True, map_location='cpu')
        # Same target cells and E/no-E status as the mixed arms; only surrounding
        # reconstructed pixels and conditional G labels differ.
        ids = torch.tensor([2*i+int(i in selected) for selected in value['selections'] for i in range(16)])
        isolated = dict(inputs={k:v[ids] for k,v in original['inputs'].items()},
            targets={k:{f:v[ids] for f,v in fields.items()} for k,fields in original['targets'].items()})
        value.update(isolated=isolated, dataset=row['dataset'], sample_id=sid)
        self.memory[sid] = value
        return value

    def prepare(self, row, path, done, binding):
        from demo.four_state_receive import PersistentRGB
        from demo import scalable_cooperation_format as fmt
        from demo.scalable_experiment import quality
        from demo.stage_c_three_path_roi_probe import LPIPSAlex
        from demo.routervc_mixedview_teacher import crop
        from demo.online_eg_eval_core import noise_pair
        self.check(); dest = done.parent; dest.mkdir(parents=True, exist_ok=True)
        for name, expected in (('path','sha256'), ('source_path','source_sha256'),
                ('reconstruction_path','reconstruction_sha256'), ('bank_path','bank_sha256'),
                ('receive_record','receive_record_sha256')):
            if digest(row[name]) != row[expected]: raise ValueError('changed source/receiver binding: '+name)
        label = read(row['path']); bank = Path(row['bank_path']).read_bytes(); info = bank_info(bank)
        with np.load(row['reconstruction_path'], allow_pickle=False) as data:
            base, enhanced = data['base'].copy(), data['enhanced'].copy()
        er = read(row['receive_record'])
        if frame_hash(base) != er['base_hash'] or frame_hash(enhanced) != er['enhanced_hash']:
            raise ValueError('actual receive pixels mismatch')
        with np.load(row['source_path'], allow_pickle=False) as data: source = data['source'].copy()
        rois = info['rois']; qualities = [r['quality'] for r in label['regions']]
        views, halo_pairs, selections, artifacts = [], [], [], {}
        for j in range(2):
            selected = mixed.selection(row['sample_id'], j); selections.append(selected)
            received = compose_candidates(base, enhanced, selected, rois)
            coverage = np.zeros(16, np.float32); coverage[selected] = 1
            scores = []
            for i, roi in enumerate(rois):
                self.check(); folder = dest/f'mix{j}/cell_{i:02d}'; folder.mkdir(parents=True, exist_ok=True)
                cell_path = folder/'result.json'; old_ref = row['cell_records'][i]
                if digest(old_ref['path']) != old_ref['sha256']: raise ValueError('old G controls changed')
                old = read(old_ref['path']); settings = old['control']
                cell_binding = dict(sample=binding, mixture=j, selected=selected, region=i,
                    control=settings, condition_hash=frame_hash(received))
                if cell_path.exists():
                    result = read(cell_path)
                    if result['binding'] != cell_binding: raise ValueError('mixed cell changed')
                    verify_artifacts(folder, result['artifacts'])
                else:
                    if self.generator is None: self.generator = PersistentRGB()
                    wire = fmt.wrap(subset_bank(bank, selected), settings)
                    atomic_bytes(folder/'teacher.acsg', wire)
                    began = time.monotonic()
                    # Only entropy-decoded B/Y enter G. GT is used BELOW for scores.
                    with torch.no_grad(): output, runtime = self.generator(received, settings)
                    self.peak = max(self.peak, torch.cuda.max_memory_allocated(),
                        *(w['runtime']['peak_cuda_allocated_bytes'] for w in runtime['windows']))
                    mask = fmt.weights(base.shape, settings)
                    np.testing.assert_array_equal(output[mask == 0], received[mask == 0])
                    noise_pair(dict(generation_runtime=old['report']['runtime']), dict(generation_runtime=runtime))
                    if self.metric is None: self.metric = LPIPSAlex(True)
                    measured = quality(crop(source, roi), crop(output, roi), self.metric)
                    atomic_npz(folder/'generated.npz', generated=crop(output, roi))
                    result = dict(binding=cell_binding, quality=measured, output_hash=frame_hash(output),
                        generation_input_hash=frame_hash(received), outside_generate_exact=True,
                        generator_reads_source=False, source_used_for_offline_scoring=True,
                        runtime=runtime, seconds=time.monotonic()-began, total_bytes=len(wire),
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
                print(f'MIXED_LABEL {row["sample_id"]} view={j+1}/2 cell={i+1}/16', flush=True)
            inputs = mixed.build_inputs(base, received, coverage)
            halo_pairs.append(mixed.build_inputs(base, received, coverage, halo=64)['local_pairs'])
            views.append(dict(inputs=inputs, targets=mixed.mixed_targets(qualities, selected, scores)))
        value = visual.batch_examples(views)
        value.update(halo_pairs=torch.cat(halo_pairs), selections=selections)
        path.parent.mkdir(parents=True, exist_ok=True); atomic_torch(path, value)
        save(done, dict(complete=True, binding=binding, cache_path=str(path), cache_sha256=digest(path),
            artifacts=artifacts, scope='measured conditional G on real mixed Y; no whole-video RD claim'))
        print(f'MIXED_SAMPLE_READY {row["sample_id"]}', flush=True)

    def release_generator(self):
        self.generator = self.metric = None
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
