"""Re-measure conditional G labels on entropy-decoded new latent B/E.

No old E labels, cut-and-paste RGB mixtures, semantic pseudo-labels, or stored
full generated videos. Six state journals resume after each measured G cell;
lossless compressed FP32 Router tensors use a bounded two-window RAM cache.
"""
from collections import Counter, OrderedDict
import gc
import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from demo.routervc_fullview_probe import read, digest, save, immutable, verify_artifacts
from demo.scalable_codec import atomic_bytes, atomic_npz
from demo.scalable_format import frame_hash
from routervc.latent import routing

REPO = Path(__file__).resolve().parents[2]
ROOT = Path('/root/autodl-fs/DCVC/runs/routervc_latent_routers_20261008')
CACHE = Path('/root/autodl-tmp/DCVC/cache/routervc_latent_routers_20261008')
DATA = Path('/root/autodl-fs/DCVC/runs/routervc_revision_20261003/mixedview_data')
INITIAL = Path('/root/autodl-fs/DCVC/runs/routervc_receiver_20261005/formal/router/core/best.pt')
COUNTS = (0, 2, 4, 8, 12, 16)
VIEW_NAMES = tuple(f'e{n}' for n in COUNTS)
CODE = ('routervc/latent/routing.py', 'routervc/latent/router_data.py',
        'routervc/latent/receiver_fit.py', 'tools/latent_router_worker.py',
        'tools/latent_router_queue.py', 'tools/run_latent_routers.sh',
        'demo/routervc_receiver_train.py', 'demo/routervc_fullview_probe.py',
        'demo/four_state_receive.py', 'demo/scalable_experiment.py',
        'demo/stage_c_three_path_roi_probe.py', 'demo/routervc_mixedview_teacher.py')


def rows(smoke=False):
    done = read(DATA/'complete.json')
    if not done['complete']:
        raise ValueError('mixed-view sources incomplete')
    result = []
    for entry in done['samples']:
        if digest(entry['view_json']) != entry['view_sha256']:
            raise ValueError('source view record changed')
        view = read(entry['view_json'])
        sample = view['sample']
        directory = Path(view['frames_dir'])
        files = [directory/f'{i:08d}.png' for i in range(17)]
        if sample['dataset'] == 'REDS':
            parent = Path(entry['view_json']).parent/'fullview'
            if digest(parent/'complete.json') != view['fullview_complete_sha256']:
                raise ValueError('REDS prepared record changed')
            recorded = read(parent/'complete.json')['artifacts']
            hashes = [recorded[f'frames/{p.name}'] for p in files]
        else:
            recorded = view['source_hashes']['files']
            hashes = [recorded[str(p)] for p in files]
        result.append(dict(sample_id=sample['sample_id'], dataset=sample['dataset'],
            sequence=sample['sequence'], router_split=sample['router_split'],
            view_kind=sample['view_kind'], files=list(map(str, files)), hashes=hashes,
            shape=[17, *reversed(sample['transform']['coded_size']), 3],
            view_json=entry['view_json'], view_sha256=entry['view_sha256']))
    if Counter(r['dataset'] for r in result) != dict(REDS=90, UVG=30) or Counter(
            r['router_split'] for r in result) != dict(train=96, validation=24):
        raise ValueError('preserve existing 120-window grouped split')
    groups = {}
    for row in result:
        group = row['dataset'], row['sequence']
        if groups.setdefault(group, row['router_split']) != row['router_split']:
            raise ValueError('source sequence crosses Router split')
    if smoke:
        return [next(r for r in result if r['dataset'] == d and r['router_split'] == 'train')
                for d in ('REDS', 'UVG')]
    reds = [r for r in result if r['dataset'] == 'REDS']
    uvg = [r for r in result if r['dataset'] == 'UVG']
    return [r for i in range(30) for r in (*reds[3*i:3*i+3], uvg[i])]


def make_protocol(smoke=False):
    from routervc.latent.packet_codec import packet_hash
    from routervc.latent import generation
    from demo.routervc_receiver_router import code_identity, load_model
    model, payload = load_model(INITIAL,
        expected_sha256='53b5b5fb9a8b5ee0c834e8af3cfbdf7c95a0e031c8c55a1dc882bb541deaca02')
    if payload['arm'] != 'core':
        raise ValueError('selected receiver initialization must be core')
    del model
    assets = generation.assets()
    return dict(format='routervc_latent_receiver_training_v1', smoke=smoke, rows=rows(smoke),
        code={n: digest(REPO/n) for n in CODE}, receiver_code=code_identity(),
        packet_profile=packet_hash(), receiver_profile=routing.identity(),
        G_assets=assets, G_assets_hash=generation.asset_hash(assets),
        initial_path=str(INITIAL), initial_sha256=digest(INITIAL),
        view_names=list(VIEW_NAMES), counts=list(COUNTS), epochs=2 if smoke else 120,
        learning_rate=1e-4, weight_decay=1e-4, ranking_weight=.1, seed=routing.SEED,
        arms=['core'], loss_scales='RMS of current TRAIN labels only; floor 1e-5',
        source_reads='sender encoding and offline scoring only',
        labels='96 measured conditional G calls per window on actual entropy-decoded Y',
        geometry='4x4 RGB G cores, 4x4 latent E addresses, two P8 packets per region bundle',
        randomness='shared seed + 65536 * spatial region; independent of selected-list order',
        G_inputs='same original ungenerated Y for all selected regions',
        transmitted_masks=False, semantic_supervision=False, freeze_UF_E_G=True,
        old_labels_reused=False, sender_training_complete=False,
        cache='lossless compressed FP32 inputs, no full generated videos')


def source(row):
    from PIL import Image
    frames = []
    for path, expected in zip(row['files'], row['hashes']):
        if digest(path) != expected:
            raise ValueError('source pixels changed: '+path)
        with Image.open(path) as image:
            frames.append(np.asarray(image.convert('RGB')))
    result = np.stack(frames)
    if list(result.shape) != row['shape']:
        raise ValueError('source geometry changed')
    return result


def selection(sample_id, count):
    if count not in COUNTS:
        raise ValueError('unsupported E region count')
    seed = int(hashlib.sha256(('mixed-v1/'+sample_id).encode()).hexdigest()[:16], 16)
    return np.random.default_rng(seed).permutation(16)[:count].tolist()


def prepare_bank(folder, row, pixels, check=lambda: None):
    from routervc.latent.packet_codec import PacketCodec, packet_hash
    from demo.four_state_receive import codec_precision
    binding = dict(row=row, packet_profile=packet_hash())
    path = folder/'encoded.json'
    if path.exists():
        result = read(path)
        if result['binding'] != binding:
            raise ValueError('encoded bank binding changed')
        verify_artifacts(folder, result['artifacts'])
        return result
    check(); began = time.monotonic()
    with codec_precision():
        codec = PacketCodec()
        b, fine, base, full, native, own, audit = codec.encode_chain(pixels)
        legacy = b+b''.join(fine)
        bottom, packets, packing = codec.repacketize(legacy, 2)
        bank = bottom+b''.join(packets[j, k] for k in range(16) for j in range(2))
        decoded_b, decoded_e, detail = codec.decode_packets(bank)
        np.testing.assert_array_equal(decoded_b, base)
        np.testing.assert_array_equal(decoded_e, full)
        if detail['base_reference_hashes'] != audit['base_reference_hashes']:
            raise ValueError('repacketization changed B temporal reference')
    del codec; gc.collect(); torch.cuda.empty_cache()
    torch.cuda.set_stream(torch.cuda.default_stream())
    atomic_bytes(folder/'bank.rvlp', bank)
    atomic_bytes(folder/'native_own.bin', native)
    result = dict(binding=binding, seconds=time.monotonic()-began,
        base_hash=frame_hash(base), full_hash=frame_hash(full),
        base_reference_hashes=audit['base_reference_hashes'], native_bytes=len(native),
        bundle_bytes=routing.bundle_bytes(bank), base_bytes=len(bottom), total_bytes=len(bank),
        same_context_endpoints_exact=True, artifacts={n:digest(folder/n)
            for n in ('bank.rvlp', 'native_own.bin')}, packing=packing)
    save(path, result)
    return result


def decode(bank):
    from demo.four_state_receive import codec_precision
    from routervc.latent.packet_codec import PacketCodec
    with codec_precision():
        codec = PacketCodec()
        base, received, detail = codec.decode_packets(bank)
    del codec; gc.collect(); torch.cuda.empty_cache()
    torch.cuda.set_stream(torch.cuda.default_stream())
    return base, received, detail


class Samples:
    def __init__(self, root, cache, protocol, check=lambda: None, progress=lambda **kw: None):
        self.root, self.cache, self.protocol = Path(root), Path(cache), protocol
        self.check, self.progress = check, progress
        self.generator = self.metric = None
        self.memory = OrderedDict()

    def prepare(self, row):
        from demo.four_state_receive import PersistentRGB, codec_precision
        from demo.routervc_receiver_router import build_inputs
        from demo.routervc_visual_router import grid_rois, METRICS
        from demo.routervc_mixedview_teacher import crop
        from demo.stage_c_three_path_roi_probe import LPIPSAlex
        from demo.scalable_experiment import quality
        folder = self.root/'samples'/row['sample_id']
        folder.mkdir(parents=True, exist_ok=True)
        done, cache = folder/'complete.json', self.cache/f'{row["sample_id"]}.npz'
        binding = dict(protocol=digest(self.root/'protocol.json'), sample=row)
        if done.exists():
            return self.verify(row)
        self.check()
        pixels = source(row)
        enc = prepare_bank(folder, row, pixels, self.check)
        bank = (folder/'bank.rvlp').read_bytes()
        rois = grid_rois(*pixels.shape[1:3])
        inputs, values = [], []
        for count in COUNTS:
            self.check()
            selected = selection(row['sample_id'], count)
            raw = routing.subset(bank, selected)
            base, received, detail = decode(raw)
            if frame_hash(base) != enc['base_hash'] or detail['base_reference_hashes'] != enc['base_reference_hashes']:
                raise ValueError('selected E changed B reference')
            current = dict(binding=binding, count=count, selected=selected,
                input_sha256=hashlib.sha256(raw).hexdigest(), input_bytes=len(raw),
                base_hash=frame_hash(base), received_hash=frame_hash(received),
                coverage=routing.coverage(routing.packets.parse(raw)).tolist())
            state_path = folder/f'e{count}.json'
            if state_path.exists():
                state = read(state_path)
                if state['binding'] != current:
                    raise ValueError('teacher state changed; cannot reuse G measurements')
            else:
                state = dict(binding=current, cells={})
            for region, roi in enumerate(rois):
                key = str(region)
                if key in state['cells']:
                    continue
                self.check()
                self.progress(phase='measure_G_labels', sample=row['sample_id'], E_regions=count,
                              G_region=region, sample_total=len(self.protocol['rows']))
                if self.generator is None:
                    self.generator = PersistentRGB()
                if self.metric is None:
                    self.metric = LPIPSAlex(True)
                settings = routing.control(received.shape, self.protocol['G_assets'], region, self.protocol['seed'])
                began = time.monotonic()
                with torch.no_grad():
                    output, reports = routing.render(received, [region], self.protocol['G_assets'],
                        self.generator, seed=self.protocol['seed'], check=self.check)
                with codec_precision():
                    before = quality(crop(pixels, roi), crop(received, roi), self.metric)
                    after = quality(crop(pixels, roi), crop(output, roi), self.metric)
                gains = [sign*(after[name]-before[name]) for name, sign in METRICS]
                if not np.isfinite(gains).all():
                    raise ValueError('nonfinite measured G gains')
                state['cells'][key] = dict(before=before, after=after, gains=gains,
                    output_hash=frame_hash(output), control=settings, seconds=time.monotonic()-began,
                    runtime=reports, outside_generate_exact=True, source_only_offline_scoring=True)
                save(state_path, state)  # One atomic journal per E density, not a huge RGB cache.
                del output
                print('LATENT_G_LABEL', row['sample_id'], count, region+1, '/16', flush=True)
            values.append([state['cells'][str(i)]['gains'] for i in range(16)])
            inputs.append(build_inputs(base, received, np.asarray(current['coverage'], np.float32)))
            del base, received
        cache.parent.mkdir(parents=True, exist_ok=True)
        packed = {f'input_{k}':torch.cat([v[k] for v in inputs]).numpy() for k in inputs[0]}
        packed['value'] = np.asarray(values, dtype=np.float32)
        atomic_npz(cache, **packed)
        result = dict(complete=True, binding=binding, cache_path=str(cache), cache_sha256=digest(cache),
            measured_cells=len(COUNTS)*16, old_labels_reused=False,
            artifacts={n:digest(folder/n) for n in ['encoded.json', 'bank.rvlp', 'native_own.bin',
                                                  *(f'e{c}.json' for c in COUNTS)]})
        save(done, result)
        return result

    def verify(self, row):
        folder = self.root/'samples'/row['sample_id']
        done = read(folder/'complete.json')
        if (not done['complete'] or done['binding'] != dict(protocol=digest(self.root/'protocol.json'), sample=row)
                or done['cache_sha256'] != digest(done['cache_path'])):
            raise ValueError('label binding/cache changed')
        verify_artifacts(folder, done['artifacts'])
        return done

    def get(self, row):
        sid = row['sample_id']
        if sid not in self.memory:
            done = self.verify(row)
            with np.load(done['cache_path'], allow_pickle=False) as f:
                inputs = {k[6:]:torch.from_numpy(f[k].copy()) for k in f.files if k.startswith('input_')}
                targets = torch.from_numpy(f['value'].copy())
            self.memory[sid] = dict(inputs=inputs, targets=dict(value=targets, weight=torch.ones_like(targets)),
                                    view_names=list(VIEW_NAMES))
        self.memory.move_to_end(sid)
        while len(self.memory) > 2:
            self.memory.popitem(last=False)
        return self.memory[sid]

    def release(self):
        self.generator = self.metric = None
        self.memory.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
