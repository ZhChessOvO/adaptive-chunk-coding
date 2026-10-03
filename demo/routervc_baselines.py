"""Separate, resumable RouterVC baseline supplement; historical pins stay intact.

UF QP8/16/24/32 uses the unchanged native codec plus an explicitly charged,
self-describing metadata sidecar. ACSE v1/v2 requires QP8, so higher-QP native
streams must NOT be mislabeled as ACSE. ``bytes`` always includes sidecar or
container/control overhead; ``native_bytes`` is a separately labeled diagnostic.
All-G uses the completed mean/BF16 RGB LoRA and no enhancement packets, either
16 original grid ROIs or one full-frame ROI. This changes execution geometry,
not the trained generator. Decoders never open the original video.

The top-level run requires tmux, holds the shared GPU mutex, and spawns fresh
processes for encoding/decoding. CPU reporting never reruns models or metrics.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from demo.scalable_codec import atomic_bytes, atomic_json, atomic_npz, file_hash
from demo.scalable_format import canonical_json, frame_hash, parse, sha256

DEFAULT = Path('/root/autodl-fs/DCVC/runs/routervc_20261003')
ADAPTER = DEFAULT.parent/'a800_online_eg_20261002/joint/adapter.pt'
MODEL_I = REPO/'checkpoints/cvpr2026_image.pth.tar'
MODEL_P = REPO/'checkpoints/cvpr2026_video_hts.pth.tar'
QPS = (8, 16, 24, 32)
REGISTRY = {'uf_qp8': '#383838', 'uf_qp16': '#326c85',
            'uf_qp24': '#3e93a8', 'uf_qp32': '#111111',
            'full_frame_g': '#ce3f8a', 'route_no_g': '#539643',
            'route_no_e': '#bc8dc9'}
CODE = ('routervc_baselines.py', 'routervc_report.py', 'routervc_format.py',
        'routervc_policy.py', 'routervc_decode.py', 'scalable_codec.py', 'scalable_format.py',
        'compact_enhancement_format.py', 'scalable_cooperation_format.py',
        'online_eg_decode.py', 'internal_condition_decode.py', 'run_routervc_baselines.sh')


def read(path):
    with Path(path).open() as stream:
        return json.load(stream)


def immutable(path, value):
    if Path(path).exists():
        if read(path) != value:
            raise ValueError(f'changed immutable input at {path}; choose a new output')
    else:
        atomic_json(Path(path), value)


def verify(folder, artifacts):
    for name, expected in artifacts.items():
        if file_hash(Path(folder)/name) != expected:
            raise ValueError(f'artifact changed: {Path(folder)/name}')


def uf_metadata(qp, shape, native, pixels, model_hashes):
    if qp not in QPS or len(shape) != 4 or shape[-1] != 3:
        raise ValueError('unsupported UF baseline QP/geometry')
    count, height, width, _ = shape
    return dict(format='RouterVC_native_UF_sidecar_v1', codec='dcvc_uf_hts_scalar',
        qp=qp, skip_thres=0., frame_count=int(count), height=int(height), width=int(width),
        display_format='rgb_u8', stream_sha256=sha256(native),
        base_rgb_sha256=frame_hash(pixels), **model_hashes)


def validate_uf_metadata(meta, data):
    expected = {'format', 'codec', 'qp', 'skip_thres', 'frame_count', 'height', 'width',
                'display_format', 'stream_sha256', 'base_rgb_sha256',
                'model_i_sha256', 'model_p_sha256'}
    if (set(meta) != expected or meta['format'] != 'RouterVC_native_UF_sidecar_v1'
            or meta['codec'] != 'dcvc_uf_hts_scalar' or type(meta['qp']) is not int or meta['qp'] not in QPS
            or meta['skip_thres'] != 0. or meta['display_format'] != 'rgb_u8'):
        raise ValueError('unknown native UF sidecar schema')
    if any(type(meta[k]) is not int or not 0 < meta[k] <= bound for k, bound in
           [('frame_count', 100000), ('height', 8192), ('width', 8192)]):
        raise ValueError('invalid UF video geometry')
    for key in ('stream_sha256', 'base_rgb_sha256', 'model_i_sha256', 'model_p_sha256'):
        value = meta[key]
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError('invalid UF digest')
        try:
            if len(bytes.fromhex(value)) != 32:
                raise ValueError()
        except ValueError:
            raise ValueError('invalid UF digest') from None
    if not data or sha256(data) != meta['stream_sha256']:
        raise ValueError('native UF payload mismatch')


def uniform_route(generated, actual_calls=0):
    """Grid visualization and execution ROI count are deliberately separate."""
    return dict(indices=list(range(16)) if generated else [],
        coverage=[0.]*16, states=['G' if generated else 'B']*16,
        boundary_edges=0, components=1 if generated else 0,
        g_calls=actual_calls, grid_G_cells=16 if generated else 0,
        actual_G_roi_calls=actual_calls, shared_policy_used=False)


def explicit_route(indices, actual_calls):
    from demo.routervc_report import grid_statistics
    states=[2 if i in indices else 0 for i in range(16)]
    stats=grid_statistics(states)
    return dict(indices=indices,coverage=[0.]*16,states=['G' if s else 'B' for s in states],
        boundary_edges=stats['boundary_edges'],components=stats['components'],
        grid_G_cells=len(indices),g_calls=actual_calls,actual_G_roi_calls=actual_calls,
        shared_policy_used=False)


def encode_uf(args):
    """Worker: original frames are legal here, but never in decode_uf."""
    from demo.chunk_enhancement_codec import configure_torch
    from demo.scalable_codec import BaseCodec
    import torch
    configure_torch();torch.cuda.reset_peak_memory_stats()
    folder = args.output;folder.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    if file_hash(args.source) != args.source_hash:
        raise ValueError('source cache changed')
    with np.load(args.source, allow_pickle=False) as cache:
        source = cache['source'].copy()
    if source.dtype != np.uint8 or source.ndim != 4 or source.shape[-1] != 3:
        raise ValueError('expected uint8 RGB source')
    codec = BaseCodec(MODEL_I, MODEL_P)
    configure_torch()
    reused = args.reuse_base is not None
    if reused:
        if args.qp != 8:
            raise ValueError('only the exact QP8 base may be reused')
        inner = parse(args.reuse_base.read_bytes())
        if inner.packets or inner.meta['base_qp'] != 8:
            raise ValueError('base-only QP8 ACSE required for reuse')
        data = inner.base
        encoded_detail = dict(reused_native_base=True, source_container_sha256=file_hash(args.reuse_base))
    else:
        data, encoded_detail = codec.encode(source, qp=args.qp)
    pixels = codec.decode(data, len(source))
    if pixels.shape != source.shape:
        raise ValueError('native UF geometry mismatch')
    if reused and frame_hash(pixels) != inner.meta['base_rgb_sha256']:
        raise ValueError('reused QP8 reconstruction differs')
    metadata = uf_metadata(args.qp, source.shape, data, pixels, codec.models)
    validate_uf_metadata(metadata, data)
    atomic_bytes(folder/'stream.bin', data)
    atomic_bytes(folder/'transmitted_meta.json', canonical_json(metadata)+b'\n')
    result = dict(complete=True, qp=args.qp, source_hash=args.source_hash,
        source_frames_read=True, encoding_reused=reused, native_encode=encoded_detail,
        native_bytes=len(data), metadata_bytes=(folder/'transmitted_meta.json').stat().st_size,
        base_hash=frame_hash(pixels), seconds=time.monotonic()-began,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        artifacts={n:file_hash(folder/n) for n in ('stream.bin', 'transmitted_meta.json')})
    result['total_bytes'] = result['native_bytes']+result['metadata_bytes']
    atomic_json(folder/'encode.json', result)


def decode_uf(args):
    """Independent worker: only transmitted native bits + charged sidecar."""
    import torch
    from demo.chunk_enhancement_codec import configure_torch
    from demo.scalable_codec import BaseCodec
    configure_torch();torch.cuda.reset_peak_memory_stats();began=time.monotonic()
    folder=args.output;out=folder/'fresh';out.mkdir(parents=True,exist_ok=True)
    data=(folder/'stream.bin').read_bytes();metadata=read(folder/'transmitted_meta.json')
    validate_uf_metadata(metadata,data)
    codec=BaseCodec(MODEL_I,MODEL_P);configure_torch()
    if any(metadata[k]!=v for k,v in codec.models.items()):
        raise ValueError('pre-shared UF model mismatch')
    base=codec.decode(data,metadata['frame_count'])
    expected=(metadata['frame_count'],metadata['height'],metadata['width'],3)
    if base.shape!=expected or frame_hash(base)!=metadata['base_rgb_sha256']:
        raise ValueError('source-free fresh UF reconstruction mismatch')
    native_bytes=(folder/'stream.bin').stat().st_size
    metadata_bytes=(folder/'transmitted_meta.json').stat().st_size
    result=dict(baseline_receiver_format=1,source_frames_read=False,pid=os.getpid(),base_hash=frame_hash(base),
        output_hash=frame_hash(base),generation_input_hash=frame_hash(base),
        stream_sha256=file_hash(folder/'stream.bin'),metadata_sha256=file_hash(folder/'transmitted_meta.json'),
        native_bytes=native_bytes,metadata_bytes=metadata_bytes,total_bytes=native_bytes+metadata_bytes,
        packet_bytes=0,generation_control_bytes=0,container_header_bytes=metadata_bytes,
        base_reference_unchanged=True,outside_generate_exact=True,generation_executed=False,
        generation_assets_validated=False,route=uniform_route(False),
        seconds=time.monotonic()-began,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        byte_scope='native stream.bin plus transmitted_meta.json; both files required and charged')
    atomic_npz(out/'reconstruction.npz',base=base,enhanced=base,reconstruction=base)
    atomic_json(out/'decode.json',result)


def decode_g(args):
    """Independent all-G worker; source-free, no E model or packets loaded."""
    import torch
    from demo.chunk_enhancement_codec import configure_torch
    from demo.scalable_codec import BaseCodec
    from demo import scalable_cooperation_format as cooperation
    from demo.online_eg_decode import identities
    from demo.internal_condition_decode import restore
    configure_torch();torch.cuda.reset_peak_memory_stats();began=time.monotonic()
    folder=args.output;out=folder/'fresh';out.mkdir(parents=True,exist_ok=True)
    data=(folder/'stream.acsg').read_bytes()
    control,_,inner,control_bytes=cooperation.parse(data)
    if inner.packets or control['protect'] or control['blend']!=1.:
        raise ValueError('G baseline requires no E/protection and blend one')
    count,height,width=(inner.meta[k] for k in ('frame_count','height','width'))
    from demo.routervc_policy import grid_rois
    grid=[[0,count,*r] for r in grid_rois(height,width)]
    full=[[0,count,0,0,width,height]]
    if control['generate']==full:
        indices=list(range(16));geometry='one_full_frame_ROI'
    else:
        if (any(r not in grid for r in control['generate'])
                or len({tuple(r) for r in control['generate']})!=len(control['generate'])):
            raise ValueError('expected a unique subset of grid ROIs or one complete frame')
        indices=[grid.index(r) for r in control['generate']]
        geometry='sixteen_grid_ROIs' if len(indices)==16 else 'fixed_partial_grid_ROIs'
    codec=BaseCodec(MODEL_I,MODEL_P);configure_torch()
    if any(inner.meta[k]!=v for k,v in codec.models.items()):
        raise ValueError('UF model mismatch')
    base=codec.decode(inner.base,count)
    if base.shape!=(count,height,width,3) or frame_hash(base)!=inner.meta['base_rgb_sha256']:
        raise ValueError('G baseline fresh base mismatch')
    del codec;gc.collect();torch.cuda.empty_cache()
    alpha=cooperation.weights(base.shape,control)
    if indices:
        expected=identities(args.adapter)
        if any(control[k]!=v for k,v in expected.items()):
            raise ValueError('G LoRA/assets/profile differ from charged control')
        generated,runtime=restore(base,control,args.adapter,[])
        output=cooperation.combine(base,generated,alpha)
    else:output,runtime=base.copy(),None
    np.testing.assert_array_equal(output[alpha==0],base[alpha==0])
    actual_calls=len(control['generate'])
    result=dict(baseline_receiver_format=1,source_frames_read=False,pid=os.getpid(),stream_sha256=file_hash(folder/'stream.acsg'),
        base_hash=frame_hash(base),generation_input_hash=frame_hash(base),output_hash=frame_hash(output),
        native_bytes=len(inner.base),container_header_bytes=inner.base_end-len(inner.base),
        generation_control_bytes=control_bytes,packet_bytes=0,total_bytes=len(data),
        base_reference_unchanged=True,outside_generate_exact=True,generation_executed=bool(indices),
        generation_assets_validated=bool(indices),route=explicit_route(indices,actual_calls),
        geometry=geometry,actual_G_roi_calls=actual_calls,
        actual_G_window_calls=len(runtime['windows']) if runtime else 0,
        generation_runtime=runtime,config=control,
        seconds=time.monotonic()-began,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        byte_scope='native UF + ACSE2 base framing + explicit ACSG2 generation control')
    assert sum(result[k] for k in ('native_bytes','container_header_bytes','generation_control_bytes'))==len(data)
    atomic_npz(out/'reconstruction.npz',base=base,enhanced=base,reconstruction=output)
    atomic_json(out/'decode.json',result)
    if torch.distributed.is_initialized():torch.distributed.destroy_process_group()


def decode_no_g(args):
    """Same received E packets; G is disabled without consulting source/Router."""
    from types import SimpleNamespace
    from demo.routervc_decode import decode
    folder=args.output
    decode(SimpleNamespace(stream=folder/'stream.rtvc',output=folder/'fresh',
        enhancement=DEFAULT.parent/'a800_online_eg_20261002/joint/enhancement.pt',
        adapter=Path('/missing/generator.pt'),router=Path('/missing/router.pt'),
        disable_generation=True,allow_incomplete_tail=False))
    report=read(folder/'fresh/decode.json')
    report.update(baseline_receiver_format=1,native_bytes=report['base_bytes'],actual_G_roi_calls=0,
        byte_scope='same original RouterVC wire, including unchanged E packets and header; G disabled locally')
    atomic_json(folder/'fresh/decode.json',report)


def parent_records(root):
    summary=read(root/'summary.json')
    if not summary.get('complete'):
        raise ValueError('main RouterVC evaluation must complete first')
    records=summary['records'];by_id={}
    for record in records:
        folder=Path(record['folder'])
        if not folder.is_absolute():folder=root/folder
        verify(folder,record['artifacts'])
        by_id.setdefault(record['sample_id'],[]).append(record)
    return records,by_id


def prepare(root,output,limit=0):
    """CPU setup only; it does not start a model or consume unlisted samples."""
    from demo import routervc_format
    from demo import scalable_cooperation_format as cooperation
    from demo.routervc_policy import grid_rois
    records,by_id=parent_records(root)
    samples=list(by_id)[:limit or None]
    protocol=dict(code={n:file_hash(REPO/'demo'/n) for n in CODE},
        main_summary=file_hash(root/'summary.json'),main_protocol=file_hash(root/'protocol.json'),
        model_i=file_hash(MODEL_I),model_p=file_hash(MODEL_P),adapter=file_hash(ADAPTER),
        samples=samples,qps=list(QPS),report_registry=REGISTRY,
        uf_bytes='native stream plus explicit charged decoding metadata sidecar',
        g_profile='same completed mean-BF16 RGB LoRA; no E; 16 grid ROIs versus 1 full-frame ROI',
        ablations='context_smooth ratio0.5/maxG4: fixed E with G off; fixed G indices with E removed',
        role='same main development samples; no new independent evaluation')
    immutable(output/'protocol.json',protocol)
    jobs=[]
    for sid in samples:
        base=next(r for r in by_id[sid] if r['method']=='base')
        original=Path(base['folder'])
        if not original.is_absolute():original=root/original
        config,inner_data,inner,_=routervc_format.parse((original/'stream.rtvc').read_bytes())
        if inner.packets or base['base_hash']!=inner.meta['base_rgb_sha256']:
            raise ValueError('main reference is not the expected packet-free base')
        sample_root=output/sid;sample_root.mkdir(parents=True,exist_ok=True)
        cached=sample_root/'base.acse'
        if cached.exists():
            if cached.read_bytes()!=inner_data:raise ValueError('changed copied base container')
        else:atomic_bytes(cached,inner_data)
        common=dict(sample_id=sid,dataset=base['dataset'],source_path=base['source_path'],
                    source_hash=base['source_hash'],base_qp8_hash=base['base_hash'])
        for qp in QPS:
            folder=sample_root/f'uf_qp{qp}';folder.mkdir(exist_ok=True)
            job=dict(common,method=f'uf_qp{qp}',kind='uf',qp=qp,ratio=0.,max_g=0,
                     folder=str(folder.resolve()),reuse_base=str(cached.resolve()) if qp==8 else None)
            immutable(folder/'job.json',job);jobs.append(job)
        count,height,width=(inner.meta[k] for k in ('frame_count','height','width'))
        rois=grid_rois(height,width)
        for method,generate in [('full_g',[[0,count,*r] for r in rois]),
                                ('full_frame_g',[[0,count,0,0,width,height]])]:
            folder=sample_root/method;folder.mkdir(exist_ok=True)
            control=routervc_format.generation_control(config,list(range(16)),rois,count)
            control['generate']=generate
            wire=cooperation.wrap(inner_data,control)
            path=folder/'stream.acsg'
            if path.exists():
                if path.read_bytes()!=wire:raise ValueError('changed G baseline wire')
            else:atomic_bytes(path,wire)
            job=dict(common,method=method,kind='g',ratio=0.,max_g=16,folder=str(folder.resolve()),
                     bytes=len(wire),stream_sha256=file_hash(path),actual_G_roi_calls=len(generate),
                     expected_G_indices=list(range(16)),
                     byte_scope='ACSE2 base plus explicit ACSG2 controls, all counted')
            immutable(folder/'job.json',job);jobs.append(job)
        chosen=next(r for r in by_id[sid] if r['method']=='context_smooth'
                    and r['ratio']==.5 and r['max_g']==4)
        chosen_folder=Path(chosen['folder'])
        if not chosen_folder.is_absolute():chosen_folder=root/chosen_folder
        full_wire=(chosen_folder/'stream.rtvc').read_bytes()
        chosen_config,chosen_inner,chosen_parsed,_=routervc_format.parse(full_wire)
        fixed_indices=chosen['decode']['route']['indices']
        folder=sample_root/'route_no_g';folder.mkdir(exist_ok=True)
        path=folder/'stream.rtvc'
        if path.exists():
            if path.read_bytes()!=full_wire:raise ValueError('changed route-no-G wire')
        else:atomic_bytes(path,full_wire)
        job=dict(common,method='route_no_g',kind='off_g',ratio=.5,max_g=0,folder=str(folder.resolve()),
            bytes=len(full_wire),stream_sha256=file_hash(path),expected_G_indices=[],
            expected_enhanced_hash=chosen['enhanced_hash'],reference_result=file_hash(chosen_folder/'result.json'),
            reference_folder=str(chosen_folder.resolve()),
            actual_G_roi_calls=0,ablation='same E bytes/positions; original G disabled, no reallocation')
        immutable(folder/'job.json',job);jobs.append(job)
        folder=sample_root/'route_no_e';folder.mkdir(exist_ok=True)
        control=routervc_format.generation_control(chosen_config,fixed_indices,rois,count)
        wire=cooperation.wrap(chosen_inner[:chosen_parsed.base_end],control)
        path=folder/'stream.acsg'
        if path.exists():
            if path.read_bytes()!=wire:raise ValueError('changed route-no-E wire')
        else:atomic_bytes(path,wire)
        job=dict(common,method='route_no_e',kind='g',ratio=0.,max_g=4,folder=str(folder.resolve()),
            bytes=len(wire),stream_sha256=file_hash(path),expected_G_indices=fixed_indices,
            reference_result=file_hash(chosen_folder/'result.json'),actual_G_roi_calls=len(fixed_indices),
            reference_folder=str(chosen_folder.resolve()),
            ablation='same G indices, order, noise seed and profile; all E removed, no G reallocation',
            byte_scope='explicit G controls charged; wire differs from shared-policy reference')
        immutable(folder/'job.json',job);jobs.append(job)
    immutable(output/'jobs.json',dict(jobs=jobs,selected_samples=samples))
    return records,jobs


def paired_noise(reference,current):
    """Same G support/order/seed must also reproduce the actual noise tensor."""
    a=reference.get('generation_runtime');b=current.get('generation_runtime')
    if a is None or b is None:
        if a is not None or b is not None:raise ValueError('unpaired skipped generation')
        return dict(paired_windows=0,diffusion_noise_equal=True)
    wa,wb=a['windows'],b['windows']
    ca,cb=a['condition_windows'],b['condition_windows']
    if not len(wa)==len(wb)==len(ca)==len(cb):raise ValueError('unpaired G window counts')
    for u,v,x,y in zip(wa,wb,ca,cb):
        if (u['start']!=v['start'] or u['crop']!=v['crop']
                or u['runtime']['seed']!=v['runtime']['seed']
                or x['diffusion_noise']!=y['diffusion_noise']):
            raise ValueError('fixed-route E removal changed G geometry/noise')
    return dict(paired_windows=len(wa),diffusion_noise_equal=True)


def validate_record(folder,job):
    """Validate a completed point without inference or metric recomputation."""
    record=read(folder/'result.json')
    if record['job']!=job:raise ValueError('resumed baseline job changed')
    verify(folder,record['artifacts'])
    d=read(folder/'fresh/decode.json')
    if record['decode']!=d or d['source_frames_read'] or d.get('baseline_receiver_format')!=1:
        raise ValueError('changed receiver evidence or source-dependent receiver')
    if record['bytes']!=d['total_bytes']:
        raise ValueError('baseline byte accounting mismatch')
    if job['kind']=='uf':
        metadata=read(folder/'transmitted_meta.json')
        validate_uf_metadata(metadata,(folder/'stream.bin').read_bytes())
        if metadata['qp']!=job['qp'] or metadata['base_rgb_sha256']!=d['base_hash']:
            raise ValueError('UF QP or fresh reconstruction metadata mismatch')
        actual=(folder/'stream.bin').stat().st_size+(folder/'transmitted_meta.json').stat().st_size
        if file_hash(folder/'stream.bin')!=d['stream_sha256'] or file_hash(folder/'transmitted_meta.json')!=d['metadata_sha256']:
            raise ValueError('UF transmitted inputs changed')
    else:
        stream=folder/('stream.rtvc' if job['kind']=='off_g' else 'stream.acsg')
        actual=stream.stat().st_size
        if file_hash(stream)!=d['stream_sha256'] or d['base_hash']!=job['base_qp8_hash']:
            raise ValueError('G transmitted input/base mismatch')
        if d['route']['indices']!=job['expected_G_indices']:
            raise ValueError('fixed ablation Generate route changed')
        if job['kind']=='off_g' and d['generation_input_hash']!=job['expected_enhanced_hash']:
            raise ValueError('fixed received enhancement changed')
    if job['method']=='route_no_e':
        reference=Path(job['reference_folder'])/'result.json'
        if file_hash(reference)!=job['reference_result']:raise ValueError('ablation reference changed')
        evidence=paired_noise(read(reference)['decode'],d)
        if record.get('paired_noise')!=evidence:raise ValueError('ablation pairing proof changed')
    if actual!=record['bytes']:raise ValueError('actual baseline file bytes differ')
    with np.load(folder/'fresh/reconstruction.npz',allow_pickle=False) as pixels:
        if frame_hash(pixels['reconstruction'])!=d['output_hash'] or frame_hash(pixels['base'])!=d['base_hash']:
            raise ValueError('fresh receiver pixels changed')
        if job['kind']!='off_g' and not np.array_equal(pixels['base'],pixels['enhanced']):
            raise ValueError('baseline unexpectedly contains enhancement')
        if job['kind']=='off_g' and not np.array_equal(pixels['enhanced'],pixels['reconstruction']):
            raise ValueError('Generate-off output differs from received enhancement')
    return record


def evaluate(run,jobs,verify_only=False):
    from demo.conditioned_generation_pipeline import execute
    from demo.scalable_experiment import quality
    from demo.stage_c_three_path_roi_probe import LPIPSAlex
    metric=None;records=[]
    for index,job in enumerate(jobs):
        run.check();folder=Path(job['folder'])
        if (folder/'result.json').exists():
            records.append(validate_record(folder,job))
            run.update(phase='verified_saved_baseline',completed=index+1,total=len(jobs));continue
        if verify_only:raise ValueError(f'missing completed baseline: {folder}')
        if job['kind']=='uf':
            if (folder/'encode.json').exists():
                encoded=read(folder/'encode.json');verify(folder,encoded['artifacts'])
                if encoded['qp']!=job['qp'] or encoded['source_hash']!=job['source_hash']:
                    raise ValueError('changed UF encoder inputs')
            else:
                extra=['--reuse-base',job['reuse_base']] if job['reuse_base'] else []
                execute(run,f'encode_{index:03d}','routervc_baselines.py',
                    ['encode-uf','--output',folder,'--source',job['source_path'],
                     '--source-hash',job['source_hash'],'--qp',job['qp'],*extra])
            command='decode-uf';distributed=False
        elif job['kind']=='off_g':command='decode-no-g';distributed=False
        else:command='decode-g';distributed=True
        if (not (folder/'fresh/decode.json').exists()
                or read(folder/'fresh/decode.json').get('baseline_receiver_format')!=1):
            execute(run,f'decode_{index:03d}','routervc_baselines.py',
                    [command,'--output',folder,'--adapter',ADAPTER],distributed=distributed)
        d=read(folder/'fresh/decode.json')
        expected_G=job['kind']=='g' and bool(job['expected_G_indices'])
        if d['source_frames_read'] or d['generation_executed']!=expected_G:
            raise ValueError('baseline receiver mode mismatch')
        if file_hash(Path(job['source_path']))!=job['source_hash']:
            raise ValueError('evaluation source changed')
        with np.load(job['source_path'],allow_pickle=False) as cache:source=cache['source'].copy()
        with np.load(folder/'fresh/reconstruction.npz',allow_pickle=False) as cache:output=cache['reconstruction'].copy()
        if source.shape!=output.shape:raise ValueError('baseline evaluation geometry mismatch')
        if metric is None:metric=LPIPSAlex(True)
        names=['fresh/decode.json','fresh/reconstruction.npz']
        names+=(['stream.bin','transmitted_meta.json','encode.json'] if job['kind']=='uf'
                else ['stream.rtvc'] if job['kind']=='off_g' else ['stream.acsg'])
        record=dict(job,job=job,bytes=d['total_bytes'],native_bytes=d['native_bytes'],
            non_native_bytes=d['total_bytes']-d['native_bytes'],quality=quality(source,output,metric),decode=d,
            artifacts={name:file_hash(folder/name) for name in names})
        if job['method']=='route_no_e':
            reference=Path(job['reference_folder'])/'result.json'
            if file_hash(reference)!=job['reference_result']:raise ValueError('ablation reference changed')
            record['paired_noise']=paired_noise(read(reference)['decode'],d)
        atomic_json(folder/'result.json',record)
        records.append(validate_record(folder,job))
        run.update(phase='baseline_complete',completed=index+1,total=len(jobs))
    return records


def supplement_report(output):
    # Extend the old reporter's declared palette, without changing pinned files.
    # This registry and BOTH source hashes are part of supplement/protocol.json.
    from demo import routervc_report
    previous_methods,previous_colors=routervc_report.METHODS,routervc_report.COLORS
    try:
        routervc_report.METHODS=tuple(dict.fromkeys((*previous_methods,*REGISTRY)))
        routervc_report.COLORS=dict(previous_colors,**REGISTRY)
        return routervc_report.report(output)
    finally:
        routervc_report.METHODS=previous_methods;routervc_report.COLORS=previous_colors


def supplement(run,root,limit=0,verify_only=False):
    """Caller owns tmux and the GPU mutex; each completed point is resumable."""
    began=time.monotonic()
    main_records,jobs=prepare(root,run.root,limit)
    before={str(Path(j['folder'])/'result.json'):file_hash(Path(j['folder'])/'result.json')
            for j in jobs if (Path(j['folder'])/'result.json').exists()}
    baselines=evaluate(run,jobs,verify_only)
    # Audit all completed points again with inference and metrics unavailable.
    evaluate(run,jobs,True)
    if any(file_hash(Path(p))!=h for p,h in before.items()):
        raise ValueError('resume changed an existing baseline result')
    source=read(root/'summary.json')
    normalized_main=[]
    for record in main_records:
        folder=Path(record['folder'])
        normalized_main.append(dict(record,folder=str(folder if folder.is_absolute() else (root/folder).resolve())))
    result=dict(complete=True,records=normalized_main+baselines,main_summary=file_hash(root/'summary.json'),
        baseline_protocol=file_hash(run.root/'protocol.json'),report_registry=REGISTRY,
        main_points=len(main_records),baseline_points=len(baselines),
        role=source.get('role','development'),native_byte_note='UF sidecars and G framing differ; total bytes drive RD, native bytes are separate.',
        geometry_note='full_g uses 16 tile calls; full_frame_g uses one ROI call despite 16 G cells in its visualization.',
        ablation_note='Fixed-route branch removals are not independently reallocated e_only/g_only; real byte changes remain counted.',
        no_new_training=True,no_new_downloads=True)
    immutable(run.root/'summary.json',result)
    supplement_report(run.root)
    done=run.root/'complete.json'
    if not done.exists():
        atomic_json(done,dict(complete=True,baseline_points=len(baselines),main_points=len(main_records),
            seconds_this_attempt=time.monotonic()-began,summary=file_hash(run.root/'summary.json'),
            readonly_completed_point_replay=True,previous_results_preserved=len(before)))
    run.update(phase='complete',completed=len(jobs),total=len(jobs))
    return result


def require_smoke(path):
    """Formal auto-continuation may only consume a matching completed smoke."""
    completed=read(path/'complete.json')
    summary=read(path/'summary.json')
    protocol=read(path/'protocol.json')
    if (not completed.get('complete') or not completed.get('readonly_completed_point_replay')
            or completed.get('baseline_points')!=8 or summary.get('baseline_points')!=8
            or file_hash(path/'summary.json')!=completed['summary']):
        raise ValueError('one-sample baseline smoke with all 8 fresh points is required')
    if protocol['code']!={name:file_hash(REPO/'demo'/name) for name in CODE}:
        raise ValueError('baseline smoke code differs from current code')
    for name,model in [('model_i',MODEL_I),('model_p',MODEL_P),('adapter',ADAPTER)]:
        if protocol[name]!=file_hash(model):raise ValueError('baseline smoke model changed')
    jobs=read(path/'jobs.json')['jobs']
    if len(jobs)!=8:raise ValueError('unexpected baseline smoke job count')
    for job in jobs:validate_record(Path(job['folder']),job)
    return dict(complete=True,path=str(path),summary_sha256=file_hash(path/'summary.json'))


def main(args):
    if args.command=='test':
        self_test();return
    if args.command in ('encode-uf','decode-uf','decode-g','decode-no-g'):
        if args.output is None:raise ValueError('worker --output is required')
        if args.command=='encode-uf':encode_uf(args)
        elif args.command=='decode-uf':decode_uf(args)
        elif args.command=='decode-no-g':decode_no_g(args)
        else:decode_g(args)
        return
    if not os.environ.get('TMUX'):
        raise RuntimeError('run/verify baseline supplements inside tmux')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    from demo.chunk_enhancement_experiment import Run
    from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
    if args.output is None:args.output=args.root/'supplement'
    run=Run(args);run.thread.start()
    try:
        if args.wait_main_complete:
            while not (args.root/'complete.json').exists():
                run.check();run.update(phase='waiting_for_main_completion');time.sleep(10)
        if not (args.root/'complete.json').exists():
            raise ValueError('main completion is missing; use --wait-main-complete to queue')
        with exclusive_native_evaluation(run):
            if args.require_smoke is not None:
                proof=require_smoke(args.require_smoke)
                immutable(args.output/'smoke_dependency.json',proof)
            supplement(run,args.root,args.limit_samples,args.command=='verify')
    except BaseException as error:
        atomic_json(args.output/'last_failure.json',dict(error=repr(error),phase=run.progress));raise
    finally:
        run.log_resources();run.stop.set();run.thread.join(timeout=3)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['run','verify','encode-uf','decode-uf','decode-g','decode-no-g','test'])
    p.add_argument('--root',type=Path,default=DEFAULT)
    p.add_argument('--output',type=Path)
    p.add_argument('--max-hours',type=float,default=12.)
    p.add_argument('--wait-main-complete',action='store_true')
    p.add_argument('--limit-samples',type=int,default=0)
    p.add_argument('--require-smoke',type=Path)
    p.add_argument('--source',type=Path)
    p.add_argument('--source-hash')
    p.add_argument('--qp',type=int,choices=QPS)
    p.add_argument('--reuse-base',type=Path)
    p.add_argument('--adapter',type=Path,default=ADAPTER)
    return p


def self_test():
    """Small CPU-only protocol tests; no assets, video dataset, or inference."""
    import unittest
    from demo.routervc_report import normalize_record

    class ProtocolTest(unittest.TestCase):
        def test_native_qps_self_describe_and_authenticate(self):
            pixels=np.zeros((17,64,64,3),np.uint8)
            weights=dict(model_i_sha256='1'*64,model_p_sha256='2'*64)
            for qp in QPS:
                meta=uf_metadata(qp,pixels.shape,b'fake native bits',pixels,weights)
                validate_uf_metadata(meta,b'fake native bits')
                self.assertEqual(meta['qp'],qp)
                with self.assertRaises(ValueError):validate_uf_metadata(meta,b'corrupt')
                with self.assertRaises(ValueError):validate_uf_metadata(dict(meta,secret_source='x'),b'fake native bits')

        def test_full_frame_calls_are_not_grid_cells(self):
            route=explicit_route(list(range(16)),1)
            self.assertEqual(route['actual_G_roi_calls'],1)
            self.assertEqual(route['grid_G_cells'],16)
            self.assertEqual(route['components'],1)
            self.assertEqual(route['boundary_edges'],0)
            value=dict(sample_id='synthetic',dataset='REDS',method='full_frame_g',ratio=0.,max_g=16,
                bytes=123,quality=dict(lpips_alex=.1,psnr_db=30.,temporal_delta_mae=1.),
                decode=dict(seconds=1.,peak_cuda_allocated_bytes=123,route=route))
            self.assertEqual(normalize_record(value,(17,64,64,3))['counts']['G'],16)

        def test_fixed_sparse_and_empty_routes(self):
            route=explicit_route([0,15],2)
            self.assertEqual(route['indices'],[0,15])
            self.assertEqual(route['components'],2)
            self.assertEqual(route['boundary_edges'],4)
            self.assertEqual(explicit_route([],0)['states'],['B']*16)
            self.assertEqual(uniform_route(False)['actual_G_roi_calls'],0)

        def test_every_added_method_has_report_color(self):
            from demo import routervc_report
            methods=set(routervc_report.METHODS)|set(REGISTRY)
            self.assertTrue({*(f'uf_qp{q}' for q in QPS),'full_g','full_frame_g',
                             'route_no_g','route_no_e'}<=methods)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ProtocolTest))
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':
    arguments=parser().parse_args()
    if arguments.limit_samples<0:raise ValueError('--limit-samples must be nonnegative')
    main(arguments)
