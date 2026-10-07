"""Stage A: native-symbol split and same-context endpoint, not deployable RD.

One native I bootstrap, one P8, fixed diagnostic windows. No training or G.
Every completed point is atomic and artifact-checked on resume. Cached context
is explicitly a tensor diagnostic; it is not a free reference in a bitstream.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from demo.scalable_codec import BaseCodec, atomic_bytes, atomic_json, atomic_npz, file_hash
from routervc.latent.split import split_symbols, restore_symbols, coarse_representatives

REPO = Path(__file__).resolve().parents[1]
DEFAULT = Path('/root/autodl-fs/DCVC/runs/routervc_latent_20261007/stage_a')
OLD_PROTOCOL = Path('/root/autodl-fs/DCVC/runs/routervc_sender_20261005/evaluation/protocol.json')
SAMPLES = ('reds-val-000-f000-n17-fullview', 'reds-val-005-f000-n17-fullview',
           'uvg-beauty-f000-historical-evaluation-crop',
           'uvg-bosphorus-f000-historical-evaluation-crop')
QPS, WIDTHS = (32, 48), (3, 9, 17)


def read(path):
    return json.loads(Path(path).read_text())


def compare(a, b):
    import torch
    if a.shape != b.shape:
        raise ValueError('comparison shape changed')
    return dict(exact=bool(torch.equal(a, b)), max_abs=float((a.float()-b.float()).abs().max()),
                changed=int((a != b).sum()), numel=a.numel())


def context_exact(bridge, proxy, expected):
    import torch
    current = bridge.context(proxy)
    # Includes unused/uninitialized buffers: compare bytes, not NaN equality.
    return all(torch.equal(current[k].contiguous().reshape(-1).view(torch.uint8),
                           expected[k].contiguous().reshape(-1).view(torch.uint8)) for k in current)


def verify_point(folder):
    result = read(folder/'complete.json')
    for name, expected in result['artifacts'].items():
        if file_hash(folder/name) != expected:
            raise ValueError(f'changed completed artifact {folder/name}')
    return result


def fixed_image(path, source, native, coarse, sid, qp):
    from PIL import Image, ImageDraw
    frames = [('Source', source[4]), ('Native full', native[4])]
    frames += [(f'B width {width}', coarse[width][4]) for width in WIDTHS]
    width = 384
    h, w = source.shape[1:3]
    height = round(width*h/w)
    canvas = Image.new('RGB', (width*len(frames), 2*height+80), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 6), f'{sid} | q*={qp}; P8 local frame 4; G off; NOT a bitrate plot', fill='black')
    for i, (title, pixels) in enumerate(frames):
        draw.text((i*width+8, 28), title, fill='black')
        im = Image.fromarray(pixels)
        canvas.paste(im.resize((width, height)), (i*width, 48))
        cw, ch = w//3, h//3
        x, y = (w-cw)//2, (h-ch)//2
        canvas.paste(im.crop((x, y, x+cw, y+ch)).resize((width, height)),
                     (i*width, height+68))
    canvas.save(path)


def one(codec, bridge, metric, source, sid, qp, folder):
    import torch
    from demo.chunk_enhancement_codec import configure_torch
    from demo.stage_c_three_path_roi_probe import tensor_from_rgb, rgb_from_tensor, evaluate_variant
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    configure_torch()
    torch.cuda.set_stream(codec.stream)
    h, w = source.shape[1:3]
    pad_r, pad_b = codec.i_net.get_padding_size(h, w, 16)
    tensors = [tensor_from_rgb(frame, codec.device) for frame in source[:9]]
    with torch.inference_mode():
        i_encoded = codec.i_net.compress(tensors[0], 32, pad_b, pad_r)
        i_ref = i_encoded['x_hat'].clone()
        codec.p_net.add_ref_feature_from_frame(i_ref)
        before = bridge.context(codec.p_net.proxy)
        p_encoded = codec.p_net.compress(torch.cat(tensors[1:9], 1), qp, False, pad_b, pad_r)
        enc = bridge.encoded(codec.p_net.proxy)
        symbols = enc['symbols'].short().cpu().numpy()
        if not torch.equal(enc['symbols'], enc['symbols'].round()):
            raise RuntimeError('native symbols are not integers')

        # Genuine native decode of the original high-q payload, same I context.
        codec.p_net.add_ref_feature_from_frame(i_ref, apply_feature_adaptor=False)
        native = codec.p_net.decompress(p_encoded['bit_stream'], {'height':h,'width':w},
                                       qp, p_encoded['ec_parallel'], False)['x_hat']
        native = [frame.clone() for frame in native]
        native_f, native_ctx = codec.p_net.proxy.get_decoded_features()
        saved = bridge.context(codec.p_net.proxy)
        full = bridge.synthesize(codec.p_net.proxy, enc['z'], enc['symbols'], qp, before)
        full_checks = dict(latent=compare(enc['latent'], full['latent']),
                           feature=compare(native_f, full['feature']),
                           context=compare(native_ctx, before['ctx']),
                           rgb_tensors=[compare(a, full[f'frame{i}']) for i,a in enumerate(native)],
                           reference_unchanged=context_exact(bridge, codec.p_net.proxy, saved))
        if not (full_checks['latent']['exact'] and full_checks['feature']['exact']
                and full_checks['context']['exact'] and full_checks['reference_unchanged']
                and all(r['exact'] for r in full_checks['rgb_tensors'])):
            atomic_json(folder/'endpoint_failed.json', full_checks)
            raise RuntimeError('native endpoint is not exact; diagnose before claiming scalability')
        native_rgb = np.stack([rgb_from_tensor(frame,h,w) for frame in native])
        variants, pixels = {}, {}
        for width in WIDTHS:
            c, r = split_symbols(symbols, width)
            restored = restore_symbols(c, r, width)
            if not np.array_equal(restored, symbols):
                raise RuntimeError('integer roundtrip failed')
            restored_gpu = torch.from_numpy(restored).to(enc['symbols']).contiguous(memory_format=torch.channels_last)
            complete = bridge.synthesize(codec.p_net.proxy, enc['z'], restored_gpu, qp, before)
            for key in full:
                if not torch.equal(full[key], complete[key]):
                    raise RuntimeError(f'complete layer changed {key}')
            reps = torch.from_numpy(coarse_representatives(c, width)).to(enc['symbols']).contiguous(memory_format=torch.channels_last)
            bottom = bridge.synthesize(codec.p_net.proxy, enc['z'], reps, qp, before)
            unchanged = context_exact(bridge, codec.p_net.proxy, saved)
            if not unchanged:
                raise RuntimeError('display synthesis mutated reference state')
            pixels[width] = np.stack([rgb_from_tensor(bottom[f'frame{i}'],h,w) for i in range(8)])
            variants[str(width)] = dict(symbols_exact=True, full_tensors_exact=True,
                reference_unchanged=unchanged, nonzero_coarse=int(np.count_nonzero(c)),
                nonzero_refinement=int(np.count_nonzero(r)),
                mean_symbol_abs_error=float(np.mean(np.abs(coarse_representatives(c,width)-symbols))),
                prior_mean_differences=[compare(full[f'means{i}'],bottom[f'means{i}']) for i in range(4)])
        torch.cuda.synchronize()
    for width in WIDTHS:
        variants[str(width)]['quality'] = evaluate_variant(source[1:9], pixels[width], metric)
    full_quality = evaluate_variant(source[1:9], native_rgb, metric)
    atomic_bytes(folder/'native_i.payload', bytes(i_encoded['bit_stream']))
    atomic_bytes(folder/'native_p.payload', bytes(p_encoded['bit_stream']))
    atomic_npz(folder/'symbols.npz', symbols=symbols, z=enc['z'].cpu().numpy())
    atomic_npz(folder/'pixels.npz', source=source[1:9], full=native_rgb,
               **{f'B{width}':pixels[width] for width in WIDTHS})
    fixed_image(folder/'fixed.png',source[1:9],native_rgb,pixels,sid,qp)
    result = dict(schema='routervc-latent-stage-A-v1',sample_id=sid,qp_star=qp,
        I_qp=32,widths=list(WIDTHS),generation=False,router=False,training=False,
        sample_role='previously-used diagnostic/development',shape=list(source[:9].shape),
        measured_frames='P frames 1..8; fixed native I QP32 reference',
        rate_scope='tensor diagnostic only; native payload sizes NOT a scalable-stream bpp',
        native_I_payload_bytes=len(i_encoded['bit_stream']),native_P_payload_bytes=len(p_encoded['bit_stream']),
        native_P_ec_parallel=int(p_encoded['ec_parallel']),
        native_endpoint_checks=full_checks,native_full=full_quality,coarse=variants,
        symbol_shape=list(symbols.shape),symbol_min=int(symbols.min()),symbol_max=int(symbols.max()),
        seconds=time.monotonic()-started,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        artifacts={name:file_hash(folder/name) for name in ('native_i.payload','native_p.payload',
                   'symbols.npz','pixels.npz','fixed.png')})
    atomic_json(folder/'complete.json', result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['run','verify'])
    p.add_argument('--output',type=Path,default=DEFAULT)
    p.add_argument('--limit',type=int,default=4)
    p.add_argument('--max-hours',type=float,default=4)
    args=p.parse_args()
    if not os.environ.get('TMUX'): raise RuntimeError('tmux required')
    if args.limit not in range(1,5): raise ValueError('limit must be 1..4')
    from demo.chunk_enhancement_experiment import Run
    from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
    from demo.stage_c_three_path_roi_probe import LPIPSAlex
    from routervc.latent.native import load_bridge
    inputs=read(OLD_PROTOCOL)['inputs']
    names=SAMPLES[:args.limit]
    code_files=[Path(__file__),*[REPO/'routervc/latent'/f for f in ('__init__.py','split.py','native.py')],
                REPO/'routervc/latent/native_bridge.cpp',REPO/'tools/run_latent_probe.sh']
    protocol=dict(schema='routervc-latent-stage-A-v1',qps=list(QPS),widths=list(WIDTHS),I_qp=32,
        samples={s:{k:inputs[s][k] for k in ('source_path','source_sha256')} for s in names},
        code={str(f.relative_to(REPO)):file_hash(f) for f in code_files},
        models={str(f):file_hash(f) for f in (REPO/'checkpoints/cvpr2026_image.pth.tar',
                                         REPO/'checkpoints/cvpr2026_video_hts.pth.tar')})
    args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/'protocol.json'
    if path.exists():
        if read(path)!=protocol: raise ValueError('protocol/source changed; use a distinct output')
    else:
        if args.command=='verify': raise RuntimeError('no completed experiment')
        atomic_json(path,protocol)
    for item in protocol['samples'].values():
        if file_hash(Path(item['source_path']))!=item['source_sha256']: raise ValueError('source changed')
    results=[]
    if args.command=='verify' or (args.output/'complete.json').exists():
        for sid in names:
            for qp in QPS: results.append(verify_point(args.output/sid/f'q{qp}'))
        done=read(args.output/'complete.json')
        if done['point_count']!=len(results): raise ValueError('wrong completion count')
        if done['summary_sha256']!=file_hash(args.output/'summary.json'): raise ValueError('summary changed')
        print(json.dumps(dict(verified_points=len(results),no_inference=True)),flush=True)
        return
    run=Run(args);run.thread.start()
    try:
        run.log_resources()
        with exclusive_native_evaluation(run):
            bridge,pins=load_bridge()
            pin_path=args.output/'native_bindings.json'
            if pin_path.exists() and read(pin_path)!=pins: raise ValueError('native ABI/binary changed')
            if not pin_path.exists(): atomic_json(pin_path,pins)
            codec=BaseCodec(REPO/'checkpoints/cvpr2026_image.pth.tar',REPO/'checkpoints/cvpr2026_video_hts.pth.tar')
            metric=LPIPSAlex(True)
            for sid in names:
                with np.load(inputs[sid]['source_path'],allow_pickle=False) as data:
                    source=data['source'][:9].copy()
                for qp in QPS:
                    run.check()
                    folder=args.output/sid/f'q{qp}';folder.mkdir(parents=True,exist_ok=True)
                    run.update(sample=sid,qp=qp,completed=len(results),phase='native_endpoint_and_split')
                    result=verify_point(folder) if (folder/'complete.json').exists() else one(codec,bridge,metric,source,sid,qp,folder)
                    results.append(result)
                    run.update(completed=len(results),total=len(names)*len(QPS))
                    print(json.dumps(dict(sample=sid,qp=qp,full=result['native_full'],
                            coarse={w:v['quality'] for w,v in result['coarse'].items()})),flush=True)
        atomic_json(args.output/'summary.json',dict(results=results))
        atomic_json(args.output/'complete.json',dict(complete=True,point_count=len(results),
                    full_native_endpoint_exact=True,seconds=time.monotonic()-run.started,
                    summary_sha256=file_hash(args.output/'summary.json')))
    finally:
        run.stop.set();run.thread.join(timeout=2);run.log_resources()


if __name__=='__main__': main()
