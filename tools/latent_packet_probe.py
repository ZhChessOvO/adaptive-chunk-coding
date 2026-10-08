"""Pixel-preserving E packing comparison, then continuous regional refinement.

All new code/profiles are separate from completed October 7 experiments. Fresh
receivers take only the stream and shared models. No G or training in this file.
"""
import argparse
import gc
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from demo.scalable_codec import atomic_bytes, atomic_json, atomic_npz, file_hash
from demo.scalable_format import frame_hash
from tools.latent_probe import SAMPLES, read
from tools.latent_stream_probe import DEFAULT as OLD_B, verify
from tools.latent_regional_probe import ORDER, COUNTS
from routervc.latent import packet_format as wire

ROOT = Path('/root/autodl-fs/DCVC/runs/routervc_latent_20261008')
OLD_C = OLD_B.parent/'stage_c'
OLD_R = OLD_B.parent/'regional'
CASES = [(sid, 17) for sid in SAMPLES]+[(SAMPLES[0], 41), (SAMPLES[0], 38)]


def receiver_guard(output):
    def audit(event, values):
        if event == 'open' and isinstance(values[0], (str, bytes)):
            path = Path(os.fsdecode(values[0])).resolve()
            if path.is_relative_to(output.resolve()):
                return
            if (path.name in ('source.npz', 'pair.npz', 'pixels.npz', 'expected.npz', 'symbols.npz', 'protocol.json')
                    or '/DCVC/data/' in str(path) or '/assets/evaluation/' in str(path)):
                raise RuntimeError('source/sender cache forbidden in packet receiver')
    sys.addaudithook(audit)


def receiver(args):
    import torch
    from routervc.latent.packet_codec import PacketCodec
    receiver_guard(args.output)
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    codec = PacketCodec()
    base, out, info = codec.decode_packets(args.stream.read_bytes(), allow_incomplete_tail=args.allow_incomplete_tail)
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_npz(args.output/'pixels.npz', base=base, reconstruction=out)
    info.update(seconds=time.monotonic()-start, pid=os.getpid(), stream_sha256=file_hash(args.stream),
                base_hash=frame_hash(base), output_hash=frame_hash(out),
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
    atomic_json(args.output/'complete.json', dict(info, artifacts={'pixels.npz': file_hash(args.output/'pixels.npz')}))


def fresh(run, stream, output, allow=False):
    if (output/'complete.json').exists():
        result = verify(output)
        if result['stream_sha256'] != file_hash(stream):
            raise ValueError('changed receiver stream')
        return result
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-m', 'tools.latent_packet_probe', 'decode',
               '--stream', str(stream), '--output', str(output)]
    if allow:
        command += ['--allow-incomplete-tail']
    with (output/'worker.log').open('a') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            start = time.monotonic()
            while child.poll() is None:
                run.check()
                if time.monotonic()-start > 900:
                    raise TimeoutError('fresh packet receiver timeout')
                time.sleep(.3)
            if child.returncode:
                raise RuntimeError(f'packet receiver failed: {output}/worker.log')
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait()
    return verify(output)


def bind_sources(mode, limit):
    inputs = read(OLD_B/'protocol.json')['inputs']
    cases = [(s, 9) for s in SAMPLES] if mode == 'single' else CASES
    cases = cases[:limit]
    records = []
    for sid, count in cases:
        if count <= 17:
            path = Path(inputs[sid]['source_path'])
            if file_hash(path) != inputs[sid]['source_sha256']:
                raise ValueError('old source changed')
            record = dict(path=str(path), sha256=file_hash(path), count=count, kind='existing_npz')
        else:
            paths = [Path('/root/autodl-fs/DCVC/data/REDS/val_sharp/000')/f'{i:08d}.png' for i in range(count)]
            record = dict(paths=[str(p) for p in paths], hashes=[file_hash(p) for p in paths], count=count,
                          kind='fullview_resize', width=1024, height=576, resize='PIL LANCZOS')
        records.append(dict(sample=sid, frames=count, source=record,
                            name=sid if mode == 'single' else f'{sid}_n{count}'))
    return records


def source_pixels(record):
    if record['kind'] == 'existing_npz':
        with np.load(record['path'], allow_pickle=False) as f:
            return f['source'][:record['count']].copy()
    from PIL import Image
    return np.stack([np.asarray(Image.open(p).convert('RGB').resize(
        (record['width'], record['height']), Image.Resampling.LANCZOS)) for p in record['paths']])


def prepare(case, folder, mode):
    import torch
    from routervc.latent.packet_codec import PacketCodec
    from routervc.latent.chain_codec import ChainCodec
    from routervc.latent import chain_format
    if (folder/'encoded.json').exists():
        enc = read(folder/'encoded.json')
        for n, h in enc['artifacts'].items():
            if file_hash(folder/n) != h:
                raise ValueError('changed encoded artifact')
        return enc
    began = time.monotonic()
    sid, count = case['sample'], case['frames']
    if mode == 'single':
        verify(OLD_B/sid)
        full = (OLD_B/sid/'w3_BE.rvl').read_bytes()
        expected_base = read(OLD_B/sid/'w3_B/complete.json')['output_hash']
        expected_full = read(OLD_B/sid/'w3_BE/complete.json')['output_hash']
        expected_refs, native_info = [], {}
    else:
        old = OLD_C/f'{sid}_n{count}'
        if (old/'complete.json').exists():
            verify(old)
            full = (old/'all_E.rvlc').read_bytes()
            enc = read(old/'encoded.json')
            with np.load(old/'expected.npz', allow_pickle=False) as f:
                base, endpoint, own = f['base'], f['full'], f['native_own']
            native = (old/'native_own_reference.bin').read_bytes()
        else:
            codec = ChainCodec()
            b, ps, base, endpoint, native, own, enc = codec.encode_chain(source_pixels(case['source']))
            full = b+b''.join(ps)
            del codec; gc.collect(); torch.cuda.empty_cache()
        atomic_bytes(folder/'legacy_full.rvlc', full)
        atomic_bytes(folder/'native_own.bin', native)
        atomic_npz(folder/'native_own.npz', reconstruction=own)
        expected_base, expected_full = frame_hash(base), frame_hash(endpoint)
        expected_refs = enc['base_reference_hashes']
        native_info = dict(bytes=len(native), same_context_chunks=enc['same_context_chunks'])
    codec = PacketCodec()
    b, ps, audit = codec.repacketize(full, 1 if mode == 'single' else 2)
    del codec; gc.collect(); torch.cuda.empty_cache()
    chunks = (count-1+7)//8
    # Region-major prefixes: E4 means 4 regions per P8, NOT 25% byte budget.
    ordered = [(j, k) for k in ORDER for j in range(chunks)]
    streams = {f'E{c}': b+b''.join(ps[j, k] for k in ORDER[:c] for j in range(chunks)) for c in COUNTS}
    if mode == 'single':
        streams['single5'] = b+ps[0, 5]
    else:
        streams['alternating'] = b+b''.join(ps[j, k] for j in range(chunks) if j%2 == 0 for k in ORDER[:4])
    if sid == SAMPLES[0]:
        streams.update(reverse=b+b''.join(ps[j, k] for j, k in reversed(ordered)),
                       reverse8=b+b''.join(ps[j, k] for j, k in reversed(ordered[:8*chunks])),
                       repeat=streams['E16']+ps[0, 5], incomplete=streams['E8']+ps[0, ORDER[8]][:-1])
    for lo, hi in zip(COUNTS, COUNTS[1:]):
        if not streams[f'E{hi}'].startswith(streams[f'E{lo}']):
            raise RuntimeError('not a literal append-only prefix')
    for name, data in streams.items():
        atomic_bytes(folder/f'{name}.rvlp', data)
    names = [f'{n}.rvlp' for n in streams]
    if mode != 'single':
        names += ['legacy_full.rvlc', 'native_own.bin', 'native_own.npz']
    result = dict(seconds=time.monotonic()-began, streams=list(streams), audit=audit,
        expected_base=expected_base, expected_full=expected_full, expected_refs=expected_refs,
        native=native_info, artifacts={n: file_hash(folder/n) for n in names})
    atomic_json(folder/'encoded.json', result)
    return result


def execute(args):
    from routervc.latent.packet_codec import packet_hash
    from demo.chunk_enhancement_experiment import Run
    from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
    from demo.stage_c_three_path_roi_probe import LPIPSAlex, evaluate_variant
    if not os.environ.get('TMUX'):
        raise RuntimeError('tmux required')
    if not 1 <= args.limit <= (4 if args.mode == 'single' else 6):
        raise ValueError('invalid sample limit')
    cases = bind_sources(args.mode, args.limit)
    args.output = args.output or ROOT/args.mode
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = dict(profile='RVLPACK2', packet_hash=packet_hash(), driver=file_hash(Path(__file__)),
        mode=args.mode, cases=cases, order=list(ORDER), counts=list(COUNTS),
        training=False, generation=False, router=False, mask_bytes=0,
        role='previously-used diagnostics; new 38/41 frame windows in REDS val000 for reset/tail only',
        timing='single=repacketize; continuous=encode if uncached plus repacketize; fresh worker includes load')
    pp = args.output/'protocol.json'
    if pp.exists():
        if read(pp) != protocol:
            raise ValueError('changed experiment protocol; choose new output')
    elif args.command == 'verify':
        raise ValueError('experiment not complete')
    else:
        atomic_json(pp, protocol)
    if args.command == 'verify' or (args.output/'complete.json').exists():
        verify(args.output)
        for case in cases:
            d = args.output/case['name']
            for name in verify(d)['points']:
                verify(d/name)
        print('Verified compact E experiment without inference', flush=True)
        return
    run = Run(args); run.thread.start()
    results = []
    try:
        with exclusive_native_evaluation(run):
            metric = LPIPSAlex(True)
            for case in cases:
                run.check()
                folder = args.output/case['name']; folder.mkdir(parents=True, exist_ok=True)
                if (folder/'complete.json').exists():
                    results.append(verify(folder)); continue
                run.update(sample=case['name'], phase='prepare_packets', completed=len(results))
                enc = prepare(case, folder, args.mode)
                source = source_pixels(case['source'])
                points = {}
                for name in enc['streams']:
                    run.update(phase='fresh_region_decode', point=name)
                    point = fresh(run, folder/f'{name}.rvlp', folder/name, name == 'incomplete')
                    if point['base_hash'] != enc['expected_base'] or point['base_reference_hashes'] != enc['expected_refs']:
                        raise RuntimeError('regional arrival affected B/reference')
                    if name in ('E16', 'reverse', 'repeat') and point['output_hash'] != enc['expected_full']:
                        raise RuntimeError('native same-context endpoint changed')
                    if point['actual_bytes'] != (folder/f'{name}.rvlp').stat().st_size:
                        raise RuntimeError('file byte mismatch')
                    if point['base_bytes']+point['E_wire_bytes']+point['ignored_tail_bytes'] != point['actual_bytes']:
                        raise RuntimeError('byte decomposition mismatch')
                    if args.mode == 'single':
                        old = read(OLD_R/case['sample']/name/'complete.json')
                        if point['output_hash'] != old['output_hash']:
                            raise RuntimeError('E packing changed decoded pixels')
                        old_score = read(OLD_R/case['sample']/'complete.json')['points'][name]
                        score = dict(quality=old_score['quality_all9'], old_bytes=old['actual_bytes'],
                                     old_E_bytes=old['E_wire_bytes'], pixel_exact_to_old=True)
                    else:
                        with np.load(folder/name/'pixels.npz', allow_pickle=False) as f:
                            output = f['reconstruction']
                        score = dict(quality=evaluate_variant(source, output, metric))
                    points[name] = dict(point, **score)
                    print(f'{case["name"]} {name}: {point["actual_bytes"]} B, '
                          f'LPIPS {score["quality"]["lpips_alex"]:.6f}', flush=True)
                for key in ('reverse8', 'incomplete'):
                    if key in points and points[key]['output_hash'] != points['E8']['output_hash']:
                        raise RuntimeError('partial arrival/fallback changed')
                native = enc['native']
                if native:
                    with np.load(folder/'native_own.npz', allow_pickle=False) as f:
                        native = dict(native, quality=evaluate_variant(source, f['reconstruction'], metric),
                                      bpp=native['bytes']*8/np.prod(source.shape[:3]))
                files = list(enc['artifacts'])+['encoded.json']+[f'{n}/complete.json' for n in points]
                result = dict(sample=case['sample'], frames=case['frames'], points=points, native=native,
                              audit=enc['audit'], artifacts={n: file_hash(folder/n) for n in files})
                atomic_json(folder/'complete.json', result); results.append(result)
                run.update(completed=len(results), total=len(cases))
        atomic_json(args.output/'summary.json', dict(results=results))
        atomic_json(args.output/'complete.json', dict(complete=True, cases=len(results),
            fresh_decodes=sum(len(r['points']) for r in results), seconds=time.monotonic()-run.started,
            artifacts={'protocol.json': file_hash(pp), 'summary.json': file_hash(args.output/'summary.json')}))
    finally:
        run.stop.set(); run.thread.join(timeout=2); run.log_resources()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('run', 'verify', 'decode'))
    p.add_argument('--mode', choices=('single', 'continuous'), default='single')
    p.add_argument('--limit', type=int, default=4)
    p.add_argument('--output', type=Path)
    p.add_argument('--stream', type=Path)
    p.add_argument('--max-hours', type=float, default=4)
    p.add_argument('--allow-incomplete-tail', action='store_true')
    args = p.parse_args()
    receiver(args) if args.command == 'decode' else execute(args)


if __name__ == '__main__':
    main()
