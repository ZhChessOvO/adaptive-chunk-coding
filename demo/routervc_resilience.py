"""Three fresh RouterVC resilience checks, without training or quality metrics.

The first two files differ only by an incomplete next E packet. The receiver
must ignore its charged tail and reproduce the complete-one-packet pixels and
route, including partial temporal E coverage. The third file has the complete
UF base, no E and G disabled, and is decoded with missing E/G/Router assets.
Malformed CRC and strict truncated parsing are rejected on CPU BEFORE any GPU
child starts. Run in tmux; the parent holds the normal global GPU mutex.
"""
import argparse
import math
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from demo import routervc_format as fmt
from demo.routervc_decode import coverage_from_packets
from demo.routervc_policy import grid_rois
from demo.routervc_audit import Reader, require
from demo.chunk_enhancement_experiment import Run, read
from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
from demo.conditioned_generation_pipeline import execute
from demo.four_state_core import immutable_json
from demo.scalable_codec import atomic_bytes, atomic_json, file_hash


RUNS = Path('/root/autodl-fs/DCVC/runs')
DEFAULT = RUNS/'routervc_resilience_20261003'
SMOKE = RUNS/'routervc_20261003_smoke'
ENHANCEMENT = RUNS/'a800_online_eg_20261002/joint/enhancement.pt'
ADAPTER = ENHANCEMENT.parent/'adapter.pt'
ROUTER = RUNS/'a800_four_state_router_20261002/context/model.pt'
CODE = ('routervc_resilience.py', 'routervc_format.py', 'routervc_policy.py',
        'routervc_decode.py', 'routervc_audit.py', 'four_state_router.py',
        'four_state_router_evaluate.py', 'run_routervc_resilience.sh')


def reject(data, *, allow_incomplete_tail=False):
    try:
        fmt.parse(data, allow_incomplete_tail=allow_incomplete_tail)
    except ValueError as error:
        return str(error)
    raise RuntimeError('malformed/truncated stream unexpectedly accepted')


def make_cases(wire):
    """Pure CPU construction and negative parsing checks; no model APIs."""
    config, inner_bytes, inner, outer = fmt.parse(wire)
    require(len(inner.packets) >= 2, 'need at least two received E packets')
    first, second = inner.packets[:2]
    require(first.meta['start'] == 0 and first.meta['count'] == 1,
            'expected region bundle to begin with its I-frame E packet')
    require(second.meta['roi'] == first.meta['roi'] and second.meta['start'] == 1,
            'expected the next packet from the same region bundle')
    require(len(second.payload) >= 2, 'second payload too small for an interior truncation')
    end = outer + first.end_offset
    clean = wire[:end]
    cutoff = outer + second.end_offset - max(1, len(second.payload)//2)
    truncated = wire[:cutoff]
    require(end < cutoff < outer + second.end_offset, 'invalid truncation position')
    _, _, a, _ = fmt.parse(clean)
    _, _, b, _ = fmt.parse(truncated, allow_incomplete_tail=True)
    require(len(a.packets) == len(b.packets) == 1 and a.packets[0].wire == b.packets[0].wire,
            'truncation did not preserve exactly one complete packet')
    require(a.base == b.base == inner.base and b.incomplete_tail_bytes == len(truncated)-len(clean),
            'ignored tail/base accounting mismatch')
    damaged = clean[:-1] + bytes([clean[-1] ^ 1])
    rejection = dict(strict_truncated=reject(truncated), crc_strict=reject(damaged),
                     crc_permissive=reject(damaged, allow_incomplete_tail=True))
    base = fmt.wrap(inner_bytes[:inner.base_end], dict(config, max_g=0))
    _, _, base_inner, _ = fmt.parse(base)
    require(base_inner.base == inner.base and not base_inner.packets, 'base-only construction changed UF')
    rois = grid_rois(inner.meta['height'], inner.meta['width'])
    expected = coverage_from_packets(a, rois).tolist()
    require(any(0 < value < 1 for value in expected), 'probe did not produce partial temporal coverage')
    return dict(first_packet=clean, truncated_next_packet=truncated, base_no_models=base), damaged, dict(
        rejection=rejection, original_packets=len(inner.packets), complete_packets=1,
        first_packet_meta=first.meta, partial_temporal_coverage=expected,
        ignored_tail_bytes=b.incomplete_tail_bytes, original_base_hash=inner.meta['base_rgb_sha256'],
        original_shape=[inner.meta['frame_count'], inner.meta['height'], inner.meta['width'], 3])


def prepare(root, smoke_root):
    completed = read(smoke_root/'complete.json')
    require(completed['complete'], 'RouterVC smoke must be complete first')
    require(file_hash(smoke_root/'summary.json') == completed['summary'], 'smoke summary changed')
    require(file_hash(smoke_root/'protocol.json') == completed['protocol'], 'smoke protocol changed')
    jobs = read(smoke_root/'jobs.json')['jobs']
    source = next(j for j in jobs if j['method'] == 'context_smooth' and j['ratio'] == .5)
    path = Path(source['folder'])/'stream.rtvc'
    require(file_hash(path) == source['stream_sha256'] and path.stat().st_size == source['bytes'],
            'source smoke stream changed')
    require(read(Path(source['folder'])/'job.json') == source, 'source smoke JOB changed')
    stream = path.read_bytes()
    cases, damaged, details = make_cases(stream)  # All CPU rejection tests precede GPU setup.
    config, _, parsed, _ = fmt.parse(stream)
    require(config['policy'] == fmt.policy_identity(), 'receiver policy changed since smoke')
    require(config['router'] == file_hash(ROUTER), 'Router weights changed since smoke')
    require(config['lora'] == file_hash(ADAPTER), 'G weights changed since smoke')
    require(parsed.meta['enhancement_model_sha256'] == file_hash(ENHANCEMENT), 'E weights changed since smoke')
    protocol = dict(code={name: file_hash(REPO/'demo'/name) for name in CODE},
        source_smoke=str(smoke_root.resolve()), source_stream=str(path.resolve()),
        source_stream_sha256=file_hash(path), source_sample_id=source['sample_id'],
        source_smoke_complete=file_hash(smoke_root/'complete.json'), details=details,
        router=file_hash(ROUTER), enhancement=file_hash(ENHANCEMENT), adapter=file_hash(ADAPTER),
        role='partial-prefix and asset-free fallback mechanism; no metric or training evidence',
        comparison='first E packet versus identical packet plus ignored incomplete next packet',
        cases={name: dict(allow_incomplete_tail=name == 'truncated_next_packet',
                          missing_all_optional_assets=name == 'base_no_models') for name in cases})
    immutable_json(root/'protocol.json', protocol)
    for name, wire in dict(cases, crc_rejected_cpu=damaged).items():
        target = root/f'{name}.rtvc'
        if target.exists():
            require(target.read_bytes() == wire, f'changed probe stream: {target}')
        else:
            atomic_bytes(target, wire)
    immutable_json(root/'cpu_rejections.json', dict(complete=True, gpu_started=False,
        **details['rejection'], artifacts={name+'.rtvc': file_hash(root/(name+'.rtvc'))
                                         for name in ('truncated_next_packet', 'crc_rejected_cpu')}))
    return protocol


def validate_saved(root, name, protocol, reader):
    path = root/f'{name}.rtvc'
    spec = protocol['cases'][name]
    wire = path.read_bytes()
    config, _, inner, header = fmt.parse(wire, allow_incomplete_tail=spec['allow_incomplete_tail'])
    folder = root/name
    decoded = reader.json(folder/'decode.json')
    reader.digest(path, decoded['stream_sha256'])
    require(decoded['config'] == config and decoded['total_bytes'] == len(wire), 'receiver wire/config mismatch')
    expected = dict(base_bytes=len(inner.base), container_header_bytes=inner.base_end-len(inner.base),
                    packet_bytes=sum(len(p.wire) for p in inner.packets),
                    incomplete_tail_bytes=inner.incomplete_tail_bytes, generation_control_bytes=header)
    require(sum(expected.values()) == len(wire) and all(decoded[k] == v for k, v in expected.items()),
            'receiver tail/packet byte accounting mismatch')
    require(decoded['source_frames_read'] is False and decoded['base_reference_unchanged'] is True
            and decoded['outside_generate_exact'] is True, 'source-free receiver invariants failed')
    require(decoded['base_hash'] == protocol['details']['original_base_hash'], 'UF base changed')
    video = reader.video(folder/'reconstruction.npz')
    require(video['shape'] == protocol['details']['original_shape'], 'receiver geometry changed')
    for field, key in (('base', 'base_hash'), ('enhanced', 'generation_input_hash'), ('reconstruction', 'output_hash')):
        require(video['hashes'][field] == decoded[key], f'saved {field} hash mismatch')
    rois = grid_rois(inner.meta['height'], inner.meta['width'])
    coverage = coverage_from_packets(inner, rois).tolist()
    require(decoded['route']['coverage'] == coverage, 'receiver partial E coverage mismatch')
    if spec['missing_all_optional_assets']:
        require(not inner.packets and coverage == [0.]*16, 'base-only file contains E')
        require(not decoded['generation_assets_validated'] and not decoded['shared_router_used']
                and not decoded['generation_executed'], 'asset-free path executed optional models')
        require(decoded['output_hash'] == decoded['generation_input_hash'] == decoded['base_hash'],
                'asset-free fallback differs from base')
    else:
        require(len(inner.packets) == 1 and coverage == protocol['details']['partial_temporal_coverage'],
                'wrong complete-packet coverage')
        require(decoded['shared_router_used'], 'partial coverage did not reach shared Router')
    return decoded


def run_checks(run, protocol, *, verify_only=False):
    root = run.root
    reader = Reader()
    records = {}
    for index, (name, spec) in enumerate(protocol['cases'].items()):
        run.check()
        folder = root/name
        folder.mkdir(exist_ok=True)
        if not (folder/'decode.json').exists():
            require(not verify_only, f'missing completed fresh receiver: {name}')
            missing = spec['missing_all_optional_assets']
            if missing:
                require(all(not Path(path).exists() for path in
                            ('/missing/routervc-E.pt', '/missing/routervc-G.pt', '/missing/routervc-router.pt')),
                        'negative-control asset paths unexpectedly exist')
            argv = ['--stream', root/f'{name}.rtvc', '--output', folder,
                '--enhancement', '/missing/routervc-E.pt' if missing else ENHANCEMENT,
                '--adapter', '/missing/routervc-G.pt' if missing else ADAPTER,
                '--router', '/missing/routervc-router.pt' if missing else ROUTER]
            if spec['allow_incomplete_tail']:
                argv += ['--allow-incomplete-tail']
            execute(run, name, 'routervc_decode.py', argv, distributed=True)
        decoded = validate_saved(root, name, protocol, reader)
        record = dict(complete=True, case=name, decoded=decoded,
            artifacts={filename: reader.digest(folder/filename)
                       for filename in ('decode.json', 'reconstruction.npz')})
        done = folder/'complete.json'
        if done.exists():
            require(read(done) == record, f'completed resilience case changed: {name}')
        else:
            require(not verify_only, f'missing case completion: {name}')
            atomic_json(done, record)
        records[name] = record
        run.update(phase='verified' if verify_only else 'checked', completed=index+1, total=3)
    clean = records['first_packet']['decoded']
    partial = records['truncated_next_packet']['decoded']
    require(clean['output_hash'] == partial['output_hash']
            and clean['generation_input_hash'] == partial['generation_input_hash']
            and clean['route'] == partial['route'], 'ignored tail changed pixels or shared routing')
    require(partial['total_bytes']-clean['total_bytes'] == protocol['details']['ignored_tail_bytes'],
            'ignored tail was not charged exactly')
    return records


def main(args):
    import os
    require(bool(os.environ.get('TMUX')), 'long resilience checks must run in tmux')
    require(math.isfinite(args.max_hours) and args.max_hours > 0, 'max-hours must be finite and positive')
    run = Run(args)
    run.thread.start()
    began = time.monotonic()
    try:
        # Prepare/reject malformed variants entirely on CPU, then obtain the
        # native codec mutex for all three genuine fresh receivers.
        protocol = prepare(args.output, args.smoke_root)
        with exclusive_native_evaluation(run):
            before = {str(p): file_hash(p) for p in args.output.glob('*/decode.json')}
            records = run_checks(run, protocol, verify_only=args.command == 'verify')
            result = dict(complete=True, fresh_decodes=3, truncated_pixels_and_route_exact=True,
                partial_temporal_coverage=protocol['details']['partial_temporal_coverage'],
                source_frames_read=False, missing_E_G_router_fallback_exact=True,
                strict_truncation_and_CRC_rejected_before_GPU=True,
                no_new_training=True, no_quality_metric_recalculation=True,
                records=records, protocol=file_hash(args.output/'protocol.json'))
            done = args.output/'complete.json'
            if done.exists():
                old = read(done)
                require(all(old[k] == v for k, v in result.items()), 'completed resilience summary changed')
            else:
                require(args.command != 'verify', 'missing final resilience completion')
                atomic_json(done, dict(result, elapsed_seconds=time.monotonic()-began))
            require(all(file_hash(Path(p)) == digest for p, digest in before.items()), 'original decoder timings changed')
            run.update(phase='verified' if args.command == 'verify' else 'complete', completed=3, total=3)
    except BaseException as error:
        atomic_json(args.output/'last_failure.json', dict(error=repr(error), phase=run.progress))
        raise
    finally:
        run.log_resources();run.stop.set();run.thread.join(timeout=3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run', 'verify'))
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--smoke-root', type=Path, default=SMOKE)
    parser.add_argument('--max-hours', type=float, default=3.)
    main(parser.parse_args())
