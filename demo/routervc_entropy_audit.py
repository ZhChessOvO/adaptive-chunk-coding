"""Audit existing rANS streams, without source pixels, training, or re-encoding E.

Fresh native decoding verifies reconstruction; a symbol-only re-encode verifies
the actual entropy bytes. Full candidate banks and selected Router streams are
reported separately. The old codecs, bitstreams and model weights stay intact.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import os
from pathlib import Path
import struct
import sys
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from demo.routervc_fullview_probe import read, digest, save, immutable, verify_artifacts

ORIGINAL = Path('/root/autodl-fs/DCVC/runs/routervc_revision_20261003/visual_evaluation_recovered')
OUTPUT = Path('/root/autodl-fs/DCVC/runs/routervc_efficiency_20261004/entropy_audit')
SCHEMA = 'routervc-current-entropy-audit-v1'
SUM_FIELDS = ('actual_bytes', 'symbols', 'nonzero', 'escaped', 'scales_above_16',
              'gaussian_proxy_bits', 'native_cdf_bits', 'bypass_bits',
              'coder_framing_rounding_bits')


def totals(records):
    value = {k: sum(r[k] for r in records) for k in SUM_FIELDS}
    value['streams'] = len(records)
    # One byte per already quantized signed-8-bit symbol, NOT raw video/float32.
    value['fixed_int8_reference_bytes'] = value['symbols']
    value['entropy_to_fixed_int8_ratio'] = value['actual_bytes'] / max(1, value['symbols'])
    value['coder_overhead_fraction'] = value['coder_framing_rounding_bits'] / max(1, 8*value['actual_bytes'])
    value['actual_minus_training_proxy_bits'] = 8*value['actual_bytes']-value['gaussian_proxy_bits']
    return value


def packet_key(packet):
    m = packet.meta
    # A subset may renumber packet IDs; geometry and payload must remain exact.
    return (m['start'], m['count'], tuple(m['roi']), m['qstep'],
            hashlib.sha256(packet.payload).hexdigest())


def packet_costs(parsed, records):
    if len(records) != 2*len(parsed.packets):
        raise ValueError('expected exactly one z and one y stream per E packet')
    rows = []
    for i, packet in enumerate(parsed.packets):
        z, y = records[2*i:2*i+2]
        nz = struct.unpack_from('<I', packet.payload)[0]
        if (z['kind'], y['kind']) != ('z', 'y') or z['actual_bytes'] != nz:
            raise ValueError('z/y stream ordering or length mismatch')
        if z['actual_bytes']+y['actual_bytes']+4 != len(packet.payload):
            raise ValueError('entropy payload bytes do not close')
        rows.append(dict(meta=packet.meta, payload_sha256=packet_key(packet)[-1],
            z=z, y=y, payload_length_bytes=4,
            outer_header_bytes=len(packet.wire)-len(packet.payload),
            packet_bytes=len(packet.wire)))
    return rows


def cost_summary(parsed, rows, outer_bytes=0):
    streams = [r[k] for r in rows for k in ('z', 'y')]
    entropy = totals(streams)
    costs = dict(base_native_bytes=len(parsed.base),
        acse_header_bytes=parsed.base_end-len(parsed.base), router_header_bytes=outer_bytes,
        packet_header_bytes=sum(r['outer_header_bytes'] for r in rows),
        payload_length_bytes=sum(r['payload_length_bytes'] for r in rows),
        entropy_bytes=entropy['actual_bytes'])
    costs['total_bytes'] = sum(costs.values())
    if costs['total_bytes'] != outer_bytes+parsed.base_end+sum(len(p.wire) for p in parsed.packets):
        raise ValueError('serialized bytes do not close')
    return dict(bytes=costs, entropy=entropy, z=totals([r['z'] for r in rows]),
                y=totals([r['y'] for r in rows]), packets=len(rows))


def selected_costs(parsed, bank_rows, outer_bytes):
    lookup = {(r['meta']['start'], r['meta']['count'], tuple(r['meta']['roi']),
               r['meta']['qstep'], r['payload_sha256']): r for r in bank_rows}
    if len(lookup) != len(bank_rows):
        raise ValueError('ambiguous bank packet')
    rows = [lookup[packet_key(p)] for p in parsed.packets]
    return cost_summary(parsed, rows, outer_bytes)


def worker(args):
    import torch
    from demo.chunk_enhancement_codec import configure_torch, decode_enhancement, load_model
    from demo.chunk_enhancement_experiment import codec
    from demo.patch_entropy_audit import AuditStreams
    from demo.scalable_format import parse, frame_hash

    class VerifiedStreams(AuditStreams):
        def decode(self, data, scales):
            symbols = super().decode(data, scales)
            if self.encode(symbols, scales) != data:
                raise ValueError('native rANS symbol re-encode differs from original bytes')
            self.records[-1]['symbol_reencode_byte_exact'] = True
            return symbols

    started = time.monotonic()
    old = read(args.prepared/'complete.json')
    bank = args.prepared/'bank.acse'
    if not old['complete'] or digest(bank) != old['artifacts']['bank.acse']:
        raise ValueError('incomplete/changed prepared bank')
    if digest(args.enhancement) != old['binding']['enhancement']:
        raise ValueError('enhancement checkpoint changed')
    configure_torch()
    torch.cuda.reset_peak_memory_stats()
    model, base_codec = load_model(args.enhancement), codec()
    model.entropy = VerifiedStreams()
    data = bank.read_bytes()
    with torch.inference_mode():
        output, report, base = decode_enhancement(model, args.enhancement, base_codec, data, return_base=True)
    torch.cuda.synchronize()
    if frame_hash(output) != old['all_E_rgb_sha256'] or frame_hash(base) != old['base_rgb_sha256']:
        raise ValueError('instrumented fresh reconstruction changed')
    parsed = parse(data)
    rows = packet_costs(parsed, model.entropy.records)
    save(args.output, dict(complete=True, sample_id=args.prepared.parent.name,
        prepared_complete_sha256=digest(args.prepared/'complete.json'), bank_sha256=digest(bank),
        checkpoint_sha256=digest(args.enhancement), reconstruction_exact=True,
        source_pixels_read=False, fresh_decode=report, packet_records=rows,
        costs=cost_summary(parsed, rows), seconds=time.monotonic()-started,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
        timing_scope='instrumented audit including symbol re-encode, not normal decoder latency'))
    print(f'ENTROPY_SAMPLE_COMPLETE {args.prepared.parent.name}', flush=True)


def binding(root):
    from demo.routervc_visual_evaluate import code_hashes
    old = read(root/'protocol.json')
    if not read(root/'complete.json')['complete'] or old['code'] != code_hashes():
        raise ValueError('completed evaluation or original source pin mismatch')
    files = ['protocol.json', 'complete.json', 'summary.json']
    for entry in old['sources']:
        folder = Path('samples')/entry['sample']['sample_id']
        files += [str(folder/'prepared'/n) for n in ('complete.json', 'bank.acse')]
        for arm in ('global_local', 'local'):
            files += [str(folder/f'{arm}_e{ratio:g}_g8'/'stream.rtvc') for ratio in (.25, .5)]
    weights = old['enhancement']
    if digest(weights['path']) != weights['sha256']:
        raise ValueError('bound enhancement checkpoint changed')
    return dict(schema=SCHEMA, original_root=str(root.resolve()),
        source_artifacts={n: digest(root/n) for n in files}, enhancement=weights,
        sources=old['sources'], original_code=old['code'],
        audit_code={n: digest(REPO/n) for n in
            ('demo/routervc_entropy_audit.py', 'demo/patch_entropy_audit.py',
             'demo/run_routervc_entropy_audit.sh')},
        scope='same 13 existing windows; banks and selected G8 E25/E50 subsets; no new RD points')


def validate_record(root, sid, original, enhancement):
    result = read(root/'samples'/f'{sid}.json')
    prepared = original/'samples'/sid/'prepared'
    if (not result['complete'] or not result['reconstruction_exact']
            or result['source_pixels_read']
            or result['prepared_complete_sha256'] != digest(prepared/'complete.json')
            or result['bank_sha256'] != digest(prepared/'bank.acse')
            or result['checkpoint_sha256'] != enhancement['sha256']):
        raise ValueError('entropy audit resume binding changed')
    return result


def finish(root, original, protocol, records):
    from demo.routervc_visual_format import parse
    selected = []
    for row, entry in zip(records, protocol['sources']):
        sid = row['sample_id']
        for arm in ('global_local', 'local'):
            for ratio in (.25, .5):
                name = f'{arm}_e{ratio:g}_g8'
                path = original/'samples'/sid/name/'stream.rtvc'
                _, _, inner, outer = parse(path.read_bytes())
                cost = selected_costs(inner, row['packet_records'], outer)
                if cost['bytes']['total_bytes'] != path.stat().st_size:
                    raise ValueError('selected stream file-size mismatch')
                selected.append(dict(sample_id=sid, dataset=entry['sample']['dataset'],
                    point=name, stream_sha256=digest(path), **cost))
    groups = {}
    for dataset in ('REDS', 'UVG'):
        groups[dataset] = {}
        for point in dict.fromkeys(r['point'] for r in selected):
            rows = [r for r in selected if r['dataset'] == dataset and r['point'] == point]
            entropy = totals([r['entropy'] for r in rows])
            # entropy.streams should count coded streams, not selected windows.
            entropy['streams'] = sum(r['entropy']['streams'] for r in rows)
            groups[dataset][point] = dict(windows=len(rows), entropy=entropy,
                total_bytes=sum(r['bytes']['total_bytes'] for r in rows),
                mean_bytes={k:sum(r['bytes'][k] for r in rows)/len(rows) for k in rows[0]['bytes']})
    save(root/'summary.json', dict(complete=True, schema=SCHEMA, windows=len(records),
        fresh_decoded_packets=sum(len(r['packet_records']) for r in records),
        exact_reencoded_entropy_streams=sum(r['costs']['entropy']['streams'] for r in records),
        source_pixels_read=False, old_files_unchanged=True, group_sums=groups, selected=selected,
        fixed_reference='one byte per existing quantized signed-8-bit symbol; not raw RGB',
        interpretation='CDF cost is cost under current predicted probabilities, not a lower bound on achievable bitrate',
        measured_decoding_seconds=sum(r['seconds'] for r in records),
        peak_cuda_allocated_bytes=max(r['peak_cuda_allocated_bytes'] for r in records)))
    files = ['protocol.json', 'summary.json']+[f"samples/{r['sample_id']}.json" for r in records]
    save(root/'complete.json', dict(complete=True, artifacts={n:digest(root/n) for n in files}))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('run', 'worker'))
    p.add_argument('--root', type=Path, default=ORIGINAL)
    p.add_argument('--output', type=Path, default=OUTPUT)
    p.add_argument('--max-hours', type=float, default=4.)
    p.add_argument('--prepared', type=Path)
    p.add_argument('--enhancement', type=Path)
    args = p.parse_args(argv)
    if args.command == 'worker':
        return worker(args)
    if not os.environ.get('TMUX'):
        raise RuntimeError('run the audit inside tmux')
    if args.output.resolve().is_relative_to(args.root.resolve()):
        raise ValueError('audit output must not modify original evaluation')
    from demo.chunk_enhancement_experiment import Run
    from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
    from demo.conditioned_generation_pipeline import execute
    run = Run(SimpleNamespace(output=args.output, command='entropy_audit', max_hours=args.max_hours))
    run.thread.start()
    try:
        protocol = binding(args.root)
        immutable(run.root/'protocol.json', protocol)
        complete = run.root/'complete.json'
        if complete.exists():
            verify_artifacts(run.root, read(complete)['artifacts'])
            # Deliberately no GPU lock, worker, scores, or timing writes on replay.
            for e in protocol['sources']:
                validate_record(run.root, e['sample']['sample_id'], args.root, protocol['enhancement'])
            print('ENTROPY_AUDIT_VERIFIED_READ_ONLY', flush=True)
            return
        rows = []
        with exclusive_native_evaluation(run):
            for index, entry in enumerate(protocol['sources']):
                run.check()
                sid = entry['sample']['sample_id']
                target = run.root/'samples'/f'{sid}.json'
                if not target.exists():
                    execute(run, f'audit_{sid}', 'routervc_entropy_audit.py',
                        ['worker', '--prepared', args.root/'samples'/sid/'prepared',
                         '--output', target, '--enhancement', protocol['enhancement']['path']])
                rows.append(validate_record(run.root, sid, args.root, protocol['enhancement']))
                run.update(completed=index+1, total=len(protocol['sources']), sample=sid)
        verify_artifacts(args.root, protocol['source_artifacts'])
        finish(run.root, args.root, protocol, rows)
        run.update(phase='complete', completed=len(rows))
        print('ENTROPY_AUDIT_COMPLETE', flush=True)
    except BaseException as error:
        save(run.root/'last_failure.json', dict(error=repr(error), progress=run.progress))
        raise
    finally:
        run.log_resources(); run.stop.set(); run.thread.join(timeout=3); run.lock.close()


if __name__ == '__main__':
    main()
