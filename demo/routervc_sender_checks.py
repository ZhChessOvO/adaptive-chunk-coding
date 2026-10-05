"""Actual final-teacher/fresh receiver and conditional sender-prefix smoke.

Only the queue owns the GPU mutex. No source array or sender asset is passed to
fresh receiver workers. Synthetic CPU contracts cannot replace this real smoke.
"""
from pathlib import Path

import numpy as np

from demo.routervc_fullview_probe import read, digest, save, immutable, verify_artifacts
from demo.conditioned_generation_pipeline import execute
from demo.scalable_format import frame_hash


def compare_teacher(expected, report, teacher_pixels, fresh_pixels):
    """Exact RGB/bytes/route checks, never approximate metric-only agreement."""
    if (report['source_frames_read'] or report['sender_router_loaded']
            or report['unreceived_candidates_read'] or not report['outside_generate_exact']
            or report['total_bytes'] != expected['total_bytes']
            or report['generation_control_bytes'] != expected['header_bytes']
            or report['packet_bytes'] != expected['E_packet_bytes']
            or report['generation_input_hash'] != expected['generation_input_hash']
            or report['output_hash'] != expected['output_hash']
            or report['route']['indices'] != expected['route']['indices']
            or bool(report['generation_executed']) != expected['generation_executed']
            or any(report[k] != 0 for k in ('explicit_E_mask_bytes', 'explicit_G_map_bytes', 'protection_mask_bytes'))):
        raise ValueError('final sender teacher differs from source-free fresh receive')
    for name in ('reconstruction', 'enhanced'):
        np.testing.assert_array_equal(teacher_pixels[name], fresh_pixels[name])
    if frame_hash(fresh_pixels['base']) != expected['base_hash']:
        raise ValueError('fresh sender teacher base differs')
    if report['generation_executed']:
        from demo.online_eg_eval_core import noise_pair
        noise_pair(report, expected, same_condition=True)


def fresh_checks(run, root, protocol, model_root):
    """Check two states and one measured addition each, in REDS and UVG."""
    teacher = protocol['teacher']
    records, generation_cases = {}, 0
    for row in protocol['rows']:
        sid = row['sample_id']
        sample = root/'samples'/sid
        for state in (0, 1):
            state_root = sample/f'state{state}'
            additions = sorted(p.name for p in state_root.glob('add_*') if (p/'result.json').exists())
            if len(additions) != 3:
                raise ValueError('smoke requires all three measured children of each state')
            for name in ('parent', additions[0]):
                source = state_root/name
                expected = read(source/'result.json')
                if not expected['complete'] or not expected['pixels_retained']:
                    raise ValueError('real sender smoke needs saved teacher RGB')
                verify_artifacts(source, expected['artifacts'])
                out = root/'fresh_teacher'/sid/f'state{state}'/name
                out.mkdir(parents=True, exist_ok=True)
                checked = out/'checked.json'
                if checked.exists():
                    result = read(checked)
                    if result['teacher'] != digest(source/'result.json'):
                        raise ValueError('saved sender teacher binding changed')
                    verify_artifacts(out, result['artifacts'])
                else:
                    if not (out/'decode.json').exists():
                        execute(run, f'sender_teacher_{sid}_{state}_{name}', 'routervc_receiver_decode.py',
                            ['--worker', '--stream', source/'stream.rvrc', '--output', out,
                             '--enhancement', teacher['enhancement']['path'],
                             '--adapter', teacher['adapter']['path'],
                             '--router', teacher['receiver']['path']], distributed=True)
                    report = read(out/'decode.json')
                    with np.load(source/'reconstruction.npz', allow_pickle=False) as a, np.load(
                            out/'reconstruction.npz', allow_pickle=False) as b:
                        compare_teacher(expected, report, a, b)
                    result = dict(complete=True, teacher=digest(source/'result.json'),
                        source_free=True, actual_G=expected['generation_executed'],
                        pixels_exact=True, final_teacher_real_decode=True,
                        measured_bytes=report['total_bytes'], header_bytes=report['generation_control_bytes'],
                        masks_bytes=0, worker_peak_cuda_bytes=report['peak_cuda_allocated_bytes'],
                        artifacts={n:digest(out/n) for n in ('decode.json', 'reconstruction.npz')})
                    save(checked, result)
                generation_cases += int(result['actual_G'])
                records[str(checked.relative_to(root))] = digest(checked)
    if generation_cases == 0:
        raise ValueError('sender real teacher smoke must exercise actual G')
    records.update(sender_prefix_checks(run, root, protocol, model_root))
    immutable(root/'fresh_audit.json', dict(complete=True, checks=records,
        teacher_fresh_checks=8, actual_teacher_G_cases=generation_cases,
        final_labels_source_free_exact=True, formal_sender_training=False,
        whole_video_RD_claim=False))


def sender_prefix_checks(run, root, protocol, model_root):
    """One conditional order, four literal prefixes and fresh on/off per arm."""
    from demo import routervc_sender_encode as encoder
    from demo import routervc_receiver_format as fmt
    from demo import routervc_receiver_policy as policy
    from demo import scalable_cooperation_format as cooperation
    from demo.routervc_light_packets import bank_info, subset_bank
    from demo.routervc_encode import compose_candidates
    from demo.scalable_codec import atomic_bytes
    teacher, records = protocol['teacher'], {}
    config = teacher['wire_config']
    for row in protocol['rows']:
        sid = row['sample_id']
        if digest(row['source_path']) != row['source_sha256']:
            raise ValueError('sender smoke source changed')
        with np.load(row['source_path'], allow_pickle=False) as data:
            source = data['source'].copy()
        with np.load(row['reconstruction_path'], allow_pickle=False) as data:
            base, all_e = data['base'].copy(), data['enhanced'].copy()
        bank = Path(row['bank_path']).read_bytes()
        info = bank_info(bank)
        candidates = encoder.authenticate_candidates(bank, source, base, all_e,
            receive_record=Path(row['receive_record']),
            expected_receive_sha256=row['receive_record_sha256'],
            expected_bank_sha256=row['bank_sha256'],
            expected_source_rgb_sha256=frame_hash(source))
        for arm in ('source', 'zero_source'):
            out = root/'fresh_sender'/sid/arm
            out.mkdir(parents=True, exist_ok=True)
            checked = out/'checked.json'
            model = model_root/arm/'best.pt'
            binding = dict(model=digest(model), protocol=digest(root/'protocol.json'), sample=row)
            if checked.exists():
                result = read(checked)
                if result['binding'] != binding:
                    raise ValueError('saved sender prefix smoke changed')
                verify_artifacts(out, result['artifacts'])
                records[str(checked.relative_to(root))] = digest(checked)
                continue
            context = encoder.load_sender(model, config, expected_sha256=digest(model),
                completion_record=model_root/'complete.json',
                expected_completion_sha256=digest(model_root/'complete.json'), allow_smoke=True)
            if (out/'plan.json').exists():
                plan = read(out/'plan.json')
                if plan['sender']['sha256'] != digest(model) or plan['config'] != config:
                    raise ValueError('saved sender order belongs to another model or receiver')
            else:
                plan = encoder.conditional_order(candidates, context)
                immutable(out/'plan.json', plan)
            total = sum(info['e_bytes'])
            budgets = (0, total//4, total//2, total)
            wires, ledgers = [], []
            for number, budget in enumerate(budgets):
                wire, ledger = encoder.encode_prefix(candidates, plan, budget)
                parsed_config, inner_bytes, _, header = fmt.parse(wire)
                expected_bytes = sum(info['e_bytes'][i] for i in ledger['indices'])
                if (parsed_config != config or inner_bytes != subset_bank(bank, ledger['indices'])
                        or ledger['packet_bytes'] != expected_bytes
                        or ledger['total_bytes'] != len(wire) or ledger['header_bytes'] != header
                        or ledger['indices'] != plan['order'][:len(ledger['indices'])]
                        or expected_bytes > budget
                        or any(ledger[k] for k in ('explicit_E_mask_bytes', 'explicit_G_map_bytes', 'protection_mask_bytes'))):
                    raise ValueError('conditional sender changed true bytes or literal prefix')
                if wires and not wire.startswith(wires[-1]):
                    raise ValueError('larger sender budget is not a literal old-stream prefix')
                atomic_bytes(out/f'prefix{number}.rvrc', wire)
                if (out/f'prefix{number}.rvrc').stat().st_size != ledger['total_bytes']:
                    raise ValueError('sender ledger differs from actual file size')
                wires.append(wire)
                ledgers.append(ledger)
            save(out/'ledgers.json', ledgers)
            # Full predicted positive-benefit prefix can legitimately be empty.
            expected_y = compose_candidates(base, all_e, ledgers[-1]['indices'], info['rois'])
            stream = out/'prefix3.rvrc'
            artifacts = ['plan.json', 'ledgers.json', *(f'prefix{i}.rvrc' for i in range(4))]
            for off in (False, True):
                fresh = out/('G_off' if off else 'G_on')
                fresh.mkdir(exist_ok=True)
                if not (fresh/'decode.json').exists():
                    argv = ['--worker', '--stream', stream, '--output', fresh,
                        '--enhancement', teacher['enhancement']['path'] if ledgers[-1]['indices'] else out/'absent_E.pt',
                        '--adapter', out/'absent_G.pt' if off else teacher['adapter']['path'],
                        '--router', out/'absent_Rg.pt' if off else teacher['receiver']['path']]
                    if off:
                        argv.append('--disable-generation')
                    execute(run, f'sender_prefix_{sid}_{arm}_{int(off)}', 'routervc_receiver_decode.py', argv,
                            distributed=True)
                report = read(fresh/'decode.json')
                _, _, inner, header = fmt.parse(stream.read_bytes())
                expected_route = [] if off else policy.route(base, expected_y, inner, config,
                    teacher['receiver']['path'], expected_policy=fmt.policy_identity())['indices']
                if (report['source_frames_read'] or report['sender_router_loaded']
                        or report['unreceived_candidates_read'] or not report['outside_generate_exact']
                        or report['generation_input_hash'] != frame_hash(expected_y)
                        or report['total_bytes'] != stream.stat().st_size
                        or report['generation_control_bytes'] != header
                        or report['route']['indices'] != expected_route
                        or any(report[k] for k in ('explicit_E_mask_bytes', 'explicit_G_map_bytes', 'protection_mask_bytes'))):
                    raise ValueError('fresh receiver differs from conditional sender actual Y')
                if off and (report['receiver_router_used'] or report['generation_assets_validated']
                            or report['generation_executed']):
                    raise ValueError('sender prefix G-off unexpectedly needs G/R_g assets')
                with np.load(fresh/'reconstruction.npz', allow_pickle=False) as data:
                    np.testing.assert_array_equal(data['base'], base)
                    np.testing.assert_array_equal(data['enhanced'], expected_y)
                    alpha = cooperation.weights(base.shape, fmt.generation_control(config,
                        expected_route, info['rois'], len(base)))
                    np.testing.assert_array_equal(data['reconstruction'][alpha == 0], expected_y[alpha == 0])
                    if off:
                        np.testing.assert_array_equal(data['reconstruction'], expected_y)
                artifacts.extend(str((fresh/n).relative_to(out)) for n in ('decode.json', 'reconstruction.npz'))
            result = dict(complete=True, binding=binding, order=plan['order'],
                literal_prefix_pairs=3, source_and_sender_assets_not_transmitted=True,
                candidate_preparation_scope='authenticated old full-E bank reuse, NOT fresh encode timing',
                masks_bytes=0, G_off_without_Rg_G_assets=True,
                artifacts={name:digest(out/name) for name in artifacts})
            save(checked, result)
            records[str(checked.relative_to(root))] = digest(checked)
    return records
