"""Real-label and new receiver-profile checks owned by the tmux queue."""
from pathlib import Path

import numpy as np
import torch

from demo.routervc_fullview_probe import read, digest, save, immutable, verify_artifacts
from demo.conditioned_generation_pipeline import execute
from demo.routervc_receiver_data import ENHANCEMENT, ADAPTER, selection
from demo.routervc_encode import compose_candidates
from demo.routervc_light_packets import subset_bank
from demo.scalable_codec import atomic_bytes
from demo.scalable_format import frame_hash


def fresh_checks(run, root, protocol, model_root):
    """Teacher equivalence plus source-free real G/G-off/no-E new wire checks."""
    from demo import routervc_receiver_format as fmt
    from demo import routervc_receiver_policy as policy
    from demo.routervc_receiver_router import ARMS
    from demo.routervc_mixedview_teacher import crop
    from demo.online_eg_eval_core import noise_pair
    records = {}
    generated_cases = 0
    for row in protocol['rows']:
        with np.load(row['reconstruction_path'], allow_pickle=False) as data:
            base, enhanced = data['base'].copy(), data['enhanced'].copy()
        rois = [r['roi'] for r in read(row['path'])['regions']]
        bank = Path(row['bank_path']).read_bytes()
        # New sparse and dense Y labels must equal independent entropy decode
        # and fresh G, not only composition of cached candidate reconstructions.
        for count, region in ((2, 0), (12, 5)):
            cell = root/'samples'/row['sample_id']/f'e{count}/cell_{region:02d}'
            expected = read(cell/'result.json')
            out = root/'fresh_teacher'/row['sample_id']/f'e{count}_r{region}'
            out.mkdir(parents=True, exist_ok=True)
            checked = out/'checked.json'
            if checked.exists():
                verify_artifacts(out, read(checked)['artifacts'])
            else:
                if not (out/'decode.json').exists():
                    execute(run, f'teacher_{row["sample_id"]}_{count}', 'online_eg_decode.py',
                        ['--stream', cell/'teacher.acsg', '--output', out,
                         '--enhancement', ENHANCEMENT, '--adapter', ADAPTER], distributed=True)
                report = read(out/'decode.json')
                if (report['source_frames_read'] or not report['outside_generate_exact']
                        or report['total_bytes'] != expected['total_bytes']
                        or report['generation_input_hash'] != expected['generation_input_hash']
                        or report['output_hash'] != expected['output_hash']):
                    raise ValueError('new mixed teacher differs from fresh decoding')
                with np.load(out/'reconstruction.npz', allow_pickle=False) as data:
                    actual = data['reconstruction'].copy()
                with np.load(cell/'generated.npz', allow_pickle=False) as data:
                    np.testing.assert_array_equal(crop(actual, rois[region]), data['generated'])
                noise_pair(report, dict(generation_runtime=expected['runtime']), same_condition=True)
                save(checked, dict(complete=True, stream=digest(cell/'teacher.acsg'),
                    source_free=True, pixels_exact=True, artifacts={n:digest(out/n)
                        for n in ('decode.json','reconstruction.npz')}))
            records[str(checked.relative_to(root))] = digest(checked)

        # Actual G-only wire profiles, never an old shared-model identifier.
        cases = [(arm, 2, False) for arm in ARMS] + [('core', 0, False), ('core', 2, True), ('core', 0, True)]
        for arm, count, off in cases:
            name = f'{arm}_e{count}_off{int(off)}'
            out = root/'fresh_receiver'/row['sample_id']/name
            out.mkdir(parents=True, exist_ok=True)
            model = model_root/arm/'best.pt'
            config = fmt.make_config(model, ADAPTER, max_g=2, boundary_lambda=0., seed=20261003)
            wire = fmt.wrap(subset_bank(bank, selection(row['sample_id'], count)), config)
            stream = out/'stream.rvrc'
            if stream.exists():
                if stream.read_bytes() != wire:
                    raise ValueError('new receiver smoke stream changed')
            else:
                atomic_bytes(stream, wire)
            checked = out/'checked.json'
            if checked.exists():
                record = read(checked)
                if record['stream'] != digest(stream):
                    raise ValueError('saved receiver smoke stream differs')
                verify_artifacts(out, record['artifacts'])
            else:
                fresh = out/'fresh'
                fresh.mkdir(exist_ok=True)
                if not (fresh/'decode.json').exists():
                    argv = ['--worker', '--stream', stream, '--output', fresh,
                        '--enhancement', out/'absent_E.pt' if count == 0 else ENHANCEMENT,
                        '--adapter', out/'absent_G.pt' if off else ADAPTER,
                        '--router', out/'absent_Rg.pt' if off else model]
                    if off:
                        argv += ['--disable-generation']
                    execute(run, f'receiver_{row["sample_id"]}_{name}',
                            'routervc_receiver_decode.py', argv, distributed=True)
                report = read(fresh/'decode.json')
                parsed, _, inner, header = fmt.parse(wire)
                expected_y = compose_candidates(base, enhanced, selection(row['sample_id'], count), rois)
                if (report['source_frames_read'] or not report['outside_generate_exact']
                        or report['total_bytes'] != len(wire) or report['generation_control_bytes'] != header
                        or report['generation_input_hash'] != frame_hash(expected_y)
                        or any(report[k] != 0 for k in ('explicit_E_mask_bytes','explicit_G_map_bytes','protection_mask_bytes'))):
                    raise ValueError('new R_g receiver violated source-free/byte contract')
                if off:
                    if report['generation_assets_validated'] or report['generation_executed']:
                        raise ValueError('G-off unexpectedly loaded G assets')
                    expected_indices = []
                else:
                    predicted = policy.route(base, expected_y, inner, parsed, model,
                                             expected_policy=fmt.policy_identity())
                    expected_indices = predicted['indices']
                if report['route']['indices'] != expected_indices:
                    raise ValueError('fresh R_g decisions differ from independent B/Y prediction')
                with np.load(fresh/'reconstruction.npz', allow_pickle=False) as data:
                    actual = data['reconstruction'].copy()
                    np.testing.assert_array_equal(data['base'], base)
                    np.testing.assert_array_equal(data['enhanced'], expected_y)
                from demo import scalable_cooperation_format as cooperation
                alpha = cooperation.weights(base.shape, fmt.generation_control(parsed,expected_indices,rois,len(base)))
                np.testing.assert_array_equal(actual[alpha == 0], expected_y[alpha == 0])
                if off:
                    np.testing.assert_array_equal(actual, expected_y)
                record = dict(complete=True, stream=digest(stream), arm=arm, e_regions=count,
                    G_off=off, source_free=True, sender_assets_required=False,
                    selected_G=expected_indices, total_bytes=len(wire), actual_header_bytes=header,
                    explicit_masks_bytes=0, actual_Y_exact=True,
                    artifacts={n:digest(out/n) for n in ('stream.rvrc','fresh/decode.json','fresh/reconstruction.npz')})
                save(checked,record)
            generated_cases += int(bool(record['selected_G']))
            records[str(checked.relative_to(root))] = digest(checked)
    if generated_cases == 0:
        raise ValueError('receiver smoke must exercise actual selected G, not only empty routes')
    immutable(root/'fresh_audit.json', dict(complete=True, checks=records,
        new_sparse_dense_labels_fresh_exact=True, G_off_without_G_Rg_assets=True,
        no_E_without_E_assets=True, receiver_without_sender_assets=True,
        actual_G_cases=generated_cases, whole_video_RD_claim=False))
