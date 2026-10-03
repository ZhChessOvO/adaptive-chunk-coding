"""Source-free, dynamic-ROI mixed-view receiver with one resident generator.

The only image inputs are independently entropy-decoded bank.acse packets and
our own verified receive cache. Sender RGB candidates/source files are never
opened. The supervising parent, not this worker, owns the global GPU mutex.
"""
import argparse
import gc
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from demo.routervc_mixedview_teacher import (
    ADAPTER, ENHANCEMENT, control, crop, digest, grid, immutable, isolate, read, save, verify,
)


def receive_e(row):
    import torch
    from demo.four_state_receive import codec_precision
    from demo.chunk_enhancement_codec import decode_enhancement, load_model
    from demo.chunk_enhancement_experiment import codec
    from demo.scalable_format import frame_hash
    if digest(Path(row['bank_path'])) != row['bank_sha256']:
        raise ValueError('received E bank bytes changed')
    began = time.monotonic()
    with codec_precision(), torch.no_grad():
        model, base_codec = load_model(ENHANCEMENT), codec()
        enhanced, detail, base = decode_enhancement(model, ENHANCEMENT, base_codec,
                                                   Path(row['bank_path']).read_bytes(), return_base=True)
        torch.cuda.synchronize()
        if (frame_hash(base) != row['expected_base_hash']
                or frame_hash(enhanced) != row['expected_E_hash']):
            raise ValueError('independent entropy decoder differs from sender expectation')
        del model, base_codec
    gc.collect(); torch.cuda.empty_cache()
    return base, enhanced, dict(complete=True, source_frames_read=False,
        sender_candidate_pixels_read=False, bank_sha256=row['bank_sha256'],
        base_hash=frame_hash(base), enhanced_hash=frame_hash(enhanced),
        seconds=time.monotonic()-began, detail=detail, base_reference_unchanged=True)


def receive(root, *, stop_after=0, verify_only=False):
    import numpy as np
    import torch
    from demo.four_state_receive import PersistentRGB
    from demo.chunk_enhancement_codec import configure_torch
    from demo.routervc_encode import subset_bank
    from demo.online_eg_eval_core import noise_pair
    from demo.scalable_codec import atomic_bytes, atomic_npz
    from demo.scalable_experiment import check_space
    from demo.scalable_format import frame_hash
    from demo import scalable_cooperation_format as fmt
    configure_torch()
    index = read(root/'receiver_index.json')
    if index['enhancement'] != digest(ENHANCEMENT) or index['adapter'] != digest(ADAPTER):
        raise ValueError('receiver E/G model identity changed')
    generator = None; new_regions = 0; completed = {}
    for row in index['samples']:
        if digest(Path(row['bank_path'])) != row['bank_sha256']:
            raise ValueError('receiver bank changed')
        sid = row['sample_id']; dest = root/'received'/sid; dest.mkdir(parents=True, exist_ok=True)
        bank = Path(row['bank_path']).read_bytes()
        e_done = dest/'E.json'
        if e_done.exists():
            er = read(e_done); verify(dest, er['artifacts'])
            if (er['bank_sha256'] != row['bank_sha256'] or er['source_frames_read']
                    or er['sender_candidate_pixels_read']):
                raise ValueError('received E cache provenance changed')
        else:
            if verify_only:
                raise ValueError('missing E cache; read-only replay may not decode')
            base, enhanced, er = receive_e(row)
            atomic_npz(dest/'received_E.npz', base=base, enhanced=enhanced)
            er['artifacts'] = {'received_E.npz':digest(dest/'received_E.npz')}; save(e_done, er)
        with np.load(dest/'received_E.npz', allow_pickle=False) as cache:
            base, enhanced = cache['base'].copy(), cache['enhanced'].copy()
        if (list(base.shape) != row['shape'] or base.shape != enhanced.shape
                or grid(base.shape[1], base.shape[2]) != row['rois']
                or frame_hash(base) != row['expected_base_hash']
                or frame_hash(enhanced) != row['expected_E_hash']):
            raise ValueError('received pixels/grid differ from byte-bound manifest')
        region_results = []
        for i, roi in enumerate(row['rois']):
            check_space(); folder = dest/f'cell_{i:02d}'; folder.mkdir(exist_ok=True)
            settings = control(index['profile'], sid, i, row['rois']); path = folder/'result.json'
            if path.exists():
                result = read(path); verify(folder, result['artifacts'])
                if (result['control'] != settings or result['bank_sha256'] != row['bank_sha256']
                        or result['encoded_manifest'] != row['encoded_manifest']):
                    raise ValueError('completed teacher region binding changed')
            else:
                if verify_only:
                    raise ValueError('missing cell; read-only replay may not generate')
                if generator is None:
                    generator = PersistentRGB()
                states, reports = {}, {}
                for state in ('G', 'EG'):
                    condition = base if state == 'G' else isolate(base, enhanced, roi)
                    inner = subset_bank(bank, [] if state == 'G' else [i])
                    wire = fmt.wrap(inner, settings); atomic_bytes(folder/f'{state}.acsg', wire)
                    began = time.monotonic(); output, runtime = generator(condition, settings)
                    mask = fmt.weights(base.shape, settings)
                    np.testing.assert_array_equal(output[mask == 0], condition[mask == 0])
                    states[state] = crop(output, roi)
                    reports[state] = dict(source_frames_read=False, outside_generate_exact=True,
                        generation_input_hash=frame_hash(condition), output_hash=frame_hash(output),
                        runtime=runtime, total_bytes=len(wire), seconds=time.monotonic()-began,
                        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
                    del output, condition, mask
                noise_pair(dict(generation_runtime=reports['G']['runtime']),
                           dict(generation_runtime=reports['EG']['runtime']))
                atomic_npz(folder/'outputs.npz', **states)
                result = dict(region=i, roi=roi, control=settings, reports=reports,
                    bank_sha256=row['bank_sha256'], encoded_manifest=row['encoded_manifest'],
                    artifacts={n:digest(folder/n) for n in ('G.acsg', 'EG.acsg', 'outputs.npz')})
                save(path, result); new_regions += 1
            region_results.append(digest(path))
            print(f'RECEIVE {sid} {i+1}/16 new_regions={new_regions}', flush=True)
            if stop_after and new_regions >= stop_after:
                if torch.distributed.is_initialized():
                    torch.distributed.destroy_process_group()
                return
        immutable(dest/'complete.json', dict(complete=True, source_frames_read=False,
                  bank_sha256=row['bank_sha256'], region_results=region_results, e_report=digest(e_done)))
        completed[sid] = digest(dest/'complete.json')
        del base, enhanced, bank; gc.collect(); torch.cuda.empty_cache()
    immutable(root/'received/complete.json', dict(complete=True, samples=completed,
              source_frames_read=False, semantic_labels_available=False))
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--stop-after', type=int, default=0)
    p.add_argument('--verify-only', action='store_true')
    args = p.parse_args()
    if not os.environ.get('TMUX') or not os.environ.get('ROUTERVC_MIXEDVIEW_PARENT'):
        raise RuntimeError('receiver must run under the supervising tmux queue')
    if args.stop_after < 0:
        raise ValueError('stop-after must be nonnegative')
    receive(args.root, stop_after=args.stop_after, verify_only=args.verify_only)
