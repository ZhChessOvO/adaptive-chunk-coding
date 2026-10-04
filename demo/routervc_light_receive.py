"""Source-free qE=2 teacher receiver; recompute EG, authenticate old G reuse."""
import argparse
import gc
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0,str(REPO))
from demo.routervc_light_teacher import ADAPTER, ENHANCEMENT, read, digest, save, immutable, verify_artifacts, crop
from demo.routervc_light_packets import bank_info, subset_bank
from demo.routervc_mixedview_teacher import isolate


def receive(root, *, stop_after=0, verify_only=False):
    import numpy as np
    import torch
    from demo.four_state_receive import PersistentRGB
    from demo.routervc_mixedview_teacher_receive import receive_e
    from demo.chunk_enhancement_codec import configure_torch
    from demo.scalable_codec import atomic_bytes, atomic_npz
    from demo.scalable_format import frame_hash, parse
    from demo.scalable_experiment import check_space
    from demo.online_eg_eval_core import noise_pair
    from demo import scalable_cooperation_format as fmt
    configure_torch(); index = read(root/'receiver_index.json')
    if index['enhancement'] != digest(ENHANCEMENT) or index['adapter'] != digest(ADAPTER):
        raise ValueError('frozen models changed')
    generator = None; new = 0; all_samples = {}
    for row in index['samples']:
        sid = row['sample_id']; dest = root/'received'/sid; dest.mkdir(parents=True,exist_ok=True)
        if digest(row['bank_path']) != row['bank_sha256']: raise ValueError('q2 bank changed')
        bank = Path(row['bank_path']).read_bytes(); info = bank_info(bank)
        if info['rois'] != row['rois']: raise ValueError('receiver grid mismatch')
        done = dest/'E.json'
        if done.exists():
            er = read(done); verify_artifacts(dest,er['artifacts'])
            if (er['bank_sha256'] != row['bank_sha256'] or er['source_frames_read']
                    or er['sender_candidate_pixels_read']): raise ValueError('receiver cache mismatch')
        else:
            if verify_only: raise ValueError('read-only replay cannot decode E')
            base, enhanced, er = receive_e(row)
            atomic_npz(dest/'received_E.npz',base=base,enhanced=enhanced)
            er['artifacts'] = {'received_E.npz':digest(dest/'received_E.npz')}; save(done,er)
        with np.load(dest/'received_E.npz',allow_pickle=False) as data:
            base, enhanced = data['base'].copy(), data['enhanced'].copy()
        if (list(base.shape) != row['shape'] or enhanced.shape != base.shape
                or frame_hash(base) != row['expected_base_hash'] or frame_hash(enhanced) != row['expected_E_hash']):
            raise ValueError('decoded pixels differ from bank')
        cells = []
        for i, roi in enumerate(row['rois']):
            check_space(); old_path = Path(row['cells'][i]['path'])
            if digest(old_path) != row['cells'][i]['sha256']: raise ValueError('old G record changed')
            old = read(old_path); verify_artifacts(old_path.parent,old['artifacts'])
            settings, _, old_parsed, _ = fmt.parse((old_path.parent/'G.acsg').read_bytes())
            # Old G is source-free, has zero E packets, identical base and G controls.
            if (settings != old['control'] or old_parsed.packets or old_parsed.base != info['parsed'].base
                    or old['reports']['G']['source_frames_read']
                    or old['reports']['G']['generation_input_hash'] != frame_hash(base)):
                raise ValueError('old G is not an identical counterfactual')
            folder = dest/f'cell_{i:02d}'; folder.mkdir(exist_ok=True); path = folder/'result.json'
            binding = dict(bank_sha256=row['bank_sha256'],encoded_manifest=row['encoded_manifest'],
                           old_G_record=row['cells'][i]['sha256'],control=settings)
            if path.exists():
                cell = read(path); verify_artifacts(folder,cell['artifacts'])
                if any(cell[k] != v for k,v in binding.items()): raise ValueError('q2 cell binding changed')
            else:
                if verify_only: raise ValueError('read-only replay cannot generate EG')
                if generator is None: generator = PersistentRGB()
                condition = isolate(base,enhanced,roi)
                wire = fmt.wrap(subset_bank(bank,[i]),settings); atomic_bytes(folder/'EG.acsg',wire)
                began = time.monotonic(); output,runtime = generator(condition,settings)
                mask = fmt.weights(base.shape,settings)
                np.testing.assert_array_equal(output[mask == 0],condition[mask == 0])
                report = dict(source_frames_read=False,outside_generate_exact=True,
                    generation_input_hash=frame_hash(condition),output_hash=frame_hash(output),
                    runtime=runtime,total_bytes=len(wire),seconds=time.monotonic()-began,
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
                noise_pair(dict(generation_runtime=old['reports']['G']['runtime']),dict(generation_runtime=runtime))
                atomic_npz(folder/'outputs.npz',EG=crop(output,roi))
                cell = dict(**binding,region=i,roi=roi,report=report,reused_G_pixels=True,
                    artifacts={n:digest(folder/n) for n in ('EG.acsg','outputs.npz')})
                save(path,cell); new += 1
                del output,condition,mask
            cells.append(digest(path))
            print(f'LIGHT_RECEIVE {sid} {i+1}/16 new_EG={new}',flush=True)
            if stop_after and new >= stop_after:
                if torch.distributed.is_initialized(): torch.distributed.destroy_process_group()
                return
        immutable(dest/'complete.json',dict(complete=True,bank_sha256=row['bank_sha256'],
                  region_results=cells,e_report=digest(done),source_frames_read=False))
        all_samples[sid] = digest(dest/'complete.json')
        del base,enhanced,bank; gc.collect(); torch.cuda.empty_cache()
    immutable(root/'received/complete.json',dict(complete=True,samples=all_samples,source_frames_read=False))
    if torch.distributed.is_initialized(): torch.distributed.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True); p.add_argument('--stop-after',type=int,default=0)
    p.add_argument('--verify-only',action='store_true'); args = p.parse_args()
    if not os.environ.get('TMUX') or not os.environ.get('ROUTERVC_LIGHT_PARENT') or args.stop_after < 0:
        raise RuntimeError('supervised tmux worker required')
    receive(args.root,stop_after=args.stop_after,verify_only=args.verify_only)
