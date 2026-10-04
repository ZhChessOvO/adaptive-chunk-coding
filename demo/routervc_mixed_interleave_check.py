"""Check that real Router updates do not perturb the resident frozen generator.

Run after the mixed smoke in tmux/torchrun. No source images are read, no formal
model is changed, and all temporary optimizers contain Router parameters only.
"""
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import time

REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))


def main():
    import numpy as np
    import torch
    from demo.routervc_mixed_train import ROOT,CACHE,INITIAL,RESUME,arm_batch
    from demo.routervc_mixed_data import Samples
    from demo.routervc_mixed_router import ARMS
    from demo import routervc_visual_router as visual
    from demo.routervc_fullview_probe import read,digest,save,immutable,verify_artifacts
    from demo.chunk_enhancement_experiment import Run
    from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
    from demo.four_state_receive import PersistentRGB,codec_precision
    from demo.scalable_format import frame_hash
    from demo.routervc_light_packets import bank_info
    from demo.routervc_encode import compose_candidates
    from demo.routervc_mixed_router import selection
    from demo.online_eg_eval_core import noise_pair
    from demo.scalable_experiment import now
    if not os.environ.get('TMUX'):raise RuntimeError('requires tmux')
    run=Run(SimpleNamespace(output=ROOT/'interleave_check',command='interleave',max_hours=1.))
    run.thread.start()
    try:
        root=ROOT/'smoke';protocol=read(root/'protocol.json')
        binding=dict(smoke=digest(root/'complete.json'),protocol=digest(root/'protocol.json'),
            script=digest(__file__),training_code=protocol['code'])
        immutable(run.root/'protocol.json',binding)
        if (run.root/'complete.json').exists():
            old=read(run.root/'complete.json')
            if old['binding']!=binding:raise ValueError('changed interleave audit')
            print('MIXED_INTERLEAVE_VERIFIED_READ_ONLY',flush=True);return
        with exclusive_native_evaluation(run):
            torch.set_num_threads(4)
            generator=PersistentRGB();data=Samples(root,CACHE/'smoke',digest(root/'protocol.json'))
            initial=torch.load(INITIAL,weights_only=True,map_location='cpu')['state_dict']
            scale=torch.load(RESUME,weights_only=True,map_location='cpu')['scale'].cuda()
            models={a:visual.VisualUtilityRouter().cuda() for a in ARMS}
            for m in models.values():m.load_state_dict(initial)
            optimizers={a:torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=1e-4) for a,m in models.items()}
            generator_ids={id(p) for p in generator.model.runner.dit.parameters()}
            if any(id(p) in generator_ids for m in models.values() for p in m.parameters()):
                raise ValueError('Router optimizer includes generator parameters')
            rows=[]
            for row in protocol['rows']:
                run.check();i,j=5,1;sid=row['sample_id']
                expected=read(root/'samples'/sid/f'mix{j}/cell_{i:02d}/result.json')
                with np.load(row['reconstruction_path'],allow_pickle=False) as f:
                    base,enhanced=f['base'].copy(),f['enhanced'].copy()
                info=bank_info(Path(row['bank_path']).read_bytes())
                received=compose_candidates(base,enhanced,selection(sid,j),info['rois'])
                settings=expected['binding']['control']
                with torch.no_grad():before,runtime_before=generator(received,settings)
                if frame_hash(before)!=expected['output_hash']:raise ValueError('resident initial output differs')
                flags=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32,
                       torch.backends.cudnn.benchmark,torch.backends.cudnn.deterministic)
                sample=data.get(row,generate=False)
                for arm in ARMS:
                    with codec_precision():
                        model,optimizer=models[arm],optimizers[arm];model.train()
                        batch=arm_batch(sample,arm,'cuda');optimizer.zero_grad(set_to_none=True)
                        loss,_=visual.masked_training_loss(model(batch['inputs']),batch['targets'],gain_scale=scale)
                        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
                after_flags=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32,
                             torch.backends.cudnn.benchmark,torch.backends.cudnn.deterministic)
                if flags!=after_flags:raise ValueError('Router update changed G precision flags')
                with torch.no_grad():after,runtime_after=generator(received,settings)
                np.testing.assert_array_equal(before,after)
                noise_pair(dict(generation_runtime=runtime_before),dict(generation_runtime=runtime_after),same_condition=True)
                rows.append(dict(sample_id=sid,updates_per_arm=len(rows)+1,output_hash=frame_hash(after),
                    pixels_exact=True,noise_exact=True,precision_flags_restored=True,
                    source_frames_read=False))
                save(run.root/'progress.json',dict(checks=rows))
                run.update(completed=len(rows),total=len(protocol['rows']))
            save(run.root/'complete.json',dict(complete=True,binding=binding,checks=rows,utc=now(),
                formal_models_modified=False,seconds=time.monotonic()-run.started))
            print('MIXED_INTERLEAVE_PASSED',flush=True)
    finally:
        run.log_resources();run.stop.set();run.thread.join(timeout=3);run.lock.close()
        if torch.distributed.is_initialized():torch.distributed.destroy_process_group()


if __name__=='__main__':main()
