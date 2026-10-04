"""Versioned bounded-pair G receiver; original visual receiver stays unchanged.

Same selected core support and feathering, different shared execution schedule.
Unmerged calls keep their original seed slot. A merged rectangle uses its first
member's slot; different shapes do NOT imply pixelwise paired diffusion noise.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from demo import routervc_format as legacy
from demo import routervc_visual_format as visual_format
from demo.routervc_generation_schedule import schedule
from demo.scalable_codec import file_hash, atomic_json

PROFILE='routervc_visual_pairmerge_area1p5_v1'
AREA_MULTIPLIER=1.5
generation_control=visual_format.generation_control


def policy_identity():
    value=dict(parent=visual_format.policy_identity(),profile=PROFILE,cap=AREA_MULTIPLIER,
        code={n:file_hash(REPO/'demo'/n) for n in
              ('routervc_scheduled_decode.py','routervc_generation_schedule.py')},
        seed='original ordinal for singleton; first ordinal for pair',
        output_support='original unmerged feather weights; no expanded writeback')
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def parse(data, *, allow_incomplete_tail=False):
    values=legacy.parse(data,allow_incomplete_tail=allow_incomplete_tail)
    if values[0]['policy']!=policy_identity():raise ValueError('pair-merge profile mismatch')
    return values


def rewrap(old_wire):
    config,inner,_,_=visual_format.parse(old_wire)
    config=dict(config,policy=policy_identity())
    value=legacy.wrap(inner,config)
    if len(value)!=len(old_wire):raise ValueError('unexpected schedule signaling overhead')
    return value


def remap_seed(seed, base_seed, slots):
    if type(seed) is not int or seed<base_seed:raise ValueError('invalid scheduler seed')
    index,window=divmod(seed-base_seed,65536)
    if index>=len(slots):raise ValueError('seed outside merged group schedule')
    return base_seed+65536*slots[index]+window


def restore_scheduled(enhanced,control,adapter,packets):
    from demo import stage_c_a800_teacher as teacher
    from demo.internal_condition_decode import restore
    if packets:raise ValueError('this fixed-weight scheduling probe uses the RGB-only joint model')
    old=control['generate']
    if not old:return enhanced.copy(),None
    if any(r[:2]!=[0,len(enhanced)] for r in old):
        raise ValueError('pair scheduler currently requires whole-window selected cores')
    plan=schedule(list(range(len(old))),[r[2:] for r in old],enhanced.shape[1:3],
                  area_multiplier=AREA_MULTIPLIER,halo=control['context'])
    new_control=dict(control,generate=[[0,len(enhanced),*g['core']] for g in plan['groups']])
    slots=[g['seed_slot'] for g in plan['groups']]
    base=teacher.PersistentSeedVR2
    class SeedSlotRunner(base):
        def restore(self,frames,seed,**kwargs):
            return super().restore(frames,seed=remap_seed(seed,control['seed'],slots),**kwargs)
    # Scoped dependency substitution, not an edit to any old codec/profile.
    with patch.object(teacher,'PersistentSeedVR2',SeedSlotRunner):
        output,runtime=restore(enhanced,new_control,adapter,packets)
    runtime['schedule']=plan
    runtime['merged_control']=new_control
    runtime['original_generation_support']=old
    runtime['seed_scope']='unchanged singleton ordinal; paired shape/noise not asserted for merged groups'
    return output,runtime


def receive(args):
    import torch
    from demo import routervc_visual_decode as original
    maxima=[];reserved=[]
    original_reset=torch.cuda.reset_peak_memory_stats
    def measured_reset(device=None):
        maxima.append(torch.cuda.max_memory_allocated(device))
        reserved.append(torch.cuda.max_memory_reserved(device))
        original_reset(device)
    with patch.object(original,'fmt',sys.modules[__name__]), \
            patch.object(original,'restore',restore_scheduled), \
            patch.object(torch.cuda,'reset_peak_memory_stats',measured_reset):
        report=original.receive(args)
    # SeedVR2 resets its counters per call. Accumulate BEFORE every reset so
    # model loading and UF/E decode are included, not only the final ROI.
    maxima.append(torch.cuda.max_memory_allocated())
    reserved.append(torch.cuda.max_memory_reserved())
    report['last_call_cuda_peak_bytes']=report['peak_cuda_allocated_bytes']
    report['peak_cuda_allocated_bytes']=max(maxima)
    report['peak_cuda_reserved_bytes']=max(reserved)
    report['memory_scope']='whole fresh worker; accumulated across every per-call counter reset'
    report['generation_all_calls_peak_cuda_bytes']=max(
        (w['runtime']['peak_cuda_allocated_bytes'] for w in report['generation_runtime']['windows']),default=0)
    report['memory_counter_segments']=len(maxima)
    atomic_json(args.output/'decode.json',report)
    return report


def main():
    import torch
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stream',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--router',type=Path,required=True);p.add_argument('--adapter',type=Path,required=True)
    p.add_argument('--enhancement',type=Path,required=True)
    args=p.parse_args();args.disable_generation=False;args.allow_incomplete_tail=False
    if not os.environ.get('TMUX') or not os.environ.get('ROUTERVC_SCHEDULE_PARENT'):
        raise RuntimeError('scheduled fresh worker requires a supervised tmux queue')
    try:receive(args)
    finally:
        if torch.distributed.is_initialized():torch.distributed.destroy_process_group()


if __name__=='__main__':main()
