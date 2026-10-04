"""Original tiled visual receiver, with whole-worker CUDA peak accounting.

No policy, pixel arithmetic or stream-profile change. Accumulate before the
upstream generator resets its per-ROI memory counters.
"""
import argparse
import os
from pathlib import Path
import sys
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0,str(REPO))


def receive(args):
    import torch
    from demo.routervc_visual_decode import receive as original
    from demo.scalable_codec import atomic_json
    peaks,reserved = [],[]
    reset = torch.cuda.reset_peak_memory_stats
    def measured_reset(device=None):
        peaks.append(torch.cuda.max_memory_allocated(device))
        reserved.append(torch.cuda.max_memory_reserved(device))
        reset(device)
    with patch.object(torch.cuda,'reset_peak_memory_stats',measured_reset):
        result = original(args)
    peaks.append(torch.cuda.max_memory_allocated()); reserved.append(torch.cuda.max_memory_reserved())
    result['last_call_cuda_peak_bytes'] = result['peak_cuda_allocated_bytes']
    result['peak_cuda_allocated_bytes'] = max(peaks)
    result['peak_cuda_reserved_bytes'] = max(reserved)
    result['memory_scope'] = 'whole fresh worker accumulated across every ROI reset; not total device VRAM'
    runtime = result['generation_runtime']
    result['generation_all_calls_peak_cuda_bytes'] = max(
        (w['runtime']['peak_cuda_allocated_bytes'] for w in runtime['windows']),default=0) if runtime else 0
    atomic_json(args.output/'decode.json',result)
    return result


if __name__ == '__main__':
    import torch
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stream',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--router',type=Path);p.add_argument('--adapter',type=Path,required=True)
    p.add_argument('--enhancement',type=Path,required=True);p.add_argument('--disable-generation',action='store_true')
    args = p.parse_args();args.allow_incomplete_tail=False
    if not os.environ.get('TMUX') or not os.environ.get('ROUTERVC_LIGHT_EVALUATION_PARENT'):
        raise RuntimeError('fresh worker requires supervised tmux evaluation')
    try: receive(args)
    finally:
        if torch.distributed.is_initialized():torch.distributed.destroy_process_group()
