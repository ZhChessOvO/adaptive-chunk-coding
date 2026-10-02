"""Small preflight for receiver-exact differentiable VAE input arithmetic."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from demo.chunk_enhancement_codec import configure_torch
from demo.scalable_generation_decode import ASSETS
from demo.stage_c_seedvr2_bridge import configure_runner
from demo.online_eg_data import encode_pixels
from demo.roi_condition_data import encode_roi

def main():
    configure_torch()
    runner,device=configure_runner(SimpleNamespace(upstream_root=REPO/'third_party/SeedVR2',
        dit_checkpoint=ASSETS['dit'],vae_checkpoint=ASSETS['vae'],positive_embedding=ASSETS['positive'],
        negative_embedding=ASSETS['negative'],lora_checkpoint=None,sample_steps=1,cfg_scale=1.,dit_dtype='bfloat16'))
    values=np.random.default_rng(12).integers(0,256,size=(17,256,256,3),dtype=np.uint8)
    pixels=torch.from_numpy(values).permute(0,3,1,2).to(device).float()
    observed=[]
    original=runner.vae.encode
    def observe(sample):
        result=original(sample)
        observed.append((sample.detach().clone(),result.posterior.mode().detach().clone()))
        return result
    runner.vae.encode=observe
    native=encode_roi(runner,values,(0,0,256,256),device).to(device)
    for grad in (False,True):
        pixels.requires_grad_(grad)
        with torch.set_grad_enabled(grad):
            result=encode_pixels(runner,pixels)
        diff=(result.detach().float()-native.float()).abs()
        print(json.dumps(dict(grad=grad,max=float(diff.max()),mean=float(diff.mean()),fraction=float((diff>0).float().mean()),
                              stride=list(result.stride()),native_stride=list(native.stride()))),flush=True)
        print(json.dumps(dict(input_error=float((observed[0][0]-observed[-1][0]).abs().max()),
            input_stride=[list(observed[0][0].stride()),list(observed[-1][0].stride())],
            posterior_error=float((observed[0][1]-observed[-1][1]).abs().max()))),flush=True)
        torch.testing.assert_close(result.detach(),native,rtol=0,atol=0)
    torch.distributed.destroy_process_group()


if __name__=='__main__':main()
