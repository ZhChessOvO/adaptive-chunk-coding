"""Opt-in real-cache CPU smoke; never modifies or promotes a formal checkpoint."""
import io
import os
from pathlib import Path
import unittest

import torch

from demo.routervc_fullview_probe import read, save, digest
from demo.routervc_mixed_queue import exact
from routervc.latent.router_data import source
from routervc.fusion.model import PrecisionFusion
from routervc.fusion.inputs import maps
from routervc.fusion.blend import fuse
from routervc.fusion.training import patch_batch, objective
from tools.fusion_pilot import load_capture


@unittest.skipUnless(os.environ.get('FUSION_FIXTURE_ROOT'), 'set FUSION_FIXTURE_ROOT for real CPU fixtures')
class RealCacheTest(unittest.TestCase):
    def test_actual_two_dataset_loss_and_resume(self):
        import lpips
        root = Path(os.environ['FUSION_FIXTURE_ROOT'])
        rows = read(root/'protocol.json')['rows']
        selected = [next(r for r in rows if r['dataset']==d and r['router_split']=='train') for d in ('REDS','UVG')]
        arrays = []
        for row in selected:
            folder=root/'samples'/row['sample_id']/'capture'
            b,y,c,patches,done=load_capture(folder)
            controls=fuse(y,c,patches,done['generated'])
            a=dict(base=b,received=y,current=c,multiband=controls['multiband'],source=source(row))
            a.update(maps(b.shape,read(folder/'received.json')['detail']['received_regions'],done['generated']))
            arrays.append(a)
        metric=lpips.LPIPS(net='alex',verbose=False).eval().requires_grad_(False)

        def trial(restart):
            torch.manual_seed(20261010)
            model=PrecisionFusion()
            optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
            losses=[]
            for step in range(4):
                v,target=patch_batch(arrays[step%2],step,'cpu')
                optimizer.zero_grad(set_to_none=True)
                loss,parts=objective(model(v),target,v,metric)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertTrue(torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(),1.)))
                optimizer.step();losses.append(parts)
                if restart and step==1:
                    f=io.BytesIO()
                    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=torch.get_rng_state()),f)
                    f.seek(0);state=torch.load(f,map_location='cpu',weights_only=True)
                    model=PrecisionFusion();model.load_state_dict(state['model'])
                    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
                    optimizer.load_state_dict(state['optimizer']);torch.set_rng_state(state['rng'])
            return dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=torch.get_rng_state(),losses=losses)

        direct,resumed=trial(False),trial(True)
        exact(direct,resumed)
        save(root.parent/'p1_cpu_smoke/complete.json',dict(complete=True,backend='CPU',
            samples=[r['sample_id'] for r in selected],steps_per_trial=4,real_lpips_backward=True,
            exact_model_optimizer_rng=True,formal_weights_used=False,losses=direct['losses'],
            test_code_sha256=digest(Path(__file__)),model_code_sha256=digest(Path(__file__).resolve().parents[1]/'routervc/fusion/model.py')))


if __name__=='__main__':unittest.main()
