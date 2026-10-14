"""CPU tests for isolated factor switches and authenticated state/report reuse."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from demo.roi_condition_data import core_image_terms
from demo.scalable_codec import atomic_npz
from routervc.generation.texture_ablation import switches, loss_terms
from routervc.generation.texture_losses import paired_lpips, phase_bias_loss
from tools.texture_ablation import numeric_state
from tools.texture_ablation_report import ARMS, draw


class AblationTests(unittest.TestCase):
    def test_four_factor_settings(self):
        self.assertEqual(switches('control'),(False,False))
        self.assertEqual(switches('repair'),(True,True))
        self.assertEqual(switches('augmentation_only'),(True,False))
        self.assertEqual(switches('phase_only'),(False,True))
        with self.assertRaises(ValueError):switches('invalid')

    def test_loss_and_gradient_match_explicit_recipe(self):
        torch.manual_seed(4)
        target=torch.rand(4,3,256,256)*2-1
        metric=lambda a,b:(a-b).square().mean((1,2,3))
        weights=dict(lpips=.5,low_frequency=.1,edge=.05,rgb_temporal=.1)
        config=dict(seed=20261014,loss=weights,phase_weight=1.)
        for arm in ('control','repair','augmentation_only','phase_only'):
            image=(target+.05*torch.randn_like(target)).requires_grad_()
            terms=core_image_terms(image,target,metric,1)
            actual,phase,spec,_=loss_terms(terms,image,target,metric,1,config,arm)
            expected_terms=core_image_terms(image,target,metric,1)
            aug,periodic=switches(arm)
            if aug:expected_terms['lpips'],expected_spec=paired_lpips(image,target,metric,1,20261014+7919)
            expected=sum(weights[k]*v for k,v in expected_terms.items())
            if periodic:expected=expected+phase_bias_loss(image[...,64:-64,64:-64],target[...,64:-64,64:-64])
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            torch.testing.assert_close(torch.autograd.grad(actual,image,retain_graph=True)[0],
                                       torch.autograd.grad(expected,image)[0],rtol=0,atol=0)
            self.assertEqual(spec,expected_spec if aug else None)

    def test_state_comparison_only_ignores_provenance(self):
        a=dict(step=6,optimizer={'a':torch.ones(1)},rng_cpu=torch.ones(2),rng_cuda=[],rng_python=(1,),
               adapter=dict(metadata={'run':'old'},state_dict={'a':torch.zeros(1)}),binding={'old':True})
        b={**a,'adapter':{**a['adapter'],'metadata':{'run':'new'}},'binding':{'new':True}}
        from demo.routervc_mixed_queue import exact
        exact(numeric_state(a),numeric_state(b))
        b['optimizer']={'a':torch.zeros(1)}
        with self.assertRaises(AssertionError):exact(numeric_state(a),numeric_state(b))

    def test_report_retains_all_four_arms(self):
        with tempfile.TemporaryDirectory(prefix='texture-ablation-test-') as directory:
            root=Path(directory);source=np.full((17,128,128,3),128,np.uint8)
            results={}
            for name in ('initial',*(a+'_000400' for a in ARMS)):
                path=root/(name+'.npz');atomic_npz(path,pixels=source)
                results[name]=dict(output_path=str(path),quality=dict(lpips_alex=.1,psnr_db=30.))
            case=dict(sample_id='SYNTHETIC FIXTURE',region=0,crop=[0,0,128,128],core=[0,0,128,128],
                      evaluation_group='known_artifact')
            draw(root,case,source,results)
            self.assertGreater((root/'comparison.png').stat().st_size,1000)
            self.assertGreater((root/'temporal_contact.png').stat().st_size,1000)
            import subprocess,json
            probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
                '-show_entries','stream=nb_frames,width','-of','json',str(root/'comparison.mp4')]))
            self.assertEqual(int(probe['streams'][0]['nb_frames']),17)
            self.assertEqual(probe['streams'][0]['width'],1536)

    def test_resource_plot_uses_device_units_and_all_mounts(self):
        import json
        from tools.texture_ablation_detail import resource_figure
        with tempfile.TemporaryDirectory(prefix='texture-resource-test-') as directory:
            root=Path(directory)
            disks={m:dict(used_percent=v) for m,v in
                   [('/root',28),('/root/autodl-tmp',76),('/root/autodl-fs',74)]}
            events=[dict(elapsed_seconds=t,gpu=f'0, A800, {mem}, 81920, 99',disks=disks)
                    for t,mem in [(0,0),(60,28672)]]
            (root/'heartbeat.jsonl').write_text('\n'.join(json.dumps(e) for e in events)+'\n')
            result=resource_figure(root,root)
            self.assertEqual(result['sampled_gpu_peak_gib'],28)
            self.assertEqual(result['queue_minutes'],1)
            self.assertEqual(result['last_disks'],disks)


if __name__=='__main__':unittest.main()
