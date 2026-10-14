"""CPU contracts for fresh initialization, paired schedules and real-pixel reports."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
import torch

from demo.routervc_mixed_queue import exact
from demo.scalable_codec import atomic_json, atomic_npz
from demo.stage_c_seedvr2_lora_utils import LoRALinear, atomic_torch_save
from demo.routervc_fullview_probe import digest
from routervc.generation.texture_ablation import learning_rate
from routervc.generation.texture_clean import ARMS, MILESTONES, fresh_payload
from tools.texture_clean import partial_audit
from tools.texture_clean_evaluate import model_paths
from tools.texture_clean_report import draw, trajectory


def template():
    return dict(format_version=1,adapter_type='fixture',rank=2,alpha=2.,last_n_blocks=1,
        target_modules=['test'],trainable_parameter_count=14,
        state_dict={'test.lora_a':torch.full((2,4),9.),'test.lora_b':torch.full((3,2),8.)},
        metadata={'old_training':True})


class CleanTests(unittest.TestCase):
    def test_fresh_init_does_not_inherit_learned_values(self):
        old=template(); other=copy.deepcopy(old)
        for value in other['state_dict'].values(): value.fill_(100)
        a,b=fresh_payload(old),fresh_payload(other)
        exact(a,b)
        self.assertNotIn('old_training',a['metadata'])
        self.assertFalse(a['metadata']['inherited_learned_parameters'])
        self.assertTrue(bool(a['state_dict']['test.lora_a'].any()))
        self.assertEqual(torch.count_nonzero(a['state_dict']['test.lora_b']),0)
        self.assertEqual(float(old['state_dict']['test.lora_b'][0,0]),8.)
        self.assertFalse(torch.equal(a['state_dict']['test.lora_a'],fresh_payload(old,22)['state_dict']['test.lora_a']))

    def test_zero_contribution_has_nonzero_B_gradient(self):
        torch.manual_seed(1)
        layer=LoRALinear(torch.nn.Linear(4,3),2,2.)
        payload=fresh_payload(template())
        with torch.no_grad():
            layer.lora_a.copy_(payload['state_dict']['test.lora_a'])
            layer.lora_b.copy_(payload['state_dict']['test.lora_b'])
        x=torch.randn(7,4)
        torch.testing.assert_close(layer(x),layer.base(x),rtol=0,atol=0)
        layer(x).square().mean().backward()
        self.assertGreater(float(layer.lora_b.grad.norm()),0.)
        self.assertIsNone(layer.base.weight.grad)

    def test_explicit_schema_rejects_unexpected_weights(self):
        value=template();value['state_dict']['extra']=torch.ones(1)
        with self.assertRaises(ValueError):fresh_payload(value)

    def test_milestones_use_fixed_2000_step_horizon(self):
        cfg=dict(initial='/tmp/initial.pt',steps=2000,learning_rate=dict(peak=1e-5,end=2e-6,warmup=100))
        self.assertEqual(MILESTONES,(100,400,1000,2000))
        self.assertAlmostEqual(learning_rate(100,cfg),1e-5)
        self.assertAlmostEqual(learning_rate(2000,cfg),2e-6)
        paths=model_paths(Path('/tmp/formal'),cfg,400,False)
        self.assertEqual(set(paths),{'clean','control_000100','phase_only_000100','control_000400','phase_only_000400'})
        self.assertNotIn('2000',' '.join(str(p) for p in paths.values()))
        paths=model_paths(Path('/tmp/smoke'),cfg,6,True)
        self.assertEqual(paths['phase_only_000006'],Path('/tmp/smoke/direct/phase_only/checkpoints/adapter_000006.pt'))

    def test_partial_audit_checks_real_steps_and_pairing(self):
        with tempfile.TemporaryDirectory(prefix='clean-audit-') as directory:
            root=Path(directory);atomic_json(root/'protocol.json',{'test':True})
            for arm in ARMS:
                folder=root/arm
                state=dict(step=1,binding=dict(protocol=digest(root/'protocol.json'),arm=arm),
                    adapter=dict(state_dict={'x.lora_b':torch.ones(1)}),
                    optimizer=dict(state={0:{'step':torch.tensor(1.)}}))
                atomic_torch_save(folder/'resume.pt',state)
                row=dict(step=1,sample='same',dataset='REDS',crop=[0,0,256,256],E_cap=.5,
                         condition={'id':1},noise={'id':2},learning_rate=1e-7)
                (folder/'steps.jsonl').write_text(json.dumps(row)+'\n')
            self.assertTrue(partial_audit(root,1)['exact_pairing'])
            row['noise']={'id':3}
            (root/'phase_only/steps.jsonl').write_text(json.dumps(row)+'\n')
            with self.assertRaises(ValueError):partial_audit(root,1)

    def test_reports_have_source_Y_old_clean_and_both_new_arms(self):
        with tempfile.TemporaryDirectory(prefix='clean-report-') as directory:
            root=Path(directory);source=np.full((17,128,128,3),128,np.uint8)
            rows={}
            for name in ('legacy','clean','control_000100','phase_only_000100'):
                path=root/(name+'.npz');atomic_npz(path,pixels=source)
                rows[name]=dict(output_path=str(path),quality=dict(lpips_alex=.1,psnr_db=30.),
                               error_phase16=dict(phase_rms_u8=0.))
            case=dict(sample_id='SYNTHETIC TEST FIXTURE',crop=[0,0,128,128],core=[0,0,128,128],
                      evaluation_group='known_artifact')
            names=draw(root,case,source,source,rows,100)
            self.assertEqual(len(names),4)
            self.assertTrue(all((root/n).stat().st_size>1000 for n in names))
            probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
                '-show_entries','stream=nb_frames,width','-of','json',str(root/'comparison.mp4')]))
            self.assertEqual(int(probe['streams'][0]['nb_frames']),17)
            self.assertEqual(probe['streams'][0]['width'],1536)
            aggregate=[dict(group='internal_validation',dataset=d,name=n,n=3,lpips=.1,psnr=30.,phase16_error=1.)
                       for d in ('REDS','UVG') for n in rows]
            trajectory(dict(aggregate=aggregate),root)
            self.assertGreater((root/'trajectory.png').stat().st_size,1000)


if __name__=='__main__':unittest.main()
