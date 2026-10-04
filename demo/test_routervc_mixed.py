"""CPU contracts for mixed labels, halo preprocessing and atomic optimization."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from demo import routervc_visual_router as visual
from demo import routervc_mixed_router as mixed
from demo.routervc_mixed_train import run_training, validate, arm_batch
from demo.routervc_mixed_queue import exact
from demo.test_routervc_visual_router import pixels, qualities
from demo.test_routervc_visual_train import tiny_config


def data_fixture(config=None):
    config=config or tiny_config();base,enhanced=pixels(64,96);qs=qualities();views=[];halos=[];isolated=[]
    rois=visual.grid_rois(64,96)
    from demo.routervc_encode import compose_candidates
    for j in range(2):
        selected=mixed.selection('fixture',j);cov=np.zeros(16,np.float32);cov[selected]=1
        received=compose_candidates(base,enhanced,selected,rois)
        # Synthetic known targets, deliberately conditional on neighbor pattern.
        generated=[dict(q['EG' if i in selected else 'G']) for i,q in enumerate(qs)]
        for q in generated:q['lpips_alex']+=.01*j
        views.append(dict(inputs=mixed.build_inputs(base,received,cov,config=config),
                          targets=mixed.mixed_targets(qs,selected,generated)))
        halos.append(mixed.build_inputs(base,received,cov,halo=64,config=config)['local_pairs'])
        for i in range(16):
            isolated.append(visual.build_training_case(base,enhanced,qs,i,i in selected,config=config))
    result=visual.batch_examples(views);result.update(halo_pairs=torch.cat(halos),
        isolated=visual.batch_examples(isolated),sample_id='fixture',dataset='REDS')
    return result


class MixedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_patterns_are_nested_reproducible_and_gt_free(self):
        a,b=mixed.selection('one',0),mixed.selection('one',1)
        self.assertEqual(len(a),4);self.assertEqual(len(b),8);self.assertEqual(a,b[:4])
        self.assertEqual(a,mixed.selection('one',0));self.assertNotEqual(b,mixed.selection('two',1))
        with self.assertRaises(ValueError):mixed.selection('one',2)

    def test_halo_is_actual_generator_clipped_extent(self):
        self.assertEqual(mixed.processing_roi([0,0,128,128],512,512,64),[0,0,192,192])
        self.assertEqual(mixed.processing_roi([128,128,128,128],512,512,64),[64,64,256,256])
        self.assertEqual(mixed.processing_roi([768,432,256,144],576,1024,64),[704,368,320,208])
        with self.assertRaises(ValueError):mixed.processing_roi([0,0,16,16],64,64,-1)

    def test_zero_halo_exact_legacy_and_other_features_preserved(self):
        base,received=pixels(64,96);cov=np.zeros(16,np.float32)
        a=mixed.build_inputs(base,received,cov);b=visual.build_receiver_inputs(base,received,cov)
        exact(a,b);c=mixed.build_inputs(base,received,cov,halo=64)
        for k in ('global_pairs','coverage','geometry'):exact(a[k],c[k])
        self.assertFalse(torch.equal(a['local_pairs'],c['local_pairs']))

    def test_neighbor_changes_are_visible_inside_same_body(self):
        b=np.zeros((3,128,128,3),np.uint8);y=b.copy();z=b.copy();z[:,40:55,40:55]=200
        cov=np.zeros(16,np.float32)
        a=mixed.build_inputs(b,y,cov,halo=0);c=mixed.build_inputs(b,z,cov,halo=0)
        exact(a['local_pairs'][:,0],c['local_pairs'][:,0])
        a=mixed.build_inputs(b,y,cov,halo=64);c=mixed.build_inputs(b,z,cov,halo=64)
        self.assertFalse(torch.equal(a['local_pairs'][:,0],c['local_pairs'][:,0]))

    def test_real_mixed_g_labels_not_summed_isolated_g(self):
        qs=qualities();generated=[dict(q['G']) for q in qs];generated[0]['lpips_alex']=.02
        t=mixed.mixed_targets(qs,[0,1],generated)
        self.assertAlmostEqual(float(t['gains']['value'][0,0,1]),qs[0]['E']['lpips_alex']-.02,places=6)
        self.assertEqual(float(t['gains']['value'][0,2,0]),0.)
        self.assertFalse(visual.supervision_metadata(t)['semantic_supervision'])
        generated[0]['lpips_alex']=None
        with self.assertRaises(ValueError):mixed.mixed_targets(qs,[0],generated)

    def test_same_number_and_coverage_of_supervised_cells(self):
        data=data_fixture()
        iso,core=arm_batch(data,'isolated_core','cpu'),arm_batch(data,'mixed_core','cpu')
        torch.testing.assert_close(iso['inputs']['coverage'].flatten(),core['inputs']['coverage'].flatten())
        self.assertEqual(iso['targets']['gains']['weight'].numel(),core['targets']['gains']['weight'].numel())
        halo=arm_batch(data,'mixed_halo','cpu')
        exact(core['targets'],halo['targets'])

    def test_regret_handles_harmful_generation_and_stable_ties(self):
        truth=np.array([1.,2.]+[-1.]*14);pred=truth.copy()
        self.assertEqual(mixed.regret(pred,truth),0.)
        self.assertGreater(mixed.regret(-pred,truth),0.)
        self.assertEqual(mixed.best_g_indices(np.ones(16),4),[0,1,2,3])

    def test_exact_mid_epoch_resume_models_optimizer_and_best_selection(self):
        config=tiny_config();data=data_fixture(config)
        protocol=dict(rows=[dict(sample_id='a',dataset='REDS',router_split='train'),
                            dict(sample_id='b',dataset='UVG',router_split='train')],
                      smoke=True,epochs=2,seed=9,learning_rate=1e-4)
        torch.manual_seed(1);initial=visual.VisualUtilityRouter(config).state_dict()
        kwargs=dict(initial_state=initial,scale=torch.ones(6),config=config,device='cpu')
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            run_training(root/'resumed',protocol,lambda _:data,stop_after=1,**kwargs)
            self.assertFalse((root/'resumed/complete.json').exists())
            saved=torch.load(root/'resumed/resume.pt',weights_only=True)
            self.assertEqual(saved['state']['cursor'],1)
            for arm in mixed.ARMS:self.assertTrue(any(not torch.equal(v,initial[k]) for k,v in saved['models'][arm].items()))
            run_training(root/'resumed',protocol,lambda _:data,**kwargs)
            run_training(root/'direct',protocol,lambda _:data,**kwargs)
            a=torch.load(root/'resumed/resume.pt',weights_only=True)
            b=torch.load(root/'direct/resume.pt',weights_only=True);exact(a,b)
            before={str(p.relative_to(root/'resumed')):(p.read_bytes(),p.stat().st_mtime_ns)
                    for p in (root/'resumed').rglob('*') if p.is_file()}
            run_training(root/'resumed',protocol,lambda _:self.fail('completed replay requested data'),**kwargs)
            after={str(p.relative_to(root/'resumed')):(p.read_bytes(),p.stat().st_mtime_ns)
                   for p in (root/'resumed').rglob('*') if p.is_file()}
            self.assertEqual(before,after)
            for arm in mixed.ARMS:
                model,payload=mixed.load_model(root/'resumed'/arm/'best.pt')
                self.assertEqual(payload['epoch'],a['state']['best'][arm]['epoch'])
                exact(model.state_dict(),a['state']['best'][arm]['weights'])

    def test_validation_uses_mixed_even_for_isolated_training_arm(self):
        cfg=tiny_config();data=data_fixture(cfg);model=visual.VisualUtilityRouter(cfg)
        rows=[dict(dataset='REDS')]
        one=validate(model,'isolated_core',rows,lambda _:data,torch.ones(6),'cpu')
        two=validate(model,'mixed_core',rows,lambda _:data,torch.ones(6),'cpu')
        self.assertEqual(one,two)


if __name__=='__main__':unittest.main()
