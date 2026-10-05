"""CPU-only paired independent R_s training, sparse validation, exact restart."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from demo import routervc_sender_router as sender
from demo import routervc_sender_train as train


def tiny_config():
    return sender.Config(global_size=32, local_size=16, channels=4, detail_channels=4, hidden=8)


def fixture():
    config = tiny_config()
    rng = torch.Generator().manual_seed(36)
    costs = torch.arange(1,17, dtype=torch.int64)[None].expand(2,16).clone()*100
    coverage = torch.zeros(2,16,1)
    coverage[1,:2] = 1
    inputs = dict(global_video=torch.rand(2,18,17,32,32,generator=rng),
        local_video=torch.rand(2,16,21,3,16,16,generator=rng),
        global_extent=torch.tensor([[0.,0.,1.,1.]]).expand(2,4).clone(),
        geometry=torch.rand(2,16,6,generator=rng), coverage=coverage,
        packet_bytes=costs[...,None].clone(), max_g=torch.full((2,1),8,dtype=torch.int64))
    value = torch.full((2,16), float('nan'))
    weight = torch.zeros(2,16)
    plans = [dict(selected=[], candidates=[0,1,2], kind='empty'),
             dict(selected=[0,1], candidates=[4,5,6], kind='random')]
    for state,plan in enumerate(plans):
        value[state,plan['candidates']] = torch.tensor([-.02,.04,.005])*(state+1)
        weight[state,plan['candidates']] = 1
    return dict(inputs=inputs, targets=dict(value=value,weight=weight,label_scope=sender.LABEL_SCOPE),
                packet_bytes=costs, plans=plans)


def protocol(smoke=True):
    rows = [dict(sample_id='a',dataset='REDS',router_split='train'),
            dict(sample_id='b',dataset='UVG',router_split='train')]
    if not smoke:
        rows += [dict(sample_id='v',dataset='UVG',router_split='validation')]
    return dict(format=train.FORMAT, rows=rows, teacher={'sender_config':asdict(tiny_config())},
        smoke=smoke, arms=list(train.ARMS), epochs=2, seed=20261005,
        learning_rate=1e-4, weight_decay=1e-4, ranking_weight=.1)


def labels(protocol):
    count = len(protocol['rows'])
    return dict(format='sender_final_labels_complete_v1', complete=True,
        protocol='0'*64, protocol_semantic_sha256=train.semantic_hash(protocol),
        samples={r['sample_id']:'1'*64 for r in protocol['rows']}, windows=count,
        measured_final_renderings=8*count, measured_marginals=6*count,
        label_scope=sender.LABEL_SCOPE, current_sender_policy_used=False)


def exact(a,b):
    if torch.is_tensor(a):
        torch.testing.assert_close(a,b,rtol=0,atol=0,equal_nan=True)
    elif isinstance(a,dict):
        assert a.keys() == b.keys()
        for key in a:
            exact(a[key],b[key])
    elif isinstance(a,(tuple,list)):
        assert type(a) is type(b) and len(a) == len(b)
        for left,right in zip(a,b):
            exact(left,right)
    else:
        assert a == b


class ZeroModel:
    def eval(self):
        return self

    def __call__(self,inputs):
        return inputs['coverage'].new_zeros((2,16))


class SenderTrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_zero_source_only_removes_source_and_errors_without_mutating_cache(self):
        data = fixture()
        original = deepcopy(data)
        source = train.arm_batch(data,'source','cpu')
        ablated = train.arm_batch(data,'zero_source','cpu')
        exact(data,original)
        exact(source['targets'],ablated['targets'])
        for key in sender.INPUT_KEYS-{'global_video','local_video'}:
            exact(source['inputs'][key],ablated['inputs'][key])
        for key,dim in (('global_video',1),('local_video',2)):
            for start,end in ((0,3),(12,18)):
                self.assertEqual(ablated['inputs'][key].narrow(dim,start,end-start).count_nonzero(),0)
            exact(source['inputs'][key].narrow(dim,3,9),ablated['inputs'][key].narrow(dim,3,9))
        exact(source['inputs']['local_video'][:,:,18:],ablated['inputs']['local_video'][:,:,18:])

    def test_unknowns_preserved_negative_retained_and_invalid_masks_rejected(self):
        data = fixture()
        batch = train.arm_batch(data,'source','cpu')
        self.assertLess(batch['targets']['value'][0,0],0)
        self.assertTrue(torch.isnan(batch['targets']['value'][0,15]))
        for edit in ('scope','count','cost','covered','plan'):
            changed=deepcopy(data)
            if edit=='scope': changed['targets']['label_scope']='direct_E_gain'
            if edit=='count': changed['targets']['weight'][0,15]=1
            if edit=='cost': changed['packet_bytes'][0,0]+=1
            if edit=='covered': changed['inputs']['coverage'][0,0]=1
            if edit=='plan': changed['plans'][0]['candidates']=[0,1,15]
            with self.assertRaises(ValueError,msg=edit):
                train.arm_batch(changed,'source','cpu')

    def test_scale_uses_train_only_signed_measured_targets_and_is_robust(self):
        p=protocol(False);data=fixture();visited=[]
        def get(row):
            visited.append(row['sample_id'])
            self.assertNotEqual(row['router_split'],'validation')
            return data
        scale=train.fit_scale(p,get)
        self.assertEqual(visited,['a','b'])
        known=data['targets']['weight']>0
        expected=np.quantile(data['targets']['value'][known].double().numpy(),.75)
        absolute=np.quantile(np.abs(data['targets']['value'][known].double().numpy()),.75)
        self.assertEqual(scale['value'],absolute)
        self.assertNotEqual(scale['value'],expected)
        self.assertEqual(scale['negative'],4)
        self.assertEqual(scale['measured_marginals'],12)
        self.assertFalse(scale['validation_used'])
        self.assertFalse(scale['receiver_scale_reused'])
        data['targets']['value'][known]=0
        scale=train.fit_scale(p,lambda _:data)
        self.assertEqual(scale['value'],1e-6)
        self.assertTrue(scale['floor_applied'])

    def test_measured_density_not_raw_gain_unknown_or_full_oracle(self):
        value=np.full(16,np.nan);weight=np.zeros(16);cost=np.ones(16,dtype=np.int64)*100
        value[:3]=[.1,.2,-.3];weight[:3]=1;cost[:3]=[100,1000,200]
        prediction=np.full(16,99.);prediction[:3]=[.1,.2,-.3]
        score=train.measured_regret(prediction,value,weight,cost)
        self.assertEqual(score['selected_region'],0) # lower raw gain, greater gain/byte
        self.assertEqual(score['regret'],0)
        self.assertEqual(score['measured_candidates'],3)
        self.assertEqual(score['selected_final_lpips_gain'],.1)
        score=train.measured_regret(-np.ones(16),value,weight,cost)
        self.assertEqual(score['no_add'],1)
        self.assertAlmostEqual(score['regret'],.1*(1300/3)/100)
        prediction[:3]=[-1,-1,1]
        score=train.measured_regret(prediction,value,weight,cost)
        self.assertEqual(score['harmful_add'],1)
        self.assertGreater(score['regret'],.1*(1300/3)/100)
        value[:3]=-1
        score=train.measured_regret(-np.ones(16),value,weight,cost)
        self.assertEqual(score['regret'],0)

    def test_validation_weights_datasets_and_states_equally(self):
        rows=[dict(dataset='REDS') for _ in range(3)]+[dict(dataset='UVG')]
        data=fixture()
        data['packet_bytes'][:]=100
        data['inputs']['packet_bytes'][:]=100
        def get(row):
            copy=deepcopy(data)
            for state in range(2):
                known=copy['targets']['weight'][state]>0
                copy['targets']['value'][state,known]=(state+1)*(1 if row['dataset']=='REDS' else 3)
            return copy
        scores=train.validate(ZeroModel(),'source',rows,get,1.,'cpu')
        self.assertEqual(scores['regret'],3.)
        self.assertEqual(scores['by_state']['empty']['regret'],2.)
        self.assertEqual(scores['by_state']['partial']['regret'],4.)
        self.assertEqual(scores['groups']['REDS']['samples'],3)
        self.assertEqual(scores['groups']['UVG']['samples'],1)
        self.assertIn('NOT full oracle',scores['scope'])

    def test_full_teacher_and_train_only_scale_required_before_model_updates(self):
        p=protocol();data=fixture();scale=train.fit_scale(p,lambda _:data)
        no_get=Mock(side_effect=AssertionError('invalid provenance must not request data'))
        with tempfile.TemporaryDirectory() as directory:
            bound=labels(p);bound['complete']=False
            with self.assertRaisesRegex(ValueError,'ALL fixed'):
                train.run_training(directory,p,no_get,scale_record=scale,labels_binding=bound,device='cpu')
            scale['validation_used']=True
            with self.assertRaisesRegex(ValueError,'TRAIN-only'):
                train.run_training(directory,p,no_get,scale_record=scale,labels_binding=labels(p),device='cpu')
            no_get.assert_not_called()

    def test_label_completion_boundary_resume_and_readonly_verification(self):
        p=protocol();data=fixture()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            train.save(root/'protocol.json',p)
            class Samples:
                def get(self,row,*,generate):
                    path=root/'samples'/row['sample_id']/'complete.json'
                    if not path.exists():
                        if not generate:
                            raise AssertionError('read-only verifier tried missing teachers')
                        train.save(path,dict(complete=True, measured_final_renderings=8,
                            measured_marginals=6,label_scope=sender.LABEL_SCOPE,current_sender_policy_used=False))
                    return data
            samples=Samples()
            def interrupt(**status):
                if status['completed_windows']==1:
                    raise InterruptedError('sample boundary')
            with self.assertRaises(InterruptedError):
                train.prepare_labels(root,p,samples,progress=interrupt)
            self.assertFalse((root/'labels.complete.json').exists())
            first=(root/'samples/a/complete.json').read_bytes()
            ready=train.prepare_labels(root,p,samples)
            self.assertEqual(ready['measured_final_renderings'],16)
            self.assertEqual(ready['measured_marginals'],12)
            self.assertEqual((root/'samples/a/complete.json').read_bytes(),first)
            self.assertEqual(train.verify_labels(root,p,samples),ready)
            train.save(root/'samples/a/complete.json',{'changed':True})
            with self.assertRaisesRegex(ValueError,'sample record changed'):
                train.verify_labels(root,p,samples)

    def test_exact_interrupted_resume_both_models_optimizers_rng_best_last(self):
        p=protocol();data=fixture();scale=train.fit_scale(p,lambda _:data)
        def get(_):
            # Exercise every saved RNG stream rather than only a deterministic
            # no-dropout model. Teacher inputs here are synthetic fixtures.
            sampled=deepcopy(data)
            sampled['inputs']['global_video']*=1+.01*(random.random()+float(np.random.random())+float(torch.rand(())))
            return sampled
        kwargs=dict(scale_record=scale,labels_binding=labels(p),device='cpu')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            before=train.capture_rng()
            train.run_training(root/'resumed',p,get,stop_after=1,**kwargs)
            exact(before,train.capture_rng())
            checkpoint=torch.load(root/'resumed/resume.pt',weights_only=True)
            self.assertEqual(checkpoint['state']['cursor'],1)
            self.assertEqual(checkpoint['state']['updates'],1)
            self.assertEqual(set(checkpoint['rng']),{'python','numpy','torch','cuda'})
            for arm in train.ARMS:
                self.assertEqual({int(v['step']) for v in checkpoint['optimizers'][arm]['state'].values()},{1})
            self.assertTrue(any(not torch.equal(checkpoint['models']['source'][key],checkpoint['models']['zero_source'][key])
                                for key in checkpoint['models']['source']))
            train.run_training(root/'resumed',p,get,**kwargs)
            train.run_training(root/'direct',p,get,**kwargs)
            left=torch.load(root/'resumed/resume.pt',weights_only=True)
            right=torch.load(root/'direct/resume.pt',weights_only=True)
            exact(left,right)
            self.assertEqual(left['state']['updates'],4)
            self.assertEqual(len(left['state']['history']),2)
            result=train.verify_training(root/'resumed',p,labels_binding=labels(p))
            self.assertTrue(result['whole_video_RD_pending'])
            self.assertFalse(result['receiver_weights_shared'])
            for arm in train.ARMS:
                model,payload=sender.load_model(root/'resumed'/arm/'best.pt')
                self.assertEqual(model.config.zero_source,arm=='zero_source')
                self.assertEqual(payload['step'],left['state']['best'][arm]['step'])
                exact(model.state_dict(),left['state']['best'][arm]['weights'])
                model,_=sender.load_model(root/'resumed'/arm/'last.pt')
                exact(model.state_dict(),left['models'][arm])
            files={str(path.relative_to(root/'resumed')):(path.read_bytes(),path.stat().st_mtime_ns)
                   for path in (root/'resumed').rglob('*') if path.is_file()}
            train.run_training(root/'resumed',p,lambda _:self.fail('completed replay requested labels'),**kwargs)
            after={str(path.relative_to(root/'resumed')):(path.read_bytes(),path.stat().st_mtime_ns)
                   for path in (root/'resumed').rglob('*') if path.is_file()}
            self.assertEqual(files,after)

    def test_crash_between_arms_replays_atomic_pair_from_identical_initial_weights(self):
        p=protocol();p['epochs']=1;data=fixture();scale=train.fit_scale(p,lambda _:data)
        kwargs=dict(scale_record=scale,labels_binding=labels(p),device='cpu')
        original=sender.training_loss;calls=0
        def fail_second(*args,**kw):
            nonlocal calls
            if torch.is_grad_enabled():
                calls+=1
                if calls==2:
                    raise InterruptedError('simulated failure between paired arms')
            return original(*args,**kw)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch.object(sender,'training_loss',side_effect=fail_second):
                with self.assertRaises(InterruptedError):
                    train.run_training(root/'resumed',p,lambda _:data,**kwargs)
            checkpoint=torch.load(root/'resumed/resume.pt',weights_only=True)
            self.assertEqual(checkpoint['state']['updates'],0)
            exact(checkpoint['models']['source'],checkpoint['models']['zero_source'])
            train.run_training(root/'resumed',p,lambda _:data,**kwargs)
            train.run_training(root/'direct',p,lambda _:data,**kwargs)
            exact(torch.load(root/'resumed/resume.pt',weights_only=True),
                  torch.load(root/'direct/resume.pt',weights_only=True))

    def test_final_checkpoint_recovers_missing_stale_exports_without_new_updates(self):
        p=protocol();data=fixture();scale=train.fit_scale(p,lambda _:data)
        kwargs=dict(scale_record=scale,labels_binding=labels(p),device='cpu')
        original=train.save
        def crash(path,value):
            if Path(path).name=='history.json' and len(value)==2:
                raise InterruptedError('final checkpoint before final JSON exports')
            return original(path,value)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch.object(train,'save',side_effect=crash):
                with self.assertRaises(InterruptedError):
                    train.run_training(root,p,lambda _:data,**kwargs)
            checkpoint=torch.load(root/'resume.pt',weights_only=True)
            self.assertEqual(checkpoint['state']['epoch'],2)
            self.assertEqual(len(train.read(root/'history.json')),1)
            (root/'initial_validation.json').unlink()
            result=train.run_training(root,p,lambda _:self.fail('final recovery accessed labels'),**kwargs)
            self.assertTrue(result['complete'])
            self.assertEqual(train.read(root/'history.json'),checkpoint['state']['history'])
            self.assertEqual(train.read(root/'initial_validation.json'),checkpoint['state']['initial_validation'])
            train.verify_training(root,p,labels_binding=labels(p))

    def test_changed_protocol_scale_and_completed_artifacts_rejected(self):
        p=protocol();data=fixture();scale=train.fit_scale(p,lambda _:data)
        kwargs=dict(scale_record=scale,labels_binding=labels(p),device='cpu')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            train.run_training(root,p,lambda _:data,stop_after=1,**kwargs)
            changed=deepcopy(p);changed['learning_rate']*=2
            with self.assertRaises(ValueError):
                train.run_training(root,changed,lambda _:data,scale_record=scale,labels_binding=labels(changed),device='cpu')
            changed_scale=dict(scale,value=scale['value']*2)
            with self.assertRaises(ValueError):
                train.run_training(root,p,lambda _:data,scale_record=changed_scale,labels_binding=labels(p),device='cpu')
            train.run_training(root,p,lambda _:data,**kwargs)
            original=train.read(root/'history.json')
            train.save(root/'history.json',original[:1])
            with self.assertRaises(ValueError):
                train.verify_training(root,p)

    def test_precision_determinism_and_rng_policy_restored(self):
        previous=train.capture_rng()
        old_deterministic=torch.are_deterministic_algorithms_enabled()
        old_tf32=torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32=True
        try:
            with train.deterministic_training(10,'cpu'):
                self.assertTrue(torch.are_deterministic_algorithms_enabled())
                self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
                random.random();np.random.random();torch.rand(2)
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
            self.assertEqual(torch.are_deterministic_algorithms_enabled(),old_deterministic)
            exact(previous,train.capture_rng())
        finally:
            torch.backends.cuda.matmul.allow_tf32=old_tf32


if __name__=='__main__':
    unittest.main()
