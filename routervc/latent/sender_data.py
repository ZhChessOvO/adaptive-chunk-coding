"""New-latent FINAL marginal teacher for the independent source-aware R_s.

Frozen adapted R_g is rerun on every actual packet-decoded parent and child.
Only original source views and authenticated new-latent banks are reused;
old feature-patch labels and pasted RGB approximations never enter this path.
"""
from collections import OrderedDict
from dataclasses import asdict
import gc
import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from demo import routervc_sender_router as sender, routervc_sender_train as fit
from demo.routervc_fullview_probe import read, digest, save, verify_artifacts
from demo.scalable_codec import atomic_bytes, atomic_npz
from demo.scalable_format import frame_hash
from routervc.latent import routing, router_data as receiver_data

REPO = receiver_data.REPO
ROOT = Path('/root/autodl-fs/DCVC/runs/routervc_latent_sender_20261009')
# The fast disk is near its 80% guard; keep this new, compact cache on the file store.
CACHE = ROOT/'input_cache'
RECEIVER = receiver_data.ROOT/'formal/router/core/best.pt'
RECEIVER_SHA = '62f02fe9a1d0fe008c0f93c5c6bf0747c7a470cf2d8eba90633795cf0bb7fe7b'
CODE = ('routervc/latent/sender_data.py', 'routervc/latent/sender.py',
        'tools/latent_sender_queue.py', 'tools/latent_sender_worker.py',
        'tools/run_latent_sender.sh', 'demo/routervc_sender_router.py',
        'demo/routervc_sender_train.py', 'demo/routervc_sender_data.py', *receiver_data.CODE)


def plans(row, source, received, costs):
    """Two states; source-error/byte, spatial-price diversity, and random proposals.

    This cheap source-visible proposal is not the teacher target or an oracle.
    All three candidates get a new whole-picture fixed-receiver measurement.
    """
    partial = receiver_data.selection(row['sample_id'], row['partial_count'])
    selected = [[], partial]
    result = []
    for index, chosen in enumerate(selected):
        rois = sender.grid_rois(*source.shape[1:3])
        errors = []
        for x,y,w,h in rois:
            difference = source[:,y:y+h,x:x+w].astype(np.float32)-received[index][:,y:y+h,x:x+w]
            errors.append(float(np.mean(difference*difference)))
        remaining = [i for i in range(16) if i not in chosen]
        first = min(remaining,key=lambda i:(-errors[i]/int(costs[i]),i))
        logcost = np.log(np.asarray(costs,dtype=float))
        span = max(float(logcost.max()-logcost.min()),1e-12)
        def diversity(i):
            return ((abs(i//4-first//4)+abs(i%4-first%4))/6.
                    +abs(logcost[i]-logcost[first])/span,-i)
        second = max((i for i in remaining if i != first),key=diversity)
        other = [i for i in remaining if i not in (first,second)]
        seed = int(hashlib.sha256(f'latent-sender/{row["sample_id"]}/{index}'.encode()).hexdigest()[:16],16)
        third = other[int(np.random.default_rng(seed).integers(len(other)))]
        result.append(dict(selected=chosen,candidates=[first,second,third],
            kind='empty' if not chosen else 'deterministic_random_prefix',
            candidate_roles=['source_reconstruction_error_per_byte','spatial_price_diversity','random']))
    return result


def make_protocol(smoke=False):
    from demo.routervc_receiver_router import load_model
    from routervc.latent.generation import assets, asset_hash
    from routervc.latent.packet_codec import packet_hash
    completed = RECEIVER.parent.parent/'complete.json'
    audit = receiver_data.ROOT/'receiver_audit_20261009/complete.json'
    verification = read(audit)
    verify_artifacts(audit.parent,verification['artifacts'])
    training = read(completed)
    if (not training['complete'] or training['updates'] != 11520
            or training['artifacts']['core/best.pt'] != RECEIVER_SHA):
        raise ValueError('explicit completed adapted receiver is required')
    model, payload = load_model(RECEIVER,expected_sha256=RECEIVER_SHA)
    if payload['binding']['protocol']['smoke'] or payload['arm'] != 'core':
        raise ValueError('do not bind smoke/halo receiver')
    del model
    rows = receiver_data.rows(False)
    counters = {}
    counts = {}
    for row in sorted(rows,key=lambda r:(r['dataset'],r['router_split'],r['sample_id'])):
        key = row['dataset'],row['router_split']
        i = counters.get(key,0); counters[key] = i+1
        counts[row['sample_id']] = (2,4,8,12)[i%4]
    rows = [dict(row,partial_count=counts[row['sample_id']]) for row in rows]
    if smoke:
        rows = [next(r for r in rows if r['dataset']==d and r['router_split']=='train'
                     and r['partial_count']==c) for d,c in (('REDS',2),('UVG',12))]
    for row in rows:
        bank_dir = receiver_data.ROOT/'formal/samples'/row['sample_id']
        encoded = read(bank_dir/'encoded.json')
        oldrow = {k:v for k,v in row.items() if k!='partial_count'}
        if encoded['binding']['row'] != oldrow or encoded['binding']['packet_profile'] != packet_hash():
            raise ValueError('new-latent bank/source binding changed')
        verify_artifacts(bank_dir,encoded['artifacts'])
        row['bank_path'] = str(bank_dir/'bank.rvlp')
        row['bank_sha256'] = encoded['artifacts']['bank.rvlp']
        row['encoded_path'] = str(bank_dir/'encoded.json')
        row['encoded_sha256'] = digest(bank_dir/'encoded.json')
    hashes = assets()
    return dict(format=fit.FORMAT, revision='new_latent_final_marginal_v1',smoke=smoke,rows=rows,
        code={n:digest(REPO/n) for n in sorted(set(CODE))},
        receiver_profile=routing.identity(),packet_profile=packet_hash(),
        teacher=dict(sender_config=asdict(sender.Config()),
            receiver=dict(path=str(RECEIVER),sha256=RECEIVER_SHA,
                complete_path=str(completed),complete_sha256=digest(completed),audit_sha256=digest(audit)),
            G_assets=hashes,G_assets_hash=asset_hash(hashes),max_g=8,seed=routing.SEED),
        arms=list(fit.ARMS),epochs=2 if smoke else 120,seed=20261009,
        learning_rate=1e-4,weight_decay=1e-4,ranking_weight=.1,
        initialization='paired seeded random independent R_s weights; not R_g or old E teacher weights',
        label_scope=sender.LABEL_SCOPE,measured_renderings_per_window=8,measured_marginals_per_window=6,
        parents='empty plus deterministic random E2/E4/E8/E12 balanced within dataset/split',
        actions='two actual P8 E packets for one latent region; no I enhancement packet',
        candidate_proposals='source-Y MSE per byte, spatial-price diversity, uniform random; not target labels',
        G_randomness='region addressed; every selected G reads identical ungenerated Y',
        labels='whole-view LPIPS(parent final)-LPIPS(child final), each reruns fixed R_g and G',
        source_used='sender inputs and offline scoring only',negative_marginals_retained=True,
        unknown_candidates_masked=True,old_labels_reused=False,transmitted_masks=False,
        semantic_supervision=False,freeze_UF_E_G_and_Rg=True,
        input_cache='lossless FP32 compressed tensors on file store; LRU two windows',
        full_allocation_evaluation_pending=True)


class Samples:
    def __init__(self,root,cache,protocol,check=lambda:None,progress=lambda **kw:None):
        self.root,self.cache,self.protocol = Path(root),Path(cache),protocol
        self.check,self.progress = check,progress
        self.memory=OrderedDict()
        self.model=self.generator=self.metric=None
        self.measured=0
        self.stop_after_renderings=0

    def _measure(self,folder,bank,selected,source,encoded):
        from demo.routervc_receiver_router import load_model
        from demo.four_state_receive import PersistentRGB,codec_precision
        from demo.stage_c_three_path_roi_probe import LPIPSAlex
        from demo.scalable_experiment import quality
        folder.mkdir(parents=True,exist_ok=True)
        teacher=self.protocol['teacher']
        inner=routing.subset(bank,selected)
        wire=routing.wrap(inner,teacher['receiver']['sha256'],teacher['G_assets_hash'],
                          max_g=teacher['max_g'],seed=teacher['seed'])
        binding=dict(protocol=digest(self.root/'protocol.json'),selected=selected,
            bank_sha256=hashlib.sha256(bank).hexdigest(),source_hash=frame_hash(source))
        if (folder/'result.json').exists():
            result=read(folder/'result.json')
            if (result['binding']!=binding or result['stream_sha256']!=hashlib.sha256(wire).hexdigest()
                    or result['total_bytes']!=len(wire)):
                raise ValueError('saved teacher state changed')
            verify_artifacts(folder,result['artifacts'])
            return result
        self.check(); began=time.monotonic()
        atomic_bytes(folder/'stream.rvlrg',wire)
        base,received,detail=receiver_data.decode(inner)
        if frame_hash(base)!=encoded['base_hash'] or detail['base_reference_hashes']!=encoded['base_reference_hashes']:
            raise ValueError('E changed B reference')
        _,config,_=routing.parse(wire)
        if self.model is None:
            self.model,_=load_model(teacher['receiver']['path'],expected_sha256=teacher['receiver']['sha256'])
        selection=routing.route(base,received,inner,config,self.model)
        if self.generator is None and selection['indices']:
            self.generator=PersistentRGB()
        if self.metric is None:
            self.metric=LPIPSAlex(True)
        with torch.no_grad():
            output,reports=routing.render(received,selection['indices'],teacher['G_assets'],
                self.generator,seed=teacher['seed'],check=self.check)
        with codec_precision():
            final=quality(source,output,self.metric)
            direct=quality(source,received,self.metric)
        if not all(np.isfinite(v) for scores in (final,direct) for v in scores.values()):
            raise ValueError('nonfinite measured quality')
        result=dict(complete=True,binding=binding,quality=final,Goff_quality=direct,route=selection,
            output_hash=frame_hash(output),base_hash=frame_hash(base),received_hash=frame_hash(received),
            total_bytes=len(wire),header_bytes=routing.HEADER_BYTES,stream_sha256=digest(folder/'stream.rvlrg'),
            runtime=reports,seconds=time.monotonic()-began,
            source_read_by_receiver=False,mask_bytes=0,actual_latent_mixture=True,
            artifacts={'stream.rvlrg':digest(folder/'stream.rvlrg')})
        save(folder/'result.json',result)
        self.measured+=1
        self.progress(phase='sender_final_teacher',measured_this_process=self.measured,state=str(folder))
        print('LATENT_SENDER_RENDER',str(folder),final,flush=True)
        if self.stop_after_renderings and self.measured>=self.stop_after_renderings:
            raise InterruptedError('intentional teacher state checkpoint test')
        return result

    def prepare(self,row):
        folder=self.root/'samples'/row['sample_id'];folder.mkdir(parents=True,exist_ok=True)
        if (folder/'complete.json').exists():
            return self.verify(row)
        self.check()
        if digest(row['bank_path'])!=row['bank_sha256'] or digest(row['encoded_path'])!=row['encoded_sha256']:
            raise ValueError('authenticated new-latent bank changed')
        source=receiver_data.source(row)
        bank=Path(row['bank_path']).read_bytes();encoded=read(row['encoded_path'])
        base,full,detail=receiver_data.decode(bank)
        if frame_hash(base)!=encoded['base_hash'] or frame_hash(full)!=encoded['full_hash']:
            raise ValueError('full-E endpoint changed')
        _,partial,_=receiver_data.decode(routing.subset(bank,receiver_data.selection(row['sample_id'],row['partial_count'])))
        costs=routing.bundle_bytes(bank)
        plan=plans(row,source,[base,partial],costs)
        inputs=[];records=[]
        for state,(p,y) in enumerate(zip(plan,(base,partial))):
            inputs.append(sender.build_inputs(source,base,y,full,
                routing.coverage(routing.packets.parse(routing.subset(bank,p['selected']))),
                np.asarray(costs,dtype=np.int64),self.protocol['teacher']['max_g']))
            parent=self._measure(folder/f's{state}/parent',bank,p['selected'],source,encoded)
            if parent['received_hash']!=frame_hash(y):
                raise ValueError('sender input differs from teacher received image')
            children={str(i):self._measure(folder/f's{state}/add{i}',bank,p['selected']+[i],source,encoded)
                      for i in p['candidates']}
            records.append(dict(parent=parent,children=children))
        # Reuse only the generic final-difference contract, not any old labels or RGB mixtures.
        from demo.routervc_sender_data import measured_targets
        target,diagnostics=measured_targets(plan,records,np.asarray(costs,dtype=np.int64))
        packed={f'input_{k}':torch.cat([i[k] for i in inputs]).numpy() for k in inputs[0]}
        packed.update(value=target['value'].numpy(),weight=target['weight'].numpy())
        cache=self.cache/(row['sample_id']+'.npz');cache.parent.mkdir(parents=True,exist_ok=True)
        atomic_npz(cache,**packed)
        save(folder/'plans.json',dict(plans=plan,diagnostics=diagnostics,packet_bytes=costs))
        names=['plans.json',*(f's{i}/{n}/result.json' for i,p in enumerate(plan)
              for n in ['parent',*(f'add{j}' for j in p['candidates'])])]
        done=dict(complete=True,binding=dict(protocol=digest(self.root/'protocol.json'),row=row),
            cache_path=str(cache),cache_sha256=digest(cache),
            measured_final_renderings=8,measured_marginals=6,label_scope=sender.LABEL_SCOPE,
            current_sender_policy_used=False,old_labels_reused=False,
            artifacts={n:digest(folder/n) for n in names})
        save(folder/'complete.json',done)
        return done

    def verify(self,row):
        folder=self.root/'samples'/row['sample_id'];done=read(folder/'complete.json')
        if (done['binding']!=dict(protocol=digest(self.root/'protocol.json'),row=row)
                or digest(done['cache_path'])!=done['cache_sha256']
                or digest(row['bank_path'])!=row['bank_sha256']):
            raise ValueError('sender cache/bank/binding changed')
        verify_artifacts(folder,done['artifacts'])
        return done

    def get(self,row,generate=False):
        sid=row['sample_id']
        if sid not in self.memory:
            done=self.prepare(row) if generate else self.verify(row)
            with np.load(done['cache_path'],allow_pickle=False) as f:
                inputs={k[6:]:torch.from_numpy(f[k].copy()) for k in f.files if k.startswith('input_')}
                target={k:torch.from_numpy(f[k].copy()) for k in ('value','weight')}
            target['label_scope']=sender.LABEL_SCOPE
            plan=read(self.root/'samples'/sid/'plans.json')
            self.memory[sid]=dict(inputs=inputs,targets=target,plans=plan['plans'],
                packet_bytes=inputs['packet_bytes'].squeeze(-1))
        self.memory.move_to_end(sid)
        while len(self.memory)>2:
            self.memory.popitem(last=False)
        return self.memory[sid]

    def release(self):
        self.generator=self.metric=self.model=None
        self.memory.clear();gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
