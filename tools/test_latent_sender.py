import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from routervc.latent import sender as runtime,sender_data as data,routing
from tools.test_latent_routing import bank
from demo.routervc_sender_router import prefix_under_budget


class LatentSenderTests(unittest.TestCase):
    def test_plans_are_reproducible_and_exclude_received(self):
        source=np.zeros((17,64,64,3),np.uint8)
        source[:,:16,:16]=255
        received=[np.zeros_like(source),np.full_like(source,12)]
        for count in (2,4,8,12):
            row=dict(sample_id='source-window',partial_count=count)
            a=data.plans(row,source,received,[30]*16)
            self.assertEqual(a,data.plans(row,source,received,[30]*16))
            self.assertEqual(a[0]['candidates'][0],0)
            self.assertEqual(len(a[1]['selected']),count)
            for plan in a:
                self.assertEqual(len(set(plan['candidates'])),3)
                self.assertFalse(set(plan['selected'])&set(plan['candidates']))

    def test_conditional_order_decodes_actual_Y_and_stops(self):
        observed=[]
        source=np.zeros((17,64,64,3),np.uint8)
        def decode(raw):
            selected=sorted({region for chunk,region in routing.packets.parse(raw)['packets']})
            # Spatially coupled stand-in: adding a region changes all Y pixels.
            y=np.full_like(source,len(selected))
            return source.copy(),y,dict(base_reference_hashes=['same'])
        def predict(model,x,b,y,full,cov,costs,max_g):
            observed.append((float(y.mean()),cov.copy()))
            gains=np.full(16,-1.,np.float32)
            if not cov.any():gains[5]=.02
            elif cov[5] and not cov[9]:gains[9]=.01
            return torch.from_numpy(gains[None])
        result=runtime.ordering(source,bank(),None,dict(max_g=8),decode=decode,predict=predict)
        self.assertEqual(result['order'],[5,9])
        self.assertEqual([v[0] for v in observed],[0.,1.,2.])
        self.assertEqual(result['steps'][1]['selected'],[5])
        self.assertTrue(result['actual_mixed_Y_decoded'])
        self.assertFalse(result['G_executed_during_planning'])

    def test_actual_bundle_costs_preserve_prefix(self):
        raw=bank();costs=np.asarray(routing.bundle_bytes(raw),np.int64)
        previous=b''
        for budget in (0,int(costs[5]),int(costs[5]+costs[9])):
            result=prefix_under_budget([5,9],costs,budget)
            current=routing.subset(raw,result['indices'])
            self.assertTrue(current.startswith(previous));previous=current
            self.assertEqual(len(current)-len(routing.subset(raw,[])),result['packet_bytes'])

    def test_sender_rejects_old_representation_before_use(self):
        payload={'binding':{'protocol':{'revision':'old_feature_patch'}}}
        with patch.object(runtime.network,'load_model',return_value=(None,payload)):
            with self.assertRaises(ValueError):runtime.load('/unused')

    def test_current_receiver_is_explicit_not_inferred_from_best_filename(self):
        self.assertEqual(len(data.RECEIVER_SHA),64)
        self.assertEqual(data.RECEIVER_SHA,'62f02fe9a1d0fe008c0f93c5c6bf0747c7a470cf2d8eba90633795cf0bb7fe7b')


if __name__=='__main__':unittest.main()
