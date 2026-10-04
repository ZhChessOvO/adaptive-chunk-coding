import unittest
from unittest.mock import patch
import numpy as np
import torch
from demo import routervc_light_packets as light
from demo.test_routervc_encode import make_bank
from demo.routervc_encode import bank_info as old_bank_info
from demo.scalable_format import parse


class LightPacketTests(unittest.TestCase):
    def setUp(self):
        self.base = np.zeros((17,64,96,3),np.uint8)
        self.enhanced = np.full_like(self.base,32)
        self.bank = make_bank(self.base,q=2.)

    def test_q2_bank_and_unchanged_q1_contract(self):
        info = light.bank_info(self.bank)
        self.assertEqual(len(info['groups']),16)
        self.assertTrue(all(len(g) == 3 for g in info['groups']))
        with self.assertRaises(ValueError): old_bank_info(self.bank)
        with self.assertRaises(ValueError): light.bank_info(make_bank(self.base,q=1.))
        with self.assertRaises(ValueError): light.bank_info(make_bank(self.base,q=2.,skip=(9,5)))

    def test_real_prefix_and_byte_costs(self):
        small = light.subset_bank(self.bank,[5])
        large = light.subset_bank(self.bank,[5,1,15])
        self.assertTrue(large.startswith(small))
        self.assertEqual(parse(large).base,parse(self.bank).base)
        self.assertEqual(len(large)-parse(large).base_end,
                         sum(light.bank_info(self.bank)['e_bytes'][i] for i in (5,1,15)))
        for ids in ([1,1],[True],[-1],[16]):
            with self.assertRaises(ValueError): light.subset_bank(self.bank,ids)

    def test_actual_views_conditional_not_additive(self):
        seen = []
        def predict(model,base,received,coverage,rois):
            seen.append((received.copy(),coverage.copy()))
            value = torch.zeros(1,16,6)
            value[0,:,0] = torch.from_numpy(coverage)*.2
            value[0,:,1] = .1+torch.from_numpy(coverage)*.03
            return value
        with patch('demo.routervc_visual_policy._model',return_value=object()), \
                patch('demo.routervc_visual_policy.predict',side_effect=predict):
            utility,predictions = light.predict_utility(self.bank,self.base,self.enhanced,'model')
        self.assertEqual(predictions.shape,(17,16,6));self.assertEqual(len(seen),17)
        np.testing.assert_allclose(utility[:,3],.33,atol=1e-6)
        self.assertEqual(sum(c.sum() for _,c in seen),16)
        for i,(pixels,_) in enumerate(seen[1:]):
            x,y,w,h = light.bank_info(self.bank)['rois'][i]
            self.assertTrue(np.all(pixels[:,y:y+h,x:x+w] == 32))
            self.assertEqual(np.count_nonzero(pixels),17*w*h*3)
        with self.assertRaises(ValueError): light.predict_utility(self.bank,self.base+1,self.enhanced,'model')


if __name__ == '__main__': unittest.main()
