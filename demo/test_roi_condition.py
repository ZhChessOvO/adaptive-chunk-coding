"""CPU invariants for RGB-first ROI training and its independent cache."""
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from demo import roi_condition_data as data
from demo.joint_condition_train import crop_for_packets as old_crop
from demo.roi_condition_train import crop_for_packets as new_crop, learning_rate
from demo.joint_condition_train import learning_rate as old_lr
from demo.scalable_codec import file_hash


class GeometryTests(unittest.TestCase):
    def test_halo_and_native_geometry(self):
        self.assertEqual(data.context_crop((17,512,512,3),(0,17,128,128,128,128)),(64,64,256,256))
        self.assertEqual(data.context_crop((33,256,384,3),(0,33,96,64,96,64)),(32,0,224,192))

    def test_frame_edge_clipping(self):
        self.assertEqual(data.context_crop((17,512,512,3),(0,17,0,0,128,128)),(0,0,192,192))

    def test_bad_geometry_rejected(self):
        for region in ((0,17,129,128,128,128),(0,18,0,0,128,128),(0,17,480,0,128,128)):
            with self.assertRaises(ValueError): data.context_crop((17,512,512,3),region)

    def test_rgb_crop_is_exact_and_independent(self):
        a=np.arange(17*64*64*3,dtype=np.uint8).reshape(17,64,64,3)
        b=data.crop_rgb(a,(8,16,32,32))
        np.testing.assert_array_equal(b,a[:,16:48,8:40])
        b[:]=0
        self.assertTrue(a[:,16:48,8:40].any())

    def test_crop_rejects_resizing_or_wrong_dtype(self):
        a=np.zeros((17,64,64,3),dtype=np.uint8)
        with self.assertRaises(ValueError): data.crop_rgb(a,(0,0,31,32))
        with self.assertRaises(ValueError): data.crop_rgb(a.astype(np.float32),(0,0,32,32))

    def test_same_crop_schedule_and_learning_rate(self):
        packets=[dict(delta=torch.ones(1),start=1,roi=(32,48,128,128))]
        for seed in range(12):
            self.assertEqual(new_crop(random.Random(seed),packets),old_crop(random.Random(seed),packets))
        for step in (1,100,1000,3000): self.assertEqual(learning_rate(step,3000),old_lr(step,3000))

    def test_image_loss_only_central_core(self):
        target=torch.zeros(17,3,256,256)
        pred=torch.ones_like(target,requires_grad=True)
        terms=data.core_image_terms(pred,target,lambda a,b:(a-b).abs().mean())
        sum(terms.values()).backward()
        self.assertEqual(int(torch.count_nonzero(pred.grad[...,:64,:])),0)
        self.assertEqual(int(torch.count_nonzero(pred.grad[...,192:,:])),0)
        self.assertGreater(int(torch.count_nonzero(pred.grad[...,64:192,64:192])),0)


class CacheTests(unittest.TestCase):
    def test_cache_reuses_condition_and_target_without_encoding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'source.npz'; received=root/'received.npz'
            rgb=np.zeros((17,256,256,3),dtype=np.uint8)
            np.savez(source,source=rgb); np.savez(received,full=rgb+10)
            entry=dict(sample_id='test',pair_hash=file_hash(source),pair_path=str(source))
            input_row=dict(path=str(received),sha256=file_hash(received))
            clean=torch.zeros(5,32,32,16,dtype=torch.bfloat16)
            raw=torch.ones_like(clean)
            with patch.object(data,'encode_roi',side_effect=[clean,raw]) as encode:
                a,key=data.cached_pair(root/'cache',None,entry,input_row,(0,0,256,256),'full','vae','cpu')
                self.assertEqual(encode.call_count,2)
            with patch.object(data,'encode_roi',side_effect=AssertionError('recomputed')):
                b,same=data.cached_pair(root/'cache',None,entry,input_row,(0,0,256,256),'full','vae','cpu')
            self.assertEqual(key,same)
            torch.testing.assert_close(a['raw'],b['raw'],rtol=0,atol=0)
            torch.testing.assert_close(a['clean'],b['clean'],rtol=0,atol=0)
            (root/'cache'/f'{key}.pt').write_bytes(b'corrupt test fixture')
            with self.assertRaises(AssertionError):
                data.cached_pair(root/'cache',None,entry,input_row,(0,0,256,256),'full','vae','cpu')


if __name__ == '__main__': unittest.main()
