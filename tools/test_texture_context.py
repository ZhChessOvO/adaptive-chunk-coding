import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from routervc.generation.context_geometry import geometry, rgb_crop, local_core, master_noise, noise_crop, window
from tools.texture_context_report import overview


class GeometryTests(unittest.TestCase):
    def test_actual_shapes_are_nested_and_same_native_pixels(self):
        for shape,old in (((17,512,512,3),(256,128,128,128)),
                          ((17,576,1024,3),(256,288,256,144))):
            geo=geometry(shape,old);a=np.arange(np.prod(shape),dtype=np.uint32).reshape(shape)
            for key in ('small','large'):
                np.testing.assert_array_equal(local_core(rgb_crop(a,geo[key]),geo[key],geo['core']),rgb_crop(a,geo['core']))
            self.assertEqual(geo['core'][2:],[128,128])

    def test_noise_is_spatially_paired_not_just_same_seed(self):
        geo=geometry((17,512,512,3),(256,128,128,128));master=master_noise(123)
        small=noise_crop(master,geo['small'],geo['large']);large=noise_crop(master,geo['large'],geo['large'])
        dx,dy=geo['small'][0]//8,geo['small'][1]//8
        torch.testing.assert_close(small,large[:,dy:dy+32,dx:dx+32],rtol=0,atol=0)
        torch.testing.assert_close(master,master_noise(123),rtol=0,atol=0)
        self.assertFalse(torch.equal(master,master_noise(124)))

    def test_rejects_padding_or_resizing(self):
        with self.assertRaises(ValueError):window((17,400,600,3),(128,128,128,128),512)
        with self.assertRaises(ValueError):noise_crop(master_noise(1),(1,0,256,256),(0,0,512,512))

    def test_overview_uses_real_source_and_marks_uvg_crop(self):
        with tempfile.TemporaryDirectory(prefix='context-figure-') as tmp:
            p=Path(tmp);v=np.full((17,512,512,3),120,np.uint8)
            c=dict(sample_id='SYNTHETIC FIXTURE',dataset='UVG')
            overview(p,c,v,v,geometry(v.shape,(256,128,128,128)))
            self.assertGreater((p/'overview.png').stat().st_size,1000)


if __name__=='__main__':unittest.main()
