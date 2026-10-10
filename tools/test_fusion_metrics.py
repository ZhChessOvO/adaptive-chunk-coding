import unittest
import numpy as np
import torch
from routervc.fusion.boundaries import edges
from routervc.fusion.metrics import crop_sites, boundary_lpips


class MetricsTest(unittest.TestCase):
    def test_eligible_frames_and_exact_identity(self):
        shape=(17,256,256,3)
        boundary=edges(shape,[(0,0)],[0,1])
        sites=crop_sites(shape,boundary)
        self.assertTrue(all(t==8 for t,x,y in sites['E_nonE']))
        pixels=np.full(shape,100,np.uint8)
        def metric(a,b):return (a-b).square().mean((1,2,3),keepdim=True)
        result=boundary_lpips(pixels,{'same':pixels},boundary,metric)
        self.assertTrue(all(v['same']==0 for v in result['metrics'].values()))

    def test_empty_categories(self):
        shape=(17,256,256,3);x=np.zeros(shape,np.uint8)
        result=boundary_lpips(x,{'same':x},[],lambda a,b:None)
        self.assertEqual(result['unique_crops'],0)
        self.assertTrue(all(v['same'] is None for v in result['metrics'].values()))


if __name__=='__main__':unittest.main()
