"""CPU-only checks for paired protocol, baseline conversion and audit failures."""
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch
import numpy as np
import torch

from demo import routervc_receiver_router as old
from demo.chunk_enhancement_codec import configure_torch
from routervc.cooperation import receiver
from tools.cooperative_review import same_receipt
from tools.cooperative_review_report import aggregate
from tools.cooperative_review_worker import sender_guard


class ReviewTests(TestCase):
    def test_conversion_is_state_independent_and_same_ranking(self):
        configure_torch();torch.manual_seed(1011)
        core=old.ReceiverGUtilityRouter()
        with patch.object(old,'load_model',return_value=(core,{})):
            model=receiver.initialize(Path('unused'),'unused').eval()
        c=model.config
        inputs=dict(local_pairs=torch.rand(1,16,c.temporal_frames,6,c.local_size,c.local_size),
            global_pairs=torch.rand(1,c.temporal_frames,6,c.global_size,c.global_size),
            geometry=torch.rand(1,16,6),coverage=torch.rand(1,16,1))
        # Geometry has the seven total geometry/coverage features expected by the old core.
        with torch.no_grad():
            expected=core(inputs)/16
            embedding=model.encode(inputs)
            for n in (0,3,8):
                selected=torch.zeros(1,16,1);selected[:,:n]=1
                actual=model.from_embedding(embedding,selected)
                torch.testing.assert_close(actual,expected,atol=1e-7,rtol=1e-5)
                self.assertEqual(actual[0,:,0].argsort().tolist(),expected[0,:,0].argsort().tolist())

    def test_receipt_checks_bytes_access_and_off(self):
        from demo.routervc_fullview_probe import digest
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'test.bin';path.write_bytes(b'abc')
            r=dict(complete=True,stream=digest(path),actual_bytes=3,header_bytes=121,mask_bytes=0,
                   source_frames_read=False,sender_router_loaded=False,generation_disabled=False,
                   base_hash='B',enhanced_hash='Y',output_hash='G',generated=[1],selected={})
            same_receipt(r,path,r)
            for key,value in [('mask_bytes',1),('source_frames_read',True),('actual_bytes',4),('header_bytes',198)]:
                with self.assertRaises(ValueError):same_receipt(dict(r,**{key:value}),path,r)
            off=dict(r,generation_disabled=True,output_hash='Y',generated=[],selected=None)
            same_receipt(off,path,r,disabled=True)
            with self.assertRaises(ValueError):same_receipt(dict(off,output_hash='G'),path,r,disabled=True)

    def test_sender_guard(self):
        for name in ('routervc_sender_20261005','routervc_latent_sender_20261009'):
            with self.assertRaises(RuntimeError):sender_guard('open',(f'/runs/{name}/best.pt',))
        sender_guard('open',('/runs/routervc_cooperation_review_20261011/old_policy.pt',))
        sender_guard('open',(42,))

    def test_summary_separates_cohorts_and_keeps_negative_results(self):
        points=[]
        for cohort in ('diagnostic','grouped_validation'):
            for dataset in ('REDS','UVG'):
                for cap in ((0.,.25,.5,1.) if cohort=='diagnostic' else (.25,)):
                    scores={}
                    for arm,value in [('old_policy',.3),('cooperative',.31 if dataset=='UVG' else .29)]:
                        scores[arm]=dict(quality=dict(lpips_alex=value,psnr_db=25.,temporal_delta_mae=2.),
                            bpp=.1,actual_bytes=100, fresh_seconds=30.,G_calls=8,peak_GiB=10.,reserved_GiB=11.)
                    points.append(dict(cohort=cohort,dataset=dataset,cap=cap,scores=scores))
        result=aggregate(points);self.assertEqual(len(result),10)
        for r in result:self.assertEqual(r['worsened'],int(r['dataset']=='UVG'))
