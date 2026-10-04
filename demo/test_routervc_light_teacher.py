"""CPU label contracts: only E/EG change, with measured q2 costs and replay."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from demo import routervc_light_teacher as teacher
from demo.routervc_light_packets import bank_info
from demo.test_routervc_encode import make_bank
from demo.scalable_codec import atomic_npz,atomic_bytes


class LightTeacherTests(unittest.TestCase):
    def fixture(self,root):
        sid='synthetic';source=root/'source.npz';old=root/'old.json'
        base=np.zeros((17,64,96,3),np.uint8)
        atomic_npz(source,source=base+8)
        received=root/'received'/sid;encoded=root/'encoded'/sid
        received.mkdir(parents=True);encoded.mkdir(parents=True)
        atomic_npz(received/'received_E.npz',base=base,enhanced=base+4)
        bank=make_bank(base,q=2.);atomic_bytes(encoded/'bank.acse',bank);info=bank_info(bank)
        regions=[]
        for i,roi in enumerate(info['rois']):
            folder=received/f'cell_{i:02d}';folder.mkdir()
            atomic_npz(folder/'outputs.npz',EG=teacher.crop(base+6,roi))
            teacher.save(folder/'result.json',dict(control={},report={'runtime':{'seconds_model_load_excluded':.2}},
                         artifacts={'outputs.npz':teacher.digest(folder/'outputs.npz')}))
            score=lambda v:dict(lpips_alex=v,psnr_db=20+v,temporal_delta_mae=1+v)
            regions.append(dict(region=i,roi=roi,quality={s:score(v) for s,v in zip(('B','E','G','EG'),(.5,.4,.3,.2))},
                costs=dict(e_packet_bytes=9999,base_container_bytes=9999,individual_stream_bytes={}),g_seconds={'G':.1,'EG':.9}))
        teacher.save(old,dict(sample_id=sid,dataset='REDS',sequence='000',router_split='train',view_kind='synthetic',
            regions=regions,source_path=str(source),source_sha256=teacher.digest(source),seconds=999.,
            e_bank_decode_seconds=999.,e_timing_scope='old q1'))
        for p in (encoded/'complete.json',received/'complete.json'):teacher.save(p,{'complete':True})
        row=dict(sample_id=sid,rois=info['rois'],source_path=str(source),source_sha256=teacher.digest(source),
                 entry=dict(path=str(old),sha256=teacher.digest(old)))
        teacher.save(root/'protocol.json',dict(samples=[row]))
        return regions,info

    def test_reuse_B_G_only_and_real_bytes_and_readonly_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old,info=self.fixture(root)
            with patch('demo.stage_c_three_path_roi_probe.LPIPSAlex'), \
                    patch('demo.scalable_experiment.quality',return_value=dict(lpips_alex=.25,psnr_db=22.,temporal_delta_mae=.8)) as quality, \
                    patch('demo.scalable_cooperation_format.wrap',side_effect=lambda inner,_:b'control'+inner):
                teacher.make_labels(root)
                self.assertEqual(quality.call_count,32)
                label=teacher.read(root/'labels/synthetic.json')
                for i,r in enumerate(label['regions']):
                    self.assertEqual(r['quality']['B'],old[i]['quality']['B'])
                    self.assertEqual(r['quality']['G'],old[i]['quality']['G'])
                    self.assertEqual(r['quality']['E']['lpips_alex'],.25)
                    self.assertEqual(r['quality']['EG']['lpips_alex'],.25)
                    self.assertEqual(r['costs']['e_packet_bytes'],info['e_bytes'][i])
                self.assertNotIn('e_bank_decode_seconds',label)
                self.assertNotIn('seconds',label)
                self.assertFalse(label['semantic_labels_available'])
                before={p:(teacher.digest(p),p.stat().st_mtime_ns) for p in (root/'labels').glob('*.json')}
                teacher.make_labels(root)
                self.assertEqual(quality.call_count,32)
                self.assertEqual(before,{p:(teacher.digest(p),p.stat().st_mtime_ns) for p in (root/'labels').glob('*.json')})
                atomic_npz(root/'received/synthetic/received_E.npz',base=np.ones((17,64,96,3),np.uint8))
                with self.assertRaises(ValueError):teacher.make_labels(root)


if __name__=='__main__':unittest.main()
