"""CPU tests of report grouping and image/video output, using synthetic pixels."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from demo.scalable_codec import atomic_npz
from tools.texture_repair_evaluate import aggregate, draw


class EvaluationTests(unittest.TestCase):
    def test_temporal_phase_detects_cancelling_grid(self):
        from tools.texture_repair_review import temporal_phase
        source=np.full((4,128,128,3),128,np.float32)
        wave=np.sin(np.arange(128,dtype=np.float32)*2*np.pi/16)[None,None,:,None]*10
        stationary=temporal_phase(source+wave,source)
        changing=temporal_phase(source+wave*np.array([1,-1,1,-1])[:,None,None,None],source)
        self.assertLess(changing['stationary_rms'],1e-4)
        self.assertGreater(changing['per_frame_rms_mean'],1)
        self.assertAlmostEqual(changing['per_frame_rms_mean'],stationary['per_frame_rms_mean'],places=5)
        self.assertGreater(changing['phase_frame_difference'],stationary['phase_frame_difference']+1)

    def test_temporal_phase_source_identity(self):
        from tools.texture_repair_review import temporal_phase
        source=np.random.default_rng(12).integers(0,256,(4,128,128,3),dtype=np.uint8)
        result=temporal_phase(source,source)
        self.assertEqual(result['per_frame_rms_mean'],0)

    def test_groups_are_not_pooled(self):
        result=dict(name='initial',quality=dict(lpips_alex=.2,psnr_db=30.,temporal_delta_mae=.1),
                    error_phase16=dict(phase_rms_u8=1.),phase16=dict(phase_rms_u8=2.))
        entries=[dict(case=dict(evaluation_group=g,dataset=d),results=[result])
                 for g,d in [('known_artifact','UVG'),('internal_validation','UVG'),
                             ('internal_validation','REDS'),('internal_validation','REDS')]]
        rows=aggregate(entries)
        self.assertEqual(len(rows),3)
        self.assertEqual({(r['group'],r['dataset']):r['n'] for r in rows},
            {('known_artifact','UVG'):1,('internal_validation','UVG'):1,('internal_validation','REDS'):2})

    def test_real_report_writer_on_synthetic_fixture(self):
        import subprocess,json
        with tempfile.TemporaryDirectory(prefix='texture-report-test-') as directory:
            folder=Path(directory)
            source=np.full((17,128,128,3),120,np.uint8)
            source[:,40:80,50:90]=180
            names=('initial','control_000400','repair_000400')
            results={}
            for name in names:
                (folder/name).mkdir();atomic_npz(folder/name/'output.npz',pixels=source)
                results[name]=dict(quality=dict(lpips_alex=.1,psnr_db=30.))
            case=dict(sample_id='SYNTHETIC UNIT TEST',region=0,crop=[0,0,128,128],core=[0,0,128,128])
            draw(folder,case,source,source,results,400)
            self.assertGreater((folder/'comparison.png').stat().st_size,1000)
            probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
                '-show_entries','stream=nb_frames,width,height','-of','json',str(folder/'comparison.mp4')]))
            self.assertEqual(int(probe['streams'][0]['nb_frames']),17)
            self.assertEqual(probe['streams'][0]['width'],1280)


if __name__=='__main__': unittest.main()
