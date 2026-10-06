"""CPU-only reporting contracts; synthetic plots are never research evidence."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from demo import routervc_sender_report as report
from demo import routervc_sender_evaluate as evaluation
from demo.scalable_codec import atomic_npz


class SenderReport(unittest.TestCase):
    def test_dataset_means_preserve_different_realized_rates(self):
        rows=[]
        for dataset,offset in (('REDS',0),('UVG',.1)):
            for arm,rate in (('fixed_old_E',.02),('source',.015),('zero_source',.01)):
                rows.append(dict(dataset=dataset,point=f'{arm}_e0.5_g8',bpp=rate,
                    lpips_alex=.3+offset,psnr_db=25.,temporal_delta_mae=4.))
        groups=report.aggregate(rows,[])
        self.assertEqual(groups['REDS']['source_e0.5_g8']['bpp'],.015)
        self.assertEqual(groups['REDS']['fixed_old_E_e0.5_g8']['bpp'],.02)
        self.assertAlmostEqual(groups['UVG']['source_e0.5_g8']['lpips_alex'],.4)
        self.assertEqual(groups['REDS']['source_e0.5_g8']['windows'],1)

    def test_fixed_images_include_every_supplied_sample(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);output=root/'report';output.mkdir()
            rows=[];baselines=[];protocol=dict(inputs={})
            pixels=np.full((17,64,96,3),100,np.uint8)
            for sid,dataset in (('reds','REDS'),('uvg','UVG')):
                source=root/(sid+'.npz');atomic_npz(source,source=pixels)
                protocol['inputs'][sid]=dict(source_path=str(source))
                for point in evaluation.point_plan():
                    dest=root/sid/point['name'];(dest/'fresh').mkdir(parents=True)
                    atomic_npz(dest/'fresh/reconstruction.npz',base=pixels,enhanced=pixels,reconstruction=pixels)
                    rows.append(dict(sample_id=sid,dataset=dataset,point=point['name'],
                        arm=point['arm'],ratio=point['ratio'],bpp=.007+point['ratio']*.04,
                        lpips_alex=.45-point['ratio']*.1,psnr_db=24.,temporal_delta_mae=4.,
                        output_folder=str(dest),Goff=dict(lpips_alex=.5),E_indices=[0],G_indices=[0,2]))
                for i,name in enumerate([f'uf_qp{q}' for q in (8,16,24,32,40,48,56)]+['wholeframe_g_one_roi']):
                    baselines.append(dict(dataset=dataset,point=name,bpp=.005*(i+1),
                        lpips_alex=.5-.02*i,psnr_db=24.+i,temporal_delta_mae=4.))
            groups,pictures=report.figures(protocol,rows,baselines,output)
            self.assertEqual(set(groups),{'REDS','UVG'})
            self.assertEqual([p['sample_id'] for p in pictures],['reds','uvg'])
            self.assertTrue((output/'rd_senders.png').is_file())
            for picture in pictures:
                self.assertEqual(picture['fixed_frame'],8)
                for key in ('image','routes'):
                    with Image.open(picture[key]) as image:
                        self.assertGreater(image.width,1000)
                        image.verify()


if __name__=='__main__':unittest.main()
