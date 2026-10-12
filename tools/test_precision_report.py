"""Small CPU fixture for the post-queue figures, without touching live outputs."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from demo.routervc_fullview_probe import save,digest,read
from demo.scalable_codec import atomic_npz
from tools.precision_report import report
from tools.precision_pilot import point_folder,bank_folder,RATIOS


class ReportTests(unittest.TestCase):
    def test_figures_and_common_delivery_exclude_infeasible_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=np.full((17,64,64,3),100,dtype=np.uint8)
            path=root/'source.png';Image.fromarray(source[0]).save(path)
            rows=[]
            for cohort in ('calibration','internal_evaluation'):
                for dataset in ('REDS','UVG'):
                    row=dict(sample_id=cohort+'_'+dataset,cohort=cohort,dataset=dataset,
                        shape=list(source.shape),files=[str(path)]*17,hashes=[digest(path)]*17)
                    rows.append(row)
                    for w in (3,5,7):
                        folder=bank_folder(root,row,w)
                        save(folder/'encoded.json',{'seconds':2.})
                        save(folder/'order.json',{'seconds':3.})
                        for cap in RATIOS:
                            dest=point_folder(root,row,w,cap);dest.mkdir(parents=True,exist_ok=True)
                            pixels=dest/'pixels.npz';atomic_npz(pixels,base=source,enhanced=source,reconstruction=source)
                            receipt=dict(stream='x',actual_bytes=100)
                            save(dest/'complete.json',receipt)
                            quality=dict(lpips_alex=.3-w*.01-cap*.05,psnr_db=25.,temporal_delta_mae=2.)
                            point=dict(cohort=cohort,dataset=dataset,sample_id=row['sample_id'],width=w,cap=cap,
                                bpp=.1+cap*.02,base_bytes=80,actual_bytes=100,quality=quality,
                                received_quality=quality,base_quality=quality,fresh_seconds=30.,
                                peak_GiB=8.,reserved_GiB=10.,receive=str(dest),stream_sha256='x',
                                reused_identical_stream=False,artifacts={str(pixels):digest(pixels)})
                            save(dest/'point.json',point)
                    if cohort=='internal_evaluation':
                        decisions=[]
                        for level in range(6):
                            decisions.append(dict(budget_bytes=80 if level<2 else 110,
                                automatic={'key':'w7_e050'},base_fraction='w5_e050',
                                fixed={str(w):None if level<2 and w==3 else f'w{w}_e050' for w in (3,5,7)}))
                        save(root/'decisions'/f'{row["sample_id"]}.json',dict(decisions=decisions))
            with patch('tools.precision_pilot.verify_point',return_value={}):
                names=report(root,dict(rows=rows,widths=[3,5,7]))
            self.assertEqual(len(names),8)
            for name in names:self.assertTrue((root/name).exists())
            summary=read(root/'summary.json')
            for entry in summary['budget_summary']:
                if entry['level']<2:
                    self.assertEqual(entry['common'],0)
                    self.assertIsNone(entry['lpips'])
            self.assertEqual(summary['logical_points'],48)


if __name__ == '__main__':unittest.main()
