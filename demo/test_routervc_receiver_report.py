"""Small synthetic CPU report checks; no formal measurements are created."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from demo import routervc_receiver_report as report
from demo import routervc_receiver_evaluate as evaluation


def rows_fixture():
    rows=[];baselines=[]
    for dataset,sid in (('REDS','reds-fixture'),('UVG','uvg-fixture')):
        for arm in ('shared','core','halo'):
            for ratio in evaluation.RATIOS:
                rows.append(dict(sample_id=sid,dataset=dataset,point=f'{arm}_e{ratio:g}_g8',arm=arm,
                    ratio=ratio,bytes=100+ratio*100,bpp=.01+ratio*.01,
                    lpips_alex=.3-ratio*.1+{'shared':0,'core':-.01,'halo':.01}[arm],
                    psnr_db=25.,temporal_delta_mae=1.,receiver_seconds=10.,policy_seconds=.1,
                    G_seconds=5.,G_peak_GiB=8.,worker_peak_GiB=9.,
                    E_indices=[] if ratio==0 else [1,2],G_indices=[3],E_packet_bytes=ratio*100))
        for qp in (8,16,24,32,40,48,56):
            baselines.append(dict(sample_id=sid,dataset=dataset,point=f'uf_qp{qp}',
                bytes=qp*10,bpp=qp/1000,lpips_alex=.5-qp*.005,psnr_db=20+qp*.1,temporal_delta_mae=1.))
        baselines.append(dict(sample_id=sid,dataset=dataset,point='wholeframe_g_one_roi',
            bytes=100,bpp=.01,lpips_alex=.2,psnr_db=25.,temporal_delta_mae=1.))
    return rows,baselines


class ReceiverReportTests(unittest.TestCase):
    def test_exact_fixed_payload_pairing_and_per_dataset_results(self):
        rows,baselines=rows_fixture()
        groups,comparison=report.aggregate(rows,baselines)
        self.assertEqual(set(groups),{'REDS','UVG','all_13_descriptive'})
        self.assertEqual(len(comparison['rows']),12)
        for dataset in ('REDS','UVG'):
            for state in ('e0','e0.25','e0.5'):
                core=comparison['groups'][dataset]['core'][state]
                self.assertAlmostEqual(core['mean_delta']['lpips_alex'],-.01)
                self.assertEqual(core['mean_delta']['bytes'],0)
                self.assertEqual(core['lower_lpips'],1)
        self.assertEqual(groups['all_13_descriptive']['core_e0_g8']['windows'],2)

    def test_reject_changed_E_and_preserve_actual_header_rate_delta(self):
        rows,baseline=rows_fixture()
        bad=deepcopy(rows)
        next(r for r in bad if r['arm']=='core')['E_indices']=[9]
        with self.assertRaises(ValueError):report.aggregate(bad,baseline)
        changed=next(r for r in rows if r['arm']=='core')
        changed['bytes']+=4;changed['bpp']+=.0004
        _,comparison=report.aggregate(rows,baseline)
        self.assertEqual(comparison['groups']['REDS']['core']['e0']['mean_delta']['bytes'],4)

    def test_rd_plot_is_generated_from_supplied_measured_table(self):
        rows,baseline=rows_fixture();groups,_=report.aggregate(rows,baseline)
        with tempfile.TemporaryDirectory() as folder:
            report.figures(groups,Path(folder))
            image=Path(folder)/'rd_fixed_E.png'
            self.assertTrue(image.exists());self.assertGreater(image.stat().st_size,1000)
        with self.assertRaises(ValueError):report.mean([],('bpp',))


if __name__=='__main__':unittest.main()
