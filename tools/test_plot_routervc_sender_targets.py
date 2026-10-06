"""Interpretation and provenance tests for inference-free sender-label plots."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from tools import plot_routervc_sender_targets as report


def records():
    parent=dict(binding=dict(selected=[]),total_bytes=100,E_packet_bytes=0,
        Goff_quality=dict(lpips_alex=.4),quality=dict(lpips_alex=.2,psnr_db=25.),route=dict(indices=[0,1]))
    child=dict(binding=dict(selected=[3]),total_bytes=200,E_packet_bytes=100,
        Goff_quality=dict(lpips_alex=.3),quality=dict(lpips_alex=.22,psnr_db=25.2),route=dict(indices=[1,0]))
    return parent,child


class TargetDiagnostics(unittest.TestCase):
    def test_negative_final_is_retained_despite_direct_improvement(self):
        row=report.marginal(*records())
        self.assertAlmostEqual(row['direct_lpips_gain'],.1)
        self.assertAlmostEqual(row['final_lpips_gain'],-.02)
        self.assertAlmostEqual(row['gain_difference'],-.12)
        self.assertEqual(row['incremental_bytes'],100)
        self.assertTrue(row['G_order_changed']);self.assertFalse(row['G_set_changed'])

    def test_only_order_and_changed_set_are_distinct(self):
        a,b=records();b['route']['indices']=[1,2]
        self.assertTrue(report.marginal(a,b)['G_set_changed'])

    def test_nonappend_state_and_incorrect_bytes_rejected(self):
        for field,value,pattern in (('binding',dict(selected=[3,4]),'exactly one'),
                                    ('E_packet_bytes',99,'increment')):
            with self.subTest(field=field):
                a,b=records();b[field]=value
                with self.assertRaisesRegex(ValueError,pattern):report.marginal(a,b)

    def test_nonfinite_metric_rejected(self):
        a,b=records();b['quality']['lpips_alex']=float('nan')
        with self.assertRaisesRegex(ValueError,'finite'):report.marginal(a,b)

    def test_dataset_split_state_groups_stay_separate(self):
        base=dict(sample_id='s',dataset='REDS',split='train',state='empty',**report.marginal(*records()))
        second=deepcopy(base);second.update(dataset='UVG',split='validation',state='partial')
        groups=report.aggregate([base,second])
        self.assertEqual(set(groups),{'REDS/train/empty','UVG/validation/partial'})
        for group in groups.values():
            self.assertEqual(group['negative_final'],1)
            self.assertEqual(group['direct_positive_final_negative'],1)
            self.assertEqual(group['G_order_only_changed'],1)

    def test_missing_label_boundary_fails_without_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);report.save(root/'protocol.json',{})
            with self.assertRaises(FileNotFoundError):report.collect(root)

    def test_figure_generation_is_cpu_only_and_supports_empty_subgroups(self):
        row=dict(sample_id='s',dataset='REDS',split='train',state='empty',**report.marginal(*records()))
        with tempfile.TemporaryDirectory() as tmp:
            report.figures([row],Path(tmp))
            self.assertGreater((Path(tmp)/'direct_vs_final.png').stat().st_size,10000)


if __name__=='__main__':unittest.main()
