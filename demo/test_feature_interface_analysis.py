import unittest
import torch
from demo.feature_interface_analysis import rounding_comparison


class NumericsTests(unittest.TestCase):
    def test_fp32_nonzero_can_be_lost_at_bf16_entry(self):
        raw=torch.ones(5,2,2,16)
        side=torch.full_like(raw,.0001)
        coverage=torch.ones(5,1,2,2)
        stats=rounding_comparison(raw,side,coverage,side.bfloat16())
        self.assertEqual(stats['raw_dtype'],'torch.float32')
        self.assertEqual(stats['effective_rms'],0.)
        self.assertEqual(stats['rounded_away_fraction'],1.)

    def test_representable_change_survives(self):
        raw=torch.ones(5,2,2,16); side=torch.full_like(raw,.125)
        stats=rounding_comparison(raw,side,torch.ones(5,1,2,2),side.bfloat16())
        self.assertEqual(stats['effective_rms'],.125)
        self.assertEqual(stats['train_style_difference_rms'],0.)
        self.assertEqual(stats['covered_changed_fraction'],1.)

    def test_no_packet_exact_and_no_input_mutation(self):
        raw=torch.randn(5,2,2,16); before=raw.clone(); side=torch.zeros_like(raw)
        stats=rounding_comparison(raw,side,torch.zeros(5,1,2,2),side.bfloat16())
        self.assertEqual(stats['effective_rms'],0.)
        self.assertEqual(stats['covered_changed_fraction'],0.)
        self.assertTrue(stats['outside_coverage_exact'])
        torch.testing.assert_close(raw,before,rtol=0,atol=0)


if __name__=='__main__': unittest.main()
