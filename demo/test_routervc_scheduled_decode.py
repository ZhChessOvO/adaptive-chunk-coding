import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from demo import routervc_scheduled_decode as s


class SeedTests(unittest.TestCase):
    def test_unmerged_seed_slot_does_not_shift_after_merge(self):
        slots=[0,2,3,5]
        self.assertEqual(s.remap_seed(100+65536*2+3,100,slots),100+65536*3+3)
        self.assertEqual(s.remap_seed(100+65536*3,100,slots),100+65536*5)

    def test_invalid_mapping_fails(self):
        with self.assertRaises(ValueError):s.remap_seed(99,100,[0])
        with self.assertRaises(ValueError):s.remap_seed(100+65536,100,[0])

    def test_peak_accumulator_does_not_lose_loading_peak_at_roi_reset(self):
        import torch
        def simulated_receiver(args):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_peak_memory_stats()
            return dict(peak_cuda_allocated_bytes=6,generation_runtime=dict(windows=[
                dict(runtime=dict(peak_cuda_allocated_bytes=7)),
                dict(runtime=dict(peak_cuda_allocated_bytes=9))]))
        with patch('torch.cuda.max_memory_allocated',side_effect=[3,17,6]), \
                patch('torch.cuda.max_memory_reserved',side_effect=[4,20,10]), \
                patch('torch.cuda.reset_peak_memory_stats') as reset, \
                patch('demo.routervc_visual_decode.receive',side_effect=simulated_receiver), \
                patch.object(s,'atomic_json'):
            result=s.receive(SimpleNamespace(output=Path('/unused')))
        self.assertEqual(reset.call_count,2)
        self.assertEqual(result['last_call_cuda_peak_bytes'],6)
        self.assertEqual(result['peak_cuda_allocated_bytes'],17)
        self.assertEqual(result['peak_cuda_reserved_bytes'],20)
        self.assertEqual(result['generation_all_calls_peak_cuda_bytes'],9)


if __name__=='__main__':unittest.main()
