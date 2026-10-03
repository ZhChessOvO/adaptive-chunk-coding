"""CPU sender contract checks, with no native codec, CUDA or model downloads."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

import numpy as np
import torch

from demo import compact_enhancement_format as compact
from demo.routervc_encode import (bank_info, compose_candidates, load_input,
    prefix_order, prepare, select, subset_bank, validate_frames)
from demo.routervc_policy import grid_rois
from demo.scalable_codec import atomic_npz
from demo.scalable_format import frame_hash, parse


def make_bank(base, *, skip=None, q=1.):
    t, h, w, _ = base.shape
    meta = dict(compact.CONSTANTS, width=w, height=h, frame_count=t,
        enhancement_codec=compact.CODECS[-1], **{k: '0'*64 for k in compact.HASHES})
    meta['base_rgb_sha256'] = frame_hash(base)
    prefix = compact.base_container(b'opaque-native-base-payload', meta)
    wires = []
    for start, count in [(0, 1)] + [(s, 8) for s in range(1, t, 8)]:
        for index, roi in enumerate(grid_rois(h, w)):
            if skip == (start, index):
                continue
            pm = dict(packet_id=len(wires)+1, start=start, count=count,
                      roi=roi, qstep=q, codec=compact.CODECS[-1])
            wires.append(compact.packet_bytes(pm, bytes([index])*(index+1), codec=pm['codec']))
    return prefix+b''.join(wires)


class StubUtility(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = None

    def forward(self, values):
        self.seen = values.clone()
        result = torch.zeros((*values.shape[:2], 6))
        # Different cells have different deterministic direct E gain.
        result[..., 0] = values[..., 36] * torch.arange(1, 17) / 100
        result[..., 1] = .1 - .04*values[..., 36]
        return result


class SenderTests(unittest.TestCase):
    def setUp(self):
        self.base = np.zeros((17, 64, 96, 3), np.uint8)
        self.enhanced = np.full_like(self.base, 64)
        self.bank = make_bank(self.base)

    def test_geometry_and_explicit_input(self):
        self.assertEqual(validate_frames(self.base), (17, 64, 96))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'source.npz'
            atomic_npz(path, source=self.base)
            pixels, manifest = load_input(path)
            np.testing.assert_array_equal(pixels, self.base)
            self.assertEqual(manifest['source_rgb_sha256'], frame_hash(self.base))
            with self.assertRaises(ValueError):
                load_input(path, count=33)
        for shape in ((16, 64, 96, 3), (18, 64, 96, 3), (17, 65, 96, 3)):
            with self.assertRaises(ValueError):
                validate_frames(np.zeros(shape, np.uint8))
        with self.assertRaises(ValueError):
            validate_frames(self.base.astype(np.float32))

    def test_literal_prefixes_33_frame_rectangle(self):
        bank = make_bank(np.zeros((33, 64, 104, 3), np.uint8))
        small, large = subset_bank(bank, [5]), subset_bank(bank, [5, 0, 15])
        self.assertTrue(large.startswith(small))
        self.assertEqual(len(parse(small).packets), 5)
        self.assertEqual(parse(small).base, parse(bank).base)
        self.assertEqual([p.meta['start'] for p in parse(small).packets], [0, 1, 9, 17, 25])
        self.assertEqual(len(large)-len(subset_bank(bank, [])),
                         sum(bank_info(bank)['e_bytes'][i] for i in (5, 0, 15)))

    def test_bank_validation(self):
        for indices in ([0, 0], [-1], [16], [True]):
            with self.assertRaises(ValueError):
                subset_bank(self.bank, indices)
        for bank in (make_bank(self.base, skip=(9, 5)), make_bank(self.base, q=2.)):
            with self.assertRaises(ValueError):
                bank_info(bank)

    def test_isolated_views_and_no_source(self):
        model = StubUtility()
        result = select(self.bank, self.base, self.enhanced, model, 500, 4)
        self.assertEqual(model.seen.shape, (17, 16, 37))
        self.assertEqual(model.seen[0, :, 36].sum(), 0)
        self.assertEqual(model.seen[1:, :, 36].sum(), 16)
        for index in range(16):
            self.assertEqual(model.seen[index+1, index, 36], 1)
            self.assertTrue(torch.equal(model.seen[index+1, index, :18], model.seen[0, index, :18]))
        self.assertFalse(result['source_frames_used_by_router'])
        self.assertFalse(result['generation_mask_transmitted'])
        self.assertLessEqual(result['e_packet_bytes'], 500)
        self.assertEqual(result['e_packet_bytes'], len(subset_bank(self.bank, result['selected_indices']))
                         - parse(self.bank).base_end)

    def test_receiver_candidate_mixture(self):
        result = select(self.bank, self.base, self.enhanced, StubUtility(), 500, 0)
        mixed = compose_candidates(self.base, self.enhanced,
                                   result['selected_indices'], grid_rois(64, 96))
        self.assertEqual(frame_hash(mixed), result['mixed_rgb_sha256'])
        for i, (x, y, w, h) in enumerate(grid_rois(64, 96)):
            expected = 64 if i in result['selected_indices'] else 0
            self.assertTrue(np.all(mixed[:, y:y+h, x:x+w] == expected))

    def test_zero_budget_and_untrusted_candidate(self):
        result = select(self.bank, self.base, self.enhanced, StubUtility(), 0, 0)
        self.assertEqual(result['selected_indices'], [])
        self.assertEqual(result['e_packet_bytes'], 0)
        with self.assertRaisesRegex(ValueError, 'base does not match'):
            select(self.bank, self.base+1, self.enhanced, StubUtility(), 100, 4)
        with self.assertRaises(ValueError):
            select(self.bank, self.base, self.enhanced, StubUtility(), -1, 4)

    def test_progressive_budget_independent_order(self):
        small = select(self.bank, self.base, self.enhanced, StubUtility(), 500, 4, mode='prefix')
        large = select(self.bank, self.base, self.enhanced, StubUtility(), 1200, 4, mode='prefix')
        self.assertEqual(small['prefix_order'], large['prefix_order'])
        self.assertTrue(subset_bank(self.bank, large['selected_indices']).startswith(
                        subset_bank(self.bank, small['selected_indices'])))
        rank, records = prefix_order(np.array([[0, 1, 0, 1], [0, 2, 0, 2]]), [10, 10], 0)
        self.assertEqual(rank, [1, 0])
        self.assertEqual([r['marginal_utility'] for r in records], [2., 1.])
        rank, _ = prefix_order(np.array([[0, -1, 0, -1], [0, -2, 0, -2]]), [10, 10], 0)
        self.assertEqual(rank, [])

    def test_prepare_atomic_resume_without_gpu(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            source = folder/'input.npz'
            atomic_npz(source, source=self.base)
            for name in ('E.pt', 'I.pt', 'P.pt'):
                from demo.scalable_codec import atomic_bytes
                atomic_bytes(folder/name, b'mock-shared-model')
            fake_codec = Mock()
            fake_codec.encode.return_value = (b'opaque-native-base-payload', {'seconds': .01})
            fake_codec.decode.return_value = self.base
            pm = parse(self.bank)
            fake_codec.metadata.return_value = pm.meta
            chunks = [dict(start=s, count=n) for s, n in ((0, 1), (1, 8), (9, 8))]
            with patch('demo.scalable_codec.BaseCodec', return_value=fake_codec) as constructor, \
                 patch('demo.chunk_enhancement_codec.configure_torch'), \
                 patch('demo.chunk_enhancement_codec.decode_features', return_value=(self.base, chunks)), \
                 patch('demo.chunk_enhancement_codec.load_model', return_value=Mock()), \
                 patch('demo.chunk_enhancement_codec.encode_enhancement', return_value=(
                     self.bank[:pm.base_end], [p.wire for p in pm.packets], self.enhanced, [])), \
                 patch('demo.chunk_enhancement_codec.decode_enhancement', return_value=(
                     self.enhanced, {'source_frames_read': False}, self.base)):
                kwargs = dict(enhancement=folder/'E.pt', model_i=folder/'I.pt', model_p=folder/'P.pt')
                first = prepare(source, folder/'run', **kwargs)
                before = (folder/'run/complete.json').read_bytes()
                second = prepare(source, folder/'run', **kwargs)
                self.assertEqual(first, second)
                self.assertEqual(constructor.call_count, 1)
                self.assertEqual(before, (folder/'run/complete.json').read_bytes())
                self.assertEqual(first['padding']['temporal_frames'], 0)
                self.assertEqual(first['candidate_bank_bytes'], len(self.bank))
                atomic_bytes(folder/'run/bank.acse', b'corrupt')
                with self.assertRaisesRegex(RuntimeError, 'changed sender artifact'):
                    prepare(source, folder/'run', **kwargs)


if __name__ == '__main__':
    unittest.main()
