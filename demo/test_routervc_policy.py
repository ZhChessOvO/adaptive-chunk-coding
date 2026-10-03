import itertools
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from demo.four_state_core import ROIS
from demo.four_state_router import FORMAT, UtilityBackbone, patch_features
from demo import routervc_policy as policy


class PolicyTests(unittest.TestCase):
    @staticmethod
    def model(context=True):
        torch.manual_seed(4)
        return UtilityBackbone(context, np.zeros(37), np.ones(37),
                               np.zeros(6), np.ones(6))

    def test_grid_rectangular_full_coverage(self):
        self.assertEqual(policy.grid_rois(512, 512), [list(r) for r in ROIS])
        for height, width in ((64, 64), (256, 384), (264, 520), (72, 80)):
            rois = policy.grid_rois(height, width)
            counts = np.zeros((height, width), np.uint8)
            for x, y, w, h in rois:
                self.assertEqual((x % 8, y % 8, w % 8, h % 8), (0, 0, 0, 0))
                self.assertGreaterEqual(min(w, h), 16)
                counts[y:y+h, x:x+w] += 1
            np.testing.assert_array_equal(counts, 1)
        for dims in ((63, 128), (64, 65), (64.0, 128), (True, 128), (-8, 64)):
            with self.subTest(dims=dims), self.assertRaises(ValueError):
                policy.grid_rois(*dims)

    def test_512_features_are_bit_exact_historical(self):
        rng = np.random.default_rng(4)
        base = rng.integers(0, 256, (17, 512, 512, 3), dtype=np.uint8)
        received = rng.integers(0, 256, base.shape, dtype=np.uint8)
        coverage = np.array([i % 2 for i in range(16)], np.float32)
        actual = policy.features(base, received, policy.grid_rois(512, 512), coverage)
        expected = np.concatenate((np.stack([patch_features(base, i) for i in range(16)]),
                                   np.stack([patch_features(received, i) for i in range(16)]),
                                   coverage[:, None]), axis=1)
        np.testing.assert_array_equal(actual.numpy(), expected[None])
        self.assertEqual(actual.dtype, torch.float32)

    def test_features_rectangular_and_coverage_validation(self):
        base = np.zeros((3, 72, 80, 3), np.uint8)
        received = base.copy()
        rois = policy.grid_rois(72, 80)
        x, y, w, h = rois[5]
        received[:, y:y+h, x:x+w] = 80
        coverage = np.zeros(16, np.float32)
        coverage[5] = 2/3
        result = policy.features(base, received, rois, coverage).numpy()[0]
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_array_equal(result[:, 36], coverage)
        untouched = [i for i in range(16) if i != 5]
        np.testing.assert_array_equal(result[untouched, :18], result[untouched, 18:36])
        for bad in (np.zeros(15), np.full(16, -1), np.full(16, 1.01),
                    np.full(16, np.nan), np.full(16, np.inf), ['0']*16):
            with self.subTest(coverage=str(bad)), self.assertRaises(ValueError):
                policy.features(base, received, rois, bad)
        invalid_pairs = ((base[:1], received[:1]), (base.astype(float), received),
                         (base, received[:2]), (base[..., 0], received[..., 0]))
        for b, r in invalid_pairs:
            with self.assertRaises(ValueError):
                policy.features(b, r, rois, coverage)
        for bad_rois in (list(reversed(rois)), rois[:-1], np.asarray(rois, float)):
            with self.assertRaises(ValueError):
                policy.features(base, received, bad_rois, coverage)

    def test_model_loading_and_cpu_prediction(self):
        model = self.model()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            torch.save(dict(format=FORMAT, config={'context': True}, model=model.state_dict()), path)
            loaded = policy.load_model(path)
            self.assertEqual(sum(p.numel() for p in loaded.parameters()), 6982)
            x = torch.randn(1, 16, 37)
            x[..., 36] = 0
            actual = policy.predict(loaded, x)
            torch.testing.assert_close(actual, model(x), rtol=0, atol=0)
            torch.testing.assert_close(actual[..., [0, 2, 4]], torch.zeros(1, 16, 3), rtol=0, atol=0)
            self.assertFalse(actual.requires_grad)
        for bad in (x[0], x.double(), torch.full_like(x, float('nan'))):
            with self.assertRaises(ValueError):
                policy.predict(model, bad)

    def test_random_small_grid_matches_independent_exhaustive(self):
        rng = np.random.default_rng(8)
        for rows, columns in ((1, 4), (2, 2), (2, 3)):
            n = rows * columns
            for _ in range(10):
                gains = rng.normal(size=n)
                cap = int(rng.integers(0, n+1))
                lam = float(rng.uniform(0, .5))
                candidates = []
                for choice in itertools.product((0, 1), repeat=n):
                    if sum(choice) > cap:
                        continue
                    boundary = sum(choice[y*columns+x] != choice[y*columns+x+1]
                                   for y in range(rows) for x in range(columns-1))
                    boundary += sum(choice[y*columns+x] != choice[(y+1)*columns+x]
                                    for y in range(rows-1) for x in range(columns))
                    gain = float((np.asarray(choice) * gains).sum())
                    bitmask = sum(flag << i for i, flag in enumerate(choice))
                    candidates.append((gain-lam*boundary, -sum(choice), -boundary, -bitmask))
                expected = max(candidates)
                result = policy._select(gains, cap, lam, rows=rows, columns=columns)
                self.assertEqual(result['bitmask'], -expected[3])
                self.assertEqual(result['objective'], expected[0])
                self.assertLessEqual(result['g_calls'], cap)

    def test_lambda_zero_topk_and_negative_empty(self):
        gains = np.array([.1*i-.45 for i in range(16)])
        for cap in range(17):
            result = policy.select_generate(gains, cap, 0)
            expected = sorted(np.argsort(-gains)[:min(cap, int((gains > 0).sum()))].tolist())
            self.assertEqual(result['indices'], expected)
        for lam in (0, .004, 2):
            result = policy.select_generate(-np.ones(16), 16, lam)
            self.assertEqual((result['indices'], result['components'], result['boundary_edges']), ([], 0, 0))
            self.assertEqual(result['objective'], 0)
        self.assertEqual(policy.select_generate(np.zeros(16), 16, 0)['indices'], [])

    def test_ties_components_and_boundary_reduction(self):
        # Equal gain prefers lower boundary degree, then smaller bitmask.
        self.assertEqual(policy.select_generate(np.ones(16), 1, 0)['indices'], [0])
        gains = np.full(16, -100.)
        gains[[5, 15]] = 1
        self.assertEqual(policy.select_generate(gains, 1, 0)['indices'], [15])
        gains = np.full(16, -100.)
        gains[[0, 1, 15]] = 1
        result = policy.select_generate(gains, 3, 0)
        self.assertEqual(result['component_indices'], [[0, 1], [15]])
        self.assertEqual(result['components'], 2)
        self.assertEqual(result['boundary_edges'], 5)
        checker = np.array([.02 if (i//4+i%4) % 2 else .019 for i in range(16)])
        plain = policy.select_generate(checker, 8, 0)
        smooth = policy.select_generate(checker, 8, .004)
        self.assertLess(smooth['boundary_edges'], plain['boundary_edges'])
        self.assertLessEqual(smooth['predicted_gain'], plain['predicted_gain'])

    def test_invalid_solver_inputs(self):
        gains = np.ones(16)
        invalid = ((gains[:-1], 4, 0), (np.full(16, np.nan), 4, 0),
                   (np.full(16, np.inf), 4, 0), (gains, -1, 0), (gains, 17, 0),
                   (gains, 1.5, 0), (gains, True, 0), (gains, 4, -.1),
                   (gains, 4, float('nan')), (gains, 4, True))
        for args in invalid:
            with self.subTest(args=str(args)), self.assertRaises(ValueError):
                policy.select_generate(*args)

    def test_received_packet_coverage_whole_bundles(self):
        from demo.four_state_core import subset
        from demo.routervc_decode import coverage_from_packets
        from demo.scalable_format import parse
        from demo.test_four_state import bank
        encoded = bank()
        rois = policy.grid_rois(512, 512)
        np.testing.assert_array_equal(coverage_from_packets(parse(encoded), rois), np.ones(16))
        for indices in ([], [5], [15, 0, 8]):
            expected = np.zeros(16, np.float32)
            expected[indices] = 1
            actual = coverage_from_packets(parse(subset(encoded, indices)), rois)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(actual.dtype, np.float32)

    def test_received_packet_coverage_partial_prefix_and_tail(self):
        from demo.four_state_core import subset
        from demo.routervc_decode import coverage_from_packets
        from demo.scalable_format import parse
        from demo.test_four_state import bank
        encoded = subset(bank(), [5])
        parsed = parse(encoded)
        rois = policy.grid_rois(512, 512)
        for end, coverage in ((parsed.base_end, 0),
                              (parsed.packets[0].end_offset, 1/17),
                              (parsed.packets[1].end_offset, 9/17),
                              (parsed.packets[2].end_offset, 1)):
            result = coverage_from_packets(parse(encoded[:end]), rois)
            expected = np.zeros(16, np.float32)
            expected[5] = coverage
            np.testing.assert_array_equal(result, expected)
        # A truncated last packet is charged by the parser but contributes no E.
        truncated = parse(encoded[:-1], allow_incomplete_tail=True)
        self.assertGreater(truncated.incomplete_tail_bytes, 0)
        self.assertEqual(coverage_from_packets(truncated, rois)[5], np.float32(9/17))

    def test_received_packet_coverage_rejects_bad_geometry(self):
        from demo.routervc_decode import coverage_from_packets
        rois = policy.grid_rois(512, 512)

        def inner(*records):
            return SimpleNamespace(meta={'frame_count': 17},
                                   packets=[SimpleNamespace(meta=p) for p in records])

        good = dict(roi=rois[5], start=0, count=1)
        for bad in (dict(good, start=-1), dict(good, start=17),
                    dict(good, count=0), dict(good, count=18),
                    dict(good, roi=[1, 0, 128, 128])):
            with self.assertRaises(ValueError):
                coverage_from_packets(inner(bad), rois)
        with self.assertRaises(ValueError):
            coverage_from_packets(inner(good, good), rois)
        with self.assertRaises(ValueError):
            coverage_from_packets(inner(dict(good, count=8), dict(good, start=7, count=8)), rois)


if __name__ == '__main__':
    unittest.main()
