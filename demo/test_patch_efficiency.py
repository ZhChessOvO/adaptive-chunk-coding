import unittest
from unittest.mock import patch
from types import SimpleNamespace
import time
import numpy as np
import torch

from demo import compact_enhancement_format as compact
from demo.scalable_format import base_container, packet_bytes, parse
from demo.chunk_enhancement_codec import pack_region, region_features, unpack_region
from demo.feature_head_enhancement import FeatureHeadEnhancement, Pad16FeatureHeadEnhancement


class CompactTests(unittest.TestCase):
    def fixture(self):
        meta = dict(compact.CONSTANTS, width=512, height=384, frame_count=19,
                    enhancement_codec=compact.CODECS[1], **{k: "a"*64 for k in compact.HASHES})
        prefix = base_container(b"test base", meta)
        packets = [packet_bytes(dict(packet_id=i+1, codec=compact.CODECS[1],
                    start=i*8+1, count=8 if i == 0 else 2, roi=[8, 16, 96, 64], qstep=0.7),
                    b"unaltered entropy payload"*10) for i in range(2)]
        return prefix+b"".join(packets)

    def test_lossless_schema_payload_and_prefixes(self):
        old = self.fixture()
        new = compact.repack(old)
        a, b = parse(old), parse(new)
        self.assertEqual(a.meta, b.meta)
        self.assertEqual(a.base, b.base)
        self.assertLess(len(new), len(old))
        self.assertEqual(compact.repack(new), new)
        for p, q in zip(a.packets, b.packets):
            self.assertEqual(p.meta, q.meta)
            self.assertEqual(p.payload, q.payload)
        for n, end in enumerate([b.base_end]+[p.end_offset for p in b.packets]):
            self.assertEqual(len(parse(new[:end]).packets), n)
        for end in range(b.packets[0].end_offset+1, len(new)):
            self.assertEqual(len(parse(new[:end], allow_incomplete_tail=True).packets), 1)
        with self.assertRaises(ValueError):
            parse(new[:-1])

    def test_integrity_and_no_silent_metadata_loss(self):
        a = parse(self.fixture())
        for meta in (dict(a.meta, extra="must not disappear"), dict(a.meta, generation="enabled")):
            with self.assertRaises(ValueError):
                compact.base_container(a.base, meta)
        data = compact.repack(self.fixture())
        for pos in (20, parse(data).base_end-1, len(data)-1):
            bad = bytearray(data)
            bad[pos] ^= 1
            with self.assertRaises(ValueError):
                parse(bytes(bad), allow_incomplete_tail=True)
        b = parse(data)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse(data+b.packets[0].wire)


class PaddingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_unchanged_on_multiple64(self):
        torch.manual_seed(17)
        old = FeatureHeadEnhancement(width=16, latent=8, hyper=4).eval()
        new = Pad16FeatureHeadEnhancement(width=16, latent=8, hyper=4).eval()
        new.load_export_state(old.export_state())
        base = torch.rand(1, 24, 64, 128)
        source = torch.rand_like(base)
        feature = torch.randn(1, 1024, 16, 24)*0.01
        with torch.no_grad():
            a, ar, _ = old.compress(source, base, feature)
            b, br, _ = new.compress(source, base, feature)
        self.assertEqual(a, b)
        torch.testing.assert_close(ar, br, atol=0, rtol=0)

    def test_odd_rectangular_short_tail_and_gradient(self):
        model = Pad16FeatureHeadEnhancement(width=16, latent=8, hyper=4).eval()
        rng = np.random.default_rng(0)
        frames = rng.integers(0, 256, (5, 96, 160, 3), dtype=np.uint8)
        feature_grid = torch.randn(1, 1024, 16, 24)*0.01
        for roi in ([8, 16, 91, 65], [152, 88, 8, 8], [0, 0, 96, 64]):
            base = pack_region(frames, 0, 5, roi, "cpu", 16)
            features = region_features({"features": feature_grid}, roi, "cpu", 4, 16)
            with torch.no_grad():
                wire, expected, _ = model.compress(base, base, features, 1.0, 5)
                actual = model.decompress(wire, base, features, 1.0, 5)
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
            x,y,w,h = roi
            np.testing.assert_array_equal(unpack_region(base, 5, roi), frames[:,y:y+h,x:x+w])
        model.train()
        value = model(base, base, features, 1.0, 5)
        (value["reconstruction"].mean()+value["bits"]*1e-6).backward()
        self.assertGreater(model.feature_synthesis[-1].weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in model.head.parameters()))


class EvaluationQueueTests(unittest.TestCase):
    def test_waits_for_training_without_cancelling_it(self):
        from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
        updates = []
        run = SimpleNamespace(started=time.monotonic(), check=lambda:None,
                              update=lambda **kw:updates.append(kw))
        with patch("demo.chunk_enhancement_evaluate.subprocess.check_output", side_effect=["12345\n", ""]), \
             patch("demo.chunk_enhancement_evaluate.Path.read_bytes",
                   return_value=b"python\0/demo/chunk_enhancement_experiment.py\0train\0"), \
             patch("demo.chunk_enhancement_evaluate.time.sleep") as sleep:
            with exclusive_native_evaluation(run):
                self.assertEqual(updates[-1]["training_pids"], [12345])
                self.assertGreaterEqual(run.gpu_wait_seconds, 0)
            sleep.assert_called_once_with(2)

    def test_exited_process_does_not_block_evaluation(self):
        from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
        run = SimpleNamespace(started=time.monotonic(), check=lambda:None, update=lambda **kw:None)
        with patch("demo.chunk_enhancement_evaluate.subprocess.check_output", return_value="12345\n"), \
             patch("demo.chunk_enhancement_evaluate.Path.read_bytes", side_effect=FileNotFoundError), \
             patch("demo.chunk_enhancement_evaluate.time.sleep") as sleep:
            with exclusive_native_evaluation(run):
                pass
            sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
