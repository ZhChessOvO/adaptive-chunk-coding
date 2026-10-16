import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from routervc.codec import split, format as wire, refinement_entropy as entropy
from routervc.routers import sender, receiver, routing
from routervc.fusion.render import render
from routervc.training.geometry import window, master_noise, noise_crop
from routervc.training.texture import phase_bias_loss
from routervc.training.vae import restore_step_log
from routervc import stream


def fake_bank(width=3):
    meta = dict(
        profile="RVL91",
        identities=["a" * 64] * 5,
        frames=17,
        height=256,
        width=256,
        q_star=48,
        bin_width=width,
        I_qp=32,
        I_parallel=1,
    )
    base = wire.base_stream(
        wire.pack_inner(meta, b"abcd", [(b"abcd", b"abcd")] * 2), "a" * 64
    )
    return base + b"".join(
        wire.packet(c, r, b"abcd", wire.digest(base))
        for r in range(16)
        for c in range(2)
    )


class ReleaseTests(unittest.TestCase):
    def test_split_exact_all_symbols(self):
        n = np.arange(-128, 128, dtype=np.int16)
        for w in (3, 5, 7, 9):
            c, r = split.split_symbols(n, w)
            np.testing.assert_array_equal(split.restore_symbols(c, r, w), n)

    def test_invalid_width(self):
        for w in (0, 2, 8, 11, True):
            with self.assertRaises(ValueError):
                entropy.validate(w)

    def test_prefix_and_bytes(self):
        bank = fake_bank()
        b = wire.subset(bank, [])
        p = wire.subset(bank, [5, 3])
        full = wire.subset(bank, [5, 3, 7])
        self.assertTrue(full.startswith(p))
        self.assertTrue(p.startswith(b))
        self.assertEqual(
            len(p), len(b) + sum(wire.bundle_bytes(bank)[i] for i in (5, 3))
        )
        self.assertEqual(wire.EHEADER.size, 22)

    def test_corrupt_packet(self):
        bad = bytearray(fake_bank())
        bad[-1] ^= 1
        with self.assertRaises(ValueError):
            wire.parse(bytes(bad))

    def test_incomplete_tail(self):
        bank = fake_bank()
        short = bank[:-1]
        with self.assertRaises(ValueError):
            wire.parse(short)
        self.assertGreater(
            wire.parse(short, allow_incomplete_tail=True)["ignored_tail_bytes"], 0
        )

    def test_coverage(self):
        cov = routing.coverage(wire.parse(wire.subset(fake_bank(), [2, 5])))
        self.assertEqual(cov.sum(), 2)
        self.assertEqual(cov[5], 1)

    def test_stream_identity(self):
        v = stream.wrap(fake_bank(), "a" * 64, "b" * 64)
        self.assertEqual(len(v) - len(fake_bank()), 121)
        self.assertEqual(stream.parse(v)[1]["max_g"], 8)
        with self.assertRaises(ValueError):
            stream.parse(v[:-1])

    def test_sender_density_stop(self):
        gains = np.zeros(16)
        gains[:3] = [1, -1, 2]
        cost = np.ones(16, dtype=np.int64) * 10
        cov = np.zeros(16)
        self.assertEqual(sender.rank_candidates(gains, cost, cov), [2, 0])
        self.assertEqual(sender.prefix_under_budget([2, 0], cost, 15)["indices"], [2])

    def test_receiver_does_not_accept_source(self):
        model = receiver.Receiver()
        with self.assertRaises(ValueError):
            model({"source": torch.zeros(1)})

    def test_receiver_no_positive_gain(self):
        model = receiver.Receiver().eval()
        for p in model.parameters():
            p.data.zero_()
        x = np.zeros((17, 256, 256, 3), np.uint8)
        self.assertEqual(receiver.select(model, x, x, np.zeros(16))["indices"], [])

    def test_fusion_no_generation_exact(self):
        x = np.random.default_rng(1).integers(0, 256, (17, 256, 256, 3), dtype=np.uint8)
        np.testing.assert_array_equal(render(x, [], []), x)

    def test_fusion_nonselected_exact(self):
        x = np.zeros((17, 256, 256, 3), np.uint8)
        patch = dict(
            region=5,
            pixels=np.ones((17, 192, 192, 3), np.uint8) * 100,
            crop=[0, 0, 192, 192],
            core=[64, 64, 64, 64],
        )
        y = render(x, [patch], [5])
        mask = np.ones(x.shape[:3], bool)
        mask[:, 64:128, 64:128] = False
        np.testing.assert_array_equal(x[mask], y[mask])

    def test_training_noise_nested(self):
        large = window([17, 576, 1024, 3], [224, 224, 128, 128], 512)
        n = noise_crop(master_noise(8), large, large)
        self.assertEqual(tuple(n.shape), (5, 64, 64, 16))

    def test_phase_zero_error(self):
        x = torch.randn(17, 3, 128, 128)
        self.assertLess(float(phase_bias_loss(x, x)), 1e-6)

    def test_log_recovery(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "steps.jsonl"
            p.write_text('{"step":1}\n{"step":2}\n{"step":')
            self.assertEqual(len(restore_step_log(p, 1)), 1)
            p.write_text('{"step":2}\n')
            with self.assertRaises(RuntimeError):
                restore_step_log(p, 2)

    def test_entropy_payload_roundtrip(self):
        y = np.random.default_rng(8).integers(-128, 128, (1, 8, 8, 8), dtype=np.int16)
        scales = np.full(y.shape, 10, np.float32)
        for w in (3, 5, 7, 9):
            c, r = split.split_symbols(y, w)
            data = entropy.encode(c, r, scales, w)
            np.testing.assert_array_equal(entropy.decode(data, c, scales, w), r)


if __name__ == "__main__":
    unittest.main()
