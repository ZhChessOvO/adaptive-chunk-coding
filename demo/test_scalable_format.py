"""CPU tests of prefix semantics and signed residual coding, without UF weights."""

import unittest

import numpy as np

from demo.scalable_format import (
    HEADER, apply_packets, base_container, decode_residual,
    encode_residual, frame_hash, haar, make_packet, packet_bytes, parse,
)


class ScalableFormatTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20260926)
        self.base = self.rng.integers(20, 230, (5, 31, 47, 3), dtype=np.uint8)
        noise = self.rng.integers(-20, 21, self.base.shape)
        self.source = np.clip(self.base.astype(np.int32) + noise, 0, 255).astype(np.uint8)
        self.meta = {"base_codec": "dcvc_uf_hts_scalar", "base_qp": 8, "skip_thres": 0.0,
                     "width": 47, "height": 31, "frame_count": 5, "display_format": "rgb_u8",
                     "base_rgb_sha256": frame_hash(self.base),
                     "model_i_sha256": "a" * 64, "model_p_sha256": "b" * 64}
        self.bottom = base_container(b"fake base for CPU format tests", self.meta)
        self.first = make_packet(self.source, self.base, packet_id=1, start=0,
                                 count=3, roi=[2, 3, 13, 11], layer=1,
                                 parent_id=0, qstep=16)
        self.one = parse(self.bottom + self.first)
        self.recon1, _ = apply_packets(self.base, self.one.packets)
        self.second = make_packet(self.source, self.recon1, packet_id=2, start=0,
                                  count=3, roi=[2, 3, 13, 11], layer=2,
                                  parent_id=1, qstep=1)

    def test_haar_signed_lossless(self):
        for levels in (1, 2, 3, 4):
            values = self.rng.integers(-255, 256, (2, 32, 48, 3), dtype=np.int32)
            np.testing.assert_array_equal(haar(haar(values, levels), levels, inverse=True), values)

    def test_odd_size_padding_and_negative_residual(self):
        residual = self.source.astype(np.int32) - self.base.astype(np.int32)
        payload = encode_residual(residual, 1)
        np.testing.assert_array_equal(decode_residual(payload, residual.shape, 1, 2), residual)

    def test_prefixes_and_exact_second_layer(self):
        complete = self.bottom + self.first + self.second
        parsed = parse(complete)
        self.assertEqual(parsed.base, parse(self.bottom).base)
        self.assertEqual(parsed.packets[0].wire, self.first)
        self.assertEqual(complete[:self.one.consumed_bytes], self.bottom + self.first)
        result, status = apply_packets(self.base, parsed.packets)
        mask = np.zeros(self.base.shape[:3], bool)
        mask[:3, 3:14, 2:15] = True
        np.testing.assert_array_equal(result[mask], self.source[mask])
        np.testing.assert_array_equal(result[~mask], self.base[~mask])
        self.assertEqual(status["applied_packet_ids"], [1, 2])
        self.assertEqual(parsed.base_end + sum(len(p.wire) for p in parsed.packets), len(complete))

    def test_missing_parent_keeps_base(self):
        packets = parse(self.bottom + self.second).packets
        output, status = apply_packets(self.base, packets)
        np.testing.assert_array_equal(output, self.base)
        self.assertEqual(status["applied_packet_ids"], [])
        self.assertEqual(status["skipped_packets"][0]["packet_id"], 2)

    def test_out_of_order_does_not_apply_stale_child(self):
        output, status = apply_packets(self.base, parse(self.bottom + self.second + self.first).packets)
        np.testing.assert_array_equal(output, self.recon1)
        self.assertEqual(status["applied_packet_ids"], [1])

    def test_independent_regions_reorder(self):
        other = make_packet(self.source, self.base, packet_id=3, start=1, count=4,
                            roi=[25, 17, 19, 13], layer=1, parent_id=0, qstep=8)
        a, _ = apply_packets(self.base, parse(self.bottom + self.first + other).packets)
        b, _ = apply_packets(self.base, parse(self.bottom + other + self.first).packets)
        np.testing.assert_array_equal(a, b)

    def test_overlapping_regions_rejected(self):
        overlap = make_packet(self.source, self.recon1, packet_id=3, start=0,
                              count=3, roi=[3, 3, 13, 11], layer=1, parent_id=0, qstep=8)
        with self.assertRaisesRegex(ValueError, "overlapping"):
            apply_packets(self.base, parse(self.bottom + self.first + overlap).packets)

    def test_partial_tail_is_never_used(self):
        complete = self.bottom + self.first + self.second
        for missing in (1, len(self.second) - 1):
            with self.assertRaisesRegex(ValueError, "truncated"):
                parse(complete[:-missing])
            parsed = parse(complete[:-missing], allow_incomplete_tail=True)
            self.assertEqual(len(parsed.packets), 1)
            self.assertEqual(parsed.incomplete_tail_bytes, len(self.second) - missing)
            out, _ = apply_packets(self.base, parsed.packets)
            np.testing.assert_array_equal(out, self.recon1)

    def test_truncated_base_always_rejected(self):
        for data in (self.bottom[:HEADER.size - 1], self.bottom[:-1]):
            with self.assertRaises(ValueError):
                parse(data, allow_incomplete_tail=True)

    def test_corruption_and_duplicate_ids(self):
        for payload in (self.bottom[:-1] + b"x",
                        self.bottom + self.first[:-1] + bytes([self.first[-1] ^ 255]),
                        self.bottom + self.first + self.first):
            with self.assertRaises(ValueError):
                parse(payload)

    def test_bounds_and_codec_validation(self):
        for field, value in (("roi", [40, 0, 20, 20]), ("start", -1),
                             ("qstep", 0), ("codec", "unknown")):
            meta = dict(self.one.packets[0].meta)
            meta[field] = value
            wire = packet_bytes(meta, self.one.packets[0].payload)
            with self.assertRaises(ValueError):
                apply_packets(self.base, parse(self.bottom + wire).packets)

    def test_wrong_decompressed_length_and_trailing_payload(self):
        residual = np.zeros((1, 4, 4, 3), dtype=np.int32)
        payload = encode_residual(residual, 1)
        for data in (payload + b"junk", payload[:-1]):
            with self.assertRaises(ValueError):
                decode_residual(data, residual.shape, 1, 2)
        with self.assertRaises(ValueError):
            decode_residual(payload, (1, 8, 8, 3), 1, 2)


if __name__ == "__main__":
    unittest.main()
