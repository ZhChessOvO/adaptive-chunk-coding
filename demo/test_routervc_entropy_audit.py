"""CPU contracts for byte accounting; native rANS is checked by the real audit."""
import struct
from types import SimpleNamespace
import unittest

from demo import routervc_entropy_audit as a


def fixture():
    payload = struct.pack('<I', 4)+b'zzzz'+b'yyyyyyyy'
    packet = SimpleNamespace(meta=dict(start=0, count=1, roi=[0,0,16,16], qstep=1., packet_id=1),
                             payload=payload, wire=b'x'*41+payload)
    parsed = SimpleNamespace(base=b'b'*10, base_end=200, packets=[packet])
    streams = []
    for kind, n in [('z',4), ('y',8)]:
        streams.append(dict(kind=kind, actual_bytes=n, symbols=n*10, nonzero=3,
            escaped=0, scales_above_16=0, gaussian_proxy_bits=n*8-3,
            native_cdf_bits=n*8-2, bypass_bits=0, coder_framing_rounding_bits=2))
    return parsed, streams


class AuditTests(unittest.TestCase):
    def test_true_file_bytes_include_all_headers(self):
        parsed, streams = fixture()
        rows = a.packet_costs(parsed, streams)
        costs = a.cost_summary(parsed, rows, outer_bytes=310)
        self.assertEqual(costs['bytes']['total_bytes'], 567)
        self.assertEqual(costs['bytes']['entropy_bytes'],12)
        self.assertEqual(costs['bytes']['payload_length_bytes'],4)
        self.assertEqual(costs['entropy']['fixed_int8_reference_bytes'],120)
        self.assertAlmostEqual(costs['entropy']['entropy_to_fixed_int8_ratio'],.1)

    def test_malformed_lengths_and_order_fail(self):
        parsed, streams = fixture()
        with self.assertRaises(ValueError): a.packet_costs(parsed, streams[:1])
        with self.assertRaises(ValueError): a.packet_costs(parsed, streams[::-1])
        streams[1]['actual_bytes'] += 1
        with self.assertRaises(ValueError): a.packet_costs(parsed, streams)

    def test_selected_packet_identity_ignores_id_not_payload(self):
        parsed, streams = fixture()
        rows = a.packet_costs(parsed, streams)
        parsed.packets[0].meta = dict(parsed.packets[0].meta, packet_id=8)
        self.assertEqual(a.selected_costs(parsed, rows,310)['packets'],1)
        parsed.packets[0].payload += b'changed'
        with self.assertRaises(KeyError): a.selected_costs(parsed, rows,310)

    def test_empty_E_still_charges_base_and_headers(self):
        parsed, _ = fixture(); parsed.packets = []
        result = a.cost_summary(parsed, [], 310)
        self.assertEqual(result['bytes']['total_bytes'],510)
        self.assertEqual(result['entropy']['actual_bytes'],0)


if __name__ == '__main__': unittest.main()
