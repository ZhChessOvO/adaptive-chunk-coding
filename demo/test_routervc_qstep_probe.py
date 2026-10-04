from types import SimpleNamespace
import unittest
from unittest.mock import patch
from demo import routervc_qstep_probe as q


def packet(index, step=2.):
    return SimpleNamespace(meta=dict(start=1,count=8,roi=[index*16,0,16,16],qstep=step),
                           wire=f'packet{index}'.encode())


class QStepTests(unittest.TestCase):
    def test_preserve_membership_and_original_prefix_order(self):
        bank = SimpleNamespace(packets=[packet(i) for i in range(3)],base=b'uf',base_end=4)
        low = SimpleNamespace(packets=[packet(2,1.)],base=b'uf')
        high = SimpleNamespace(packets=[packet(2,1.),packet(0,1.)],base=b'uf')
        with patch('demo.scalable_format.parse',return_value=bank):
            a = q.select_same_packets(b'HEADfullbank',low)
            b = q.select_same_packets(b'HEADfullbank',high)
        self.assertEqual(a,b'HEADpacket2')
        self.assertEqual(b,b'HEADpacket2packet0')
        self.assertTrue(b.startswith(a))

    def test_cannot_reuse_q1_payload_or_changed_base(self):
        bank=SimpleNamespace(packets=[packet(0,1.)],base=b'uf',base_end=4)
        ref=SimpleNamespace(packets=[packet(0,1.)],base=b'uf')
        with patch('demo.scalable_format.parse',return_value=bank):
            with self.assertRaises(ValueError):q.select_same_packets(b'HEAD',ref)
            bank.packets=[packet(0)];ref.base=b'other'
            with self.assertRaises(ValueError):q.select_same_packets(b'HEAD',ref)


if __name__=='__main__':unittest.main()
