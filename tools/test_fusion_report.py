import struct
import unittest
import zlib

from tools.fusion_report import change, check_envelope, MODES


class FusionReportTests(unittest.TestCase):
    def test_lpips_direction(self):
        self.assertAlmostEqual(change(.3,.27)['reduction_percent'],10.)
        self.assertLess(change(.2,.21)['reduction_percent'],0)
        with self.assertRaises(ValueError): change(0,.1)

    def test_envelope_counts_and_identities(self):
        original=b'original entropy-coded stream'; profile='12'*32; checkpoint='34'*32
        for mode in MODES:
            head=struct.pack('<8s32s32sB',b'RVLFUS01',bytes.fromhex(profile),
                bytes.fromhex(checkpoint if mode=='learned' else '0'*64),MODES.index(mode))
            wire=head+struct.pack('<I',zlib.crc32(head))+original
            check_envelope(wire,original,mode,profile,checkpoint)
            for damaged in (wire+b'x',wire[:73]+b'\x00'*4+wire[77:],wire[:-1]+b'X'):
                with self.assertRaises(ValueError):check_envelope(damaged,original,mode,profile,checkpoint)
            with self.assertRaises(ValueError):check_envelope(wire,original,mode,'ff'*32,checkpoint)


if __name__=='__main__':unittest.main()
