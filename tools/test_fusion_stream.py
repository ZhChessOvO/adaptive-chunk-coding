import unittest
from unittest.mock import patch
from routervc.fusion import stream


class StreamTest(unittest.TestCase):
    def test_counts_prefix_and_corruption(self):
        with patch.object(stream.routing, 'parse'):
            b = b'example-received-prefix'
            for mode in stream.MODES:
                sha = '1'*64 if mode == 'learned' else '0'*64
                short = stream.wrap(b, mode, sha); long = stream.wrap(b+b'Epacket', mode, sha)
                self.assertEqual(len(short)-len(b), 77)
                self.assertTrue(long.startswith(short))
                self.assertEqual(stream.parse(short), (b, dict(mode=mode, model_sha256=sha)))
                damaged = bytearray(short); damaged[76] ^= 1
                with self.assertRaises(ValueError): stream.parse(bytes(damaged))

    def test_model_required_only_for_learned(self):
        with patch.object(stream.routing, 'parse'):
            with self.assertRaises(ValueError): stream.wrap(b'x', 'learned')
            with self.assertRaises(ValueError): stream.wrap(b'x', 'current', '1'*64)


if __name__ == '__main__': unittest.main()
