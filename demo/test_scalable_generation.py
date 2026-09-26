import copy
import unittest

import numpy as np

from demo import compact_enhancement_format as compact
from demo import scalable_generation_format as g
from demo.scalable_format import parse


class GenerationTests(unittest.TestCase):
    def fixture(self):
        meta = dict(compact.CONSTANTS,width=128,height=128,frame_count=33,
                    enhancement_codec=compact.CODECS[-1],**{k:"a"*64 for k in compact.HASHES})
        prefix = compact.base_container(b"test base",meta)
        wire = compact.packet_bytes(dict(packet_id=1,codec=compact.CODECS[-1],start=1,
                    count=8,roi=[64,64,64,64],qstep=1.), b"payload",codec=compact.CODECS[-1])
        c = dict(seed=17,strength=.5,window=17,stride=8,context=64,feather=8,
                 generate=[[0,33,0,0,128,128]], protect=[[0,33,0,0,32,32]],
                 **{k:"b"*64 for k in g.HASHES})
        return prefix+wire,c

    def test_roundtrip_every_literal_prefix(self):
        inner,c = self.fixture()
        wire = g.wrap(inner,c)
        loaded,old,parsed,end = g.parse(wire)
        self.assertEqual(c,loaded)
        self.assertEqual(inner,old)
        for boundary in [parsed.base_end]+[p.end_offset for p in parsed.packets]:
            self.assertEqual(g.wrap(inner[:boundary],c),wire[:end+boundary])
        with self.assertRaises(ValueError):
            g.parse(wire[:-1])
        self.assertEqual(len(g.parse(wire[:-1],allow_incomplete_tail=True)[2].packets),0)

    def test_corruption_and_unknown_controls(self):
        inner,c = self.fixture()
        wire = g.wrap(inner,c)
        for n in (0,5,17,len(wire)-1):
            bad = bytearray(wire)
            bad[n] ^= 1
            with self.assertRaises(ValueError):
                g.parse(bytes(bad))
        for k,v in (("extra",1),("strength",float("nan")),("window",33),("seed",-1),
                    ("generate",[[0,34,0,0,128,128]]),("dit","g"*64)):
            with self.assertRaises(ValueError):
                g.wrap(inner,dict(c,**{k:v}))

    def test_protection_enhancement_and_inner_feather(self):
        inner,c = self.fixture()
        mask = g.weights((33,128,128,3),c,parse(inner))
        self.assertTrue(np.all(mask[:,0,:] == 0))
        self.assertTrue(np.all(mask[:,:32,:32] == 0))
        self.assertTrue(np.all(mask[1:9,64:,64:] == 0))
        self.assertEqual(mask[10,80,80],1.)
        self.assertTrue(0 < mask[10,4,40] < 1)

    def test_33_frame_windows_and_overlap_rejection(self):
        self.assertEqual(g.windows(33),[0,8,16])
        self.assertEqual(g.windows(22),[0,5])
        inner,c = self.fixture()
        c["generate"].append(copy.deepcopy(c["generate"][0]))
        with self.assertRaises(ValueError):
            g.wrap(inner,c)


if __name__ == "__main__":
    unittest.main()
