"""Small protocol invariants, independent of GPU/large-model assets."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from demo.four_state_core import (ROIS, STATES, aggregate_bytes, crop, isolation,
    control, seed_for, subset, costs, immutable_json, verify)
from demo import compact_enhancement_format as compact
from demo import scalable_cooperation_format as fmt
from demo.scalable_format import parse
from demo.scalable_codec import file_hash


def bank():
    meta = dict(compact.CONSTANTS, width=512, height=512, frame_count=17,
                enhancement_codec=compact.CODECS[-1], **{k:'0'*64 for k in compact.HASHES})
    prefix = compact.base_container(b'test-native-payload', meta)
    wires = []
    for t,n in ((0,1),(1,8),(9,8)):
        for i, roi in enumerate(ROIS):
            m = dict(packet_id=len(wires)+1, start=t,count=n,roi=roi,qstep=1.,codec=compact.CODECS[-1])
            wires.append(compact.packet_bytes(m, b'a'*(i+1), codec=m['codec']))
    return prefix+b''.join(wires)


class ProtocolTests(unittest.TestCase):
    def test_coverage(self):
        mass = np.zeros((512,512),int)
        for x,y,w,h in ROIS: mass[y:y+h,x:x+w] += 1
        np.testing.assert_array_equal(mass,1)

    def test_isolation(self):
        base = np.zeros((2,512,512,3),np.uint8)
        full = np.full_like(base, 42)
        isolated = isolation(base,full,5)
        self.assertEqual(np.count_nonzero(isolated),2*128*128*3)
        np.testing.assert_array_equal(crop(isolated,5),42)
        self.assertFalse(base.any())

    def test_independent_packets_prefix(self):
        b = bank()
        small, big = subset(b,[5]), subset(b,[5,0])
        self.assertTrue(big.startswith(small))
        p = parse(small)
        self.assertEqual(len(p.packets),3)
        self.assertEqual([v.meta['roi'] for v in p.packets], [ROIS[5]]*3)
        self.assertEqual(p.base,parse(b).base)

    def test_bad_selection(self):
        for s in ([1,1],[-1],[16]):
            with self.assertRaises(ValueError): subset(bank(),s)

    def test_joint_state_cost_not_double_count(self):
        b=bank(); profile={k:'0'*64 for k in fmt.HASHES}
        cc=[costs(b,profile,'sample',i) for i in range(16)]
        for states in (['B']*16,['E']*16,['EG']*16,list(STATES)*4):
            e=[i for i,s in enumerate(states) if s in ('E','EG')]
            g=[i for i,s in enumerate(states) if s in ('G','EG')]
            wire=subset(b,e)
            c=control(profile,'sample',0)
            c['generate']=[[0,17,*ROIS[i]] for i in g]
            if g: wire=fmt.wrap(wire,c)
            self.assertEqual(len(wire),aggregate_bytes(cc[0]['base_container_bytes'],
                [v['e_packet_bytes'] for v in cc],states,cc[0]['g_shared_bytes'],cc[0]['g_region_bytes']))

    def test_seeds(self):
        self.assertEqual(seed_for('a',5),seed_for('a',5))
        self.assertNotEqual(seed_for('a',5),seed_for('a',6))
        self.assertLess(seed_for('a',5)+65536*16,2**63)

    def test_codec_precision_scoped_and_exception_safe(self):
        from demo.four_state_receive import codec_precision
        old = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            with self.assertRaisesRegex(RuntimeError,'probe'):
                with codec_precision():
                    self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
                    self.assertFalse(torch.backends.cudnn.allow_tf32)
                    raise RuntimeError('probe')
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
            self.assertTrue(torch.backends.cudnn.allow_tf32)
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old

    def test_immutable_resume_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'done.json'
            immutable_json(p,{'complete':True})
            before=p.stat().st_mtime_ns
            immutable_json(p,{'complete':True})
            self.assertEqual(before,p.stat().st_mtime_ns)
            with self.assertRaises(RuntimeError): immutable_json(p,{'complete':False})
            verify(p.parent,{'done.json':file_hash(p)})
            with self.assertRaises(RuntimeError): verify(p.parent,{'done.json':'0'*64})


if __name__ == '__main__': unittest.main()
