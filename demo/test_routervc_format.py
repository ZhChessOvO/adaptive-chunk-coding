import unittest
from demo import routervc_format as fmt
from demo.test_four_state import bank
from demo.four_state_core import subset


def config():
    return dict({k:'0'*64 for k in fmt.HASHES},seed=1,max_g=8,boundary_lambda=.004,
                strength=1.,blend=1.,window=17,stride=8,context=64,feather=16)


class Tests(unittest.TestCase):
    def test_roundtrip_and_real_prefix(self):
        c=config(); b=bank()
        small=fmt.wrap(subset(b,[5]),c);large=fmt.wrap(subset(b,[5,1]),c)
        self.assertTrue(large.startswith(small))
        got,inner,parsed,cost=fmt.parse(small)
        self.assertEqual(got,c);self.assertEqual(inner,subset(b,[5]))
        self.assertEqual(len(small),len(inner)+cost)
        self.assertEqual(cost,310)
        self.assertNotIn('generate',got)

    def test_corrupt_and_unknown(self):
        w=fmt.wrap(bank(),config())
        for data in (w[:5],w[:40],w[:20]+bytes([w[20]^1])+w[21:],b'BAD!'+w[4:]):
            with self.assertRaises(ValueError):fmt.parse(data)
        for key,value in [('max_g',17),('max_g',True),('boundary_lambda',float('nan')),
                          ('window',9),('seed',-1),('generate',[])]:
            with self.assertRaises(ValueError):fmt.wrap(bank(),dict(config(),**{key:value}))

    def test_truncated_packet_tail(self):
        w=fmt.wrap(subset(bank(),[5]),config())
        with self.assertRaises(ValueError):fmt.parse(w[:-1])
        _,_,parsed,_=fmt.parse(w[:-1],allow_incomplete_tail=True)
        self.assertEqual(len(parsed.packets),2)
        self.assertGreater(parsed.incomplete_tail_bytes,0)


if __name__=='__main__':unittest.main()
