import unittest
from routervc.latent import chain_format as c


class ChainFormatTests(unittest.TestCase):
    def base(self,frames=17):
        m=dict(profile='RVLC1',identities=['ab'*32]*5,frames=frames,height=576,width=1024,
               q_star=48,bin_width=3,I_qp=32,I_parallel=1)
        return c.pack(m,b'i'*4,[(b'z'*4,b'y'*4)]*2)

    def test_all_arrival_patterns(self):
        b=self.base();a=c.packet(0,b'a'*8);z=c.packet(1,b'z'*8)
        for extra,keys,duplicates in [(b'',set(),0),(a,{0},0),(z,{1},0),(a+z,{0,1},0),
                                      (z+a,{0,1},0),(a+z+a,{0,1},1)]:
            p=c.parse(b+extra)
            self.assertEqual(p['base_end'],len(b));self.assertEqual(set(p['enhancements']),keys)
            self.assertEqual(p['duplicates'],duplicates)

    def test_incomplete_tail(self):
        b=self.base(12);e=c.packet(1,b'x'*8)
        for i in range(1,len(e)):
            with self.assertRaises(ValueError): c.parse(b+e[:i])
            self.assertEqual(c.parse(b+e[:i],True)['enhancements'],{})
        with self.assertRaises(ValueError): c.parse(b[:-1],True)

    def test_corrupt_and_conflict(self):
        b=self.base();a=c.packet(0,b'a'*8)
        for extra in [a+c.packet(0,b'b'*8),c.packet(3,b'a'*8),a[:-1]+b'!']:
            with self.assertRaises(ValueError): c.parse(b+extra,True)


if __name__=='__main__': unittest.main()
