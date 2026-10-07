import unittest
import numpy as np
from routervc.latent import entropy as e
from routervc.latent import format as f
from routervc.latent.split import split_symbols,restore_symbols


class EntropyTests(unittest.TestCase):
    def test_rans_signed_edges(self):
        symbols=np.arange(-128,128,dtype=np.int16)
        scales=np.linspace(.11,16,256)
        for width in (3,9,17):
            c,r=split_symbols(symbols,width)
            b=e.encode_coarse(c,scales,width)
            got=e.decode_coarse(b,scales,width)
            np.testing.assert_array_equal(got,c)
            extra=e.encode_fine(c,r,scales,width)
            np.testing.assert_array_equal(restore_symbols(got,e.decode_fine(extra,got,scales,width),width),symbols)
            self.assertEqual(b,e.encode_coarse(got,scales,width))

    def test_zero_and_sparse(self):
        rng=np.random.default_rng(107)
        for width in (3,9,17):
            for values in [np.zeros(100,dtype=np.int16),rng.integers(-20,21,(2,3,10),dtype=np.int16)]:
                scales=np.full(values.shape,1.)
                c,r=split_symbols(values,width)
                b=e.encode_coarse(c,scales,width);extra=e.encode_fine(c,r,scales,width)
                got=e.decode_coarse(b,scales,width)
                np.testing.assert_array_equal(restore_symbols(got,e.decode_fine(extra,got,scales,width),width),values)

    def test_group_truncation(self):
        c,r=split_symbols(np.array([-4,0,4]),3);scales=np.ones(3)
        extra=e.encode_fine(c,r,scales,3)
        for wire in (extra[:-1],extra+b'x'):
            with self.assertRaises(ValueError): e.decode_fine(wire,c,scales,3)

    def test_cdf_valid(self):
        for width in (3,9,17):
            for coarse in (None,-3,0,3):
                cdf,length=e.tables(width,coarse)
                self.assertTrue(np.all(cdf[:,0]==0))
                self.assertTrue(np.all(cdf[:,-1]==65536))
                self.assertTrue(np.all(np.diff(cdf)>0))
                self.assertTrue(np.all(length==cdf.shape[1]))


class FormatTests(unittest.TestCase):
    def base(self):
        return f.base_stream(qp=48,width=3,height=576,image_width=1024,i_parallel=1,
            identities=['ab'*32]*4,intra=b'i'*4,z=b'z'*4,coarse=b'y'*4)

    def test_literal_prefix_and_accounting(self):
        b=self.base();extra=f.enhancement_packet(b'f'*12)
        p=f.parse(b+extra)
        self.assertEqual(p.base_end,len(b));self.assertEqual(p.fine,b'f'*12)
        self.assertIsNone(f.parse(b).fine)
        self.assertEqual(len(b),f.HEADER.size+f.IDENTITIES+12+f.BASE_DIGEST)
        self.assertEqual(len(extra),f.EHEADER.size+12)

    def test_every_truncation(self):
        b=self.base();extra=f.enhancement_packet(b'f'*12)
        for i in range(len(b)):
            with self.assertRaises(ValueError): f.parse(b[:i],allow_incomplete_tail=True)
        for i in range(1,len(extra)):
            with self.assertRaises(ValueError): f.parse(b+extra[:i])
            p=f.parse(b+extra[:i],allow_incomplete_tail=True)
            self.assertIsNone(p.fine);self.assertEqual(p.ignored_tail_bytes,i)

    def test_corruption_and_duplicates(self):
        b=self.base();extra=f.enhancement_packet(b'f'*12)
        for wire in (b[:-1]+b'!',b+extra[:-1]+b'!',b+extra+extra):
            with self.assertRaises(ValueError): f.parse(wire,allow_incomplete_tail=True)


if __name__=='__main__': unittest.main()
