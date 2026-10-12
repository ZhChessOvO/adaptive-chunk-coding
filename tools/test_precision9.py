"""Real entropy and framing tests, isolated from frozen historical profiles."""
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from routervc.precision9 import entropy as compact, format as wire, stream
from routervc.latent import compact_entropy as old
from routervc.latent.split import split_symbols, restore_symbols, coarse_representatives
from routervc.precision9 import budget


def fake_bank(width=3):
    inner = wire.pack_inner(dict(profile='RVL91', identities=['a'*64]*5, frames=17,
        height=512, width=512, q_star=48, bin_width=width, I_qp=32, I_parallel=1),
        b'abcd', [(b'zzzz', b'yyyy')]*2)
    base = wire.base_stream(inner, 'b'*64)
    return base+b''.join(wire.packet(j, k, bytes([j,k,3,4]), wire.digest(base))
                        for k in range(16) for j in range(2))


class Precision9Tests(unittest.TestCase):
    def test_parent_points_are_not_reencoded_or_overwritten(self):
        from demo.routervc_fullview_probe import save
        from tools.precision9_pilot import bank_folder,point_folder,plan
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'new';parent=Path(tmp)/'parent'
            save(root/'protocol.json',dict(parent_root=str(parent)))
            row=dict(sample_id='fixed',cohort='internal_evaluation')
            self.assertEqual(bank_folder(root,row,9),root/'banks/fixed/w9')
            self.assertEqual(bank_folder(root,row,7),parent/'banks/fixed/w7')
            self.assertEqual(point_folder(root,row,7,.5),parent/'internal_evaluation/fixed/w7/e050')
            with patch('tools.precision9_pilot.execute') as execute:
                with self.assertRaises(ValueError):plan(None,root,row,7)
                execute.assert_not_called()
            self.assertFalse(parent.exists())

    def test_user_width_bound(self):
        for w in (3,5,7,9): compact.validate(w)
        for w in (2,8,11,17,3.,True):
            with self.assertRaises(ValueError): compact.validate(w)

    def test_all_symbols_scales_and_endpoint_bins(self):
        values = np.tile(np.arange(-128,128,dtype=np.int16), (128,1))
        scales = np.broadcast_to(np.geomspace(.01,100,128)[:,None], values.shape)
        for w in (3,5,7,9):
            c,r = split_symbols(values,w)
            raw = compact.encode(c,r,scales,w)
            decoded = compact.decode(raw,c,scales,w)
            np.testing.assert_array_equal(restore_symbols(c,decoded,w), values)
            reps = coarse_representatives(c,w)
            self.assertTrue(np.isfinite(reps).all())
            for bad in (raw[:-1],raw+b'0',b''):
                with self.assertRaises(ValueError): compact.decode(bad,c,scales,w)

    def test_width3_is_bit_exact_old_entropy(self):
        rng = np.random.default_rng(1012)
        values = rng.integers(-128,128,(2,32,32),dtype=np.int16)
        scales = np.exp(rng.uniform(-3,3,values.shape))
        c,r = split_symbols(values,3)
        np.testing.assert_array_equal(compact.cdfs(3), old.cdfs())
        self.assertEqual(compact.encode(c,r,scales,3), old.encode(c,r,scales))

    def test_original_profile_remains_isolated(self):
        from routervc.precision import entropy as original, format as old_wire, stream as old_stream
        from tools.test_precision import fake_bank as old_bank
        with self.assertRaises(ValueError): original.validate(9)
        self.assertEqual(wire.HEADER.size,old_wire.HEADER.size)
        self.assertEqual(stream.HEADER_BYTES,old_stream.HEADER_BYTES)
        for w in (3,5,7):
            self.assertEqual(len(fake_bank(w)),len(old_bank(w)))
            np.testing.assert_array_equal(compact.cdfs(w),original.cdfs(w))
        with self.assertRaises(ValueError):old_wire.parse(fake_bank(9))
        with self.assertRaises(ValueError):wire.parse(old_bank(3))
        rng=np.random.default_rng(9)
        y=rng.integers(-128,128,(4,32,32),dtype=np.int16)
        scales=np.exp(rng.uniform(-4,4,y.shape))
        for w in (3,5,7):
            c,r=split_symbols(y,w)
            self.assertEqual(compact.encode(c,r,scales,w),original.encode(c,r,scales,w))
        bank=fake_bank(9)
        p=wire.parse(bank);base=wire.subset(bank,[])
        last=wire.packet(1,15,p['packets'][1,15],wire.digest(base))
        for bad in (bank[:-1],bank+last[:-1],wire.subset(fake_bank(7),[])+last):
            with self.assertRaises(ValueError):wire.parse(bad)

    def test_framing_prefix_bytes_and_no_masks(self):
        for width in (3,5,7,9):
            bank = fake_bank(width); costs = wire.bundle_bytes(bank)
            previous = None
            for n in (0,1,7,16):
                inner = wire.subset(bank,list(range(n)))
                raw = stream.wrap(inner,'a'*64,'b'*64)
                self.assertEqual(len(raw),121+len(wire.subset(bank,[]))+sum(costs[:n]))
                self.assertEqual(stream.parse(raw)[0],inner)
                self.assertEqual(wire.parse(inner)['info']['bin_width'],width)
                if previous is not None: self.assertTrue(raw.startswith(previous))
                previous = raw

    def test_corruption_duplicates_and_cross_width(self):
        bank = fake_bank(); p = wire.parse(bank); base = wire.subset(bank,[])
        first = wire.packet(0,0,p['packets'][0,0],wire.digest(base))
        self.assertEqual(wire.parse(bank+first)['duplicates'],1)
        conflicting = wire.packet(0,0,b'1234',wire.digest(base))
        for data in (bank+conflicting, bank[:-1], bank[:50], bank[:-1]+bytes([bank[-1]^1]),
                     wire.subset(fake_bank(5),[])+first):
            with self.assertRaises(ValueError): wire.parse(data)
        partial = wire.parse(bank[:-1],allow_incomplete_tail=True)
        self.assertEqual(len(partial['packets']),31)
        self.assertGreater(partial['ignored_tail_bytes'],0)

    def test_config_and_base_validation(self):
        bank = fake_bank()
        for options in ({'max_g':17},{'max_g':True},{'seed':-1},{'seed':1.2}):
            with self.assertRaises(ValueError): stream.wrap(bank,'a'*64,'b'*64,**options)
        inner = wire.parse(bank)['base']
        with self.assertRaises(ValueError): wire.parse_inner(inner+b'extra')

    def test_selector_no_evaluation_scores_and_infeasible_b(self):
        rows=[dict(cohort='calibration',width=w,cap=cap,sample_id=f's{i}',sender_features=[1.,2.,3.,4.],
            quality={'lpips_alex':.2+.01*w+.03*cap}) for w in (3,5,7,9) for cap in (0.,.5) for i in range(4)]
        table=budget.fit(rows)
        candidates=[dict(width=w,cap=cap,sender_features=[1.,2.,3.,4.],actual_bytes=100-w*5+int(20*cap),
            base_bytes=100-w*5,key=f'{w}-{cap}') for w in (3,5,7,9) for cap in (0.,.5)]
        self.assertFalse(budget.choose(table,candidates,50)['feasible'])
        self.assertEqual(budget.choose(table,candidates,100)['key'],'3-0.0')
        self.assertEqual(budget.fixed(candidates,100,3),'3-0.5')
        with self.assertRaises(ValueError):budget.predict(table,dict(candidates[0],quality=.01))
        with self.assertRaises(ValueError):budget.fit([dict(rows[0],cohort='internal_evaluation')])
        self.assertIsNone(budget.base_fraction(candidates,0))


if __name__ == '__main__': unittest.main()
