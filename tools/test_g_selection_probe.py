import unittest
import numpy as np
from tools.g_selection_probe import choices


class ChoicesTests(unittest.TestCase):
    def test_cap_and_nonpositive(self):
        q=choices(np.arange(16)-6,[0,1,8,9])
        self.assertEqual(q['old'],[0,1,8,9]);self.assertEqual(q['positive_old'],[8,9])
        self.assertEqual(q['top8'],list(range(15,7,-1)))
        self.assertTrue(all(len(v)<=8 for v in q.values()))

    def test_empty_and_ties(self):
        self.assertEqual(choices([0]*16,[])['top8'],[])
        self.assertEqual(choices([1]*16,[15])['top8'],list(range(8)))

    def test_invalid(self):
        for gains,old in (([1]*15,[]),([np.nan]*16,[]),([1]*16,[0,0]),([1]*16,[16])):
            with self.assertRaises(ValueError):choices(gains,old)


if __name__=='__main__':unittest.main()
