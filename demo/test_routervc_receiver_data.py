"""Fast CPU contracts for six-state supervision and bounded compact caches."""
import tempfile
from pathlib import Path
import unittest

import torch

from demo import routervc_receiver_data as data
from demo import routervc_mixed_router as mixed
from demo.routervc_fullview_probe import save, digest
from demo.chunk_enhancement_codec import atomic_torch


class ReceiverDataTest(unittest.TestCase):
    def test_nested_states_reuse_historical_mixtures(self):
        for sid in ('reds_train_001', 'uvg_beauty_032'):
            selected = [data.selection(sid,c) for c in data.COUNTS]
            for count, values in zip(data.COUNTS,selected):
                self.assertEqual(len(values),count)
                self.assertEqual(len(set(values)),count)
            for before, after in zip(selected,selected[1:]):
                self.assertEqual(before,after[:len(before)])
            self.assertEqual(selected[2],mixed.selection(sid,0))
            self.assertEqual(selected[3],mixed.selection(sid,1))

    def test_invalid_region_count(self):
        with self.assertRaises(ValueError):
            data.selection('sample',3)

    def fixture(self, root, cache, sid):
        row = dict(sample_id=sid,dataset='REDS')
        path=cache/f'{sid}.pt'
        atomic_torch(path,dict(inputs={'coverage':torch.zeros(1,16,1)}))
        dest=root/'samples'/sid
        dest.mkdir(parents=True)
        save(dest/'complete.json',dict(binding=dict(protocol='test',sample=row),
             cache_sha256=digest(path),artifacts={},reused_artifacts={}))
        return row

    def test_bounded_lru_and_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'formal';cache=Path(directory)/'cache'
            cache.mkdir()
            samples=data.Samples(root,cache,'test',max_cached=2)
            rows=[self.fixture(root,cache,str(i)) for i in range(3)]
            for row in rows:
                self.assertEqual(samples.get(row,generate=False)['sample_id'],row['sample_id'])
            self.assertEqual(list(samples.memory),['1','2'])
            samples.get(rows[1],generate=False)
            self.assertEqual(list(samples.memory),['2','1'])
            samples.get(rows[0],generate=False)
            self.assertEqual(list(samples.memory),['1','0'])
            self.assertEqual(len(samples.verified),3)

    def test_cache_tampering_and_no_implicit_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'formal';cache=Path(directory)/'cache'
            cache.mkdir()
            row=self.fixture(root,cache,'one')
            samples=data.Samples(root,cache,'test')
            atomic_torch(cache/'one.pt',dict(changed=True))
            with self.assertRaisesRegex(ValueError,'tensors changed'):
                samples.get(row,generate=False)
            with self.assertRaisesRegex(ValueError,'cannot generate'):
                samples.get(dict(sample_id='missing'),generate=False)


if __name__=='__main__':
    unittest.main()
