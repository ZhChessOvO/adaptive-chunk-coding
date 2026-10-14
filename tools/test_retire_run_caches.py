import unittest
from tools.retire_run_caches import RUNS, REAL_RUNS, selected


class SelectionTests(unittest.TestCase):
    def test_only_old_cache_files(self):
        root=RUNS/'routervc_revision_20261003/mixedview_teacher/encoded/a/prepared'
        self.assertTrue(selected(root/'candidates.npz'))
        self.assertTrue(selected(REAL_RUNS/(root/'candidates.npz').relative_to(RUNS)))
        for name in ('source.npz','complete.json','bank.acse','adapter.pt','preview.png'):
            self.assertFalse(selected(root/name))

    def test_active_and_final_evidence_never_selected(self):
        for rel in ('routervc_fusion_20261010/p1_data/samples/x/received.npz',
                    'routervc_texture_context_20261015/training/received_E.npz',
                    'routervc_receiver_20261005/evaluation/samples/x/generated.npz',
                    'routervc_revision_20261003/mixedview_data/outputs.npz'):
            self.assertFalse(selected(RUNS/rel))


if __name__=='__main__':unittest.main()
