import unittest
from unittest.mock import patch
import tools.retire_run_caches as retirement
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

    def test_expanded_scope_removes_pixels_not_compact_evidence(self):
        fields=('GROUPS','PROTECTED','PROTECTED_FILES','AUDIT','REASON')
        with patch.multiple(retirement,**{k:getattr(retirement,k) for k in fields}), \
             patch.object(retirement,'dependency_inventory',return_value={}):
            retirement.expanded_scope()
            root=RUNS/'routervc_precision_20261012/internal_evaluation/x'
            self.assertTrue(selected(root/'pixels.npz'))
            for name in ('result.json','fixed_frame.png','stream.bin','adapter.pt','source.npz'):
                self.assertFalse(selected(root/name))
            self.assertTrue(selected(RUNS/'routervc_latent_sender_20261009/input_cache/formal/sample.npz'))
            self.assertFalse(selected(RUNS/'routervc_texture_context_20261015/probe/x/output.npz'))
            self.assertFalse(selected(RUNS/'routervc_fusion_20261010/p1_data/samples/x/received.npz'))

    def test_current_dependencies_protected_under_both_mount_aliases(self):
        root=RUNS/'routervc_precision_20261012/internal_evaluation/x/pixels.npz'
        real=REAL_RUNS/root.relative_to(RUNS)
        fields=('GROUPS','PROTECTED','PROTECTED_FILES','AUDIT','REASON')
        with patch.multiple(retirement,**{k:getattr(retirement,k) for k in fields}), \
             patch.object(retirement,'dependency_inventory',return_value={str(root):{'resolved':str(real)}}):
            retirement.expanded_scope()
            self.assertFalse(selected(root));self.assertFalse(selected(real))


if __name__=='__main__':unittest.main()
