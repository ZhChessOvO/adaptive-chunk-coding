"""CPU tests for binding, no-mask bytes, inactive compatibility and failures."""
import copy
import subprocess
import sys
from unittest import TestCase
from unittest.mock import patch
import torch
from routervc.generation import integration as g
from tools.g_integration import validate


def bundle():
    from demo.internal_condition_model import FORMAT
    return dict(rank=8,alpha=8.,target_modules=['x'],last_n_blocks=8,trainable_parameter_count=4,
        state_dict={'x':torch.ones(2,2)},feature_format='old',feature_enabled=False,feature_state={},
        internal_condition_format=FORMAT,branch_kind='internal',branch_mode='off',branch_state={})


def receipt():
    condition={k:'same' for k in ('before_vae','after_vae','conditions','diffusion_noise',
        'effective_condition','before_diffusion','rgb_condition_unchanged')}
    return dict(base_hash='B',enhanced_hash='Y',actual_bytes=200,bpp=.1,header_bytes=121,mask_bytes=0,
        source_frames_read=False,sender_router_loaded=False,sender_checkpoint_access_guard=True,
        generation_disabled=False,output_hash='G',generated=[2],selected={'indices':[2]},
        generation_runtime=[dict(crop=[0,0,256,256],core=[64,64,128,128],
            runtime=dict(condition_windows=[condition]))])


class IntegrationTests(TestCase):
    def test_export_preserves_candidate_and_copies_only_disabled_interface(self):
        legacy=bundle(); candidate={k:v for k,v in bundle().items() if k not in g.BRANCH_KEYS}
        candidate['state_dict']['x'].fill_(2)
        result=g.compatible_bundle(candidate,legacy)
        self.assertTrue(torch.equal(result['state_dict']['x'],candidate['state_dict']['x']))
        self.assertFalse(torch.equal(result['state_dict']['x'],legacy['state_dict']['x']))
        self.assertNotIn('branch_state',candidate); self.assertEqual(result['branch_mode'],'off')
        result['state_dict']['x'].zero_(); self.assertTrue(candidate['state_dict']['x'].all())

    def test_active_interface_rejected(self):
        for key,value in (('branch_mode','actual'),('feature_enabled',True)):
            old=bundle();old[key]=value
            with self.assertRaises(ValueError): g.compatible_bundle(bundle(),old)

    def test_architecture_nonfinite_rejected(self):
        for change in ('rank','tensor'):
            candidate=bundle()
            if change=='rank': candidate['rank']=4
            else: candidate['state_dict']['x'][0,0]=float('nan')
            with self.assertRaises(ValueError): g.compatible_bundle(candidate,bundle())

    def test_envelope_changes_identity_not_payload_or_size(self):
        with patch.object(g.stream,'wrap'),patch.object(g,'identity',return_value='11'*32):
            wire=g.wrap(b'payload','22'*32,'33'*32)
            inner,info=g.parse(wire)
            self.assertEqual(inner,b'payload');self.assertEqual(len(wire),121+7)
            self.assertEqual(info['assets_sha256'],'33'*32)
            different=g.wrap(inner,'22'*32,'44'*32)
            self.assertEqual(len(different),len(wire));self.assertNotEqual(different,wire)
            with self.assertRaises(ValueError):g.stream.parse(wire)

    def test_corruption_and_wrong_profile_rejected(self):
        with patch.object(g.stream,'wrap'),patch.object(g,'identity',return_value='11'*32):
            wire=g.wrap(b'payload','22'*32,'33'*32)
            for i in (0,9,44,105,117):
                bad=bytearray(wire);bad[i]^=1
                with self.assertRaises(ValueError):g.parse(bytes(bad))
            with self.assertRaises(ValueError):g.parse(wire[:80])

    def test_contract_and_legacy_exactness(self):
        r=receipt();validate(r,r,legacy=True)
        validate(dict(r,output_hash='candidate'),r)
        with self.assertRaises(ValueError):validate(dict(r,output_hash='candidate'),r,legacy=True)
        for key,value in (('mask_bytes',1),('base_hash','changed'),('actual_bytes',201),
                          ('source_frames_read',True),('generated',[3])):
            with self.assertRaises(ValueError):validate(dict(r,**{key:value}),r)

    def test_pairing_checks_actual_noise(self):
        r=receipt();wrong=copy.deepcopy(r)
        wrong['generation_runtime'][0]['runtime']['condition_windows'][0]['diffusion_noise']='changed'
        with self.assertRaises(ValueError):validate(wrong,r)

    def test_g_off_contract(self):
        r=receipt();off=dict(r,generation_disabled=True,output_hash='Y',generated=[],selected=None)
        validate(off,r,disabled=True)
        with self.assertRaises(ValueError):validate(dict(off,output_hash='G'),r,disabled=True)

    def test_source_and_sender_read_guard(self):
        code="""
from pathlib import Path
import sys
from tools.latent_router_worker import receiver_guard
from tools.cooperative_sender_decode import deny_sender_checkpoint
receiver_guard(Path('/tmp/routervc-test-output'),Path('/tmp/routervc-test-output/stream.rvlg'))
sys.addaudithook(deny_sender_checkpoint)
for name in ('/root/autodl-fs/DCVC/data/a.png','/tmp/source.npz','/tmp/protocol.json',
             '/root/autodl-fs/DCVC/runs/sender_test/best.pt'):
    try: open(name,'rb')
    except RuntimeError: pass
    else: raise AssertionError(name)
"""
        subprocess.run([sys.executable,'-c',code],check=True,capture_output=True)
