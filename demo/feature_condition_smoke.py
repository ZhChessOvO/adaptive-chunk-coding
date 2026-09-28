"""Fresh-process checks of the new profile, without receiver source access."""
from pathlib import Path

import numpy as np
import torch

from demo import scalable_cooperation_format as fmt
from demo.chunk_enhancement_experiment import read
from demo.conditioned_generation_pipeline import execute
from demo.feature_condition_cache import PREVIOUS
from demo.feature_condition_decode import identities
from demo.feature_condition_model import FeatureCondition, FORMAT
from demo.patch_prefix_probe import load_frames
from demo.scalable_codec import atomic_bytes, atomic_json, file_hash
from demo.stage_c_seedvr2_lora_utils import atomic_torch_save

OLD = Path('/root/autodl-fs/DCVC/runs/a800_scalable_cooperation_20260928')


def decode_smoke(root, run):
    reference = read(OLD/'summary.json')['results'][0]
    sid = reference['sample']['sample_id']
    zero = torch.load(PREVIOUS/'image/adapter.pt',weights_only=True,map_location='cpu')
    zero.update(feature_format=FORMAT,feature_enabled=True,
                feature_state=FeatureCondition().state_dict())
    zero_path = root/'zero_adapter.pt'
    atomic_torch_save(zero_path,zero)
    trained = root/'feature_resume/adapter.pt'
    results = {}
    for label, mode, adapter, disabled in [
        ('zero_full','full',zero_path,False),('zero_none','none',zero_path,False),
        ('trained_full','full',trained,False),('repeat_full','full',trained,False),
        ('off_full','full',trained,True)]:
        old_name = 'cooperate_l05' if mode == 'full' else 'generate_l05'
        control,inner,_,_ = fmt.parse((OLD/sid/f'{old_name}.acsg').read_bytes())
        control.update(identities(adapter)); control['strength'] = 1.
        data = fmt.wrap(inner,control)
        assert len(data) == (OLD/sid/f'{old_name}.acsg').stat().st_size
        dest = root/'decode_smoke'/label
        dest.mkdir(parents=True,exist_ok=True)
        stream = dest/'stream.acsg'
        atomic_bytes(stream,data)
        argv = ['--stream',stream,'--output',dest,'--adapter',
                Path('/nonexistent/no_generator.pt') if disabled else adapter]
        if disabled:
            argv += ['--disable-generation']
        execute(run,label,'feature_condition_decode.py',argv,distributed=True)
        pixels = load_frames(dest/'reconstruction.npz')
        report = read(dest/'decode.json')
        if label.startswith('zero'):
            np.testing.assert_array_equal(pixels,
                load_frames(PREVIOUS/'evaluation'/sid/f'image_{mode}/reconstruction.npz'))
        elif label == 'off_full':
            np.testing.assert_array_equal(pixels,load_frames(OLD/sid/'enhance_q1/reconstruction.npz'))
            assert not report['generation_assets_validated']
        elif label == 'repeat_full':
            np.testing.assert_array_equal(pixels,load_frames(root/'decode_smoke/trained_full/reconstruction.npz'))
        if label == 'trained_full':
            assert any(w['feature_side_rms'] > 0 for w in report['generation_runtime']['windows'])
        assert not report['source_frames_read'] and report['feature_bytes_added'] == 0
        results[label] = dict(stream_sha256=file_hash(stream),report=report)
    atomic_json(root/'decode_check.json',dict(complete=True,zero_adapter_full_exact=True,
        no_E_zero_branch_exact=True,repeat_exact=True,G_off_without_weights_exact=True,
        original_E_payload_unchanged=True,results=results))
