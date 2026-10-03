"""Isolated visual-router profile using the unchanged, charged RTVC v1 syntax.

This changes the shared policy identity, not UF/E/G weights or ACSE2 packets.
There is no per-region G/protection map. Historical modules remain untouched.
"""
import hashlib
import json
from pathlib import Path

from demo import routervc_format as legacy
from demo.scalable_codec import file_hash

REPO = Path(__file__).resolve().parents[1]
PROFILE = 'routervc_visual_perceptual_v1'
HASHES = legacy.HASHES
CODE = ('routervc_visual_format.py', 'routervc_visual_encode.py',
        'routervc_visual_decode.py', 'routervc_visual_policy.py',
        'routervc_visual_router.py', 'routervc_encode.py', 'routervc_decode.py',
        'routervc.py', 'routervc_format.py', 'routervc_policy.py',
        'four_state_router.py', 'four_state_router_evaluate.py',
        'scalable_codec.py', 'scalable_cooperation_format.py',
        'chunk_enhancement_codec.py', 'chunk_enhancement_model.py',
        'feature_head_enhancement.py', 'online_eg_decode.py')


def code_identity():
    return {name: file_hash(REPO/'demo'/name) for name in CODE}


def policy_identity():
    from demo.routervc_visual_policy import policy_identity as router_identity
    binding = dict(profile=PROFILE, code=code_identity(), router_policy=router_identity(),
                   semantic_heads_used=False)
    return hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()


def validate(config):
    legacy.validate(config)
    if config['policy'] != policy_identity():
        raise ValueError('visual Router policy/profile mismatch')


def make_config(router, adapter, **kwargs):
    config = legacy.make_config(router, adapter, **kwargs)
    config['policy'] = policy_identity()
    validate(config)
    return config


def wrap(inner, config):
    validate(config)
    return legacy.wrap(inner, config)


def parse(data, *, allow_incomplete_tail=False):
    result = legacy.parse(data, allow_incomplete_tail=allow_incomplete_tail)
    validate(result[0])
    return result


generation_control = legacy.generation_control


def supervised(args, callback):
    """CLI-only tmux/heartbeat/mutex scope; children must not reacquire the lock."""
    import math
    import os
    from demo.chunk_enhancement_experiment import Run
    from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
    from demo.scalable_codec import atomic_json
    if not os.environ.get('TMUX'):
        raise RuntimeError('visual Router encode/decode must run in tmux')
    if not math.isfinite(args.max_hours) or args.max_hours <= 0:
        raise ValueError('max-hours must be finite and positive')
    run = Run(args); run.thread.start()
    try:
        with exclusive_native_evaluation(run):
            result = callback(args, run)
            run.update(phase='complete', completed=1, total=1)
        print(json.dumps(dict(complete=True, command=args.command, output=str(args.output))), flush=True)
        return result
    except BaseException as error:
        atomic_json(args.output/f'{args.command}.last_failure.json', dict(error=repr(error), progress=run.progress))
        raise
    finally:
        run.log_resources(); run.stop.set(); run.thread.join(timeout=3)
