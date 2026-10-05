"""RVRC v1: asymmetric RouterVC receiver policy over unchanged ACSE2 packets.

Only the G receiver model is identified here: the source-aware sender model is
not a decoding dependency. No E/G/protection mask or sender prediction is sent.
The binary layout mirrors the older wrapper, but has its own magic and policy.
Always charge the header returned by parse(), not an assumed historical size.
"""
import hashlib
import json
from pathlib import Path
import zlib

from demo import routervc_format as legacy
from demo.scalable_format import parse as parse_inner
from demo.scalable_codec import file_hash
from demo.online_eg_decode import identities

REPO = Path(__file__).resolve().parents[1]
PROFILE = 'routervc_asymmetric_receiver_v1'
MAGIC = b'RVRC'
VERSION = 1
HEADER = legacy.HEADER
CONTROL = legacy.CONTROL
HASHES = ('receiver_router', 'policy', *legacy.cooperation.HASHES)
KEYS = (legacy.KEYS - {'router'}) | {'receiver_router'}
CONTROL_KEYS = ('seed', 'max_g', 'boundary_lambda', 'strength', 'blend',
                'window', 'stride', 'context', 'feather')
CODE = ('routervc_receiver_format.py', 'routervc_receiver_policy.py',
        'routervc_receiver_decode.py', 'routervc_receiver_router.py',
        'routervc_visual_router.py', 'routervc_mixed_router.py',
        'routervc_format.py', 'routervc_policy.py', 'routervc.py',
        'routervc_encode.py', 'scalable_format.py', 'scalable_codec.py',
        'scalable_cooperation_format.py', 'compact_enhancement_format.py',
        'chunk_enhancement_codec.py', 'chunk_enhancement_model.py',
        'feature_head_enhancement.py', 'online_eg_decode.py',
        'internal_condition_decode.py')


def code_identity():
    return {name: file_hash(REPO/'demo'/name) for name in CODE}


def policy_identity():
    from demo.routervc_receiver_policy import policy_identity as router_identity
    binding = dict(profile=PROFILE, magic=MAGIC.decode(), version=VERSION,
                   code=code_identity(), receiver_policy=router_identity())
    return hashlib.sha256(json.dumps(binding, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def validate(config):
    if type(config) is not dict or set(config) != KEYS:
        raise ValueError('unknown/missing asymmetric receiver configuration')
    # Reuse the unchanged scalar/geometry contract, without accepting its wire.
    translated = {('router' if k == 'receiver_router' else k): v
                  for k, v in config.items()}
    legacy.validate(translated)
    if config['policy'] != policy_identity():
        raise ValueError('asymmetric receiver policy/profile mismatch')


def make_config(receiver_router, adapter, max_g=8, boundary_lambda=0., seed=20261005):
    config = dict(identities(adapter), receiver_router=file_hash(receiver_router),
                  policy=policy_identity(), seed=seed, max_g=max_g,
                  boundary_lambda=boundary_lambda, strength=1., blend=1.,
                  window=17, stride=8, context=64, feather=16)
    validate(config)
    return config


def wrap(inner, config):
    validate(config)
    if len(inner) < 5 or inner[:4] != b'ACSE' or inner[4] != 2:
        raise ValueError('asymmetric receiver requires ACSE2')
    parse_inner(inner)
    body = CONTROL.pack(*(config[k] for k in CONTROL_KEYS))
    body += b''.join(bytes.fromhex(config[k]) for k in HASHES)
    return HEADER.pack(MAGIC, VERSION, len(body), zlib.crc32(body)) + body + inner


def parse(data, *, allow_incomplete_tail=False):
    if len(data) < HEADER.size:
        raise ValueError('truncated asymmetric receiver header')
    magic, version, size, crc = HEADER.unpack_from(data)
    if magic != MAGIC or version != VERSION or size != CONTROL.size + 32*len(HASHES):
        raise ValueError('unknown asymmetric receiver header')
    end = HEADER.size + size
    body = data[HEADER.size:end]
    if len(body) != size or zlib.crc32(body) != crc:
        raise ValueError('corrupt asymmetric receiver header')
    config = dict(zip(CONTROL_KEYS, CONTROL.unpack_from(body)))
    offset = CONTROL.size
    for key in HASHES:
        config[key] = body[offset:offset+32].hex()
        offset += 32
    validate(config)
    inner = data[end:]
    if len(inner) < 5 or inner[:4] != b'ACSE' or inner[4] != 2:
        raise ValueError('asymmetric receiver requires ACSE2')
    return config, inner, parse_inner(inner, allow_incomplete_tail=allow_incomplete_tail), end


generation_control = legacy.generation_control


def supervised(args, callback):
    """CLI parent owns tmux, heartbeats and the shared native GPU mutex."""
    import math
    import os
    from demo.chunk_enhancement_experiment import Run
    from demo.chunk_enhancement_evaluate import exclusive_native_evaluation
    from demo.scalable_codec import atomic_json
    if not os.environ.get('TMUX'):
        raise RuntimeError('asymmetric receiver must run in tmux')
    if not math.isfinite(args.max_hours) or args.max_hours <= 0:
        raise ValueError('max-hours must be finite and positive')
    run = Run(args)
    run.thread.start()
    try:
        with exclusive_native_evaluation(run):
            result = callback(args, run)
            run.update(phase='complete', completed=1, total=1)
        print(json.dumps(dict(complete=True, command=args.command, output=str(args.output))), flush=True)
        return result
    except BaseException as error:
        atomic_json(args.output/f'{args.command}.last_failure.json',
                    dict(error=repr(error), progress=run.progress))
        raise
    finally:
        run.log_resources()
        run.stop.set()
        run.thread.join(timeout=3)
