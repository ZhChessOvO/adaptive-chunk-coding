"""RouterVC v1: shared receiver policy around append-only ACSE2 packets.

No G map is transmitted. The fixed header identifies models, deterministic
policy and budgets; it does not contain encoder-only predictions or images.
ACSE2 is unmodified, so complete packet prefixes remain independently valid.
"""
import hashlib
import math
from pathlib import Path
import struct
import zlib

from demo.scalable_format import parse as parse_inner
from demo.scalable_codec import file_hash
from demo.online_eg_decode import identities
from demo import scalable_cooperation_format as cooperation

HEADER = struct.Struct('<4sBII')
CONTROL = struct.Struct('<QBddd4H')
HASHES = ('router', 'policy', *cooperation.HASHES)
KEYS = set(HASHES) | {'seed', 'max_g', 'boundary_lambda', 'strength', 'blend',
                     'window', 'stride', 'context', 'feather'}
REPO = Path(__file__).resolve().parents[1]


def policy_identity():
    # Explicit pin: changing routing/features/decoder requires a new profile.
    names = ('routervc_format.py', 'routervc_policy.py', 'routervc_decode.py',
             'four_state_router.py', 'four_state_router_evaluate.py')
    return hashlib.sha256(''.join(file_hash(REPO/'demo'/n) for n in names).encode()).hexdigest()


def make_config(router, adapter, max_g=8, boundary_lambda=0., seed=20261003):
    return dict(identities(adapter), router=file_hash(router), policy=policy_identity(),
                seed=seed, max_g=max_g, boundary_lambda=boundary_lambda,
                strength=1., blend=1., window=17, stride=8, context=64, feather=16)


def validate(config):
    if set(config) != KEYS:
        raise ValueError('unknown/missing RouterVC configuration')
    for key in HASHES:
        value = config[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('invalid model/profile digest')
        try:
            if len(bytes.fromhex(value)) != 32: raise ValueError()
        except ValueError: raise ValueError('invalid model/profile digest') from None
    if type(config['seed']) is not int or not 0 <= config['seed'] < 2**63:
        raise ValueError('invalid seed')
    if type(config['max_g']) is not int or not 0 <= config['max_g'] <= 16:
        raise ValueError('invalid G call budget')
    for key, upper in [('boundary_lambda', 1.), ('strength', 1.), ('blend', 1.)]:
        value = config[key]
        if type(value) not in (int,float) or not math.isfinite(value) or not 0 <= value <= upper:
            raise ValueError('invalid strength/boundary parameter')
    if any(type(config[k]) is not int for k in ('window','stride','context','feather')):
        raise ValueError('noninteger execution profile')
    if tuple(config[k] for k in ('window','stride','context','feather')) != (17,8,64,16):
        raise ValueError('unsupported RouterVC execution profile')
    if config['strength'] != 1.:
        raise ValueError('RouterVC v1 requires full-strength jointly trained LoRA')


def wrap(inner, config):
    validate(config)
    if len(inner) < 5 or inner[:4] != b'ACSE' or inner[4] != 2:
        raise ValueError('RouterVC requires ACSE2')
    parse_inner(inner)
    body = CONTROL.pack(*(config[k] for k in ('seed','max_g','boundary_lambda',
        'strength','blend','window','stride','context','feather')))
    body += b''.join(bytes.fromhex(config[k]) for k in HASHES)
    return HEADER.pack(b'RTVC',1,len(body),zlib.crc32(body))+body+inner


def parse(data, *, allow_incomplete_tail=False):
    if len(data) < HEADER.size: raise ValueError('truncated RouterVC header')
    magic,version,size,crc = HEADER.unpack_from(data)
    if magic != b'RTVC' or version != 1 or size != CONTROL.size+32*len(HASHES):
        raise ValueError('unknown RouterVC header')
    end=HEADER.size+size;body=data[HEADER.size:end]
    if len(body)!=size or zlib.crc32(body)!=crc: raise ValueError('corrupt RouterVC header')
    config=dict(zip(('seed','max_g','boundary_lambda','strength','blend',
                     'window','stride','context','feather'),CONTROL.unpack_from(body)))
    offset=CONTROL.size
    for key in HASHES:
        config[key]=body[offset:offset+32].hex();offset+=32
    validate(config)
    inner=data[end:]
    if len(inner)<5 or inner[:4]!=b'ACSE' or inner[4]!=2: raise ValueError('requires ACSE2')
    return config,inner,parse_inner(inner,allow_incomplete_tail=allow_incomplete_tail),end


def generation_control(config, indices, rois, count):
    return dict({k:config[k] for k in cooperation.HASHES},seed=config['seed'],
        strength=config['strength'],window=config['window'],stride=config['stride'],
        context=config['context'],feather=config['feather'],blend=config['blend'],
        processing_scale=1,protect=[],generate=[[0,count,*rois[i]] for i in indices])
