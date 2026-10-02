"""Sender only: new E weights, exact entropy packets, preserved UF base bytes."""
import argparse
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo.four_state_core import (ENHANCEMENT, ROIS, rows, verify, costs,
                                  immutable_json)
from demo.chunk_enhancement_codec import configure_torch, encode_enhancement, load_model
from demo.chunk_enhancement_experiment import read
from demo.scalable_codec import atomic_bytes, atomic_json, file_hash
from demo.scalable_format import frame_hash
from demo.scalable_experiment import check_space


def prepare(root):
    configure_torch()
    protocol = read(root/'protocol.json')
    model = None
    entries = []
    for row in rows(protocol['smoke']):
        sid = row['sample_id']
        dest = root/'encoded'/sid
        dest.mkdir(parents=True, exist_ok=True)
        done = dest/'complete.json'
        if done.exists():
            record = read(done)
            verify(dest, record['artifacts'])
            assert record['pair_hash'] == row['pair_hash']
        else:
            check_space()
            for path, key in [('pair_path','pair_hash'), ('base_path','base_file_hash'),
                              ('feature_path','feature_hash')]:
                assert file_hash(Path(row[path])) == row[key]
            with np.load(row['pair_path'], allow_pickle=False) as cache:
                source, base = cache['source'].copy(), cache['base'].copy()
            assert source.shape == base.shape == (17, 512, 512, 3)
            chunks = torch.load(row['feature_path'], weights_only=True, map_location='cpu')['chunks']
            if model is None:
                model = load_model(ENHANCEMENT).requires_grad_(False)
            begin = time.monotonic()
            prefix, packets, expected, details = encode_enhancement(model, ENHANCEMENT,
                Path(row['base_path']).read_bytes(), source, base, chunks, ROIS, 1., compact=True)
            bank = prefix+b''.join(packets)
            atomic_bytes(dest/'packets.acse', bank)
            record = dict(sample_id=sid, pair_hash=row['pair_hash'],
                expected_base_hash=frame_hash(base), expected_E_hash=frame_hash(expected),
                encoding_seconds=time.monotonic()-begin, packet_details=details,
                per_region_cost=[costs(bank, protocol['profile'], sid, i) for i in range(16)],
                artifacts={'packets.acse':file_hash(dest/'packets.acse')})
            atomic_json(done, record)
        # Receiver index intentionally contains no source image/cache paths.
        entries.append(dict(sample_id=sid, encoded_manifest=file_hash(done)))
        print(f'ENCODE {len(entries)}/{len(protocol["samples"])} {sid}', flush=True)
    immutable_json(root/'encoded/index.json', dict(entries=entries, complete=True))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    prepare(p.parse_args().root)
