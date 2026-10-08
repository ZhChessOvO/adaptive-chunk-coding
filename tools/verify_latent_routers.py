"""Deep CPU-only verification; never fill missing labels or run inference."""
import argparse
from pathlib import Path

import numpy as np
import torch

from demo.routervc_fullview_probe import read, digest, verify_artifacts
from demo.routervc_mixed_queue import exact
from demo.routervc_receiver_router import load_model
from routervc.latent import router_data as data, routing


def verify_stage(root):
    root = Path(root)
    top = read(root/'complete.json')
    if not top['complete']:
        raise ValueError('receiver stage incomplete')
    verify_artifacts(root,top['artifacts'])
    protocol = read(root/'protocol.json')
    for name, expected in protocol['code'].items():
        if digest(data.REPO/name) != expected:
            raise ValueError('training source changed: '+name)
    samples = data.Samples(root,Path('/unused-read-only-cache'),protocol)
    labels = read(root/'labels.complete.json')
    if labels['protocol'] != digest(root/'protocol.json'):
        raise ValueError('label protocol differs')
    checked = 0
    for row in protocol['rows']:
        folder = root/'samples'/row['sample_id']
        if digest(folder/'complete.json') != labels['samples'][row['sample_id']]:
            raise ValueError('sample completion differs')
        record = samples.verify(row)
        bank = (folder/'bank.rvlp').read_bytes()
        with np.load(record['cache_path'],allow_pickle=False) as tensors:
            targets = tensors['value'].copy()
        for view,count in enumerate(data.COUNTS):
            state = read(folder/f'e{count}.json')
            chosen = data.selection(row['sample_id'],count)
            inner = routing.subset(bank,chosen)
            import hashlib
            if (state['binding']['input_sha256'] != hashlib.sha256(inner).hexdigest()
                    or state['binding']['input_bytes'] != len(inner)
                    or state['binding']['selected'] != chosen
                    or sorted(map(int,state['cells'])) != list(range(16))):
                raise ValueError('incomplete or changed teacher state')
            measured = np.array([state['cells'][str(i)]['gains'] for i in range(16)],np.float32)
            np.testing.assert_array_equal(targets[view],measured)
            for region in range(16):
                cell = state['cells'][str(region)]
                if cell['control'] != routing.control(row['shape'],protocol['G_assets'],region,protocol['seed']):
                    raise ValueError('G teacher processing/noise changed')
                if not cell['outside_generate_exact'] or not cell['source_only_offline_scoring']:
                    raise ValueError('invalid teacher isolation')
                expected = [cell['before']['lpips_alex']-cell['after']['lpips_alex'],
                    cell['after']['psnr_db']-cell['before']['psnr_db'],
                    cell['before']['temporal_delta_mae']-cell['after']['temporal_delta_mae']]
                np.testing.assert_array_equal(np.array(cell['gains']),np.array(expected))
                checked += 1
    if checked != labels['measured_G_cells']:
        raise ValueError('measured cell total differs')
    for name in ('resumed','direct') if protocol['smoke'] else ('router',):
        folder = root/name
        done = read(folder/'complete.json')
        if done['config'] != digest(folder/'config.json'):
            raise ValueError('fit config changed')
        verify_artifacts(folder,done['artifacts'])
        resume = torch.load(folder/'resume.pt',weights_only=True,map_location='cpu')
        _, best = load_model(folder/'core/best.pt',expected_sha256=done['artifacts']['core/best.pt'])
        if (resume['state']['epoch'] != protocol['epochs'] or resume['state']['cursor'] != 0
                or resume['binding']['protocol'] != protocol or best['binding'] != resume['binding']):
            raise ValueError('training did not reach the declared terminal state')
        exact(best['state_dict'],resume['state']['best']['weights'])
        exact(best['selection'],resume['state']['best']['score'])
    if protocol['smoke']:
        exact(torch.load(root/'resumed/resume.pt',weights_only=True),
              torch.load(root/'direct/resume.pt',weights_only=True))
        for path in (root/'fresh').glob('*/*/complete.json'):
            verify_artifacts(path.parent,read(path)['artifacts'])
        interrupted = read(root/'label_resume_checked.json')
        import hashlib, json
        for path,cells in interrupted['previous_cell_hashes'].items():
            current = read(path)['cells']
            for i,expected in cells.items():
                if hashlib.sha256(json.dumps(current[i],sort_keys=True).encode()).hexdigest() != expected:
                    raise ValueError('interrupted teacher cell changed')
    return dict(complete=True,stage=str(root),measured_cells=checked,
                compressed_targets_match_measured_scores=True,training_checkpoint_and_best_verified=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',type=Path)
    args=p.parse_args()
    print(verify_stage(args.stage),flush=True)
