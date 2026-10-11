"""Paired warm-start R_s adaptation on measured cooperative final marginals.

Both arms start from the same trained source-aware sender with reset AdamW.
Only the zero_source arm removes source-bearing channels. The frozen sender's
actual prefix supplies teacher states; this module never declares those labels
to be independent of the current sender policy. All labels precede optimization.
"""
from collections import Counter
from dataclasses import asdict, replace
import math
from pathlib import Path
import time

import torch

from demo import routervc_sender_router as sender
from demo import routervc_sender_train as old
from demo.chunk_enhancement_codec import atomic_torch
from demo.four_state_receive import codec_precision
from demo.routervc_fullview_probe import digest, immutable, read, save, verify_artifacts

FORMAT = 'cooperative_sender_p2b_v1'
LABEL_FORMAT = 'cooperative_sender_labels_v1'
SCALE_FORMAT = 'cooperative_sender_train_final_delta_scale_v1'
ARMS = ('source', 'zero_source')
arm_batch = old.arm_batch
validate = old.validate
semantic_hash = old.semantic_hash
cpu_tree = old.cpu_tree


def _protocol(protocol):
    train, valid = old.split_rows(protocol)
    if (protocol.get('format') != FORMAT or tuple(protocol.get('arms', ())) != ARMS
            or type(protocol.get('smoke')) is not bool
            or type(protocol.get('epochs')) is not int or protocol['epochs'] <= 0
            or type(protocol.get('seed')) is not int):
        raise ValueError('invalid cooperative sender training protocol')
    if not protocol['smoke'] and (
            len(protocol['rows']) != 60 or len(train) != 48 or len(valid) != 12
            or Counter(r['dataset'] for r in protocol['rows']) != dict(REDS=45, UVG=15)
            or protocol['epochs'] != 120):
        raise ValueError('formal sender pilot requires 60 windows, 45/15 views, 48/12 split and 120 epochs')
    for name in ('learning_rate', 'weight_decay', 'ranking_weight'):
        value = protocol.get(name)
        if (type(value) not in (int, float) or not math.isfinite(value)
                or value < 0 or (name == 'learning_rate' and value == 0)):
            raise ValueError('invalid sender optimization setting: '+name)
    initialization = protocol.get('initialization', {})
    if (initialization.get('kind') != 'same_trained_source_sender_for_both_arms'
            or initialization.get('optimizer_reset') is not True
            or not isinstance(initialization.get('path'), str)
            or not sender._valid_hash(initialization.get('sha256'))):
        raise ValueError('explicit trained source sender and reset optimizer binding required')
    config = sender.Config(**protocol['teacher']['sender_config'])
    if config.zero_source:
        raise ValueError('teacher cache must retain source; ablate inside zero_source only')
    return train, valid, config


def _validate_label_binding(protocol, value):
    count = len(protocol['rows'])
    samples = value.get('samples', {})
    if (value.get('format') != LABEL_FORMAT or value.get('complete') is not True
            or value.get('protocol_sha256') != semantic_hash(protocol)
            or not isinstance(samples, dict)
            or set(samples) != {r['sample_id'] for r in protocol['rows']}
            or not all(sender._valid_hash(v) for v in samples.values())
            or value.get('windows') != count
            or value.get('measured_final_renderings') != 8*count
            or value.get('measured_marginals') != 6*count
            or value.get('label_scope') != sender.LABEL_SCOPE
            or value.get('current_sender_policy_used') is not True
            or value.get('no_UF_E_G_Rg_F_updates') is not True):
        raise ValueError('all measured actual-trajectory cooperative labels must be complete before fitting')


def fit_scale(protocol, get, check=lambda: None):
    """Fit the common loss scale using all six signed TRAIN labels per row."""
    record = old.fit_scale(protocol, get, check)
    record.update(format=SCALE_FORMAT, protocol_sha256=semantic_hash(protocol))
    return record


def _validate_scale(protocol, scale):
    train, _ = old.split_rows(protocol)
    if (scale.get('format') != SCALE_FORMAT
            or scale.get('protocol_sha256') != semantic_hash(protocol)
            or scale.get('train_samples') != [r['sample_id'] for r in train]
            or scale.get('measured_marginals') != 6*len(train)
            or scale.get('validation_used') is not False
            or scale.get('receiver_scale_reused') is not False
            or scale.get('loss_only_normalization') is not True
            or scale.get('label_scope') != sender.LABEL_SCOPE
            or not sender._valid_hash(scale.get('observations_sha256'))
            or type(scale.get('value')) not in (int, float)
            or not math.isfinite(scale['value']) or scale['value'] <= 0):
        raise ValueError('fresh finite TRAIN-only cooperative final-marginal scale required')


def _initial_model(protocol):
    initialization = protocol['initialization']
    model, payload = sender.load_model(initialization['path'],
                                      expected_sha256=initialization['sha256'])
    if (payload['step'] <= 0 or model.config.zero_source
            or payload['binding'].get('arm') != 'source'
            or asdict(model.config) != protocol['teacher']['sender_config']):
        raise ValueError('initialization must be a trained compatible source-aware R_s')
    # load_model intentionally freezes deployment models; fitting must re-enable gradients.
    return model.requires_grad_(True)


def run_training(output, protocol, get, *, scale_record, labels_binding,
                 device='cuda', check=lambda: None, progress=lambda **kw: None, stop_after=0):
    """Atomically update both arms; replay the pair if interrupted between arms."""
    train, valid, config = _protocol(protocol)
    _validate_label_binding(protocol, labels_binding)
    _validate_scale(protocol, scale_record)
    if type(stop_after) is not int or stop_after < 0:
        raise ValueError('stop_after must be a nonnegative paired update count')
    architectures = {arm: asdict(replace(config, zero_source=(arm == 'zero_source'))) for arm in ARMS}
    binding = dict(protocol=protocol, architectures=architectures, labels=labels_binding,
                   scale=scale_record, device_type=torch.device(device).type)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    immutable(output/'config.json', binding)
    if (output/'complete.json').exists():
        return verify_training(output, protocol, labels_binding=labels_binding, scale_record=scale_record)
    check()
    # Receipts bind the rendered states; refitting also catches a stale supplied
    # scalar/observation ledger before any model or optimizer update.
    if fit_scale(protocol, get, check) != scale_record:
        raise ValueError('sender TRAIN observations changed after scale fitting')
    with old.deterministic_training(protocol['seed'], device):
        return _run_training(output, binding, get, train, valid, device, check, progress, stop_after)


def _run_training(output, binding, get, train, valid, device, check, progress, stop_after):
    protocol, scale = binding['protocol'], float(binding['scale']['value'])
    initial = _initial_model(protocol)
    initial_weights = cpu_tree(initial.state_dict())
    models, optimizers = {}, {}
    for arm in ARMS:
        model = (initial if arm == 'source' else
                 sender.SenderUtilityRouter(sender.Config(**binding['architectures'][arm])))
        model.load_state_dict(initial_weights, strict=True)
        models[arm] = model.to(device).requires_grad_(True)
        optimizers[arm] = torch.optim.AdamW(model.parameters(), lr=protocol['learning_rate'],
                                           weight_decay=protocol['weight_decay'])
    state = dict(epoch=0, cursor=0, updates=0, history=[], best={}, initial_validation=None,
                 epoch_losses={arm: 0. for arm in ARMS})
    resume = output/'resume.pt'
    if resume.exists():
        loaded = torch.load(resume, weights_only=True, map_location='cpu')
        if loaded['binding'] != binding:
            raise ValueError('cooperative sender resume binding changed')
        for arm in ARMS:
            models[arm].load_state_dict(loaded['models'][arm], strict=True)
            optimizers[arm].load_state_dict(loaded['optimizers'][arm])
        state = cpu_tree(loaded['state'])
        if (not 0 <= state['epoch'] <= protocol['epochs']
                or not 0 <= state['cursor'] <= len(train)
                or state['updates'] != state['epoch']*len(train)+state['cursor']
                or len(state['history']) != state['epoch']
                or (state['epoch'] == protocol['epochs'] and state['cursor'] != 0)):
            raise ValueError('invalid cooperative sender checkpoint cursor/history')
        old.restore_rng(loaded['rng'], device)

    def checkpoint():
        atomic_torch(resume, dict(binding=binding, state=cpu_tree(state), rng=old.capture_rng(device),
            models={arm: cpu_tree(model.state_dict()) for arm, model in models.items()},
            optimizers={arm: cpu_tree(opt.state_dict()) for arm, opt in optimizers.items()}))

    def export(arm, weights, kind, step, score, epoch):
        model = models[arm]
        current = cpu_tree(model.state_dict())
        model.load_state_dict(weights, strict=True)
        model_binding = dict(protocol=protocol, architecture=binding['architectures'][arm], arm=arm,
                             labels=binding['labels'], scale=binding['scale'])
        atomic_torch(output/arm/(kind+'.pt'), sender.export_payload(model, model_binding, step,
            dict(epoch=epoch, validation=score, scope='measured candidate ranking; not full-video RD')))
        model.load_state_dict(current, strict=True)

    ranking = protocol['ranking_weight']
    if state['initial_validation'] is None:
        state['initial_validation'] = {arm: validate(models[arm], arm, valid, get, scale,
            device, check, ranking) for arm in ARMS}
        checkpoint()
    if stop_after and state['updates'] >= stop_after:
        return None
    began = time.monotonic()
    for epoch in range(state['epoch'], protocol['epochs']):
        order = torch.randperm(len(train), generator=torch.Generator().manual_seed(
            protocol['seed']+epoch)).tolist()
        for position in range(state['cursor'], len(order)):
            check()
            row = train[order[position]]
            data, losses = get(row), {}
            for arm in ARMS:
                model, optimizer = models[arm], optimizers[arm]
                batch = arm_batch(data, arm, device)
                with codec_precision():
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    loss, _ = sender.training_loss(model(batch['inputs']), batch['targets'],
                        batch['packet_bytes'], gain_scale=scale, ranking_weight=ranking)
                    if not torch.isfinite(loss):
                        raise RuntimeError('nonfinite cooperative sender loss')
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                    if not torch.isfinite(norm):
                        raise RuntimeError('nonfinite cooperative sender gradient')
                    optimizer.step()
                losses[arm] = float(loss.detach())
            for arm in ARMS:
                state['epoch_losses'][arm] += losses[arm]
            state.update(cursor=position+1, updates=state['updates']+1)
            checkpoint()
            status = dict(phase='smoke_sender_optimization' if protocol['smoke'] else 'formal_sender_optimization',
                epoch=epoch+1, epochs=protocol['epochs'], window=position+1, train_windows=len(train),
                updates_per_arm=state['updates'], sample_id=row['sample_id'], losses=losses,
                both_sender_optimizers_updated=True, all_final_labels_ready=True, checkpoint_loadable=True,
                current_sender_policy_used=True, no_UF_E_G_Rg_F_updates=True,
                seconds_this_process=time.monotonic()-began)
            save(output/'progress.json', status)
            progress(**status)
            print('COOPERATIVE_SENDER_TRAIN', status, flush=True)
            if stop_after and state['updates'] >= stop_after:
                return None
        scores = {arm: validate(models[arm], arm, valid, get, scale, device, check, ranking)
                  for arm in ARMS}
        for arm in ARMS:
            previous, score = state['best'].get(arm), scores[arm]
            if previous is None or (score['regret'], score['loss']) < (
                    previous['score']['regret'], previous['score']['loss']):
                state['best'][arm] = dict(epoch=epoch+1, step=state['updates'], score=score,
                                         weights=cpu_tree(models[arm].state_dict()))
        state['history'].append(dict(epoch=epoch+1, validation=scores,
            train_loss={arm: state['epoch_losses'][arm]/len(train) for arm in ARMS}))
        state.update(epoch=epoch+1, cursor=0, epoch_losses={arm: 0. for arm in ARMS})
        checkpoint()
        save(output/'history.json', state['history'])
        save(output/'initial_validation.json', state['initial_validation'])
        for arm in ARMS:
            best = state['best'][arm]
            export(arm, best['weights'], 'best', best['step'], best['score'], best['epoch'])
        print(f'COOPERATIVE_SENDER_EPOCH {epoch+1}/{protocol["epochs"]}', flush=True)
    # Resume.pt is authoritative even if interruption occurred during final exports.
    save(output/'history.json', state['history'])
    save(output/'initial_validation.json', state['initial_validation'])
    for arm in ARMS:
        best = state['best'][arm]
        export(arm, models[arm].state_dict(), 'last', state['updates'],
               state['history'][-1]['validation'][arm], protocol['epochs'])
        export(arm, best['weights'], 'best', best['step'], best['score'], best['epoch'])
    names = ['resume.pt', 'history.json', 'initial_validation.json']+[
        f'{arm}/{kind}.pt' for arm in ARMS for kind in ('best', 'last')]
    result = dict(complete=True, config=digest(output/'config.json'), epochs=protocol['epochs'],
        updates_per_arm=state['updates'], parameters_per_arm=sum(p.numel() for p in models['source'].parameters()),
        selected={arm: dict(epoch=v['epoch'], step=v['step'], validation=v['score'])
                  for arm, v in state['best'].items()}, artifacts={name: digest(output/name) for name in names},
        paired_arms=list(ARMS), initialization=protocol['initialization'], optimizer_reset=True,
        current_sender_policy_used=True, no_UF_E_G_Rg_F_updates=True,
        source_ablation_complete=True, receiver_weights_shared=False, labels_scope=sender.LABEL_SCOPE,
        full_allocation_evaluation_pending=True, whole_video_RD_pending=True,
        semantic_supervision=False, transmitted_masks=False)
    save(output/'complete.json', result)
    return verify_training(output, protocol, labels_binding=binding['labels'], scale_record=binding['scale'])


def verify_training(output, protocol, *, labels_binding=None, scale_record=None):
    """Read-only check of final checkpoint, all exports, and measured-label bindings."""
    train, _, config = _protocol(protocol)
    output = Path(output)
    binding, result = read(output/'config.json'), read(output/'complete.json')
    _validate_label_binding(protocol, binding['labels'])
    _validate_scale(protocol, binding['scale'])
    architectures = {arm: asdict(replace(config, zero_source=(arm == 'zero_source'))) for arm in ARMS}
    if (binding['protocol'] != protocol or binding['architectures'] != architectures
            or (labels_binding is not None and binding['labels'] != labels_binding)
            or (scale_record is not None and binding['scale'] != scale_record)
            or result.get('complete') is not True or result['config'] != digest(output/'config.json')
            or result['epochs'] != protocol['epochs'] or tuple(result['paired_arms']) != ARMS
            or result.get('initialization') != protocol['initialization']
            or result.get('optimizer_reset') is not True
            or result.get('current_sender_policy_used') is not True
            or result.get('no_UF_E_G_Rg_F_updates') is not True):
        raise ValueError('completed cooperative sender binding changed')
    verify_artifacts(output, result['artifacts'])
    loaded = torch.load(output/'resume.pt', weights_only=True, map_location='cpu')
    state = loaded['state']
    if (loaded['binding'] != binding or state['epoch'] != protocol['epochs'] or state['cursor'] != 0
            or len(state['history']) != protocol['epochs']
            or state['updates'] != len(train)*protocol['epochs']
            or result['updates_per_arm'] != state['updates']
            or read(output/'history.json') != state['history']
            or read(output/'initial_validation.json') != state['initial_validation']):
        raise ValueError('cooperative sender final checkpoint/history/updates are inconsistent')
    for arm in ARMS:
        best = state['best'][arm]
        if result['selected'][arm] != dict(epoch=best['epoch'], step=best['step'], validation=best['score']):
            raise ValueError('cooperative sender best selection changed')
        optimizer = loaded['optimizers'][arm]
        if not optimizer['state'] or any(int(v['step']) != state['updates'] for v in optimizer['state'].values()):
            raise ValueError('cooperative sender optimizer did not complete every paired update')
        for kind in ('best', 'last'):
            model, payload = sender.load_model(output/arm/(kind+'.pt'))
            expected = best['weights'] if kind == 'best' else loaded['models'][arm]
            step = best['step'] if kind == 'best' else state['updates']
            epoch = best['epoch'] if kind == 'best' else protocol['epochs']
            score = best['score'] if kind == 'best' else state['history'][-1]['validation'][arm]
            expected_binding = dict(protocol=protocol, architecture=architectures[arm], arm=arm,
                                    labels=binding['labels'], scale=binding['scale'])
            if (payload['binding'] != expected_binding or asdict(model.config) != architectures[arm]
                    or payload['step'] != step or payload['selection']['epoch'] != epoch
                    or payload['selection']['validation'] != score):
                raise ValueError('cooperative sender exported checkpoint binding changed')
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    return result
