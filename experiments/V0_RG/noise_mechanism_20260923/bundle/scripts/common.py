"""Frozen V0_RG interfaces. Never trains a model or refits a noisy preprocessor."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

_REFERENCES = {}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def array_hash(array):
    arr = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode())
    digest.update(json.dumps(list(arr.shape)).encode())
    digest.update(memoryview(arr).cast('B'))
    return digest.hexdigest()


def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj).__name__)


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False, default=_json_default), encoding='utf-8')
    temporary.replace(path)


def write_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def code_hashes():
    # Only inference/measurement code changes invalidate existing probes.
    # Re-rendering a figure or hardening aggregation must not force re-inference.
    root = Path(__file__).parent
    return {name: sha256(root / name) for name in ('common.py', 'metrics.py', 'probe.py')}


def condition_slug(kind, severity, seed):
    return '{}_{}_seed{}'.format(kind, format(float(severity), '.8g'), int(seed))


def load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding='utf-8-sig'))
    if config.get('new_training_allowed') is not False:
        raise ValueError('This bundle only supports frozen checkpoints; new_training_allowed must be false.')
    if not set(config['allowed_gpu_ids']).issubset({1, 4, 5}):
        raise ValueError('GPU pool must be a subset of physical GPUs 1,4,5.')
    config['_config_path'] = str(path)
    config['_config_sha256'] = sha256(path)
    return config


def load_reference(config):
    path = str(Path(config['reference_runner']).resolve())
    if path not in _REFERENCES:
        if not Path(path).is_file():
            raise FileNotFoundError('Audited reference evaluator is required: ' + path)
        sys.path.insert(0, config['remote_repo'])
        name = '_v0rg_raw_reference_' + hashlib.sha256(path.encode()).hexdigest()[:12]
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        _REFERENCES[path] = module
    return _REFERENCES[path]


def load_case(config, dataset, model_seed, variant='full'):
    if dataset not in config['datasets'] or variant not in config['variants']:
        raise ValueError('Dataset or variant was not registered in config.json.')
    if int(model_seed) not in config['model_seeds']:
        raise ValueError('Unregistered model seed.')
    ref = load_reference(config)
    cohort = json.loads(Path(config['cohort']).read_text(encoding='utf-8-sig'))
    entry = next(e for e in cohort['entries'] if e['dataset'] == dataset)
    if entry['input_kind'] != 'general':
        raise ValueError('This four-case raw-input protocol accepts only registered general NPZ inputs.')
    checkpoint = Path(entry['checkpoint_root']).parents[1] / variant / 'current' / ('seed_' + str(model_seed))
    required = ['model.pt', 'preprocessor.npz', 'final_summary.json', 'test_indices.npy',
                'labels_true_test.npy', 'embedding_all.npy', 'predictions_all.npy', 'cluster_centers.npy']
    missing = [f for f in required if not (checkpoint / f).is_file()]
    if missing:
        raise FileNotFoundError('{} missing {}'.format(checkpoint, missing))
    summary = json.loads((checkpoint / 'final_summary.json').read_text(encoding='utf-8'))
    cfg = ref.V0_RGConfig(**{k: v for k, v in summary['config'].items() if k in ref.CONFIG_KEYS})
    state = ref.load_frozen_preprocessor(checkpoint / 'preprocessor.npz', entry['input_kind'], cfg.target_sum)
    original, labels = ref.load_data(Path(entry['data_path']))
    original = original.toarray() if hasattr(original, 'toarray') else np.asarray(original)
    raw = np.ascontiguousarray(original, dtype=np.float32)
    changed_by_cast = 0
    for start in range(0, len(raw), 512):
        changed_by_cast += int(np.count_nonzero(original[start:start+512] != raw[start:start+512]))
    x = ref.frozen_transform(raw, state)
    test_indices = np.asarray(np.load(checkpoint / 'test_indices.npy', allow_pickle=False), dtype=np.int64)
    if test_indices.ndim != 1 or len(test_indices) == 0 or len(np.unique(test_indices)) != len(test_indices):
        raise ValueError('Invalid or repeated test indices.')
    if np.any(test_indices < 0) or np.any(test_indices >= len(raw)):
        raise ValueError('Test indices outside input rows.')
    if not np.array_equal(labels[test_indices], np.load(checkpoint / 'labels_true_test.npy', allow_pickle=False)):
        raise ValueError('Source sample/label order differs from checkpoint.')
    saved_embedding = np.load(checkpoint / 'embedding_all.npy', allow_pickle=False)
    saved_predictions = np.asarray(np.load(checkpoint / 'predictions_all.npy', allow_pickle=False), dtype=np.int64)
    centers = np.asarray(np.load(checkpoint / 'cluster_centers.npy', allow_pickle=False), dtype=np.float32)
    if len(saved_embedding) != len(raw) or saved_predictions.shape != (len(raw),):
        raise ValueError('Saved all-row embeddings/predictions do not align with input rows.')
    if centers.ndim != 2 or centers.shape[1] != saved_embedding.shape[1]:
        raise ValueError('Readout dimension mismatch.')
    if not all(np.isfinite(a).all() for a in (x, saved_embedding, centers)):
        raise ValueError('Non-finite frozen artifact.')
    provenance = dict(protocol=config['protocol'], config_sha256=config['_config_sha256'],
                      model_sha256=sha256(checkpoint/'model.pt'),
                      preprocessor_sha256=sha256(checkpoint/'preprocessor.npz'),
                      readout_sha256=sha256(checkpoint/'cluster_centers.npy'),
                      test_indices_sha256=sha256(checkpoint/'test_indices.npy'),
                      source_data_sha256=sha256(Path(entry['data_path'])),
                      checkpoint_config_sha256=hashlib.sha256(json.dumps(summary['config'], sort_keys=True).encode()).hexdigest(),
                      reference_runner_sha256=sha256(Path(config['reference_runner'])),
                      v0rg_source_sha256={str(p.relative_to(Path(config['remote_repo']))): sha256(p)
                                         for p in sorted((Path(config['remote_repo'])/'methods'/'TopoGate'/'V0_RG').rglob('*.py'))},
                      code_sha256=code_hashes(), checkpoint=str(checkpoint))
    return dict(config=config, ref=ref, entry=entry, checkpoint=checkpoint, summary=summary,
                cfg=cfg, state=state, raw=raw, raw_original_dtype=str(original.dtype),
                raw_cast_fraction=float(changed_by_cast / original.size), labels=labels, x=x,
                test_indices=test_indices, saved_embedding=saved_embedding,
                saved_predictions=saved_predictions, centers=centers, provenance=provenance,
                dataset=dataset, model_seed=int(model_seed), variant=variant)


def checked_device(device, allowed=(1,4,5)):
    import torch
    device = torch.device(device)
    if device.type == 'cpu':
        return device
    if device.type != 'cuda' or device.index is None:
        raise ValueError('Use cpu or an explicit CUDA index.')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    if visible:
        values = visible.split(',')
        if not all(x.strip().isdigit() for x in values):
            raise ValueError('Cannot verify physical GPU IDs from this CUDA_VISIBLE_DEVICES value.')
        if device.index >= len(values):
            raise ValueError('Logical CUDA index exceeds CUDA_VISIBLE_DEVICES.')
        physical = int(values[device.index])
    else:
        physical = device.index
    if physical not in allowed:
        raise ValueError('Forbidden physical GPU {} (allowed {}).'.format(physical, allowed))
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable.')
    return device


def load_encoder(case, device):
    device = checked_device(device, case['config']['allowed_gpu_ids'])
    model = case['ref'].load_model(case['checkpoint'], case['cfg'], case['x'].shape[1], device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def encode(case, model, matrix, device):
    device = checked_device(device, case['config']['allowed_gpu_ids'])
    return case['ref']._extract_embedding(model, np.ascontiguousarray(matrix, dtype=np.float32),
                                          case['cfg'].batch_size, device)


def replay_clean(case, model, device):
    z = encode(case, model, case['x'], device)
    expected = case['saved_embedding']
    if z.shape != expected.shape:
        raise ValueError('Clean embedding shape mismatch.')
    error = float(np.max(np.abs(z - expected)))
    cfg = case['config']
    if not np.allclose(z, expected, atol=cfg['clean_replay_atol'], rtol=cfg['clean_replay_rtol']):
        raise ValueError('Clean embedding replay failed; max abs error {}'.format(error))
    pred = case['ref'].nearest_centroid(z, case['centers'])
    if not np.array_equal(pred, case['saved_predictions']):
        raise ValueError('Clean native centroid assignments do not reproduce saved predictions.')
    return z, dict(status='passed', embedding_abs_max_error=error, exact_prediction_match=True,
                   atol=cfg['clean_replay_atol'], rtol=cfg['clean_replay_rtol'])


def corrupt(case, kind, severity, noise_seed):
    noisy_raw, _ = case['ref'].corrupt_raw_input(case['raw'], kind, float(severity), int(noise_seed),
                                               case['entry']['input_kind'])
    noisy_x = case['ref'].frozen_transform(noisy_raw, case['state'])
    # Both raw operands refer to the float32 representation entering the same corruption function.
    metadata = dict(raw_input_sha256=array_hash(noisy_raw), model_input_sha256=array_hash(noisy_x),
                    clean_raw_sha256=array_hash(case['raw']),
                    raw_fraction_changed=float(np.mean(noisy_raw != case['raw'])),
                    model_fraction_changed=float(np.mean(noisy_x != case['x'])),
                    cast_fraction_before_noise=case['raw_cast_fraction'],
                    original_dtype=case['raw_original_dtype'], corruption_dtype=str(case['raw'].dtype),
                    corruption_scope='raw_input', label_use='posthoc_only')
    return noisy_raw, noisy_x, metadata
