"""Parent-authored asset verification only; no fitting, labels, or selection."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'
import sys, json, hashlib
from pathlib import Path
import numpy as np
import torch

manifest = json.loads(Path(sys.argv[1]).read_text())
base = Path(manifest['remote_base'])
sys.path[:0] = [str(base), '/home/luolie/ToPoGate']
from v0rg_tunable_multihop.input_adapter import load_npz_matrix_only
from v0rg_tunable_multihop.tuning import SplitPreprocessor
from v0rg_tunable_multihop.model import WeightedAutoEncoder
from v0rg_tunable_multihop.trainer import _extract_embedding

torch.set_num_threads(2)
rows = []
for name, case in manifest['cases'].items():
    path = Path(case['feature_path'])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == case['feature_sha256']
    loaded = load_npz_matrix_only(path)
    assert loaded.labels is None and loaded.profile['labels_accessed'] is False
    prep = SplitPreprocessor(case['input_kind'], case['feature_limit']).fit(loaded.X)
    assert prep.fingerprint == case['preprocessing_hash'], name
    X = prep.transform(loaded.X)
    config = case['expected_config_template']
    selected_depth = case['variant'].split('_')[0]
    for cell in manifest['depth_cells']:
        if cell['dataset'] != name or cell['depth'] not in ('H1', selected_depth):
            continue
        assert cell['status'] == 'reuse'
        seed = cell['seed']; artifacts = cell['artifacts']
        model = WeightedAutoEncoder(num_genes=X.shape[1], hidden_size=config['hidden_size'],
            dropout=config['dropout'], masked_data_weight=config['masked_data_weight'],
            mask_loss_weight=config['mask_loss_weight'])
        state = torch.load(artifacts[f'model_seed_{seed}.pt'], map_location='cpu', weights_only=True)
        model.load_state_dict(state)
        with torch.no_grad():
            actual = _extract_embedding(model, X, config['batch_size'], torch.device('cpu'))
        saved_file = np.load(artifacts[f'embedding_seed_{seed}.npz'])
        key = 'embedding' if 'embedding' in saved_file.files else 'X'
        if key not in saved_file.files:
            raise ValueError(saved_file.files)
        saved = saved_file[key]
        maximum = float(np.max(np.abs(actual - saved)))
        relative = float(np.linalg.norm(actual - saved) / max(np.linalg.norm(saved), 1e-12))
        rows.append(dict(dataset=name, depth=cell['depth'], seed=seed,
            preprocessing_hash=prep.fingerprint, max_absolute_difference=maximum,
            relative_frobenius_difference=relative,
            clean_embedding_agrees=bool(np.allclose(actual,saved,rtol=1e-4,atol=1e-4))))
        print(json.dumps(rows[-1]), flush=True)
out = dict(kind='read_only_checkpoint_preflight', labels_loaded=False, new_training=False,
           device='cpu', rows=rows, all_clean_embeddings_agree=all(r['clean_embedding_agrees'] for r in rows))
Path(sys.argv[2]).write_text(json.dumps(out, indent=2))
