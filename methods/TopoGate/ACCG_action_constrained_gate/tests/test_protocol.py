from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from methods.TopoGate.ACCG_action_constrained_gate.config import load_config
from methods.TopoGate.ACCG_action_constrained_gate.protocol import _load_reusable_branchpoint
from methods.TopoGate.V25_systematic_mechanism_study import e1_protocol as e1


ROOT = Path(__file__).resolve().parents[4]


def _write_branchpoint(root: Path, X: np.ndarray, *, seed: int = 42, n_clusters: int = 3) -> Path:
    main = load_config(ROOT / "methods/TopoGate/ACCG_action_constrained_gate/configs/accg_joint.yaml")
    root.mkdir(parents=True, exist_ok=True)
    (root / "resolved_config.json").write_text(
        json.dumps(main.to_dict(), indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    torch.save(
        {
            "seed": int(seed),
            "n_clusters": int(n_clusters),
            "model_input_shape": [int(value) for value in X.shape],
            "model_input_hash": e1._hash_array(X),
            "epoch": int(main.v21.warmup_epochs),
            "model_state": {"model": {}},
            "rng": {"torch": torch.get_rng_state()},
        },
        root / "branchpoint.pt",
    )
    return root / "branchpoint.pt"


def test_ablation_branchpoint_reuse_checks_panel_identity(tmp_path: Path) -> None:
    X = np.arange(24, dtype=np.float32).reshape(8, 3)
    source = _write_branchpoint(tmp_path / "main", X)
    control = load_config(ROOT / "methods/TopoGate/ACCG_action_constrained_gate/configs/accg_coordinate.yaml")
    loaded_source, branchpoint = _load_reusable_branchpoint(
        source.parent,
        config=control,
        seed=42,
        n_clusters=3,
        X_model=X,
    )
    assert loaded_source == source
    assert branchpoint["model_input_hash"] == e1._hash_array(X)

    with pytest.raises(ValueError, match="seed"):
        _load_reusable_branchpoint(source, config=control, seed=123, n_clusters=3, X_model=X)
    with pytest.raises(ValueError, match="n_clusters"):
        _load_reusable_branchpoint(source, config=control, seed=42, n_clusters=4, X_model=X)
    changed = X.copy()
    changed[0, 0] += 1.0
    with pytest.raises(ValueError, match="model_input_hash"):
        _load_reusable_branchpoint(source, config=control, seed=42, n_clusters=3, X_model=changed)

