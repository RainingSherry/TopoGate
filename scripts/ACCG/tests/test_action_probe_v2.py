from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]


def _load_module():
    path = ROOT / "scripts/ACCG/evaluate_action_probes_v2.py"
    spec = importlib.util.spec_from_file_location("evaluate_action_probes_v2", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v2_family_holdout_and_row_bootstrap_are_grouped() -> None:
    module = _load_module()
    rng = np.random.default_rng(3)
    target = np.tile(np.asarray([0, 1, 0, 1], dtype=np.int64), 40)
    baseline = rng.normal(size=(target.size, 3))
    joint = target.astype(np.float64) + rng.normal(scale=0.05, size=target.size)
    families = np.repeat(np.asarray(["a", "b"]), target.size // 2)
    holdout = module._family_holdout_scores(baseline, joint, target, families)
    assert holdout["valid"] is True
    assert holdout["auc_joint"] > 0.95
    rows = np.repeat(np.arange(target.size // 4), 4)
    bootstrap = module._row_group_bootstrap(
        target,
        holdout["baseline_score"],
        holdout["joint_score"],
        rows,
        seed=42,
        replicates=100,
    )
    assert bootstrap["valid"] is True
    assert bootstrap["delta_auc_ci_low"] > 0.0


def test_v2_world_roles_do_not_require_joint_gain_for_w1_or_w3() -> None:
    module = _load_module()
    summary = {
        "valid": True,
        "auc_joint": 0.51,
        "delta_auc": -0.01,
        "delta_pr": -0.01,
        "bootstrap": {"delta_auc_ci_low": -0.1},
    }
    assert module._decision(module.CONTROL_WORLD, summary)["passes"] is None
    assert module._decision(module.BOUNDARY_WORLD, summary)["passes"] is None
    assert module._decision(module.PRIMARY_WORLD, summary)["passes"] is False
