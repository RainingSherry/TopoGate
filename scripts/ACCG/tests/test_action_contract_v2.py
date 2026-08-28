from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _load_module():
    path = ROOT / "scripts/ACCG/evaluate_action_contract_v2.py"
    spec = importlib.util.spec_from_file_location("evaluate_action_contract_v2", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v2_roles_keep_w1_out_of_joint_positive_gate() -> None:
    module = _load_module()
    assert module.PRIMARY_WORLD == "W5_joint_interaction"
    assert module.SECONDARY_WORLD == "W2_rare_coherent_signal"
    assert module.NEGATIVE_CONTROL_WORLD == "W1_isolated_corruption"
    assert module.PRIMARY_AUC_FLOOR == 0.65
    assert module.SECONDARY_AUC_FLOOR == 0.60
    assert module.FAMILY_AUC_FLOOR == module.PRIMARY_AUC_FLOOR


def test_v2_has_a_fresh_unseen_generator_family() -> None:
    config = (ROOT / "methods/TopoGate/ACCG_action_constrained_gate/configs/synthetic_contract_v2.yaml").read_text()
    assert "gamma_sparse" in config
