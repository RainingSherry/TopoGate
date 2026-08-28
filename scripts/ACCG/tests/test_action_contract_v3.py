from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _load_module():
    path = ROOT / "scripts/ACCG/evaluate_action_contract_v3.py"
    spec = importlib.util.spec_from_file_location("evaluate_action_contract_v3", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v3_uses_incremental_endpoint_and_keeps_a_floor() -> None:
    module = _load_module()
    assert module.PRIMARY_AUC_FLOOR == 0.60
    assert module.SECONDARY_AUC_FLOOR == 0.60
    assert module.BOOTSTRAP_REPLICATES == 1000


def test_v3_fresh_seed_contract_is_documented() -> None:
    candidates = (
        ROOT / "reports/ACCG/SYNTHETIC_CONTRACT_V3.md",
        ROOT / "review-stage/ACCG_SYNTHETIC_CONTRACT_V3.md",
    )
    contract = next((path for path in candidates if path.is_file()), None)
    assert contract is not None
    text = contract.read_text()
    assert "[3032, 3033, 3034, 3035, 3036]" in text
    assert "amendments after v3" in text
