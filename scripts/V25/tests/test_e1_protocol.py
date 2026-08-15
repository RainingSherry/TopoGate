from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from methods.TopoGate.V25_systematic_mechanism_study.e1_protocol import _resolve_adam_foreach, _resolve_adam_fused, run_e1
from methods.TopoGate.V25_systematic_mechanism_study.e2_metrics import (
    CoordinateMetricAccumulator,
    aggregate_selected_vs_eligible,
)


ROOT = Path(__file__).resolve().parents[3]


def test_e1_cpu_smoke_contract_and_adam_probe(tmp_path: Path) -> None:
    import scipy.sparse as sp

    values = np.asarray(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 4.0, 3.0, 2.0],
            [6.0, 5.0, 4.0, 3.0],
            [0.5, 1.5, 2.5, 3.5],
            [5.5, 4.5, 3.5, 2.5],
        ],
        dtype=np.float32,
    )
    from methods.TopoGate.V25_systematic_mechanism_study.e1_protocol import E1Config, _build_components, _load_state

    config = E1Config(epochs=3, warmup_epochs=1, batch_size=3, hidden_size=4, graph_svd_min_dim=2, graph_svd_max_dim=3, neighbor_k=2, stats_block_size=2, cluster_n_init=2)
    result = run_e1(
        values,
        sp.csr_matrix(values),
        n_clusters=2,
        config=config,
        seed=42,
        device="cpu",
        evaluation_labels=np.asarray([0, 0, 1, 1, 0, 1]),
        output_dir=tmp_path,
    )
    assert result["status"] == "completed"
    assert result["audit"]["TR_shared_schedule_hashes"] == {"donor": True, "eligible": True, "budget": True, "selection_noise": True}
    assert result["audit"]["none_contract"]["assignment_forward_calls"] == 0
    assert result["audit"]["none_contract"]["js_forward_calls"] == 0
    assert result["audit"]["none_contract"]["shadow_assignment_calls"] > 0
    assert set(result["one_step"]) == {"N", "R", "T"}
    assert set(result["arms"]["T"]["gradient_probe"]) == {"T0", "T1", "T2"}
    assert all(value is not None for value in result["pairs"].values())
    assert all(item["metrics"]["labels_used_after_fit_only"] is True for item in result["arms"].values())
    assert all(item["metrics"]["labels_used_after_fit_only"] is True for item in result["one_step"].values())
    assert result["audit"]["topology_statistics_storage"] == "memmap"
    assert result["audit"]["optimizer"] == "Adam"
    assert (tmp_path / "branchpoint.pt").is_file()
    assert (tmp_path / "pairs" / "N_R.json").is_file()
    branchpoint = torch.load(tmp_path / "branchpoint.pt", map_location="cpu", weights_only=False)
    assert all(
        tensor.device.type == "cpu"
        for tensor in branchpoint["model_state"]["optimizer"]["state"].values()
        for tensor in tensor.values()
        if isinstance(tensor, torch.Tensor)
    )
    restored = _build_components(torch.as_tensor(values), 2, config, 42, torch.device("cpu"))
    _load_state(restored, branchpoint["model_state"])
    for name in ("model", "head", "gate"):
        restored_state = restored[name].state_dict()
        expected_state = branchpoint["model_state"][name]
        assert restored_state.keys() == expected_state.keys()
        assert all(torch.equal(restored_state[key].cpu(), expected_state[key].cpu()) for key in restored_state)


def test_e1_label_free_k_boundary_requires_explicit_k_without_labels() -> None:
    from types import SimpleNamespace

    from scripts.V25.run_e1_matched_protocol import _resolve_n_clusters

    unlabeled = SimpleNamespace(labels=None)
    with pytest.raises(ValueError, match="--n-clusters is required"):
        _resolve_n_clusters(unlabeled, None)
    assert _resolve_n_clusters(unlabeled, 3) == (3, "explicit_n_clusters")


def test_e2_feature_audit_requires_passing_pilot_gate(tmp_path: Path) -> None:
    from scripts.V25.build_e2_feature_audit import _require_pilot_gate

    confirmation_root = tmp_path / "E1" / "confirmation"
    pilot_audit = tmp_path / "E1" / "pilot" / "Audit" / "phase_summary.json"
    pilot_audit.parent.mkdir(parents=True)
    pilot_audit.write_text(
        json.dumps({"phase_gate": {"passes": False, "material_dataset_count": 1}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="requires a passing pilot"):
        _require_pilot_gate(confirmation_root)

    pilot_audit.write_text(
        json.dumps({"phase_gate": {"passes": True, "material_dataset_count": 2}}),
        encoding="utf-8",
    )
    path, payload = _require_pilot_gate(confirmation_root)
    assert path == pilot_audit
    assert payload["phase_gate"]["material_dataset_count"] == 2


def test_e1_runner_requires_retain_a2(tmp_path: Path) -> None:
    from scripts.V25.run_e1_matched_protocol import _require_e1_authorization

    decision = tmp_path / "A2_decision.json"
    decision.write_text(json.dumps({"decision": "cancel_e1"}), encoding="utf-8")
    with pytest.raises(ValueError, match="retain_e1"):
        _require_e1_authorization(decision)
    decision.write_text(json.dumps({"decision": "retain_e1"}), encoding="utf-8")
    payload, digest = _require_e1_authorization(decision)
    assert payload["decision"] == "retain_e1"
    assert len(digest) == 64


def test_e1_repeated_run_is_deterministic(tmp_path: Path) -> None:
    import scipy.sparse as sp
    from methods.TopoGate.V25_systematic_mechanism_study.e1_protocol import E1Config

    values = np.arange(48, dtype=np.float32).reshape(12, 4)
    config = E1Config(epochs=2, warmup_epochs=1, batch_size=4, hidden_size=4, graph_svd_min_dim=2, graph_svd_max_dim=3, neighbor_k=2, stats_block_size=2, cluster_n_init=2)
    first = run_e1(values, sp.csr_matrix(values), n_clusters=2, config=config, seed=42, device="cpu", output_dir=tmp_path / "a")
    second = run_e1(values, sp.csr_matrix(values), n_clusters=2, config=config, seed=42, device="cpu", output_dir=tmp_path / "b")
    assert first["audit"] == second["audit"]
    assert first["pairs"] == second["pairs"]
    assert json.loads((tmp_path / "a" / "one_step.json").read_text()) == json.loads((tmp_path / "b" / "one_step.json").read_text())
    for arm in ("N", "R", "T"):
        assert np.array_equal(np.load(tmp_path / "a" / arm / "embedding_final.npy"), np.load(tmp_path / "b" / arm / "embedding_final.npy"))


def test_coordinate_audit_uses_dataset_seed_as_inference_unit() -> None:
    selected = np.asarray([[True, False, False], [False, True, False]])
    eligible = np.asarray([[True, True, False], [True, True, True]])
    result = aggregate_selected_vs_eligible(
        selected,
        eligible,
        {"variance": np.arange(6, dtype=np.float64).reshape(2, 3)},
        dataset_id="toy",
        seed=42,
    )
    assert result["statistical_unit"] == "dataset_seed_summary"
    assert result["coordinate_distribution_is_descriptive_only"] is True
    assert result["selected_coordinate_count"] == 2
    assert result["eligible_not_selected_coordinate_count"] == 3
    assert result["metrics"]["variance"]["selected_n_coordinates"] == 2


def test_streaming_coordinate_accumulator_matches_batch_summary() -> None:
    selected = np.asarray([[True, False, False], [False, True, False]])
    eligible = np.asarray([[True, True, False], [True, True, True]])
    values = np.arange(6, dtype=np.float64).reshape(2, 3)
    batch = aggregate_selected_vs_eligible(
        selected,
        eligible,
        {"variance": values},
        dataset_id="toy",
        seed=42,
    )
    streaming = CoordinateMetricAccumulator("toy", 42)
    streaming.update(selected[:1], eligible[:1], {"variance": values[:1]})
    streaming.update(selected[1:], eligible[1:], {"variance": values[1:]})
    result = streaming.finalize()
    assert result["statistical_unit"] == "dataset_seed_summary"
    assert result["selected_coordinate_count"] == batch["selected_coordinate_count"]
    assert result["eligible_not_selected_coordinate_count"] == batch["eligible_not_selected_coordinate_count"]
    assert result["metrics"]["variance"]["difference"] == batch["metrics"]["variance"]["difference"]


def test_high_dimensional_cuda_resource_path_disables_only_foreach_workspace() -> None:
    from methods.TopoGate.V25_systematic_mechanism_study.e1_protocol import E1Config

    config = E1Config()
    assert _resolve_adam_foreach(62061, config, torch.device("cuda")) is False
    assert _resolve_adam_fused(62061, config, torch.device("cuda")) is True
    assert _resolve_adam_foreach(2000, config, torch.device("cuda")) is None
    assert _resolve_adam_fused(2000, config, torch.device("cuda")) is False
    assert _resolve_adam_foreach(62061, E1Config(adam_foreach=True), torch.device("cuda")) is True
    assert _resolve_adam_fused(62061, E1Config(adam_foreach=True), torch.device("cuda")) is False


def test_scrna_count_adapter_is_an_explicit_frozen_v25_protocol() -> None:
    from methods.TopoGate.V21_assignment_adversarial_gate.input_adapter import prepare_dual_input

    values = np.asarray([[0.0, 1.0], [2.0, 0.0], [1.0, 3.0]], dtype=np.float32)
    prepared = prepare_dual_input(values, dataset_name="toy_scrna", input_protocol="scRNA_count")
    assert prepared.profile["input_protocol"] == "scRNA_count"
    assert prepared.profile["labels_used"] is False
    assert prepared.X_model.shape == values.shape
