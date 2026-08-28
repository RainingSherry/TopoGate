from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

from scripts.V25 import audit_e1_phase
from scripts.V25.audit_v25_contract import audit as audit_v25_contract
from scripts.V25.build_a1_failure_atlas import artifact_complete_replay_rows, paired_atlas_rows
from scripts.V25.build_a2_triage import (
    build_holdout_manifest,
    triage_decision,
)
from scripts.V25.freeze_claim import freeze_claim
from scripts.V25.preflight_holdout import preflight
from scripts.V25.build_holdout_e1_manifest import build_manifest as build_holdout_e1_manifest
from scripts.V25.build_holdout_manifest import build_manifest as build_claim_holdout_manifest
from scripts.V25.summarize_e1 import classify_effect, phase_gate
from scripts.V25.build_e1_manifest import CONFIRMATION, PILOT, SEEDS, build_jobs


ROOT = Path(__file__).resolve().parents[3]


def test_a0_actual_counts_and_boundary_separation() -> None:
    summary = json.loads((ROOT / "result/V25_systematic_mechanism_study/A0/registry_summary.json").read_text())
    assert summary["v1_v22_rows"] == 2209
    assert summary["v1_v22_paired_rows"] == 1637
    assert summary["v1_v22_units"] == 431
    assert summary["v23_v24_boundary_records"] == 2
    assert summary["replay_eligible_rows"] == 0


def test_formal_contract_audit_covers_holdout_adapter_fields() -> None:
    result = audit_v25_contract(ROOT / "result/V25_systematic_mechanism_study")
    assert result["status"] == "audit_ok"
    assert result["checks"]["holdout_adapter_contract_complete"] is True


def test_a0_registry_persists_provenance_and_label_k_boundary_fields() -> None:
    path = ROOT / "result/V25_systematic_mechanism_study/A0/mechanism_evidence_registry.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        fields = set(next(csv.reader(handle)))
    required = {
        "source_hash",
        "preprocess_hash",
        "k_source",
        "k_hash",
        "labels_used_for_fit",
        "k_used_for_fit",
        "label_k_isolation_status",
        "measurement_timing",
        "causal_status",
        "artifact_status",
        "reused_from",
        "alternative_explanation",
    }
    assert required <= fields


def test_a1_seed_is_part_of_pairing_key() -> None:
    rows = [
        {
            "record_type": "intervention_record",
            "version": "V21",
            "source_batch": "b",
            "dataset_id": "d",
            "input_protocol": "p",
            "readout": "r",
            "seeds": "42",
            "variant": "topology",
            "variant_family": "learned",
            "paired_delta_ari": "0.4",
            "paired_control": "scmae",
            "ari_mean": "0.6",
            "status": "completed",
        },
        {
            "record_type": "intervention_record",
            "version": "V21",
            "source_batch": "b",
            "dataset_id": "d",
            "input_protocol": "p",
            "readout": "r",
            "seeds": "123",
            "variant": "topology",
            "variant_family": "learned",
            "paired_delta_ari": "0.5",
            "paired_control": "scmae",
            "ari_mean": "0.7",
            "status": "completed",
        },
        {
            "record_type": "intervention_record",
            "version": "V21",
            "source_batch": "b",
            "dataset_id": "d",
            "input_protocol": "p",
            "readout": "r",
            "seeds": "42",
            "variant": "scmae",
            "variant_family": "none",
            "paired_delta_ari": "",
            "paired_control": "",
            "ari_mean": "0.2",
            "status": "completed",
        },
        {
            "record_type": "intervention_record",
            "version": "V21",
            "source_batch": "b",
            "dataset_id": "d",
            "input_protocol": "p",
            "readout": "r",
            "seeds": "123",
            "variant": "scmae",
            "variant_family": "none",
            "paired_delta_ari": "",
            "paired_control": "",
            "ari_mean": "0.1",
            "status": "completed",
        },
    ]
    paired = paired_atlas_rows(rows)
    assert [row["control_ari"] for row in paired] == [0.2, 0.1]
    assert all(row["causal_status"] == "observational" for row in paired)
    assert all(row["confidence"] == "low_observational_summary" for row in paired)


def test_a1_replay_gate_excludes_metadata_only_rows() -> None:
    rows = [
        {"artifact_status": "registry_only", "replay_eligible": "False"},
        {"artifact_status": "source_file_present", "replay_eligible": "True"},
        {"artifact_status": "artifact_complete", "replay_eligible": "True"},
    ]
    assert len(artifact_complete_replay_rows(rows)) == 1


def test_a2_vetoes_when_v21_heterogeneity_is_absent() -> None:
    a0 = {"v1_v22_rows": 2}
    a1 = {"paired_rows": 2}
    v21 = {
        "audit_ok": True,
        "expected_jobs": 6,
        "completed_valid_jobs": 6,
        "selection_uses_labels": True,
        "per_dataset": [
            {"delta_vs_scmae_only": 0.01},
            {"delta_vs_scmae_only": 0.02},
            {"delta_vs_scmae_only": 0.0},
        ],
    }
    decision, details = triage_decision(a0, a1, v21)
    assert decision == "cancel_e1"
    assert details["checks"]["v21_heterogeneity"] is False


def test_holdout_contract_is_outcome_independent_and_adapter_frozen(tmp_path: Path) -> None:
    valid = tmp_path / "valid.npz"
    valid.write_bytes(b"placeholder")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "datasets": [
                    {
                        "dataset_id": "new_text",
                        "name": "new_text",
                        "family": "sparse_text",
                        "input_protocol": "shared_text",
                        "source_path": str(valid),
                        "selection_uses_labels_or_outcomes": False,
                    },
                    {
                        "dataset_id": "new_scrna",
                        "name": "new_scrna",
                        "family": "scRNA_count_unlabelled",
                        "input_protocol": "scRNA_count",
                        "source_path": str(valid),
                        "selection_uses_labels_or_outcomes": False,
                    },
                    {
                        "dataset_id": "cnae9",
                        "name": "cnae9",
                        "family": "sparse_text",
                        "input_protocol": "shared_text",
                        "source_path": str(valid),
                        "selection_uses_labels_or_outcomes": False,
                    },
                ]
            }
        )
    )
    holdout = build_holdout_manifest((manifest,), ROOT)
    candidates = {row["dataset_id"]: row for row in holdout["candidates"]}
    excluded = {row["dataset_id"]: row for row in holdout["excluded"]}
    assert candidates["new_text"]["holdout_eligible"] is True
    assert candidates["new_scrna"]["holdout_eligible"] is True
    assert candidates["new_scrna"]["input_adapter"] == "prepare_dual_input"
    assert excluded["cnae9"]["exclusion_reason"] == "overlaps_v21_development_panel"


def test_e1_four_state_rule_and_heterogeneous_phase_gate() -> None:
    assert classify_effect([0.06, 0.05, -0.005])["state"] == "Positive"
    assert classify_effect([-0.06, -0.05, 0.005])["state"] == "Negative"
    assert classify_effect([0.01, -0.01, 0.0])["state"] == "Observed-Small"
    assert classify_effect([0.06, -0.06, 0.0])["state"] == "Inconclusive"
    summaries = {
        "d1": {"I_d": {"state": "Positive"}, "S_d": {"state": "Observed-Small"}},
        "d2": {"I_d": {"state": "Negative"}, "S_d": {"state": "Observed-Small"}},
        "d3": {"I_d": {"state": "Observed-Small"}, "S_d": {"state": "Observed-Small"}},
    }
    gate = phase_gate(summaries)
    assert gate["passes"] is True
    assert gate["same_sign_across_datasets_not_required"] is True


def test_e1_manifest_panel_counts_are_three_arm_and_seed_frozen(tmp_path: Path) -> None:
    pilot = build_jobs("pilot", PILOT, tmp_path)
    confirmation = build_jobs("confirmation", CONFIRMATION, tmp_path)
    assert len(pilot) == 3 * len(SEEDS)
    assert len(confirmation) == 3 * len(SEEDS)
    assert {job["seed"] for job in pilot} == set(SEEDS)
    assert all(job["arms"] == ["N", "R", "T"] for job in pilot + confirmation)
    assert all(job["selection_uses_labels_or_outcomes"] is False for job in pilot + confirmation)


def test_e1_phase_audit_excludes_invalid_panels_and_requires_all_declared_seeds(tmp_path: Path, monkeypatch) -> None:
    complete_root = tmp_path / "complete"
    jobs = []
    for seed in (42, 123, 7):
        for arm in ("N", "R", "T"):
            jobs.append({"phase": "pilot", "panel_run_key": f"panel::dataset_a::{seed}", "dataset": "dataset_a", "seed": seed, "arm": arm})
    (complete_root / "manifest_snapshot.json").parent.mkdir(parents=True, exist_ok=True)
    (complete_root / "manifest_snapshot.json").write_text(json.dumps({"manifest_id": "toy_manifest", "phases": {"pilot": {"expected_panel_jobs": 3, "expected_arm_jobs": 9, "jobs": jobs}}}))
    for seed in (42, 123, 7, 99):
        panel = complete_root / "dataset_a" / f"seed{seed}"
        (panel / "T").mkdir(parents=True)
        (panel / "summary.json").write_text(json.dumps({"status": "completed"}))
        if seed != 99:
            (panel / "T" / "gradient_probe.json").write_text(json.dumps({"T0": {"cos": 1.0}}))
            (panel / "one_step.json").write_text(json.dumps({arm: {"metrics": {}, "loss": 0.0} for arm in ("N", "R", "T")}))

    def fake_panel_audit(panel: Path):
        seed = int(panel.name.removeprefix("seed"))
        valid = seed in {42, 123, 7}
        return (
            {"dataset": panel.parent.name, "seed": seed, "panel_run_key": f"panel::{panel.parent.name}::{seed}", "panel_path": str(panel), "audit_ok": valid},
            {"I_full_ARI": float(seed) / 1000.0, "S_full_ARI": -float(seed) / 1000.0},
        )

    monkeypatch.setattr(audit_e1_phase, "_panel_audit", fake_panel_audit)
    payload = audit_e1_phase.audit_phase(complete_root)
    assert payload["invalid_or_incomplete_panel_count"] == 1
    assert len(payload["pair_rows"]) == 3
    assert payload["datasets"]["dataset_a"]["inference_status"] == "complete_valid_seed_set"
    assert payload["datasets"]["dataset_a"]["seeds"] == [7, 42, 123]
    assert payload["coverage_complete"] is False
    assert payload["unexpected_panel_keys"] == ["panel::dataset_a::99"]

    incomplete_root = tmp_path / "incomplete"
    jobs_b = [
        {"phase": "pilot", "panel_run_key": f"panel::dataset_b::{seed}", "dataset": "dataset_b", "seed": seed, "arm": arm}
        for seed in (42, 123, 7)
        for arm in ("N", "R", "T")
    ]
    (incomplete_root / "manifest_snapshot.json").parent.mkdir(parents=True, exist_ok=True)
    (incomplete_root / "manifest_snapshot.json").write_text(json.dumps({"manifest_id": "toy_manifest", "phases": {"pilot": {"expected_panel_jobs": 3, "expected_arm_jobs": 9, "jobs": jobs_b}}}))
    for seed in (42, 123):
        panel = incomplete_root / "dataset_b" / f"seed{seed}"
        (panel / "T").mkdir(parents=True)
        (panel / "summary.json").write_text(json.dumps({"status": "completed"}))
        (panel / "T" / "gradient_probe.json").write_text(json.dumps({"T0": {"cos": 1.0}}))
        (panel / "one_step.json").write_text(json.dumps({arm: {"metrics": {}, "loss": 0.0} for arm in ("N", "R", "T")}))
    payload = audit_e1_phase.audit_phase(incomplete_root)
    assert payload["datasets"]["dataset_b"]["inference_status"] == "inconclusive_invalid_or_incomplete_panel_set"
    assert payload["datasets"]["dataset_b"]["I_d"]["state"] == "Inconclusive"
    assert payload["coverage_complete"] is False
    assert len(payload["missing_expected_panel_keys"]) == 1


def test_e1_panel_audit_recomputes_primary_pairs_from_predictions(tmp_path: Path) -> None:
    source = tmp_path / "source.npz"
    labels = np.asarray([0, 0, 1, 1])
    np.savez(source, X=np.eye(4, dtype=np.float32), y=labels)
    panel = tmp_path / "dataset" / "seed42"
    for arm, predictions in {
        "N": np.asarray([0, 0, 1, 1]),
        "R": np.asarray([0, 1, 1, 0]),
        "T": np.asarray([0, 1, 0, 1]),
    }.items():
        (panel / arm).mkdir(parents=True, exist_ok=True)
        ari = adjusted_rand_score(labels, predictions)
        np.save(panel / arm / "predictions.npy", predictions)
        (panel / arm / "metrics.json").write_text(
            json.dumps({"ari": float(ari), "labels_used_after_fit_only": True})
        )
    aris = {
        arm: adjusted_rand_score(labels, np.load(panel / arm / "predictions.npy"))
        for arm in ("N", "R", "T")
    }
    (panel / "summary.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "protocol_id": "v25_e1_v21_matched_nrt_v1",
                "seed": 42,
                "pairs": {
                    "I_full_ARI": aris["R"] - aris["N"],
                    "S_full_ARI": aris["T"] - aris["R"],
                    "I_1step_ARI": 0.0,
                    "S_1step_ARI": 0.0,
                },
            }
        )
    )
    digest = __import__("hashlib").sha256(source.read_bytes()).hexdigest()
    (panel / "audit.json").write_text(
        json.dumps(
            {
                "protocol_id": "v25_e1_v21_matched_nrt_v1",
                "labels_used_during_fit": False,
                "TR_shared_schedule_hashes": {key: True for key in ("donor", "eligible", "budget", "selection_noise")},
                "none_contract": {"assignment_forward_calls": 0, "js_forward_calls": 0},
                "branchpoint": {"warmup_branchpoint_before_first_assignment": True, "head_initialised": True},
            }
        )
    )
    (panel / "manifest_record.json").write_text(json.dumps({"dataset": "dataset", "seed": 42, "source_sha256": digest}))
    (panel / "runner_profile.json").write_text(json.dumps({"dataset": "dataset", "seed": 42, "data_path": str(source)}))
    (panel / "one_step.json").write_text(
        json.dumps(
            {
                arm: {
                    "metrics": {
                        "ari": 0.0,
                        "labels_used_after_fit_only": True,
                    },
                    "loss": 0.0,
                }
                for arm in ("N", "R", "T")
            }
        )
    )

    row, pairs = audit_e1_phase._panel_audit(panel)
    assert row["audit_ok"] is True
    assert row["primary_ari_recomputed_from_saved_predictions"] is True
    assert pairs["I_full_ARI"] == aris["R"] - aris["N"]
    assert pairs["S_full_ARI"] == aris["T"] - aris["R"]


def test_e1_phase_audit_rejects_missing_dataset_duplicate_and_wrong_dataset(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "coverage"
    jobs = [
        {"phase": "pilot", "panel_run_key": f"panel::{dataset}::{seed}", "dataset": dataset, "seed": seed, "arm": arm}
        for dataset in ("d1", "d2", "d3")
        for seed in (42, 123, 7)
        for arm in ("N", "R", "T")
    ]
    root.mkdir(parents=True)
    (root / "manifest_snapshot.json").write_text(
        json.dumps({"manifest_id": "toy_manifest", "phases": {"pilot": {"expected_panel_jobs": 9, "expected_arm_jobs": 27, "jobs": jobs}}})
    )
    panel_specs = [("d1", seed) for seed in (42, 123, 7)]
    panel_specs += [("d1_alias", 42)]  # reports the d1/42 key: duplicate panel
    panel_specs += [("d2", seed) for seed in (42, 123, 7)]
    panel_specs += [("wrong", 42)]  # remains an unexpected dataset
    for dataset, seed in panel_specs:
        panel = root / dataset / f"seed{seed}"
        (panel / "summary.json").parent.mkdir(parents=True, exist_ok=True)
        (panel / "summary.json").write_text(json.dumps({"status": "completed"}))

    def fake_panel_audit(panel: Path):
        seed = int(panel.name.removeprefix("seed"))
        reported_dataset = "d1" if panel.parent.name == "d1_alias" else panel.parent.name
        key = f"panel::{reported_dataset}::{seed}"
        return (
            {"dataset": reported_dataset, "seed": seed, "panel_run_key": key, "panel_path": str(panel), "audit_ok": True},
            {"I_full_ARI": 0.04, "S_full_ARI": -0.04},
        )

    monkeypatch.setattr(audit_e1_phase, "_panel_audit", fake_panel_audit)
    payload = audit_e1_phase.audit_phase(root)
    assert payload["coverage_complete"] is False
    assert payload["missing_expected_panel_keys"] == sorted(f"panel::d3::{seed}" for seed in (42, 123, 7))
    assert payload["duplicate_panel_keys"] == ["panel::d1::42"]
    assert payload["unexpected_panel_keys"] == ["panel::wrong::42"]
    assert payload["phase_gate"]["expected_dataset_count"] == 3


def test_claim_freeze_requires_explicit_predeclared_family(tmp_path: Path) -> None:
    root = tmp_path / "V25"
    (root / "A2").mkdir(parents=True)
    (root / "A2" / "A2_decision.json").write_text(json.dumps({"decision": "retain_e1"}))
    (root / "A2" / "measurement_schema.json").write_text(
        json.dumps(
            {
                "delta_threshold": 0.03,
                "threshold_sensitivity": [0.02, 0.03, 0.05],
                "claim_activation": {
                    "selection": ["E1_NRT"],
                    "generic_intervention": ["E1_NRT"],
                    "objective_compatibility": ["E1_NRT", "E2-B", "E2-C"],
                    "local_global": ["E3_frozen_matched_pair"],
                },
            }
        )
    )
    payload = freeze_claim(root, "selection", None)
    assert payload["primary_endpoint_key"] == "S_full_ARI"
    assert (root / "PhaseC" / "FROZEN_PAPER_CLAIM.md").is_file()
    assert json.loads((root / "PhaseC" / "FROZEN_PAPER_CLAIM.json").read_text())["claim_family"] == "selection"


def test_claim_freeze_cannot_switch_after_phase_c(tmp_path: Path) -> None:
    root = tmp_path / "V25"
    (root / "A2").mkdir(parents=True)
    (root / "A2" / "A2_decision.json").write_text(json.dumps({"decision": "retain_e1"}))
    (root / "A2" / "measurement_schema.json").write_text(
        json.dumps(
            {
                "delta_threshold": 0.03,
                "claim_activation": {
                    "selection": ["E1_NRT"],
                    "generic_intervention": ["E1_NRT"],
                    "objective_compatibility": ["E1_NRT", "E2-B", "E2-C"],
                    "local_global": ["E3_frozen_matched_pair"],
                },
            }
        )
    )
    freeze_claim(root, "selection", None)
    import pytest

    with pytest.raises(ValueError, match="already frozen"):
        freeze_claim(root, "local_global", None)


def test_holdout_preflight_freezes_adapter_and_k_without_outcomes(tmp_path: Path) -> None:
    root = tmp_path / "V25"
    (root / "A2").mkdir(parents=True)
    (root / "PhaseC").mkdir(parents=True)
    (root / "PhaseC" / "FROZEN_PAPER_CLAIM.json").write_text(
        json.dumps({"claim_family": "selection", "primary_endpoint": "S_full_ARI", "activation_subset": ["E1_NRT"]})
    )
    source = tmp_path / "holdout.npz"
    np.savez(source, X=np.asarray([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]], dtype=np.float32), y=np.asarray([0, 1, 1]))
    import hashlib

    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    (root / "A2" / "holdout_candidate_manifest.json").write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "dataset_id": "toy_holdout",
                        "domain": "sparse_text",
                        "holdout_eligible": True,
                        "source_path": str(source),
                        "source_hash": digest,
                        "input_protocol": "shared_text",
                        "outcome_selection_declared": False,
                    }
                ]
            }
        )
    )
    (root / "A2" / "A2_decision.json").write_text(json.dumps({"decision": "retain_e1"}))
    payload = preflight(root, ["toy_holdout"], {})
    assert payload["datasets"][0]["K_source"] == "benchmark_oracle_from_y"
    assert payload["datasets"][0]["adapter_valid"] is True
    assert payload["datasets"][0]["input_adapter"] == "prepare_dual_input"
    assert payload["datasets"][0]["feature_selection"] == "adapter_default_label_free"
    assert payload["datasets"][0]["normalization"] == "prepare_dual_input_frozen"
    assert payload["datasets"][0]["max_features"] == "adapter_default"
    assert payload["datasets"][0]["graph_input"] == "X_graph_from_prepare_dual_input"
    assert payload["datasets"][0]["model_input"] == "X_model_from_prepare_dual_input"
    assert (root / "PhaseD" / "holdout_activation_manifest.json").is_file()


def test_holdout_preflight_respects_a2_veto(tmp_path: Path) -> None:
    root = tmp_path / "V25"
    (root / "A2").mkdir(parents=True)
    (root / "PhaseC").mkdir(parents=True)
    (root / "A2" / "A2_decision.json").write_text(json.dumps({"decision": "cancel_e1"}))
    (root / "PhaseC" / "FROZEN_PAPER_CLAIM.json").write_text(
        json.dumps({"claim_family": "selection", "primary_endpoint": "S_full_ARI", "activation_subset": ["E1_NRT"]})
    )
    import pytest

    with pytest.raises(ValueError, match="retain_e1"):
        preflight(root, [], {})


def test_holdout_e1_manifest_is_claim_bound_and_three_arm(tmp_path: Path) -> None:
    root = tmp_path / "V25"
    (root / "A2").mkdir(parents=True)
    (root / "PhaseC").mkdir(parents=True)
    (root / "A2" / "A2_decision.json").write_text(json.dumps({"decision": "retain_e1"}))
    claim = root / "PhaseC" / "FROZEN_PAPER_CLAIM.json"
    claim.write_text(json.dumps({"claim_family": "selection", "primary_endpoint": "S_full_ARI", "activation_subset": ["E1_NRT"]}))
    import hashlib

    claim_hash = hashlib.sha256(claim.read_bytes()).hexdigest()
    (root / "PhaseD" ).mkdir(parents=True)
    (root / "PhaseD" / "holdout_activation_manifest.json").write_text(
        json.dumps(
            {
                "claim_freeze_sha256": claim_hash,
                "claim_family": "selection",
                "primary_endpoint": "S_full_ARI",
                "activation_subset": ["E1_NRT"],
                "datasets": [
                    {
                        "dataset_id": "toy",
                        "input_protocol": "shared_text",
                        "source_path": "/tmp/toy.npz",
                        "current_source_sha256": "abc",
                        "input_adapter": "prepare_dual_input",
                        "feature_selection": "adapter_default_label_free",
                        "normalization": "prepare_dual_input_frozen",
                        "max_features": "adapter_default",
                        "graph_input": "X_graph_from_prepare_dual_input",
                        "model_input": "X_model_from_prepare_dual_input",
                        "K_source": "explicit_n_clusters",
                        "n_clusters": 2,
                        "preflight_status": "valid",
                        "adapter_valid": True,
                    }
                ],
            }
        )
    )
    payload = build_holdout_e1_manifest(root)
    assert payload["manifest_id"] == "v25_holdout_e1_manifest_v1"
    assert payload["expected_panel_jobs"] == 3
    assert payload["expected_arm_jobs"] == 9
    assert all(row["arms"] == ["N", "R", "T"] for row in payload["jobs"])
    assert all(row["input_adapter"] == "prepare_dual_input" for row in payload["jobs"])
    assert all(row["max_features"] == "adapter_default" for row in payload["jobs"])


def test_claim_dependent_holdout_manifest_uses_matched_pair_for_local_global(tmp_path: Path) -> None:
    root = tmp_path / "V25"
    (root / "A2").mkdir(parents=True)
    (root / "PhaseC").mkdir(parents=True)
    (root / "PhaseD").mkdir(parents=True)
    (root / "A2" / "A2_decision.json").write_text(json.dumps({"decision": "retain_e1"}))
    claim = root / "PhaseC" / "FROZEN_PAPER_CLAIM.json"
    claim.write_text(
        json.dumps(
            {
                "claim_family": "local_global",
                "primary_endpoint": "1[delta_kNN_purity > 0 and delta_ARI <= 0]",
                "activation_subset": ["E3_frozen_matched_pair"],
            }
        )
    )
    import hashlib

    claim_hash = hashlib.sha256(claim.read_bytes()).hexdigest()
    (root / "PhaseD" / "holdout_activation_manifest.json").write_text(
        json.dumps(
            {
                "claim_freeze_sha256": claim_hash,
                "datasets": [
                    {
                        "dataset_id": "toy",
                        "domain": "sparse_text",
                        "input_protocol": "shared_text",
                        "source_path": "/tmp/toy.npz",
                        "current_source_sha256": "abc",
                        "preflight_status": "valid",
                        "adapter_valid": True,
                        "K_source": "explicit_n_clusters",
                        "n_clusters": 2,
                    }
                ],
            }
        )
    )
    payload = build_claim_holdout_manifest(root)
    assert payload["claim_family"] == "local_global"
    assert payload["arms"] == ["matched_pair"]
    assert payload["expected_panel_jobs"] == 3
    assert payload["expected_arm_jobs"] == 3
