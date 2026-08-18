import json
from pathlib import Path

from scripts.sparse_corruption_principle_probe import c2_matrix, protocol


def test_c2_positive_control_is_label_free_and_exact_budget():
    result = c2_matrix.positive_control()
    assert result["status"] == "completed_valid"
    assert result["labels_used"] is False
    assert all(arm["exact_budget"] for arm in result["arms"].values())


def test_c2_runner_rejects_forbidden_or_non_single_gpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert c2_matrix._cuda_visible_is_legal() is False
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    assert c2_matrix._cuda_visible_is_legal() is False
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    assert c2_matrix._cuda_visible_is_legal() is True


def test_c2_resolved_contract_has_dense_h0_firewall():
    config = protocol.resolved_config()
    assert config["c2_matrix_authorized"] is True
    assert "dense H0" in config["support_interpretation_firewall"]
    assert json.loads(json.dumps(config))["formal_matrix"]["runs"] == 54


def test_existing_run_reuse_rechecks_current_source_hashes(tmp_path: Path, monkeypatch):
    dataset = "Mouse_retina"
    principle = "P0_Random"
    seed = 42
    source = {
        "H0_sha256": "h0-source",
        "budget_manifest_sha256": "budget-source",
    }
    label_source = {"labels_sha256": "labels-source"}
    monkeypatch.setattr(c2_matrix, "_load_h0", lambda _: (None, source))
    monkeypatch.setattr(c2_matrix, "_load_labels", lambda _: (None, label_source))
    run_dir = tmp_path / dataset / principle / f"seed{seed}"
    run_dir.mkdir(parents=True)
    c2_matrix.write_json(
        run_dir / "summary.json",
        {
            "status": "completed_valid",
            "stage": "C2_static_matrix",
            "protocol_id": protocol.C2_PROTOCOL_ID,
            "dataset": dataset,
            "principle": principle,
            "seed": seed,
            "source": {
                "H0_sha256": source["H0_sha256"],
                "budget_manifest_sha256": source["budget_manifest_sha256"],
                "labels_sha256": label_source["labels_sha256"],
            },
        },
    )
    c2_matrix.write_json(run_dir / "audit.json", {"audit_ok": True})
    c2_matrix.write_json(
        run_dir / "resolved_config.json",
        {"dataset": dataset, "principle": principle, "seed": seed},
    )
    assert c2_matrix._existing_run_valid(run_dir, dataset, principle, seed) is True
    summary = json.loads((run_dir / "summary.json").read_text())
    summary["source"]["H0_sha256"] = "changed-current-input"
    c2_matrix.write_json(run_dir / "summary.json", summary)
    assert c2_matrix._existing_run_valid(run_dir, dataset, principle, seed) is False


def test_aggregate_does_not_promote_partial_seed_cell(tmp_path: Path):
    root = tmp_path / "C2_static_matrix"
    for dataset in protocol.DEVELOPMENT_PANEL:
        for principle in protocol.PRINCIPLES:
            for seed in protocol.PRIMARY_SEEDS:
                if dataset == "Mouse_retina" and principle == "P1_SupportPreserve" and seed == 7:
                    continue
                run_dir = root / dataset / principle / f"seed{seed}"
                run_dir.mkdir(parents=True)
                c2_matrix.write_json(
                    run_dir / "summary.json",
                    {
                        "dataset": dataset,
                        "role": protocol.ROLE_BY_DATASET[dataset],
                        "principle": principle,
                        "seed": seed,
                        "status": "completed_valid",
                        "metrics": {"ARI": 0.1, "NMI": 0.1, "ACC": 0.1, "L_rec": 1.0},
                        "corruption_audit": {
                            "effective_changed_coordinate_rate_mean": 0.2,
                            "support_change_rate_mean": 0.1,
                            "value_change_rate_mean": 0.1,
                            "total_absolute_change_mean": 1.0,
                            "exact_budget_all_epochs": True,
                        },
                    },
                )
    result = c2_matrix.aggregate(root, c2_matrix.positive_control(), score_manifest={}, gpu_pool=(2,))
    partial = [
        row
        for row in result["dataset_rows"]
        if row.get("dataset") == "Mouse_retina" and row.get("principle") == "P1_SupportPreserve"
    ]
    assert len(partial) == 1
    assert partial[0]["status"] == "incomplete_compute"
    assert result["decision"]["status"] == "incomplete_compute"

    missing_dir = root / "Mouse_retina" / "P1_SupportPreserve" / "seed7"
    missing_dir.mkdir(parents=True, exist_ok=True)
    c2_matrix.write_json(
        missing_dir / "summary.json",
        {
            "dataset": "Mouse_retina",
            "role": protocol.ROLE_BY_DATASET["Mouse_retina"],
            "principle": "P1_SupportPreserve",
            "seed": 7,
            "status": "completed_valid",
            "metrics": {"ARI": 0.1, "NMI": 0.1, "ACC": 0.1, "L_rec": 1.0},
            "corruption_audit": {
                "effective_changed_coordinate_rate_mean": 0.2,
                "support_change_rate_mean": 0.1,
                "value_change_rate_mean": 0.1,
                "total_absolute_change_mean": 1.0,
                "exact_budget_all_epochs": True,
            },
        },
    )
    complete = c2_matrix.aggregate(root, c2_matrix.positive_control(), score_manifest={}, gpu_pool=(2,))
    assert complete["decision"]["status"] != "incomplete_compute"
    assert all(row["status"] == "completed_valid" for row in complete["dataset_rows"])
