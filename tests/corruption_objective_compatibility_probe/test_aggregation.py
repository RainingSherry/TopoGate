from __future__ import annotations

from pathlib import Path

from scripts.corruption_objective_compatibility_probe import analysis, protocol


def _summary(dataset: str, arm: str, seed: int, ari: float, *, stage: str, objective: str = "O0_GlobalMSE") -> dict:
    return {
        "project_id": protocol.PROJECT_ID,
        "protocol_id": protocol.PROTOCOL_ID,
        "stage": stage,
        "dataset": dataset,
        "arm": arm,
        "objective": objective,
        "seed": seed,
        "status": "completed_valid",
        "metrics": {"ARI": ari},
        "source": {},
    }


def _write(path: Path, summary: dict) -> None:
    analysis.write_json(path / "summary.json", summary)
    analysis.write_json(path / "audit.json", {"audit_ok": True})


def test_e1_aggregate_rejects_partial_seed_cells(tmp_path: Path) -> None:
    root = tmp_path / "e1"
    nofit = tmp_path / "nofit"
    for dataset in protocol.DEVELOPMENT_PANEL:
        for arm in protocol.E1_ARMS:
            for seed in protocol.PRIMARY_SEEDS:
                if dataset == "cnae9" and arm == "P2_SupportTarget" and seed == 7:
                    continue
                _write(root / dataset / arm / f"seed{seed}", _summary(dataset, arm, seed, 0.5, stage="E1"))
                _write(nofit / dataset / arm / f"seed{seed}", _summary(dataset, arm, seed, 0.5, stage="E1b_nofit"))
    result = analysis.aggregate_e1(root, nofit)
    assert result["status"] == "incomplete_compute"
    row = next(row for row in result["dataset_rows"] if row["dataset"] == "cnae9")
    assert row["status"] == "incomplete_compute"
    assert result["gate"]["g1_cross_domain_opportunity"] is False


def test_e1_aggregate_requires_both_deltas_and_seed_pairing(tmp_path: Path) -> None:
    root = tmp_path / "e1"
    nofit = tmp_path / "nofit"
    for dataset in protocol.DEVELOPMENT_PANEL:
        for arm in protocol.E1_ARMS:
            for seed in protocol.PRIMARY_SEEDS:
                # Non-biological P0=0.10, P2=0.20, Clean=0.18 gives only
                # delta-random material; delta-clean is intentionally small.
                ari = {"P0_Random": 0.10, "P2_SupportTarget": 0.20, "Clean": 0.18}[arm]
                _write(root / dataset / arm / f"seed{seed}", _summary(dataset, arm, seed, ari, stage="E1"))
                _write(nofit / dataset / arm / f"seed{seed}", _summary(dataset, arm, seed, ari, stage="E1b_nofit"))
    result = analysis.aggregate_e1(root, nofit)
    assert result["status"] == "completed_valid"
    assert result["gate"]["g1_winner_count"] == 0
    assert result["gate"]["g1_cross_domain_opportunity"] is False


def test_existing_valid_run_rechecks_current_hashes(tmp_path: Path, monkeypatch) -> None:
    current = {"H0_sha256": "h0", "budget_manifest_sha256": "budget", "labels_sha256": "labels"}
    monkeypatch.setattr(analysis, "current_source_hashes", lambda dataset: dict(current))
    run_dir = tmp_path / "run"
    summary = _summary("Mouse_retina", "Clean", 42, 0.2, stage="E1")
    summary["labels_used_during_fit"] = False
    summary["source"] = dict(current)
    _write(run_dir, summary)
    assert analysis.existing_valid_run(run_dir, dataset="Mouse_retina", arm="Clean", objective="O0_GlobalMSE", seed=42, stage="E1")
    current["H0_sha256"] = "changed"
    assert not analysis.existing_valid_run(run_dir, dataset="Mouse_retina", arm="Clean", objective="O0_GlobalMSE", seed=42, stage="E1")
