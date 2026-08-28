from __future__ import annotations

import json
from pathlib import Path

from scripts.v9_regime.summarize import summarize


def _write_run(root: Path, dataset: str, variant: str, seed: int, ari: float) -> None:
    path = root / dataset / variant / f"seed{seed}" / "run_record.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "dataset_id": dataset,
                "variant": variant,
                "seed": seed,
                "status": "completed",
                "metrics": {"ari": ari, "nmi": ari / 2.0},
            }
        ),
        encoding="utf-8",
    )


def test_summary_reports_paired_and_control_counts(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    for seed, full, nomix in ((42, 0.20, 0.10), (123, 0.15, 0.10), (7, 0.18, 0.17)):
        _write_run(runs, "d1", "full", seed, full)
        _write_run(runs, "d1", "nomix", seed, nomix)
        _write_run(runs, "d1", "random", seed, full - 0.01)
    output = tmp_path / "summary"
    result = summarize(runs, output)
    assert result["completed_runs"] == 9
    assert result["paired_dataset_rows"] == 1
    assert result["control_pair_rows"] == 3
    assert result["status_counts"] == {"completed": 9}
    assert (output / "control_deltas.csv").exists()


def test_summary_can_pair_full_with_scmae_reference(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    for seed, full, scmae in ((42, 0.20, 0.10), (123, 0.15, 0.10), (7, 0.18, 0.17)):
        _write_run(runs, "d1", "full", seed, full)
        _write_run(runs, "d1", "scmae", seed, scmae)
    output = tmp_path / "summary_scmae"
    result = summarize(runs, output, reference_variant="scmae")
    assert result["paired_reference_variant"] == "scmae"
    assert result["paired_seed_rows"] == 3
    paired_header = (output / "paired_deltas.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "scmae_ari" in paired_header
