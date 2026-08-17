from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[3]


def _load_script(name: str):
    path = ROOT / "scripts/ACCG" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_manifest_builds_main_jobs_and_reuses_controls_for_ablations(tmp_path: Path) -> None:
    datasets = []
    for index in range(8):
        path = tmp_path / f"data{index}.npz"
        np.savez_compressed(path, X=np.eye(8, dtype=np.float32))
        datasets.append(
            {
                "dataset_id": f"d{index}",
                "name": f"dataset-{index}",
                "source_path": str(path),
                "domain": "scRNA" if index < 4 else "text",
                "source_family": f"family-{index}",
                "input_protocol": "clubench_bridge" if index < 4 else "shared_text",
                "license": "test-only",
                "n_clusters": 2,
            }
        )
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(
        yaml.safe_dump({"datasets": datasets, "development_subset": ["d0", "d4"]}), encoding="utf-8"
    )
    module = _load_script("build_real_manifest.py")
    manifest = module.build_manifest(spec_path, tmp_path / "outputs")
    assert manifest["expected_main_panels"] == 24
    assert manifest["expected_ablation_arms"] == 24
    ablations = [job for job in manifest["jobs"] if job["role"] == "ablation"]
    assert ablations
    assert all(job["reused_controls"] == ["N", "R", "T_s"] for job in ablations)


def test_real_matrix_runner_is_dry_by_default(tmp_path: Path) -> None:
    manifest = {
        "manifest_id": "accg_locked_real_panel_v1",
        "selection_uses_labels_or_outcomes": False,
        "jobs": [
            {
                "run_key": "dry",
                "role": "main",
                "dataset_id": "d",
                "output_dir": str(tmp_path / "never-created"),
                "record": {"name": "d"},
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/ACCG/run_matrix.py"), "--manifest", str(manifest_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert '"execute": false' in completed.stdout.lower()
    assert not (tmp_path / "never-created").exists()


def test_real_manifest_requires_labels_or_explicit_k(tmp_path: Path) -> None:
    datasets = []
    for index in range(8):
        path = tmp_path / f"unlabelled{index}.npz"
        np.savez_compressed(path, X=np.eye(8, dtype=np.float32))
        row = {
            "dataset_id": f"d{index}",
            "name": f"dataset-{index}",
            "source_path": path.name,
            "domain": "scRNA" if index < 4 else "text",
            "source_family": f"family-{index}",
            "input_protocol": "clubench_bridge",
            "license": "test-only",
        }
        if index:
            row["n_clusters"] = 2
        datasets.append(row)
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump({"datasets": datasets}), encoding="utf-8")
    module = _load_script("build_real_manifest.py")
    with pytest.raises(ValueError, match="needs labels or an explicit n_clusters"):
        module.build_manifest(spec_path, tmp_path / "outputs")


def test_ablation_job_is_blocked_until_full_canonical_main_panel_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script("run_matrix.py")
    main = tmp_path / "main"
    main.mkdir()
    (main / "branchpoint.pt").write_bytes(b"partial")
    job = {
        "run_key": "ablation",
        "role": "ablation",
        "dataset_id": "d",
        "seed": 42,
        "output_dir": str(tmp_path / "ablation"),
        "reused_from": str(main),
        "record": {"name": "dataset", "source_path": "unused", "source_sha256": "sha"},
        "config": "unused",
        "config_sha256": "config-sha",
    }

    def _forbidden(*_args, **_kwargs):
        raise AssertionError("subprocess must not launch before the canonical main panel is complete")

    monkeypatch.setattr(module.subprocess, "run", _forbidden)
    result = module._run_job(job, gpu=None, epochs=None, warmup_epochs=None, force=False)
    assert result["status"] == "blocked_missing_canonical_control"
    assert not (tmp_path / "ablation").exists()


def test_synthetic_matrix_runner_is_dry_by_default(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix_only.npz"
    np.savez_compressed(matrix, X=np.eye(8, dtype=np.float32))
    manifest = {
        "manifest_id": "accg_synthetic_w0_w5_v1",
        "config": {"n_clusters": 2},
        "worlds": ["W0_matched_null"],
        "records": [
            {
                "dataset_id": "synthetic-dry",
                "family": "lognormal_sparse",
                "world": "W0_matched_null",
                "seed": 42,
                "matrix_path": str(matrix),
            }
        ],
    }
    manifest_path = tmp_path / "synthetic_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output_root = tmp_path / "never-created-synthetic"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/ACCG/run_synthetic_matrix.py"),
            "--manifest",
            str(manifest_path),
            "--output-root",
            str(output_root),
            "--worlds",
            "W0_matched_null",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert '"execute": false' in completed.stdout.lower()
    assert not output_root.exists()


def test_real_summary_recomputes_saved_outer_metrics(tmp_path: Path) -> None:
    module = _load_script("summarize_matrix.py")
    output = tmp_path / "panel"
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    output.mkdir()
    np.save(output / "labels_true.npy", labels)
    (output / "branchpoint.pt").write_bytes(b"test")
    (output / "summary.json").write_text(
        json.dumps({"status": "completed", "seed": 42, "variant": "accg_joint"}), encoding="utf-8"
    )
    (output / "runner_profile.json").write_text(
        json.dumps(
            {
                "dataset": "dataset",
                "dataset_sha256": "source-sha",
                "config_sha256": "config-sha",
                "labels_used_during_fit": False,
                "branchpoint_reused": False,
            }
        ),
        encoding="utf-8",
    )
    (output / "resolved_config.json").write_text(json.dumps({"variant": "accg_joint"}), encoding="utf-8")
    (output / "audit.json").write_text(
        json.dumps(
            {
                "matched_schedule": {
                    "T_s": {"donor": True, "eligible": True, "budget": True, "selection_noise": True},
                    "T_c": {"donor": True, "eligible": True, "budget": True, "selection_noise": True},
                }
            }
        ),
        encoding="utf-8",
    )
    for arm in ("N", "R", "T_s", "T_c"):
        arm_dir = output / arm
        arm_dir.mkdir()
        np.save(arm_dir / "predictions.npy", labels)
        (arm_dir / "metrics.json").write_text(json.dumps({"ari": 1.0, "nmi": 1.0}), encoding="utf-8")
        (arm_dir / "structural_audit.json").write_text("{}", encoding="utf-8")
    job = {
        "run_key": "main",
        "role": "main",
        "seed": 42,
        "record": {"name": "dataset", "source_sha256": "source-sha"},
        "config_sha256": "config-sha",
    }
    assert module._audit_job(job, output) == []
    (output / "T_c/metrics.json").write_text(json.dumps({"ari": 0.0, "nmi": 1.0}), encoding="utf-8")
    assert "T_c:ari_mismatch" in module._audit_job(job, output)


def test_real_summary_keeps_unlabelled_operational_panel_out_of_confirmatory_aggregate(
    tmp_path: Path,
) -> None:
    module = _load_script("summarize_matrix.py")
    output = tmp_path / "operational"
    output.mkdir()
    (output / "branchpoint.pt").write_bytes(b"test")
    (output / "summary.json").write_text(
        json.dumps({"status": "completed", "seed": 42, "variant": "accg_joint"}), encoding="utf-8"
    )
    (output / "runner_profile.json").write_text(
        json.dumps(
            {
                "dataset": "unlabelled",
                "dataset_sha256": "source-sha",
                "config_sha256": "config-sha",
                "labels_used_during_fit": False,
                "branchpoint_reused": False,
            }
        ),
        encoding="utf-8",
    )
    (output / "resolved_config.json").write_text(
        json.dumps({"variant": "accg_joint"}), encoding="utf-8"
    )
    (output / "audit.json").write_text(
        json.dumps(
            {
                "matched_schedule": {
                    "T_s": {"donor": True, "eligible": True, "budget": True, "selection_noise": True},
                    "T_c": {"donor": True, "eligible": True, "budget": True, "selection_noise": True},
                }
            }
        ),
        encoding="utf-8",
    )
    for arm in ("N", "R", "T_s", "T_c"):
        arm_dir = output / arm
        arm_dir.mkdir()
        np.save(arm_dir / "predictions.npy", np.asarray([0, 1], dtype=np.int64))
        (arm_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "readout": "clean_embedding_known_k_kmeans",
                    "labels_used_for_fit": False,
                    "labels_used_for_readout": False,
                    "n_clusters": 2,
                }
            ),
            encoding="utf-8",
        )
        (arm_dir / "structural_audit.json").write_text(
            json.dumps(
                {
                    "joint_delta_mean": 0.0,
                    "constraint_violation_rate": 0.0,
                    "constraint_infeasible_rate": 0.0,
                    "budget_fill": 1.0,
                }
            ),
            encoding="utf-8",
        )
    manifest = {
        "manifest_id": "accg_locked_real_panel_v2",
        "jobs": [
            {
                "run_key": "operational",
                "role": "main",
                "dataset_id": "unlabelled",
                "seed": 42,
                "output_dir": str(output),
                "record": {
                    "name": "unlabelled",
                    "domain": "scRNA",
                    "source_family": "public_source",
                    "source_sha256": "source-sha",
                    "labels_present": False,
                    "K_source": "explicit_n_clusters",
                },
                "config_sha256": "config-sha",
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    payload = module.summarize(manifest_path)
    assert payload["status"] == "complete"
    assert payload["main_dataset_count"] == 0
    assert payload["operational_dataset_count"] == 1
    assert payload["operational_run_rows"][0]["k_source"] == "explicit_n_clusters"
