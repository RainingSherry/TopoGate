"""Persistent 32+8+5 V0_RG retuning for five audited biological datasets.

This runner creates an isolated experiment root, runs exactly one writer per
dataset, and refreshes only server-side CSV summaries from verified final runs.
It never computes a full content SHA-256 for a raw biological source file.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
PROTOCOL_ID = "topogate_unified_biology_missing5_20260916"
ALLOWED_GPUS = (1, 2, 3, 5)
DATA_ROOT = Path("/data/luolie/biopipeline/scCluBench/data")
OLD_TUNING_ROOT = Path("/data/luolie/ToPoGate/result/v0_per_dataset_tuning_v1_full_20260903_40pergpu")
DEFAULT_PYTHON = Path("/data/luolie/conda/envs/topogate_missing5_20260916/bin/python")


DATASETS: tuple[dict[str, Any], ...] = (
    {
        "dataset_id": "Pollen",
        "path": str(DATA_ROOT / "processed/Pollen.h5ad"),
        "matrix_key": "X",
        "label_key": "obs/resolved_label",
        "input_kind": "log1p_expression",
        "n_clusters": 11,
        "panel": "biology",
        "source_group": "Pollen",
        "representation": "processed_Pollen_h5ad_X_log1p_expression",
        "source_reference": "scCluBench processed/Pollen.h5ad",
        "development_use_history": "historical V0 tuning; not reused as a score",
        "old_winner_path": str(OLD_TUNING_ROOT / "worker66/Pollen/selected_config.json"),
    },
    {
        "dataset_id": "Shekhar",
        "path": str(DATA_ROOT / "processed_scmae/Shekhar.h5ad"),
        "matrix_key": "X",
        "label_key": "obs/resolved_label",
        "input_kind": "raw_count",
        "n_clusters": 18,
        "panel": "biology",
        "source_group": "Shekhar",
        "representation": "processed_scmae_Shekhar_h5ad_X_raw_count",
        "source_reference": "scCluBench processed_scmae/Shekhar.h5ad",
        "development_use_history": "historical V0 tuning; not reused as a score",
        "old_winner_path": str(OLD_TUNING_ROOT / "worker70/Shekhar/selected_config.json"),
    },
    {
        "dataset_id": "Wang",
        "path": str(DATA_ROOT / "processed/Wang.h5ad"),
        "matrix_key": "X",
        "label_key": "obs/resolved_label",
        "input_kind": "log1p_expression",
        "n_clusters": 2,
        "panel": "biology",
        "source_group": "Wang",
        "representation": "processed_Wang_h5ad_X_log1p_expression",
        "source_reference": "scCluBench processed/Wang.h5ad",
        "development_use_history": "historical V0 tuning; not reused as a score",
        "old_winner_path": str(OLD_TUNING_ROOT / "worker66/Wang/selected_config.json"),
    },
    {
        "dataset_id": "hrvatin",
        "path": str(DATA_ROOT / "scMAE/hrvatin.h5"),
        "matrix_key": "X",
        "label_key": "Y",
        "input_kind": "log1p_expression",
        "n_clusters": 8,
        "panel": "biology",
        "source_group": "hrvatin",
        "representation": "scMAE_hrvatin_h5_X_log1p_expression",
        "source_reference": "scCluBench scMAE/hrvatin.h5",
        "development_use_history": "historical V0 tuning; not reused as a score",
        "old_winner_path": str(OLD_TUNING_ROOT / "worker71/hrvatin/selected_config.json"),
    },
    {
        "dataset_id": "hrvatin_geo",
        "path": str(DATA_ROOT / "scMAE/hrvatin_geo/GSE102827_MATRIX.csv.gz"),
        "labels_path": str(DATA_ROOT / "scMAE/hrvatin_geo/GSE102827_cell_type_assignments.csv.gz"),
        "input_adapter": "hrvatin_geo_csv",
        "label_column": "maintype",
        "input_kind": "log1p_expression",
        "n_clusters": 8,
        "panel": "biology",
        "source_group": "hrvatin_geo",
        "representation": "canonical_GSE102827_cell_by_gene_log1p_expression",
        "source_reference": "GSE102827 canonical matrix plus cell_type_assignments",
        "development_use_history": "new canonical GEO representation; no historical warm start",
    },
)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


def _run_text(command: list[str], *, cwd: Path | None = None) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()


def _old_winner_mapping(dataset: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Map only compatible 2026-09-03 fields; never import a historical score."""
    path_text = dataset.get("old_winner_path")
    if not path_text:
        return None, {
            "dataset_id": dataset["dataset_id"],
            "classification": "no_historical_warm_start",
            "reason": "hrvatin_geo has no 2026-09-03 per-dataset winner",
        }
    source = Path(path_text)
    if not source.exists():
        return None, {
            "dataset_id": dataset["dataset_id"],
            "classification": "historical_winner_unavailable",
            "path": str(source),
            "reason": "selected_config.json is absent",
        }
    historic = json.loads(source.read_text())
    raw = historic.get("selected_config", {})
    mode = str(historic.get("selected_parameterization", ""))
    compatible = {
        key: raw[key]
        for key in (
            "lr", "mask_ratio", "tau", "pseudo_weight", "neighbor_k", "mix_neighbors",
            "edge_reliability_mode", "gamma_sim", "gamma_mutual", "gamma_snn",
            "gamma_distance", "gate_min", "gate_max", "beta_mutual", "beta_snn",
            "beta_perturb",
        )
        if key in raw
    }
    compatible["neighbor_estimator"] = "current"
    compatible["auxiliary_weighting"] = "gate"
    compatible["beta_uncertainty"] = 0.0
    if compatible.get("edge_reliability_mode") != "sim_mutual_snn_distance":
        compatible["gamma_distance"] = 0.0
    mapping_rule = "direct compatible RG fields"
    if mode == "fixed":
        strength = float(raw.get("effective_mix_strength", 0.05))
        compatible.update(
            {
                "gate_center": None,
                "gate_adaptivity": None,
                "gate_min": strength,
                "gate_max": strength,
                "edge_reliability_mode": "none",
                "gamma_sim": 0.0,
                "gamma_mutual": 0.0,
                "gamma_snn": 0.0,
                "gamma_distance": 0.0,
                "beta_mutual": 0.0,
                "beta_snn": 0.0,
                "beta_perturb": 0.0,
            }
        )
        mapping_rule = "legacy fixed F1 mapped to constant gate plus base neighbor probabilities"
    audit = {
        "dataset_id": dataset["dataset_id"],
        "classification": "warm_start_only",
        "historical_path": str(source),
        "historical_selected_candidate": historic.get("selected_candidate"),
        "historical_parameterization": mode,
        "mapping_rule": mapping_rule,
        "compatible_overrides": compatible,
        "historical_scores_imported": False,
        "historical_protocol_reused": False,
    }
    return compatible, audit


def build_manifest() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    datasets = []
    audit = []
    for template in DATASETS:
        row = dict(template)
        warm_start, record = _old_winner_mapping(row)
        row.pop("old_winner_path", None)
        row["warm_start_configs"] = [warm_start] if warm_start is not None else []
        datasets.append(row)
        audit.append(record)
    return {
        "protocol_id": PROTOCOL_ID,
        "raw_source_full_sha256": "not_computed",
        "datasets": datasets,
    }, audit


def _environment(python: Path) -> dict[str, Any]:
    packages = "numpy scipy scikit-learn torch scanpy anndata h5py pandas optuna PyYAML"
    version_code = (
        "import importlib.metadata as m, json; "
        f"print(json.dumps({{p: m.version(p) for p in {packages!r}.split()}}))"
    )
    try:
        package_versions = json.loads(_run_text([str(python), "-c", version_code]))
    except subprocess.CalledProcessError as exc:
        package_versions = {"error": exc.output}
    try:
        gpu_info = _run_text(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used", "--format=csv,noheader"]
        ).splitlines()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        gpu_info = [f"unavailable: {exc}"]
    try:
        disk_info = _run_text(["df", "-h", "/data"]).splitlines()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        disk_info = [f"unavailable: {exc}"]
    return {
        "repository": str(REPOSITORY_ROOT),
        "git_commit": _run_text(["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT),
        "git_branch": _run_text(["git", "branch", "--show-current"], cwd=REPOSITORY_ROOT),
        "python": str(python),
        "python_version": _run_text([str(python), "--version"]),
        "package_versions": package_versions,
        "gpus": gpu_info,
        "disk_budget": disk_info,
        "allowed_physical_gpus": list(ALLOWED_GPUS),
        "environment_variable_whitelist": ["CUDA_VISIBLE_DEVICES", "PYTHONPATH"],
    }


def prepare(root: Path, python: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name in ("inventory", "scheduler/logs", "scheduler/status", "scheduler/jobs", "datasets", "summaries", "reports", "checksums"):
        (root / name).mkdir(parents=True, exist_ok=True)
    manifest, audit = build_manifest()
    protocol = {
        "protocol_id": PROTOCOL_ID,
        "model": "TopoGate V0_RG rg_full only",
        "split": {"train": 0.6, "validation": 0.2, "test": 0.2, "split_seed": 9102026},
        "search": {"candidates": 32, "screen_seed": 42, "refine_seeds": [123, 7], "selection": "mean validation ARI, then sample std, then config hash"},
        "final": {"seeds": [42, 123, 7, 2025, 3407], "metrics": ["ARI", "NMI", "ACC"], "std_ddof": 1},
        "fixed": {
            "epochs": 80, "hidden_size": 128, "batch_size": 256, "dropout": 0,
            "masked_data_weight": 0.75, "mask_loss_weight": 0.70, "knn_pca_dim": 50,
            "n_top_features": 2000, "kmeans_n_init": 20, "neighbor_estimator": "current",
            "auxiliary_weighting": "gate", "beta_uncertainty": 0,
        },
        "labels_used_during_fit": False,
        "selection_labels": "validation_only",
        "test_used_for_selection": False,
        "raw_source_full_sha256": "not_computed",
        "gpu_identity_excluded_from_scientific_config": True,
    }
    write_json(root / "protocol.json", protocol)
    write_json(root / "dataset_manifest.json", manifest)
    write_json(root / "environment.json", _environment(python))
    write_json(root / "inventory/historical_warm_start_audit.json", audit)
    atomic_text(
        root / "README.md",
        "# Five biology V0_RG retuning\n\n"
        "This root contains only the new 32+8+5 `rg_full` protocol for Pollen, Shekhar, Wang, hrvatin, and hrvatin_geo. "
        "Historical scores are never reused. Resume one dataset with:\n\n"
        f"`{python} scripts/V0_RG/run_missing5_biology.py --root {root} --python {python} run-dataset --dataset-id Pollen --gpu 5`\n",
    )
    update_summaries(root)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], preferred_headers: list[str]) -> None:
    headers = list(preferred_headers)
    for row in rows:
        for key in row:
            if key not in headers:
                headers.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows([{key: _csv_value(value) for key, value in row.items()} for row in rows])
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def update_summaries(root: Path) -> None:
    manifest = _read_json(root / "dataset_manifest.json") or {"datasets": []}
    ids = [row["dataset_id"] for row in manifest["datasets"]]
    trial_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    report_rows: list[str] = []
    for dataset_id in ids:
        dataset_dir = root / "datasets" / dataset_id
        selected = _read_json(dataset_dir / "selected.json")
        if selected:
            winner = selected["winner"]
            selected_rows.append(
                {
                    "dataset_id": dataset_id,
                    "selection_hash": selected.get("selection_hash"),
                    "validation_ari_mean": winner.get("validation_ari_mean"),
                    "validation_ari_std": winner.get("validation_ari_std"),
                    "winner_config_hash": winner.get("config_hash"),
                    "winner_config": winner.get("config"),
                }
            )
            for rank, candidate in enumerate(selected.get("shortlist", []), start=1):
                ranking_rows.append({"dataset_id": dataset_id, "rank": rank, **candidate})
        for stage in ("validation", "final"):
            for record_path in sorted(dataset_dir.glob(f"{stage}/*/seed_*/record.json")):
                record = _read_json(record_path)
                if not record:
                    continue
                row = {
                    "dataset_id": dataset_id,
                    "stage": stage,
                    "seed": record.get("seed"),
                    "status": record.get("status"),
                    "attempt": record.get("attempt"),
                    "config_hash": record.get("config_hash"),
                    "validation_ari": record.get("validation_ari"),
                    "wall_seconds": record.get("wall_seconds"),
                    "record_path": str(record_path),
                }
                if record.get("test_metrics"):
                    row.update({key: record["test_metrics"].get(key) for key in ("ari", "nmi", "acc")})
                    final_rows.append({"dataset_id": dataset_id, "seed": record["seed"], **record["test_metrics"], "record_path": str(record_path)})
                if record.get("status") == "failed":
                    failure_rows.append({"dataset_id": dataset_id, "stage": stage, "seed": record.get("seed"), "error_type": record.get("error_type"), "error": record.get("error"), "record_path": str(record_path)})
                trial_rows.append(row)
                artifact = _read_json(record_path.parent / "artifact_manifest.json")
                if artifact:
                    for item in artifact.get("generated_artifacts", []):
                        artifact_rows.append({"dataset_id": dataset_id, "stage": stage, "seed": record.get("seed"), **item, "run_dir": str(record_path.parent)})
        for failure_path in dataset_dir.glob("failures/*.json"):
            failure = _read_json(failure_path)
            if failure:
                failure_rows.append({"dataset_id": dataset_id, "stage": failure_path.stem, **failure, "record_path": str(failure_path)})
        final_summary = _read_json(dataset_dir / "final_summary.json")
        completed = _read_json(dataset_dir / "completed.json")
        if final_summary and completed:
            metrics = final_summary.get("metrics", {})
            summary_rows.append(
                {
                    "dataset_id": dataset_id,
                    "status": final_summary.get("status"),
                    "selection_hash": final_summary.get("selection_hash"),
                    "final_seed_count": final_summary.get("n_seeds"),
                    "ARI_mean": metrics.get("ari", {}).get("mean"),
                    "ARI_std": metrics.get("ari", {}).get("std"),
                    "NMI_mean": metrics.get("nmi", {}).get("mean"),
                    "NMI_std": metrics.get("nmi", {}).get("std"),
                    "ACC_mean": metrics.get("acc", {}).get("mean"),
                    "ACC_std": metrics.get("acc", {}).get("std"),
                    "summary_path": str(dataset_dir / "final_summary.json"),
                }
            )
            report_rows.append(f"- {dataset_id}: completed, {final_summary.get('n_seeds')} verified final seeds")
        else:
            report_rows.append(f"- {dataset_id}: incomplete")
    summaries = root / "summaries"
    write_csv(summaries / "trials_long.csv", trial_rows, ["dataset_id", "stage", "seed", "status", "attempt", "config_hash", "validation_ari", "ari", "nmi", "acc", "wall_seconds", "record_path"])
    write_csv(summaries / "validation_ranking.csv", ranking_rows, ["dataset_id", "rank", "config_hash", "validation_ari_mean", "validation_ari_std", "seeds", "config"])
    write_csv(summaries / "selected_hyperparameters.csv", selected_rows, ["dataset_id", "selection_hash", "winner_config_hash", "validation_ari_mean", "validation_ari_std", "winner_config"])
    write_csv(summaries / "final_metrics_per_seed.csv", final_rows, ["dataset_id", "seed", "ari", "nmi", "acc", "ami", "f1_macro", "fmi", "record_path"])
    write_csv(summaries / "final_summary_per_dataset.csv", summary_rows, ["dataset_id", "status", "final_seed_count", "ARI_mean", "ARI_std", "NMI_mean", "NMI_std", "ACC_mean", "ACC_std", "selection_hash", "summary_path"])
    write_csv(summaries / "failures.csv", failure_rows, ["dataset_id", "stage", "seed", "error_type", "error", "record_path"])
    write_csv(summaries / "artifact_manifest.csv", artifact_rows, ["dataset_id", "stage", "seed", "path", "size", "sha256", "run_dir"])
    write_csv(root / "checksums/artifact_manifest.csv", artifact_rows, ["dataset_id", "stage", "seed", "path", "size", "sha256", "run_dir"])
    atomic_text(
        root / "reports/FINAL_REPORT.md",
        "# Five-dataset retuning status\n\n"
        "Only completed rows contain five final test seeds. Statistics use sample standard deviation (`ddof=1`).\n\n"
        + "\n".join(report_rows)
        + "\n",
    )


def run_dataset(root: Path, python: Path, dataset_id: str, gpu: int) -> int:
    if gpu not in ALLOWED_GPUS:
        raise ValueError(f"GPU {gpu} is not allowed")
    manifest = _read_json(root / "dataset_manifest.json")
    if manifest is None or dataset_id not in {row["dataset_id"] for row in manifest["datasets"]}:
        raise ValueError(f"dataset_id is absent from this experiment manifest: {dataset_id}")
    scheduler = root / "scheduler"
    log_path = scheduler / "logs" / f"{dataset_id}.log"
    status_path = scheduler / "status" / f"{dataset_id}.json"
    command_base = [
        str(python), "-m", "methods.TopoGate.V0_RG.tuning",
        "--manifest", str(root / "dataset_manifest.json"), "--output-dir", str(root / "datasets"),
        "--dataset-id", dataset_id, "--device", "cuda", "--gpu", str(gpu),
        "--epochs", "80", "--trials", "32", "--feature-limit", "2000",
        "--split-seed", "9102026", "--search-seed", "9102026", "--max-attempts", "3",
    ]
    status = {
        "dataset_id": dataset_id,
        "status": "running",
        "pid": os.getpid(),
        "physical_gpu": gpu,
        "log_path": str(log_path),
        "restart_command": " ".join(command_base + ["--stage", "search"]),
        "started_unix": time.time(),
    }
    write_json(status_path, status)
    clean_environment = dict(os.environ)
    clean_environment.pop("CUDA_VISIBLE_DEVICES", None)
    exit_code = 1
    try:
        with log_path.open("a", encoding="utf-8") as log:
            for stage in ("search", "final"):
                command = command_base + ["--stage", stage]
                log.write(f"\n[{time.strftime('%FT%T%z')}] {' '.join(command)}\n")
                log.flush()
                completed = subprocess.run(command, cwd=REPOSITORY_ROOT, env=clean_environment, stdout=log, stderr=subprocess.STDOUT, check=False)
                if completed.returncode != 0:
                    status.update(status="failed", stage=stage, exit_code=completed.returncode, finished_unix=time.time())
                    write_json(status_path, status)
                    return completed.returncode
        status.update(status="completed", stage="final", exit_code=0, finished_unix=time.time())
        write_json(status_path, status)
        exit_code = 0
        return 0
    finally:
        update_summaries(root)
        if exit_code != 0 and status.get("status") == "running":
            status.update(status="failed", stage="unknown", exit_code=exit_code, finished_unix=time.time())
            write_json(status_path, status)


def dispatch(root: Path, python: Path) -> None:
    assignments = {"Pollen": 5, "Wang": 5, "Shekhar": 1, "hrvatin": 2, "hrvatin_geo": 3}
    for dataset_id, gpu in assignments.items():
        status = _read_json(root / "scheduler/status" / f"{dataset_id}.json")
        if status and status.get("status") in {"running", "completed"}:
            continue
        command = [
            str(python), str(Path(__file__).resolve()), "--root", str(root), "--python", str(python),
            "run-dataset", "--dataset-id", dataset_id, "--gpu", str(gpu),
        ]
        process = subprocess.Popen(
            command,
            cwd=REPOSITORY_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        write_json(
            root / "scheduler/jobs" / f"{dataset_id}.json",
            {"dataset_id": dataset_id, "pid": process.pid, "physical_gpu": gpu, "command": command, "launched_unix": time.time()},
        )


def smoke(root: Path, python: Path, gpu: int) -> None:
    """Exercise each registered loader without persisting any raw matrix copy."""
    if gpu not in ALLOWED_GPUS:
        raise ValueError(f"GPU {gpu} is not allowed")
    from methods.TopoGate.V0_RG.config import V0_RGConfig
    from methods.TopoGate.V0_RG.input_adapter import load_matrix
    from methods.TopoGate.V0_RG.run import clustering_metrics, resolve_runtime_device
    from methods.TopoGate.V0_RG.trainer import fit_predict, predict_from_saved_state
    from methods.TopoGate.V0_RG.tuning import SplitPreprocessor

    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    runtime_device = resolve_runtime_device("cuda", gpu)
    manifest = _read_json(root / "dataset_manifest.json")
    rows = []
    progress_path = root / "reports/smoke_progress.json"
    write_json(progress_path, {"status": "running", "runtime_device": runtime_device, "datasets": rows})
    for row in manifest["datasets"]:
        loaded = load_matrix(
            row["path"], row.get("labels_path"), matrix_key=row.get("matrix_key", "X"),
            label_key=row.get("label_key"), label_column=row.get("label_column"), input_adapter=row.get("input_adapter"),
        )
        n_rows = min(96, loaded.X.shape[0])
        fit_n = max(8, int(n_rows * 0.7))
        preprocessor = SplitPreprocessor(row["input_kind"], min(64, loaded.X.shape[1])).fit(loaded.X[:fit_n])
        fit_X = preprocessor.transform(loaded.X[:fit_n])
        score_X = preprocessor.transform(loaded.X[fit_n:n_rows])
        config = replace(
            V0_RGConfig(protocol_id="missing5_smoke", hidden_size=8, epochs=1, batch_size=16,
                         n_top_features=min(64, loaded.X.shape[1]), beta_uncertainty=0.0),
            gate_center=None, gate_adaptivity=None, gate_min=0.05, gate_max=0.05,
            edge_reliability_mode="none",
        )
        prediction, embedding, diagnostics = fit_predict(
            score_X, fit_X=fit_X, n_clusters=2, config=config, seed=42, device=runtime_device,
        )
        recreated, _ = predict_from_saved_state(
            score_X, config=config, device=runtime_device,
            model_state_dict=diagnostics["model_state_dict"], cluster_centers=diagnostics["cluster_centers"],
        )
        labels = loaded.labels[fit_n:n_rows]
        _, encoded = __import__("numpy").unique(labels.astype(str), return_inverse=True)
        rows.append(
            {
                "dataset_id": row["dataset_id"], "input_shape": list(loaded.X.shape), "profile": loaded.profile,
                "smoke_prediction_reload_equal": bool((prediction == recreated).all()),
                "smoke_metrics": clustering_metrics(encoded, prediction), "embedding_shape": list(embedding.shape),
            }
        )
        write_json(progress_path, {"status": "running", "runtime_device": runtime_device, "datasets": rows})
    write_json(root / "reports/smoke.json", {"status": "passed", "runtime_device": runtime_device, "datasets": rows})
    write_json(progress_path, {"status": "passed", "runtime_device": runtime_device, "datasets": rows})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("prepare")
    run_parser = subparsers.add_parser("run-dataset")
    run_parser.add_argument("--dataset-id", required=True)
    run_parser.add_argument("--gpu", required=True, type=int)
    dispatch_parser = subparsers.add_parser("dispatch")
    dispatch_parser.set_defaults(action="dispatch")
    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--gpu", default=5, type=int)
    subparsers.add_parser("summarize")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.root, args.python)
    elif args.action == "run-dataset":
        raise SystemExit(run_dataset(args.root, args.python, args.dataset_id, args.gpu))
    elif args.action == "dispatch":
        dispatch(args.root, args.python)
    elif args.action == "smoke":
        smoke(args.root, args.python, args.gpu)
    else:
        update_summaries(args.root)


if __name__ == "__main__":
    main()
