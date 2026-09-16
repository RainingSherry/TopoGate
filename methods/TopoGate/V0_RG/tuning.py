"""Per-dataset validation-ARI tuning; train-only preprocessing and readout.

Run search first, inspect selected.json, then run final with the frozen selection.
Imports of Optuna/Torch are delayed so manifests and selection can be audited on CPU.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import time
import traceback
from typing import Any

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler

from .config import V0_RGConfig

PROTOCOL = "topogate_unified_perdataset_inductive_v1"
SELECTION_SEEDS = (42, 123, 7)
FINAL_SEEDS = (42, 123, 7, 2025, 3407)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def content_digest(path):
    """Hash code or generated artifacts; never call this for source data."""
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def code_digest():
    root = Path(__file__).parent
    paths = sorted(root.glob("*.py"))
    paths += sorted((root.parents[1] / "NeighborMix_scMAE").glob("*.py"))
    return digest({str(p.relative_to(root.parents[2])): content_digest(p) for p in paths})


def source_fingerprint(path: str | Path) -> dict[str, Any]:
    """Metadata-only source fingerprint required by the no-raw-SHA policy."""
    source = Path(path).resolve()
    metadata = {"path": str(source), "size": int(source.stat().st_size), "mtime_ns": int(source.stat().st_mtime_ns)}
    return {**metadata, "metadata_hash": digest(metadata), "content_sha256": "not_computed"}


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def write_numpy(path: str | Path, values: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values)
    temporary.replace(path)


def write_torch(path: str | Path, payload: dict[str, Any]) -> None:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def split_rows(n, seed=9102026):
    """Fixed, X-only random split; no labels used to allocate samples."""
    if n < 10:
        raise ValueError("fewer than 10 rows: record as ineligible for this split protocol")
    rows = np.random.default_rng(seed).permutation(n)
    a, b = int(n * .6), int(n * .8)
    return rows[:a], rows[a:b], rows[b:]


class SplitPreprocessor:
    """Same fitted feature map for fit/validation/test, with explicit input semantics.

    Biological feature selection uses Seurat HVG with the existing documented
    variance fallback. General matrices use train variance only when capped.
    Scaling is always fitted on training rows; count normalization is row-local.
    """
    def __init__(self, input_kind, n_top_features=2000, target_sum=10000.):
        if input_kind not in {"general", "raw_count", "log1p_expression"}:
            raise ValueError("input_kind must be general, raw_count or log1p_expression")
        if n_top_features <= 0 or not math.isfinite(target_sum) or target_sum <= 0:
            raise ValueError("invalid preprocessing limits")
        self.input_kind = input_kind
        self.n_top_features = n_top_features
        self.target_sum = target_sum

    def _normalize(self, X):
        X = sp.csr_matrix(X, dtype=np.float32, copy=True)
        X.sum_duplicates()
        X.eliminate_zeros()
        if not np.isfinite(X.data).all():
            raise ValueError("input contains non-finite values; resolve before the experiment")
        if self.input_kind != "general" and np.any(X.data < 0):
            raise ValueError("biological expression must be nonnegative")
        if self.input_kind == "raw_count":
            if not np.allclose(X.data, np.rint(X.data), atol=1e-5, rtol=0):
                raise ValueError("raw_count requires integer-valued counts")
            sums = np.asarray(X.sum(axis=1)).ravel()
            scale = np.divide(self.target_sum, sums, out=np.zeros_like(sums), where=sums > 0)
            X = X.multiply(scale[:, None]).tocsr()
            X.data = np.log1p(X.data)
        return X

    def fit(self, X):
        X = self._normalize(X)
        self.width = X.shape[1]
        count = min(self.width, self.n_top_features)
        if self.input_kind != "general":
            from .input_adapter import _hvg_subset
            _, self.features, self.selection = _hvg_subset(X, count)
        else:
            mean = np.asarray(X.mean(axis=0)).ravel().astype(np.float64)
            variance = np.asarray(X.multiply(X).mean(axis=0)).ravel() - mean ** 2
            if not np.isfinite(variance).all():
                raise ValueError("feature variance overflow")
            self.features = np.sort(np.lexsort((np.arange(self.width), -variance))[:count])
            self.selection = {"strategy": "all" if count == self.width else "train_variance"}
        self.scaler = StandardScaler().fit(X[:, self.features].toarray())
        self.fingerprint = digest({"input_kind": self.input_kind, "width": self.width,
            "features": self.features.tolist(), "mean": self.scaler.mean_.tolist(),
            "scale": self.scaler.scale_.tolist(), "selection": self.selection,
            "target_sum": self.target_sum})
        return self

    def transform(self, X):
        X = self._normalize(X)
        if X.shape[1] != self.width:
            raise ValueError("input feature width changed")
        result = self.scaler.transform(X[:, self.features].toarray()).astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("non-finite transformed matrix")
        return np.ascontiguousarray(result)

    def payload(self, *, fit_scope: str) -> dict[str, Any]:
        return {
            "input_kind": self.input_kind,
            "target_sum": self.target_sum,
            "n_top_features": self.n_top_features,
            "width": int(self.width),
            "features": self.features.tolist(),
            "selection": self.selection,
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
            "fingerprint": self.fingerprint,
            "fit_scope": fit_scope,
        }


def initial_candidates(base):
    """Four mechanism anchors; endpoint parity is at operator, not legacy-run level."""
    common = dict(gate_center=None, gate_adaptivity=None, auxiliary_weighting="gate",
                  neighbor_estimator="current", variant="rg_full")
    constant = replace(base, **common, gate_min=.1, gate_max=.1, neighbor_k=5,
                       mix_neighbors=4, edge_reliability_mode="none")
    topology = replace(base, **common, gate_min=0., gate_max=.15, neighbor_k=10,
                       mix_neighbors=4, edge_reliability_mode="sim_mutual_snn_distance")
    return [constant, topology,
            replace(constant, edge_reliability_mode="sim_mutual_snn_distance"),
            replace(topology, edge_reliability_mode="none")]


def suggest_config(trial, base):
    center = trial.suggest_float("gate_center", .02, .2, log=True)
    adapt = trial.suggest_categorical("gate_adaptivity", [0., .25, .5, 1.])
    edge = trial.suggest_categorical("edge", [False, True])
    return replace(base, gate_center=center, gate_adaptivity=adapt,
        gate_min=0., gate_max=.15, variant="rg_full", auxiliary_weighting="gate",
        neighbor_k=trial.suggest_categorical("neighbor_k", [5, 10, 20, 30]),
        mix_neighbors=trial.suggest_categorical("mix_neighbors", [2, 4]),
        neighbor_estimator="current", gamma_sim=0., gamma_distance=0.,
        edge_reliability_mode="sim_mutual_snn" if edge else "none",
        gamma_mutual=trial.suggest_categorical("gamma_mutual", [0., .5, 1., 2., 4.]) if edge else 0.,
        gamma_snn=trial.suggest_categorical("gamma_snn", [0., .5, 1., 2., 4.]) if edge else 0.,
        beta_mutual=trial.suggest_categorical("beta_mutual", [0., 1., 2., 4.]) if adapt else 0.,
        beta_snn=trial.suggest_categorical("beta_snn", [0., 1., 2., 4.]) if adapt else 0.,
        beta_perturb=trial.suggest_categorical("beta_perturb", [0., 1., 2., 4.]) if adapt else 0.,
        beta_uncertainty=0.,
        tau=trial.suggest_float("tau", .1, 1., log=True),
        pseudo_weight=trial.suggest_float("pseudo_weight", .05, 1., log=True),
        mask_ratio=trial.suggest_float("mask_ratio", .2, .6),
        lr=trial.suggest_float("lr", 3e-4, 3e-3, log=True))


def rank_candidates(records, seeds):
    """Rank only complete, finite, same-budget validation results. No test field accepted."""
    if len({r["identity"] for r in records}) > 1:
        raise ValueError("mixed experiment identities")
    groups = {}
    for r in records:
        if r.get("stage") != "validation" or "test_metrics" in r:
            raise ValueError("selection accepts validation records only")
        if r["status"] != "completed":
            continue
        key = digest(r["config"])
        groups.setdefault(key, {})
        seed = int(r["seed"])
        if seed in groups[key]:
            raise ValueError("duplicate candidate/seed record")
        if not math.isfinite(float(r["validation_ari"])):
            raise ValueError("non-finite validation score")
        groups[key][seed] = r
    ranked = []
    for key, by_seed in groups.items():
        if not all(s in by_seed for s in seeds):
            continue
        rows = [by_seed[s] for s in seeds]
        if len({r["identity"] for r in rows}) != 1:
            raise ValueError("mixed experiment identity")
        scores = [r["validation_ari"] for r in rows]
        ranked.append({"config": rows[0]["config"], "config_hash": key,
                       "validation_ari_mean": float(np.mean(scores)),
                       "validation_ari_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
                       "seeds": list(seeds)})
    if len({r["config"]["epochs"] for r in ranked}) > 1:
        raise ValueError("cannot select across training budgets")
    return sorted(ranked, key=lambda r: (-r["validation_ari_mean"], r["validation_ari_std"], r["config_hash"]))


def load_manifest(path):
    data = json.loads(Path(path).read_text())
    rows = data["datasets"]
    seen = set()
    for row in rows:
        for key in ("dataset_id", "path", "input_kind", "n_clusters", "panel", "source_group", "representation"):
            if key not in row:
                raise ValueError(f"manifest row missing {key}")
        identifier = row["dataset_id"]
        if not identifier or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for c in identifier):
            raise ValueError("dataset_id must be a safe unique directory name")
        if identifier in seen:
            raise ValueError("duplicate dataset_id")
        seen.add(identifier)
        if row["panel"] not in {"clubench", "biology"} or int(row["n_clusters"]) < 2:
            raise ValueError("invalid panel or registered K")
    return rows


def _legacy_run_dataset(row, out, args):
    from .input_adapter import load_matrix
    from .trainer import fit_predict
    from sklearn.metrics import adjusted_rand_score

    out.mkdir(parents=True, exist_ok=True)
    loaded = load_matrix(row["path"], row.get("labels_path"))
    X, y = loaded.X, loaded.labels
    if y is None or np.asarray(y).ndim != 1 or len(y) != X.shape[0]:
        raise ValueError("benchmark needs one external label per row")
    train, val, test = split_rows(len(y), args.split_seed)
    if int(row["n_clusters"]) > len(train):
        raise ValueError("registered K exceeds training size")
    environment = {k: importlib.metadata.version(k) for k in ("numpy", "scipy", "scikit-learn", "torch", "PyYAML")}
    if row["input_kind"] != "general":
        environment["scanpy"] = importlib.metadata.version("scanpy")
    identity = {"protocol": PROTOCOL, "dataset": row, "data_source": source_fingerprint(row["path"]),
        "labels_source": source_fingerprint(row["labels_path"]) if row.get("labels_path") else "embedded",
        "split_hash": digest([train.tolist(), val.tolist(), test.tolist()]),
        "code_hash": code_digest(), "epochs": args.epochs, "trials": args.trials,
        "search_seed": args.search_seed, "feature_limit": args.feature_limit,
        "environment": environment, "python": platform.python_version(), "device": args.device}
    identity_hash = digest(identity)
    meta_path = out / "experiment.json"
    if meta_path.exists() and json.loads(meta_path.read_text()) != identity:
        raise ValueError("experiment identity changed; use a new output directory")
    write_json(meta_path, identity)
    write_json(out / "split.json", {"train": train.tolist(), "validation": val.tolist(), "test": test.tolist()})
    base = V0_RGConfig(protocol_id=PROTOCOL, epochs=args.epochs, n_top_features=args.feature_limit)
    final = args.stage == "final"
    selected_path = out / "selected.json"
    if final:
        selected = json.loads(selected_path.read_text())
        expected_digest = selected.pop("selection_hash")
        if digest(selected) != expected_digest or selected["identity"] != identity_hash:
            raise ValueError("frozen selection integrity or identity mismatch")
        candidates = [V0_RGConfig(**selected["winner"]["config"])]
        fit_rows, score_rows = np.concatenate([train, val]), test
        marker = out / "final_started.json"
        marker_payload = {"selection_hash": expected_digest, "identity": identity_hash}
        if marker.exists() and json.loads(marker.read_text()) != marker_payload:
            raise ValueError("final evaluation already started with a different selection")
        write_json(marker, marker_payload)
    else:
        if (out / "final_started.json").exists():
            raise ValueError("test evaluation has started; this experiment cannot be retuned")
        fit_rows, score_rows = train, val
    prep = SplitPreprocessor(row["input_kind"], args.feature_limit).fit(X[fit_rows])
    fit_X, score_X = prep.transform(X[fit_rows]), prep.transform(X[score_rows])
    write_json(out / ("final_preprocessing.json" if final else "selection_preprocessing.json"),
        {"hash": prep.fingerprint, "features": prep.features.tolist(), "selection": prep.selection,
         "fit_scope": "train_validation" if final else "train"})

    def evaluate(config, seed):
        key = digest(asdict(config))
        path = out / ("final" if final else "validation") / key / f"seed_{seed}.json"
        if path.exists():
            record = json.loads(path.read_text())
            if record["identity"] != identity_hash or record["preprocessing_hash"] != prep.fingerprint:
                raise ValueError("cached run identity mismatch")
            if record["status"] == "completed":
                return record
            raise RuntimeError(f"previous run failed: {path}; inspect before retrying")
        record = {"identity": identity_hash, "config": asdict(config), "seed": seed,
            "stage": "test" if final else "validation", "status": "running",
            "preprocessing_hash": prep.fingerprint, "labels_used_during_fit": False,
            "labels_used_for_selection": not final, "selection_labels": "validation_only",
            "known_k": int(row["n_clusters"]), "K_source": "manifest",
            "test_used_for_selection": False}
        write_json(path, record)
        start = time.monotonic()
        try:
            pred, embedding, diagnostics = fit_predict(score_X, fit_X=fit_X,
                n_clusters=int(row["n_clusters"]), config=config, seed=seed, device=args.device)
            if not np.isfinite(embedding).all():
                raise ValueError("non-finite embedding")
            score = float(adjusted_rand_score(y[score_rows], pred))
            if final:
                from .run import clustering_metrics
                _, encoded = np.unique(y[score_rows], return_inverse=True)
                record["test_metrics"] = clustering_metrics(encoded, pred)
            else:
                record["validation_ari"] = score
            record.update(status="completed", wall_seconds=time.monotonic()-start,
                          core_summary=diagnostics["core_summary"],
                          gate_summary=diagnostics["gate_summary"], edge_summary=diagnostics["edge_summary"])
            write_json(path, record)
            return record
        except Exception as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            write_json(path, record)
            raise

    if final:
        records = [evaluate(candidates[0], s) for s in FINAL_SEEDS]
        keys = ("ari", "nmi", "acc")
        write_json(out / "final_summary.json", {"identity": identity_hash, "n_seeds": len(records),
            "selection_hash": expected_digest, "test_n": len(test),
            "metrics": {k: {"mean": float(np.mean([r["test_metrics"][k] for r in records])),
                              "std": float(np.std([r["test_metrics"][k] for r in records], ddof=1))} for k in keys}})
        return
    if selected_path.exists():
        # Selection is immutable within this experiment, even before test evaluation.
        raise ValueError("selection already frozen; run final or inspect selected.json")
    import optuna
    anchors = initial_candidates(base)
    for overrides in row.get("warm_start_configs", []):
        locked = {"protocol_id", "epochs", "hidden_size", "batch_size", "dropout",
                  "masked_data_weight", "mask_loss_weight", "n_top_features", "target_sum",
                  "kmeans_n_init", "num_workers", "variant"}
        if set(overrides) & locked:
            raise ValueError("warm start must contain only search parameters, not protocol/backbone")
        anchors.append(replace(base, **overrides))
    anchors = list({digest(asdict(c)): c for c in anchors}.values())
    if len(anchors) > args.trials:
        raise ValueError("warm starts exceed the candidate budget")
    screened = [evaluate(c, SELECTION_SEEDS[0]) for c in anchors]
    sampler = optuna.samplers.TPESampler(seed=args.search_seed, n_startup_trials=8)
    study = optuna.create_study(storage=f"sqlite:///{(out / 'search.db').resolve()}",
                               study_name=identity_hash, direction="maximize", sampler=sampler,
                               load_if_exists=True)
    def objective(trial):
        config = suggest_config(trial, base)
        trial.set_user_attr("config", asdict(config))
        return evaluate(config, SELECTION_SEEDS[0])["validation_ari"]
    remaining = max(0, args.trials - len(anchors) - len(study.trials))
    study.optimize(objective, n_trials=remaining)
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            screened.append(evaluate(V0_RGConfig(**trial.user_attrs["config"]), SELECTION_SEEDS[0]))
    # Exact duplicate proposals count against the search budget but do not retrain.
    unique = {digest(r["config"]): r for r in screened}
    shortlisted = rank_candidates(list(unique.values()), (SELECTION_SEEDS[0],))[:4]
    refined = [evaluate(V0_RGConfig(**r["config"]), s) for r in shortlisted for s in SELECTION_SEEDS]
    ranked = rank_candidates(refined, SELECTION_SEEDS)
    if not ranked:
        raise ValueError("no complete candidate available")
    selected = {"identity": identity_hash, "selection_metric": "validation_ARI_mean",
        "selection_labels": "validation_only", "labels_used_for_selection": True,
        "test_used_for_selection": False, "winner": ranked[0], "shortlist": ranked}
    selected["selection_hash"] = digest(selected)
    write_json(selected_path, selected)


def _package_versions(input_kind: str) -> dict[str, str]:
    packages = ["numpy", "scipy", "scikit-learn", "torch", "PyYAML", "h5py", "pandas"]
    if input_kind != "general":
        packages.extend(["anndata", "scanpy"])
    result = {}
    for package in packages:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not_installed"
    return result


def _profile_for_identity(profile: dict[str, Any]) -> dict[str, Any]:
    """Keep data structure and row-map provenance without hashing source contents."""
    normalized = dict(profile)
    for key in ("source_cell_ids", "source_cell_column_indices"):
        if key in normalized:
            values = normalized.pop(key)
            normalized[f"{key}_count"] = len(values)
            normalized[f"{key}_fingerprint"] = digest(values)
    return normalized


def build_experiment_identity(
    *,
    row: dict[str, Any],
    loaded_profile: dict[str, Any],
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Scientific identity only. Runtime device and physical GPU are excluded."""
    source = {
        "matrix": source_fingerprint(row["path"]),
        "labels": (
            source_fingerprint(row["labels_path"])
            if row.get("labels_path")
            else "embedded_in_matrix_source"
        ),
        "structure": _profile_for_identity(loaded_profile),
    }
    return {
        "protocol": PROTOCOL,
        "dataset": row,
        "source": source,
        "split_hash": digest(
            {"train": train.tolist(), "validation": validation.tolist(), "test": test.tolist()}
        ),
        "code_hash": code_digest(),
        "epochs": int(args.epochs),
        "trials": int(args.trials),
        "search_seed": int(args.search_seed),
        "feature_limit": int(args.feature_limit),
        "environment": _package_versions(row["input_kind"]),
        "python": platform.python_version(),
    }


@contextmanager
def dataset_lock(out: Path):
    """Prevent concurrent writers from mutating one dataset experiment directory."""
    import fcntl

    lock_path = out / ".dataset.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"dataset lock is already held: {lock_path}") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid(), "acquired_unix": time.time()}))
        handle.flush()
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _run_dir(out: Path, stage: str, config: V0_RGConfig, seed: int) -> Path:
    return out / stage / digest(asdict(config)) / f"seed_{int(seed)}"


def _required_run_artifacts(
    run_dir: Path, final: bool, *, include_completion: bool = True
) -> list[Path]:
    names = [
        "record.json",
        "completed.json",
        "config.json",
        "metrics.json",
        "training_history.json",
        "model_state.pt",
        "checkpoint_latest.pt",
        "cluster_centers.npy",
        "embedding.npy",
        "predictions.npy",
        "labels_true.npy",
        "row_indices.npy",
        "node_gate.npy",
        "neighbor_indices.npy",
        "neighbor_base_probs.npy",
        "edge_weights.npy",
        "edge_reliability.npy",
        "artifact_manifest.json",
    ]
    if final:
        names.append("verification.json")
    return [
        run_dir / name for name in names
        if include_completion or name != "completed.json"
    ]


def _artifact_manifest(run_dir: Path) -> dict[str, Any]:
    artifacts = []
    for path in sorted(run_dir.iterdir()):
        if path.is_file() and path.name not in {"artifact_manifest.json", "completed.json"}:
            artifacts.append(
                {
                    "path": path.name,
                    "size": int(path.stat().st_size),
                    "sha256": content_digest(path),
                }
            )
    return {"generated_artifacts": artifacts, "generated_artifact_count": len(artifacts)}


def _completed_record(
    run_dir: Path,
    *,
    identity_hash: str,
    preprocessing_hash: str,
    final: bool,
) -> dict[str, Any] | None:
    record_path = run_dir / "record.json"
    marker_path = run_dir / "completed.json"
    if not record_path.exists() or not marker_path.exists():
        return None
    record = json.loads(record_path.read_text())
    marker = json.loads(marker_path.read_text())
    if (
        record.get("status") != "completed"
        or record.get("identity") != identity_hash
        or record.get("preprocessing_hash") != preprocessing_hash
        or marker.get("identity") != identity_hash
        or marker.get("preprocessing_hash") != preprocessing_hash
    ):
        raise ValueError(f"cached run identity mismatch: {run_dir}")
    missing = [str(path) for path in _required_run_artifacts(run_dir, final) if not path.exists()]
    if missing:
        raise ValueError(f"completed marker has missing artifacts: {missing}")
    return record


def _next_attempt(run_dir: Path, max_attempts: int) -> tuple[int, Path]:
    attempts = run_dir / "attempts"
    existing = sorted(path for path in attempts.glob("attempt_*") if path.is_dir()) if attempts.exists() else []
    attempt = len(existing) + 1
    if attempt > int(max_attempts):
        raise RuntimeError(f"run exhausted {max_attempts} attempts: {run_dir}")
    directory = attempts / f"attempt_{attempt}"
    directory.mkdir(parents=True, exist_ok=False)
    return attempt, directory


def _write_run_artifacts(
    run_dir: Path,
    *,
    config: V0_RGConfig,
    row_indices: np.ndarray,
    encoded_labels: np.ndarray,
    predictions: np.ndarray,
    embedding: np.ndarray,
    diagnostics: dict[str, Any],
    state: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    write_json(run_dir / "config.json", asdict(config))
    write_numpy(run_dir / "row_indices.npy", np.asarray(row_indices, dtype=np.int64))
    write_numpy(run_dir / "labels_true.npy", np.asarray(encoded_labels, dtype=np.int64))
    write_numpy(run_dir / "predictions.npy", np.asarray(predictions, dtype=np.int64))
    write_numpy(run_dir / "embedding.npy", np.asarray(embedding, dtype=np.float32))
    write_numpy(run_dir / "cluster_centers.npy", np.asarray(diagnostics["cluster_centers"], dtype=np.float32))
    for name in (
        "node_gate",
        "neighbor_indices",
        "neighbor_base_probs",
        "edge_weights",
        "edge_reliability",
        "pseudo_perturbation",
    ):
        write_numpy(run_dir / f"{name}.npy", np.asarray(diagnostics[name]))
    write_json(run_dir / "training_history.json", diagnostics["training_history"])
    write_json(
        run_dir / "diagnostics.json",
        {
            "core_summary": diagnostics["core_summary"],
            "gate_summary": diagnostics["gate_summary"],
            "edge_summary": diagnostics["edge_summary"],
            "graph_profile": diagnostics["graph_profile"],
        },
    )
    write_json(run_dir / "metrics.json", metrics)
    write_torch(run_dir / "model_state.pt", state)


def _verify_saved_final(
    run_dir: Path,
    *,
    score_X: np.ndarray,
    config: V0_RGConfig,
    encoded_labels: np.ndarray,
    predictions: np.ndarray,
    recorded_metrics: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    from .run import clustering_metrics
    from .trainer import predict_from_saved_state
    import torch

    state = torch.load(run_dir / "model_state.pt", map_location="cpu", weights_only=False)
    centers = np.load(run_dir / "cluster_centers.npy", allow_pickle=False)
    reproduced_predictions, _ = predict_from_saved_state(
        score_X,
        config=config,
        device=device,
        model_state_dict=state["model_state_dict"],
        cluster_centers=centers,
    )
    if not np.array_equal(predictions, reproduced_predictions):
        raise ValueError("saved final model predictions differ from the recorded predictions")
    reproduced_metrics = clustering_metrics(encoded_labels, reproduced_predictions)
    for key in ("ari", "nmi", "acc"):
        if float(reproduced_metrics[key]) != float(recorded_metrics[key]):
            raise ValueError(f"saved final model {key} differs from recorded metric")
    return {
        "status": "passed",
        "prediction_equal": True,
        "metrics_exact": True,
        "metrics": {key: reproduced_metrics[key] for key in ("ari", "nmi", "acc")},
    }


def run_dataset(row: dict[str, Any], out: Path, args: argparse.Namespace) -> dict[str, Any] | None:
    """Run one dataset under held-out tuning with persistent, verifiable artifacts."""
    from .input_adapter import load_matrix
    from .run import clustering_metrics
    from .trainer import fit_predict

    out.mkdir(parents=True, exist_ok=True)
    with dataset_lock(out):
        loaded = load_matrix(
            row["path"],
            row.get("labels_path"),
            matrix_key=row.get("matrix_key", "X"),
            label_key=row.get("label_key"),
            label_column=row.get("label_column"),
            input_adapter=row.get("input_adapter"),
        )
        X, raw_labels = loaded.X, loaded.labels
        if raw_labels is None or np.asarray(raw_labels).ndim != 1 or len(raw_labels) != X.shape[0]:
            raise ValueError("benchmark requires one verified external label per matrix row")
        label_classes, encoded_y = np.unique(np.asarray(raw_labels, dtype=str), return_inverse=True)
        encoded_y = encoded_y.astype(np.int64)
        if int(row["n_clusters"]) != int(label_classes.size):
            raise ValueError(
                f"registered K={row['n_clusters']} disagrees with the verified label count={label_classes.size}"
            )
        train, validation, test = split_rows(len(encoded_y), args.split_seed)
        if int(row["n_clusters"]) > len(train):
            raise ValueError("registered K exceeds the training sample count")

        identity = build_experiment_identity(
            row=row,
            loaded_profile=loaded.profile,
            train=train,
            validation=validation,
            test=test,
            args=args,
        )
        identity_hash = digest(identity)
        experiment_path = out / "experiment.json"
        if experiment_path.exists() and json.loads(experiment_path.read_text()) != identity:
            raise ValueError("experiment identity changed; create a new output directory")
        write_json(experiment_path, identity)
        write_json(out / "dataset_profile.json", loaded.profile)
        write_json(
            out / "labels_mapping.json",
            {
                "classes": label_classes.astype(str).tolist(),
                "n_classes": int(label_classes.size),
                "source": row.get("label_key") or row.get("label_column") or "embedded_or_labels_path",
            },
        )
        write_json(
            out / "split.json",
            {
                "split_seed": int(args.split_seed),
                "train": train.tolist(),
                "validation": validation.tolist(),
                "test": test.tolist(),
                "split_hash": identity["split_hash"],
            },
        )
        if "source_cell_ids" in loaded.profile:
            write_json(
                out / "source_row_mapping.json",
                {
                    "source_cell_ids": loaded.profile["source_cell_ids"],
                    "source_cell_column_indices": loaded.profile["source_cell_column_indices"],
                    "mapping_fingerprint": digest(
                        {
                            "ids": loaded.profile["source_cell_ids"],
                            "columns": loaded.profile["source_cell_column_indices"],
                        }
                    ),
                },
            )
        write_json(
            out / "runtime.json",
            {
                "last_runtime_device": args.runtime_device,
                "last_physical_gpu": int(args.gpu),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "pid": os.getpid(),
                "recorded_unix": time.time(),
            },
        )

        final = args.stage == "final"
        selected_path = out / "selected.json"
        if final:
            if not selected_path.exists():
                raise ValueError("cannot evaluate final test before selected.json is frozen")
            selected = json.loads(selected_path.read_text())
            selection_hash = selected.pop("selection_hash", None)
            if selection_hash is None or digest(selected) != selection_hash or selected["identity"] != identity_hash:
                raise ValueError("frozen selection integrity or identity mismatch")
            candidates = [V0_RGConfig(**selected["winner"]["config"])]
            fit_rows, score_rows = np.concatenate([train, validation]), test
            marker = out / "final_started.json"
            marker_payload = {"selection_hash": selection_hash, "identity": identity_hash}
            if marker.exists() and json.loads(marker.read_text()) != marker_payload:
                raise ValueError("test evaluation already started with a different selection")
            write_json(marker, marker_payload)
            preprocessing_file = out / "final_preprocessing.json"
            fit_scope = "train_validation"
        else:
            if (out / "final_started.json").exists():
                raise ValueError("test evaluation has started; this experiment cannot be retuned")
            fit_rows, score_rows = train, validation
            preprocessing_file = out / "selection_preprocessing.json"
            fit_scope = "train"

        preprocessor = SplitPreprocessor(row["input_kind"], args.feature_limit).fit(X[fit_rows])
        fit_X = preprocessor.transform(X[fit_rows])
        score_X = preprocessor.transform(X[score_rows])
        write_json(preprocessing_file, preprocessor.payload(fit_scope=fit_scope))

        def evaluate(config: V0_RGConfig, seed: int) -> dict[str, Any]:
            stage = "final" if final else "validation"
            run_dir = _run_dir(out, stage, config, seed)
            cached = _completed_record(
                run_dir,
                identity_hash=identity_hash,
                preprocessing_hash=preprocessor.fingerprint,
                final=final,
            )
            if cached is not None:
                return cached
            run_dir.mkdir(parents=True, exist_ok=True)
            attempt_number, attempt_dir = _next_attempt(run_dir, args.max_attempts)
            record = {
                "identity": identity_hash,
                "config": asdict(config),
                "config_hash": digest(asdict(config)),
                "seed": int(seed),
                "stage": stage,
                "status": "running",
                "attempt": int(attempt_number),
                "preprocessing_hash": preprocessor.fingerprint,
                "labels_used_during_fit": False,
                "labels_used_for_selection": not final,
                "selection_labels": "validation_only",
                "known_k": int(row["n_clusters"]),
                "K_source": "verified_manifest",
                "test_used_for_selection": False,
                "runtime": {"device": args.runtime_device, "physical_gpu": int(args.gpu)},
            }
            write_json(run_dir / "record.json", record)
            write_json(attempt_dir / "record.json", record)
            start = time.monotonic()
            try:
                def checkpoint_callback(payload: dict[str, Any]) -> None:
                    payload = dict(payload)
                    payload.update(
                        {
                            "identity": identity_hash,
                            "config": asdict(config),
                            "seed": int(seed),
                            "preprocessing_hash": preprocessor.fingerprint,
                        }
                    )
                    write_torch(run_dir / "checkpoint_latest.pt", payload)

                predictions, embedding, diagnostics = fit_predict(
                    score_X,
                    fit_X=fit_X,
                    n_clusters=int(row["n_clusters"]),
                    config=config,
                    seed=seed,
                    device=args.runtime_device,
                    checkpoint_callback=checkpoint_callback,
                )
                if predictions is None or not np.isfinite(embedding).all():
                    raise ValueError("training did not produce finite predictions and embeddings")
                metrics = clustering_metrics(encoded_y[score_rows], predictions)
                state = {
                    "identity": identity_hash,
                    "config": asdict(config),
                    "seed": int(seed),
                    "model_state_dict": diagnostics["model_state_dict"],
                    "optimizer_state_dict": diagnostics["optimizer_state_dict"],
                    "rng_state": diagnostics["rng_state"],
                    "cluster_centers": diagnostics["cluster_centers"],
                }
                _write_run_artifacts(
                    run_dir,
                    config=config,
                    row_indices=score_rows,
                    encoded_labels=encoded_y[score_rows],
                    predictions=predictions,
                    embedding=embedding,
                    diagnostics=diagnostics,
                    state=state,
                    metrics=metrics,
                )
                if final:
                    verification = _verify_saved_final(
                        run_dir,
                        score_X=score_X,
                        config=config,
                        encoded_labels=encoded_y[score_rows],
                        predictions=predictions,
                        recorded_metrics=metrics,
                        device=args.runtime_device,
                    )
                    write_json(run_dir / "verification.json", verification)
                    record["test_metrics"] = metrics
                    record["verification"] = verification
                else:
                    record["validation_ari"] = float(metrics["ari"])
                    record["validation_metrics"] = metrics
                record.update(
                    status="completed",
                    wall_seconds=time.monotonic() - start,
                    core_summary=diagnostics["core_summary"],
                    gate_summary=diagnostics["gate_summary"],
                    edge_summary=diagnostics["edge_summary"],
                )
                write_json(run_dir / "record.json", record)
                write_json(attempt_dir / "record.json", record)
                write_json(run_dir / "artifact_manifest.json", _artifact_manifest(run_dir))
                missing = [str(path) for path in _required_run_artifacts(run_dir, final, include_completion=False) if not path.exists()]
                if missing:
                    raise RuntimeError(f"required run artifacts were not persisted: {missing}")
                write_json(
                    run_dir / "completed.json",
                    {
                        "identity": identity_hash,
                        "preprocessing_hash": preprocessor.fingerprint,
                        "config_hash": record["config_hash"],
                        "seed": int(seed),
                        "stage": stage,
                        "completed_unix": time.time(),
                    },
                )
                return record
            except Exception as exc:
                record.update(
                    status="failed",
                    wall_seconds=time.monotonic() - start,
                    error_type=type(exc).__name__,
                    error=str(exc),
                    traceback=traceback.format_exc(),
                )
                write_json(run_dir / "record.json", record)
                write_json(attempt_dir / "record.json", record)
                raise

        if final:
            records = [evaluate(candidates[0], seed) for seed in FINAL_SEEDS]
            metric_keys = ("ari", "nmi", "acc")
            summary = {
                "identity": identity_hash,
                "status": "completed",
                "n_seeds": len(records),
                "seeds": list(FINAL_SEEDS),
                "selection_hash": selection_hash,
                "test_n": int(len(test)),
                "metrics": {
                    key: {
                        "mean": float(np.mean([record["test_metrics"][key] for record in records])),
                        "std": float(np.std([record["test_metrics"][key] for record in records], ddof=1)),
                    }
                    for key in metric_keys
                },
                "per_seed": [
                    {"seed": int(record["seed"]), **{key: record["test_metrics"][key] for key in metric_keys}}
                    for record in records
                ],
            }
            write_json(out / "final_summary.json", summary)
            write_json(
                out / "completed.json",
                {
                    "identity": identity_hash,
                    "selection_hash": selection_hash,
                    "final_seed_count": len(records),
                    "verified_final_seed_count": len(records),
                    "completed_unix": time.time(),
                },
            )
            return summary

        if selected_path.exists():
            raise ValueError("selection already frozen; run final or inspect selected.json")
        import optuna

        base = V0_RGConfig(
            protocol_id=PROTOCOL,
            epochs=args.epochs,
            n_top_features=args.feature_limit,
            beta_uncertainty=0.0,
        )
        anchors = initial_candidates(base)
        for overrides in row.get("warm_start_configs", []):
            locked = {
                "protocol_id", "epochs", "hidden_size", "batch_size", "dropout",
                "masked_data_weight", "mask_loss_weight", "n_top_features", "target_sum",
                "kmeans_n_init", "num_workers", "variant",
            }
            if set(overrides) & locked:
                raise ValueError("warm start may contain only compatible search parameters")
            anchors.append(replace(base, **overrides))
        if len(anchors) > args.trials:
            raise ValueError("anchors and warm starts exceed the candidate budget")
        screened = []
        for config in anchors:
            try:
                screened.append(evaluate(config, SELECTION_SEEDS[0]))
            except Exception:
                continue
        sampler = optuna.samplers.TPESampler(seed=args.search_seed, n_startup_trials=8)
        study = optuna.create_study(
            storage=f"sqlite:///{(out / 'search.db').resolve()}",
            study_name=identity_hash,
            direction="maximize",
            sampler=sampler,
            load_if_exists=True,
        )

        def objective(trial: Any) -> float:
            config = suggest_config(trial, base)
            trial.set_user_attr("config", asdict(config))
            try:
                return float(evaluate(config, SELECTION_SEEDS[0])["validation_ari"])
            except Exception as exc:
                trial.set_user_attr("failure", f"{type(exc).__name__}: {exc}")
                return -1.0

        remaining = max(0, int(args.trials) - len(anchors) - len(study.trials))
        study.optimize(objective, n_trials=remaining)
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE and "config" in trial.user_attrs:
                try:
                    screened.append(evaluate(V0_RGConfig(**trial.user_attrs["config"]), SELECTION_SEEDS[0]))
                except Exception:
                    continue
        unique = {digest(record["config"]): record for record in screened}
        shortlist = rank_candidates(list(unique.values()), (SELECTION_SEEDS[0],))[:4]
        refined = []
        for ranked in shortlist:
            config = V0_RGConfig(**ranked["config"])
            for seed in SELECTION_SEEDS:
                try:
                    refined.append(evaluate(config, seed))
                except Exception:
                    continue
        ranking = rank_candidates(refined, SELECTION_SEEDS)
        if not ranking:
            raise ValueError("no candidate completed all three validation seeds")
        selected = {
            "identity": identity_hash,
            "selection_metric": "validation_ARI_mean",
            "selection_labels": "validation_only",
            "labels_used_for_selection": True,
            "test_used_for_selection": False,
            "winner": ranking[0],
            "shortlist": ranking,
            "candidate_budget": int(args.trials),
            "screened_anchor_proposals": len(anchors),
            "tpe_proposals": len(study.trials),
        }
        selected["selection_hash"] = digest(selected)
        write_json(selected_path, selected)
        write_json(out / "validation_ranking.json", {"identity": identity_hash, "ranking": ranking})
        return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-id", help="omit to process all manifest rows sequentially")
    parser.add_argument("--stage", choices=["search", "final"], required=True)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--feature-limit", type=int, default=2000)
    parser.add_argument("--split-seed", type=int, default=9102026)
    parser.add_argument("--search-seed", type=int, default=9102026)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()
    if args.trials < 4:
        parser.error("--trials must include at least the four anchors")
    rows = load_manifest(args.manifest)
    if args.dataset_id:
        rows = [r for r in rows if r["dataset_id"] == args.dataset_id]
        if not rows:
            parser.error("dataset-id is absent from manifest")
    if args.stage == "search":
        import optuna  # Fail before training if the optional search dependency is absent.
    from .run import resolve_runtime_device
    if args.max_attempts < 1:
        parser.error("--max-attempts must be positive")
    args.runtime_device = resolve_runtime_device(args.device, args.gpu)
    failures = []
    for row in rows:
        try:
            run_dataset(row, Path(args.output_dir) / row["dataset_id"], args)
        except Exception as exc:
            failures.append({"dataset_id": row["dataset_id"], "error": f"{type(exc).__name__}: {exc}"})
            print(json.dumps(failures[-1]), flush=True)
    if failures:
        for failure in failures:
            write_json(
                Path(args.output_dir) / failure["dataset_id"] / "failures" / f"{args.stage}.json",
                failure,
            )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
