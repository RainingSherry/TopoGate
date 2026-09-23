"""Per-dataset validation-ARI tuning; train-only preprocessing and readout.

Run search first, inspect selected.json, then run final with the frozen selection.
Imports of Optuna/Torch are delayed so manifests and selection can be audited on CPU.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler

from .config import V0_RGConfig

PROTOCOL = "topogate_unified_perdataset_inductive_v1"
SELECTION_SEEDS = (42, 123, 7)
FINAL_SEEDS = (42, 123, 7, 2025, 3407)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def code_digest():
    root = Path(__file__).parent
    paths = sorted(root.glob("*.py"))
    paths += sorted((root.parents[1] / "NeighborMix_scMAE").glob("*.py"))
    return digest({str(p.relative_to(root.parents[2])): file_digest(p) for p in paths})


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))
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
                       "validation_ari_std": float(np.std(scores)), "seeds": list(seeds)})
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


def run_dataset(row, out, args):
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
    identity = {"protocol": PROTOCOL, "dataset": row, "data_hash": file_digest(row["path"]),
        "labels_hash": file_digest(row["labels_path"]) if row.get("labels_path") else "embedded",
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
            if final:
                np.save(path.parent / f"embedding_seed_{seed}.npy", embedding)
                np.save(path.parent / f"predictions_seed_{seed}.npy", pred)
                import torch
                torch.save(diagnostics["model_state_dict"], path.parent / f"model_seed_{seed}.pt")
                np.save(path.parent / f"cluster_centers_seed_{seed}.npy", diagnostics["cluster_centers"])
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
    args.device = resolve_runtime_device(args.device, args.gpu)
    failures = []
    for row in rows:
        try:
            run_dataset(row, Path(args.output_dir) / row["dataset_id"], args)
        except Exception as exc:
            failures.append({"dataset_id": row["dataset_id"], "error": f"{type(exc).__name__}: {exc}"})
            print(json.dumps(failures[-1]), flush=True)
    if failures:
        write_json(Path(args.output_dir) / f"failures_{args.stage}.json", failures)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
