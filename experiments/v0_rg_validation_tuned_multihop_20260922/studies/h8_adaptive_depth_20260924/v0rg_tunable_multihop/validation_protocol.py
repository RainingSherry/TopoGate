"""Feature-only validation protocol for expanded V0_RG path experiments.

The split-preparation process is the only component allowed to read benchmark
labels.  It writes a feature-only matrix plus separate validation and final-test
sidecars.  The tuning runner never receives a test-label path.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.model_selection import StratifiedShuffleSplit

from .config import V0_RGConfig, load_config
from .input_adapter import encode_labels, load_matrix, load_npz_matrix_only
from .run import clustering_metrics
from .trainer import fit_predict
from .tuning import SplitPreprocessor


PROTOCOL_ID = "v0_rg_expanded_multihop_validation_only_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _write_npz(path: Path, **payload: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)


def _write_feature_only(path: Path, X: np.ndarray | sp.spmatrix) -> None:
    if sp.issparse(X):
        matrix = sp.csr_matrix(X)
        _write_npz(
            path,
            data=np.asarray(matrix.data),
            indices=np.asarray(matrix.indices, dtype=np.int64),
            indptr=np.asarray(matrix.indptr, dtype=np.int64),
            shape=np.asarray(matrix.shape, dtype=np.int64),
        )
    else:
        _write_npz(path, X=np.asarray(X))


def prepare_isolated_split(source_path: str | Path, output_dir: str | Path, *, split_seed: int = 9102026) -> dict:
    """Create immutable feature/label sidecars from a benchmark once."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    loaded = load_matrix(source_path)
    labels, _ = encode_labels(loaded.labels)
    if labels is None:
        raise ValueError("the trusted split preparer requires benchmark labels")
    classes, counts = np.unique(labels, return_counts=True)
    rows = np.arange(labels.size, dtype=np.int64)
    if np.all(counts >= 5):
        split_strategy = "stratified_shuffle_60_20_20"
        first = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=int(split_seed))
        train_idx, remainder_idx = next(first.split(rows, labels))
        second = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=int(split_seed) + 1)
        validation_local, test_local = next(second.split(remainder_idx, labels[remainder_idx]))
        validation_idx = remainder_idx[validation_local]
        test_idx = remainder_idx[test_local]
    else:
        split_strategy = "classwise_rare_60_20_20_best_effort"
        rng = np.random.default_rng(int(split_seed))
        train_parts, validation_parts, test_parts = [], [], []
        for class_value in classes:
            class_rows = rng.permutation(rows[labels == class_value])
            count = int(class_rows.size)
            if count >= 3:
                validation_count = max(1, int(round(0.2 * count)))
                test_count = max(1, int(round(0.2 * count)))
                while validation_count + test_count >= count:
                    if validation_count >= test_count and validation_count > 1:
                        validation_count -= 1
                    elif test_count > 1:
                        test_count -= 1
                    else:
                        break
                validation_parts.append(class_rows[:validation_count])
                test_parts.append(class_rows[validation_count:validation_count + test_count])
                train_parts.append(class_rows[validation_count + test_count:])
            elif count == 2:
                train_parts.append(class_rows[:1])
                validation_parts.append(class_rows[1:])
            else:
                train_parts.append(class_rows)
        train_idx = np.sort(np.concatenate(train_parts))
        validation_idx = np.sort(np.concatenate(validation_parts))
        test_idx = np.sort(np.concatenate(test_parts)) if test_parts else np.empty(0, dtype=np.int64)
        if validation_idx.size == 0 or test_idx.size == 0:
            raise ValueError("rare-class fallback could not form both validation and final-test sets")
    feature_path = output / "features_only.npz"
    validation_path = output / "validation_labels.npz"
    test_path = output / "test_labels.npz"
    _write_feature_only(feature_path, loaded.X)
    _write_npz(validation_path, indices=validation_idx, labels=labels[validation_idx])
    _write_npz(test_path, indices=test_idx, labels=labels[test_idx])
    common = {
        "protocol_id": PROTOCOL_ID,
        "split_seed": int(split_seed),
        "split_strategy": split_strategy,
        "minimum_class_count": int(counts.min()),
        "n_samples": int(labels.size),
        "n_clusters": int(classes.size),
        "K_provenance": "benchmark_oracle_from_y",
        "feature_path": str(feature_path.resolve()),
        "feature_sha256": _sha256(feature_path),
        "train_indices": train_idx.tolist(),
        "validation_indices_sha256": hashlib.sha256(validation_idx.tobytes()).hexdigest(),
        "validation_labels_sha256": _sha256(validation_path),
        "test_indices_sha256": hashlib.sha256(test_idx.tobytes()).hexdigest(),
        "test_labels_sha256": _sha256(test_path),
        "split_sizes": {"train": int(train_idx.size), "validation": int(validation_idx.size), "test": int(test_idx.size)},
    }
    tuning_manifest = dict(common)
    tuning_manifest.update({
        "validation_labels_path": str(validation_path.resolve()),
        "test_labels_loaded": False,
        "test_label_path_exposed": False,
    })
    final_manifest = dict(common)
    final_manifest.update({"test_labels_path": str(test_path.resolve()), "test_labels_loaded": True})
    _write_json(output / "tuning_manifest.json", tuning_manifest)
    _write_json(output / "final_scoring_manifest.json", final_manifest)
    return tuning_manifest


def run_validation_only(
    tuning_manifest_path: str | Path,
    output_dir: str | Path,
    *,
    config: V0_RGConfig,
    seed: int,
    input_kind: str = "general",
    feature_limit: int = 2000,
    device: str = "cpu",
) -> dict:
    """Fit transductively on feature-only X and score validation sidecar only."""
    manifest = json.loads(Path(tuning_manifest_path).read_text(encoding="utf-8"))
    forbidden = {"test_labels_path", "test_label_path"}.intersection(manifest)
    if forbidden or manifest.get("test_label_path_exposed", True):
        raise ValueError("tuning manifest exposes a final-test label path")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    loaded = load_npz_matrix_only(manifest["feature_path"])
    X = loaded.X
    prep = SplitPreprocessor(input_kind, int(feature_limit)).fit(X)
    X_all = prep.transform(X)
    predictions, embedding, diagnostics = fit_predict(
        X_all, fit_X=X_all, n_clusters=int(manifest["n_clusters"]), config=config, seed=int(seed), device=device
    )
    sidecar = np.load(manifest["validation_labels_path"], allow_pickle=False)
    indices = np.asarray(sidecar["indices"], dtype=np.int64)
    labels = np.asarray(sidecar["labels"], dtype=np.int64)
    if indices.size != labels.size or np.any(indices < 0) or np.any(indices >= X_all.shape[0]):
        raise ValueError("invalid validation-label sidecar")
    record = {
        "protocol_id": PROTOCOL_ID,
        "status": "completed",
        "seed": int(seed),
        "config": asdict(config),
        "n_clusters": int(manifest["n_clusters"]),
        "K_provenance": manifest["K_provenance"],
        "evaluation_mode": "transductive_feature_only",
        "graph_scope": "all_samples_unlabeled",
        "labels_used_during_fit": False,
        "labels_used_for_selection": True,
        "selection_label_scope": "validation_only",
        "test_labels_loaded": False,
        "preprocessing_hash": prep.fingerprint,
        "feature_sha256": manifest["feature_sha256"],
        "validation_labels_sha256": manifest["validation_labels_sha256"],
        "validation_metrics": clustering_metrics(labels, predictions[indices]),
    }
    _write_npz(output / f"predictions_seed_{seed}.npz", predictions=np.asarray(predictions, dtype=np.int64))
    _write_npz(output / f"embedding_seed_{seed}.npz", embedding=np.asarray(embedding, dtype=np.float32))
    torch.save(diagnostics["model_state_dict"], output / f"model_seed_{seed}.pt")
    _write_json(output / f"record_seed_{seed}.json", record)
    _write_json(output / f"hop_sampling_seed_{seed}.json", diagnostics["hop_sampling"])
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="V0_RG feature-only validation protocol")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--source", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--split-seed", type=int, default=9102026)
    validate = sub.add_parser("validate")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--output", required=True)
    validate.add_argument("--config", required=True)
    validate.add_argument("--seed", type=int, required=True)
    validate.add_argument("--input-kind", default="general")
    validate.add_argument("--feature-limit", type=int, default=2000)
    validate.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_isolated_split(args.source, args.output, split_seed=args.split_seed)
    else:
        run_validation_only(args.manifest, args.output, config=load_config(args.config), seed=args.seed,
                            input_kind=args.input_kind, feature_limit=args.feature_limit, device=args.device)


if __name__ == "__main__":
    main()
