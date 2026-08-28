#!/usr/bin/env python3
"""Run the registered V15 corruption conditions on one dataset.

This is an exploratory Stage-2/Stage-3 launcher. It never uses labels to fit or
choose a condition. Labels, when present in the NPZ, are passed only to the
post-fit metric writer and to the benchmark K protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import scipy.sparse as sp
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.TopoGate.V15_counterfactual_gate.config import load_config
from methods.TopoGate.V15_counterfactual_gate.run import fit_v15, load_npz


DEFAULT_MODES = ("clean", "feature_mask", "heavy_tail_noise", "random_graph_replacement", "row_contamination", "compound")


def _array_hash(X: np.ndarray | sp.spmatrix) -> str:
    digest = hashlib.sha256()
    if sp.issparse(X):
        matrix = sp.csr_matrix(X, dtype=np.float32)
        for value in (matrix.data, matrix.indices, matrix.indptr, np.asarray(matrix.shape, dtype=np.int64)):
            digest.update(np.asarray(value).tobytes())
    else:
        digest.update(np.asarray(X, dtype=np.float32).tobytes())
    return digest.hexdigest()


def _feature_mask(X: np.ndarray | sp.spmatrix, rng: np.random.Generator, fraction: float) -> tuple[Any, np.ndarray]:
    if sp.issparse(X):
        output = sp.csr_matrix(X, dtype=np.float32, copy=True)
        selected = rng.random(output.data.size) < fraction
        output.data[selected] = 0.0
        output.eliminate_zeros()
        return output, selected.astype(np.uint8)
    output = np.asarray(X, dtype=np.float32).copy()
    observed = output != 0.0
    selected = observed & (rng.random(output.shape) < fraction)
    output[selected] = 0.0
    return output, selected.astype(np.uint8)


def _heavy_tail_noise(X: np.ndarray | sp.spmatrix, rng: np.random.Generator, scale: float) -> tuple[Any, np.ndarray]:
    if sp.issparse(X):
        output = sp.csr_matrix(X, dtype=np.float32, copy=True)
        if output.data.size:
            feature_scale = np.asarray(output.power(2).mean(axis=0)).ravel() ** 0.5
            row_ids = np.repeat(np.arange(output.shape[0]), np.diff(output.indptr))
            col_ids = output.indices
            local_scale = feature_scale[col_ids]
            output.data += (scale * local_scale * rng.standard_t(df=2.0, size=output.data.size)).astype(np.float32)
        return output, np.ones(output.shape[0], dtype=np.uint8)
    output = np.asarray(X, dtype=np.float32).copy()
    feature_scale = np.std(output, axis=0).astype(np.float32) + 1e-6
    noise = rng.standard_t(df=2.0, size=output.shape).astype(np.float32) * (scale * feature_scale[None, :])
    output += noise
    return output, np.ones(output.shape[0], dtype=np.uint8)


def _row_contamination(X: np.ndarray | sp.spmatrix, rng: np.random.Generator, fraction: float) -> tuple[Any, np.ndarray]:
    n = int(X.shape[0])
    count = min(n, max(1, int(round(n * fraction)))) if n else 0
    rows = rng.choice(n, size=count, replace=False) if count else np.empty(0, dtype=np.int64)
    donor = rng.choice(n, size=count, replace=True) if count else np.empty(0, dtype=np.int64)
    if sp.issparse(X):
        output = sp.csr_matrix(X, dtype=np.float32, copy=True).tolil()
        for target, source in zip(rows.tolist(), donor.tolist()):
            output[target] = X.getrow(int(source))
        return output.tocsr(), np.isin(np.arange(n), rows).astype(np.uint8)
    output = np.asarray(X, dtype=np.float32).copy()
    if count:
        output[rows] = output[donor]
    return output, np.isin(np.arange(n), rows).astype(np.uint8)


def corrupt(
    X: np.ndarray | sp.spmatrix,
    mode: str,
    seed: int,
    *,
    feature_fraction: float,
    row_fraction: float,
    noise_scale: float,
) -> tuple[np.ndarray | sp.spmatrix, dict[str, Any], np.ndarray]:
    rng = np.random.default_rng(seed)
    if mode == "clean" or mode == "random_graph_replacement":
        return X, {"mode": mode, "changed": False, "mask_kind": "row"}, np.zeros(int(X.shape[0]), dtype=np.uint8)
    if mode == "feature_mask":
        output, selected = _feature_mask(X, rng, feature_fraction)
        return output, {"mode": mode, "changed": True, "feature_fraction": feature_fraction, "selected_count": int(np.sum(selected)), "mask_kind": "feature"}, selected
    if mode == "heavy_tail_noise":
        output, selected = _heavy_tail_noise(X, rng, noise_scale)
        return output, {"mode": mode, "changed": True, "noise_scale": noise_scale, "affected_rows": int(np.sum(selected)), "mask_kind": "row"}, selected
    if mode == "row_contamination":
        output, selected = _row_contamination(X, rng, row_fraction)
        return output, {"mode": mode, "changed": True, "row_fraction": row_fraction, "contaminated_rows": int(np.sum(selected)), "mask_kind": "row"}, selected
    if mode == "compound":
        output, first = _feature_mask(X, rng, feature_fraction)
        output, second = _heavy_tail_noise(output, rng, noise_scale)
        output, third = _row_contamination(output, rng, row_fraction)
        return output, {
            "mode": mode,
            "changed": True,
            "feature_fraction": feature_fraction,
            "noise_scale": noise_scale,
            "row_fraction": row_fraction,
            "selected_feature_count": int(np.sum(first)),
            "contaminated_rows": int(np.sum(third)),
            "affected_rows_noise": int(np.sum(second)),
            "mask_kind": "row",
        }, third
    raise ValueError(f"unknown corruption mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "result" / "V15" / "corruption_diagnostics")
    parser.add_argument("--config", type=Path, default=ROOT / "methods" / "TopoGate" / "V15_counterfactual_gate" / "configs" / "topogate_v15.yaml")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--n-clusters", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    parser.add_argument("--feature-fraction", type=float, default=0.2)
    parser.add_argument("--row-fraction", type=float, default=0.1)
    parser.add_argument("--noise-scale", type=float, default=0.2)
    parser.add_argument("--graph-replacement-fraction", type=float, default=1.0)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    args = parser.parse_args()
    X, y = load_npz(args.data_path)
    n_clusters = int(np.unique(y).size) if args.n_clusters is None and y is not None else args.n_clusters
    if n_clusters is None:
        raise ValueError("--n-clusters is required when the NPZ has no labels")
    k_protocol = "benchmark_oracle_from_y" if args.n_clusters is None and y is not None else "explicit"
    overrides: dict[str, Any] = {}
    for value in args.overrides:
        if "=" not in value:
            raise ValueError(f"override must be key=value: {value}")
        key, raw = value.split("=", 1)
        overrides[key] = yaml.safe_load(raw)
    dataset_name = args.dataset_name or args.data_path.stem
    for mode in [value.strip() for value in args.modes.split(",") if value.strip()]:
        corrupted, metadata, corruption_mask = corrupt(
            X,
            mode,
            args.seed,
            feature_fraction=args.feature_fraction,
            row_fraction=args.row_fraction,
            noise_scale=args.noise_scale,
        )
        mode_overrides = dict(overrides)
        mode_overrides["seed"] = args.seed
        mode_overrides["graph_replacement_fraction"] = args.graph_replacement_fraction if mode == "random_graph_replacement" else float(mode_overrides.get("graph_replacement_fraction", 0.0))
        config = load_config(args.config, mode_overrides)
        suffix = f"__graph{args.graph_replacement_fraction:g}" if mode == "random_graph_replacement" else ""
        save_dir = args.output_root / f"{dataset_name}__{mode}{suffix}__seed{args.seed}"
        metadata.update(
            {
                "original_source": str(args.data_path.resolve()),
                "original_sha256": hashlib.sha256(args.data_path.read_bytes()).hexdigest(),
                "corrupted_input_sha256": _array_hash(corrupted),
                "labels_used_during_fit": False,
                "graph_replacement_fraction": args.graph_replacement_fraction if mode == "random_graph_replacement" else 0.0,
            }
        )
        fit_v15(
            corrupted,
            int(n_clusters),
            y,
            config=config,
            save_dir=save_dir,
            dataset_name=dataset_name,
            source_path=args.data_path,
            k_protocol=k_protocol,
            run_metadata={"corruption": metadata},
        )
        (save_dir / "corruption.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        np.save(save_dir / "corruption_mask.npy", corruption_mask)
        print(json.dumps({"mode": mode, "output": str(save_dir)}))


if __name__ == "__main__":
    main()
