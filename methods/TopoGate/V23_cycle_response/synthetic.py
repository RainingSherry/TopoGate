from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import file_sha256


def _sparsify_rows(values: np.ndarray, keep_fraction: float) -> np.ndarray:
    keep = max(1, int(round(values.shape[1] * float(keep_fraction))))
    indices = np.argpartition(np.abs(values), -keep, axis=1)[:, -keep:]
    result = np.zeros_like(values, dtype=np.float32)
    rows = np.arange(values.shape[0])[:, None]
    result[rows, indices] = values[rows, indices]
    return result


def generate_worlds(
    *,
    n_samples: int = 3000,
    n_features: int = 1000,
    n_clusters: int = 6,
    latent_rank: int = 16,
    zero_fraction: float = 0.90,
    seed: int = 42,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if n_samples % n_clusters != 0:
        raise ValueError("n_samples must be divisible by n_clusters")
    if not 0.0 < zero_fraction < 1.0:
        raise ValueError("zero_fraction must be in (0, 1)")
    rng = np.random.default_rng(int(seed))
    per_cluster = n_samples // n_clusters
    labels = np.repeat(np.arange(n_clusters, dtype=np.int64), per_cluster)
    base_loading = rng.normal(0.0, 1.0 / np.sqrt(latent_rank), size=(n_features, latent_rank))

    dependency_rows: list[np.ndarray] = []
    mean_rows: list[np.ndarray] = []
    for cluster in range(n_clusters):
        latent = rng.normal(size=(per_cluster, latent_rank))
        loading_delta = np.zeros_like(base_loading)
        active_features = rng.choice(n_features, size=max(1, n_features // 5), replace=False)
        loading_delta[active_features] = rng.normal(
            0.0,
            0.75 / np.sqrt(latent_rank),
            size=(active_features.size, latent_rank),
        )
        dependency = latent @ (base_loading + loading_delta).T
        dependency += rng.normal(0.0, 0.10, size=dependency.shape)
        dependency_rows.append(dependency)

        shifted_latent = latent.copy()
        shifted_latent[:, cluster % latent_rank] += 2.5
        mean_only = shifted_latent @ base_loading.T
        mean_only += rng.normal(0.0, 0.10, size=mean_only.shape)
        mean_rows.append(mean_only)

    dependency = _sparsify_rows(np.vstack(dependency_rows), 1.0 - zero_fraction)
    mean_only = _sparsify_rows(np.vstack(mean_rows), 1.0 - zero_fraction)
    conditional_null = dependency.copy()
    for cluster in range(n_clusters):
        rows = np.flatnonzero(labels == cluster)
        for feature in range(n_features):
            conditional_null[rows, feature] = conditional_null[rng.permutation(rows), feature]
    global_null = dependency.copy()
    for feature in range(n_features):
        global_null[:, feature] = global_null[rng.permutation(n_samples), feature]
    return {
        "cluster_specific_dependency": (dependency, labels.copy()),
        "mean_only_shared_dependency": (mean_only, labels.copy()),
        "conditional_dependency_destroyed": (conditional_null, labels.copy()),
        "global_structure_destroyed_sanity": (global_null, labels.copy()),
    }


def write_panel(output_root: str | Path, *, seed: int, **kwargs: int | float) -> dict[str, object]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    worlds = generate_worlds(seed=seed, **kwargs)
    for name, (matrix, labels) in worlds.items():
        world_dir = root / name / f"seed{seed}"
        world_dir.mkdir(parents=True, exist_ok=True)
        matrix_path = world_dir / "matrix_only.npz"
        labels_path = world_dir / "labels_true.npy"
        np.savez_compressed(matrix_path, X=matrix.astype(np.float32))
        np.save(labels_path, labels.astype(np.int64))
        records.append(
            {
                "dataset_id": f"synthetic__{name}__seed{seed}",
                "world": name,
                "matrix_path": str(matrix_path.resolve()),
                "matrix_sha256": file_sha256(matrix_path),
                "labels_path": str(labels_path.resolve()),
                "labels_sha256": file_sha256(labels_path),
                "input_protocol": "clubench_bridge",
                "n_samples": int(matrix.shape[0]),
                "n_features": int(matrix.shape[1]),
                "zero_fraction": float(np.mean(matrix == 0.0)),
                "labels_used_during_generation": True,
                "labels_accessible_during_fit": False,
                "labels_accessible_during_profile": False,
            }
        )
    manifest = {
        "manifest_id": f"v23_synthetic_death_tests_seed{seed}",
        "generation_config": {
            "seed": int(seed),
            **{key: value for key, value in kwargs.items()},
        },
        "selection_uses_results": False,
        "records": records,
    }
    (root / f"manifest_seed{seed}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate V23 synthetic mechanism worlds")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-samples", type=int, default=3000)
    parser.add_argument("--n-features", type=int, default=1000)
    parser.add_argument("--n-clusters", type=int, default=6)
    parser.add_argument("--latent-rank", type=int, default=16)
    parser.add_argument("--zero-fraction", type=float, default=0.90)
    args = parser.parse_args()
    manifest = write_panel(
        args.output_root,
        seed=args.seed,
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_clusters=args.n_clusters,
        latent_rank=args.latent_rank,
        zero_fraction=args.zero_fraction,
    )
    print(json.dumps({"manifest_id": manifest["manifest_id"], "records": len(manifest["records"])}, indent=2))


if __name__ == "__main__":
    main()
