from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import V24Q1Config, WORLD_NAMES


def _balanced_labels(n_samples: int, n_clusters: int) -> np.ndarray:
    return np.repeat(np.arange(n_clusters, dtype=np.int64), n_samples // n_clusters)


def _block_support_template(
    n_rows: int,
    n_blocks: int,
    active_blocks: int,
    rng: np.random.Generator,
) -> np.ndarray:
    blocks = np.zeros((n_rows, n_blocks), dtype=np.bool_)
    for row in range(n_rows):
        blocks[row, rng.choice(n_blocks, size=active_blocks, replace=False)] = True
    return blocks


def _expand_blocks(blocks: np.ndarray, block_size: int) -> np.ndarray:
    return np.repeat(blocks, block_size, axis=1)


def _rank_columns(values: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(values, axis=0, kind="mergesort"), axis=0, kind="mergesort")


def _signed_correlation(block_size: int, rho: float, cluster: int, block: int, seed: int) -> np.ndarray:
    if cluster == 0:
        signs = np.ones(block_size, dtype=np.float64)
    else:
        rng = np.random.default_rng(seed + 10_007 * cluster + 97 * block)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=block_size)
        signs[0] = 1.0
    return (1.0 - rho) * np.eye(block_size, dtype=np.float64) + rho * np.outer(signs, signs)


def _w4_dependency_only(config: V24Q1Config, rng: np.random.Generator, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact support and nonzero marginals, with class-specific block dependence."""

    per_cluster = config.n_samples // config.n_clusters
    template_blocks = _block_support_template(
        per_cluster,
        config.n_blocks,
        config.active_blocks_per_sample,
        rng,
    )
    template_support = _expand_blocks(template_blocks, config.block_size)
    value_templates: list[np.ndarray] = []
    for feature in range(config.n_features):
        count = int(template_support[:, feature].sum())
        values = rng.lognormal(mean=0.0, sigma=0.55, size=count).astype(np.float32)
        value_templates.append(np.sort(values))

    rows: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for cluster in range(config.n_clusters):
        permutation = rng.permutation(per_cluster)
        support_blocks = template_blocks[permutation]
        support = _expand_blocks(support_blocks, config.block_size)
        values = np.zeros((per_cluster, config.n_features), dtype=np.float32)
        for block in range(config.n_blocks):
            active_rows = np.flatnonzero(support_blocks[:, block])
            if not active_rows.size:
                continue
            correlation = _signed_correlation(
                config.block_size,
                config.dependency_rho,
                cluster,
                block,
                seed,
            )
            latent = rng.multivariate_normal(
                mean=np.zeros(config.block_size),
                cov=correlation,
                size=active_rows.size,
            )
            ranks = _rank_columns(latent)
            start = block * config.block_size
            for offset, feature in enumerate(range(start, start + config.block_size)):
                values[active_rows, feature] = value_templates[feature][ranks[:, offset]]
        if not np.array_equal(values > 0.0, support):
            raise RuntimeError("W4 support/value construction drifted")
        rows.append(values)
        labels.append(np.full(per_cluster, cluster, dtype=np.int64))
    return np.vstack(rows), np.concatenate(labels)


def _apply_mean_shift(values: np.ndarray, labels: np.ndarray, config: V24Q1Config) -> np.ndarray:
    result = values.copy()
    group_width = config.n_features // config.n_clusters
    for cluster in range(config.n_clusters):
        start = cluster * group_width
        end = config.n_features if cluster == config.n_clusters - 1 else (cluster + 1) * group_width
        rows = labels == cluster
        selected = result[rows, start:end] > 0.0
        result_block = result[rows, start:end]
        result_block[selected] += 1.25
        result[rows, start:end] = result_block
    return result


def _class_weighted_support(config: V24Q1Config, rng: np.random.Generator) -> np.ndarray:
    per_cluster = config.n_samples // config.n_clusters
    all_rows: list[np.ndarray] = []
    group_width = config.n_blocks // config.n_clusters
    for cluster in range(config.n_clusters):
        weights = np.ones(config.n_blocks, dtype=np.float64)
        start = cluster * group_width
        end = config.n_blocks if cluster == config.n_clusters - 1 else (cluster + 1) * group_width
        weights[start:end] = 7.0
        weights /= weights.sum()
        rows = np.zeros((per_cluster, config.n_blocks), dtype=np.bool_)
        for row in range(per_cluster):
            rows[row, rng.choice(config.n_blocks, size=config.active_blocks_per_sample, replace=False, p=weights)] = True
        all_rows.append(rows)
    return np.vstack(all_rows)


def _iid_amplitudes(support: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    values = np.zeros(support.shape, dtype=np.float32)
    values[support] = rng.lognormal(mean=0.0, sigma=0.55, size=int(support.sum())).astype(np.float32)
    return values


def _shared_support(config: V24Q1Config, rng: np.random.Generator) -> np.ndarray:
    """Return class-balanced support with exactly the same support multiset per class."""

    per_cluster = config.n_samples // config.n_clusters
    template = _block_support_template(
        per_cluster,
        config.n_blocks,
        config.active_blocks_per_sample,
        rng,
    )
    blocks = np.vstack([template[rng.permutation(per_cluster)] for _ in range(config.n_clusters)])
    return _expand_blocks(blocks, config.block_size)


def _iid_support(config: V24Q1Config, rng: np.random.Generator) -> np.ndarray:
    """Draw an independent exact-sparsity support pattern for every sample."""

    blocks = _block_support_template(
        config.n_samples,
        config.n_blocks,
        config.active_blocks_per_sample,
        rng,
    )
    return _expand_blocks(blocks, config.block_size)


def _mean_only_world(config: V24Q1Config, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Independent amplitudes with class-specific first moments only."""

    labels = _balanced_labels(config.n_samples, config.n_clusters)
    return _apply_mean_shift(_iid_amplitudes(_shared_support(config, rng), rng), labels, config), labels


def _marginal_only_world(config: V24Q1Config, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Independent amplitudes with class-specific marginal dispersion, not dependence."""

    labels = _balanced_labels(config.n_samples, config.n_clusters)
    support = _shared_support(config, rng)
    values = np.zeros(support.shape, dtype=np.float32)
    base_sigma = 0.55
    # Center each log-normal law at one. Changing sigma therefore changes only
    # the nonzero marginal shape, while conditional feature draws stay iid.
    values[support] = rng.lognormal(
        mean=-0.5 * base_sigma * base_sigma,
        sigma=base_sigma,
        size=int(support.sum()),
    ).astype(np.float32)
    group_width = config.n_features // config.n_clusters
    for cluster in range(config.n_clusters):
        start = cluster * group_width
        end = config.n_features if cluster == config.n_clusters - 1 else (cluster + 1) * group_width
        rows = np.flatnonzero(labels == cluster)
        local_support = support[np.ix_(rows, np.arange(start, end))]
        sigma = 1.05
        replacement = rng.lognormal(
            mean=-0.5 * sigma * sigma,
            sigma=sigma,
            size=int(local_support.sum()),
        ).astype(np.float32)
        block = values[np.ix_(rows, np.arange(start, end))]
        block[local_support] = replacement
        values[np.ix_(rows, np.arange(start, end))] = block
    if not np.array_equal(values > 0.0, support):
        raise RuntimeError("marginal-only support/value construction drifted")
    return values, labels


def _w5_mixed(config: V24Q1Config, rng: np.random.Generator, seed: int) -> tuple[np.ndarray, np.ndarray]:
    labels = _balanced_labels(config.n_samples, config.n_clusters)
    support_blocks = _class_weighted_support(config, rng)
    support = _expand_blocks(support_blocks, config.block_size)
    values = np.zeros((config.n_samples, config.n_features), dtype=np.float32)
    for cluster in range(config.n_clusters):
        cluster_rows = np.flatnonzero(labels == cluster)
        for block in range(config.n_blocks):
            local = cluster_rows[support_blocks[cluster_rows, block]]
            if not local.size:
                continue
            corr = _signed_correlation(config.block_size, config.dependency_rho, cluster, block, seed)
            latent = rng.multivariate_normal(np.zeros(config.block_size), corr, size=local.size)
            start = block * config.block_size
            values[local, start : start + config.block_size] = np.exp(0.55 * latent).astype(np.float32)
    if not np.array_equal(values > 0.0, support):
        raise RuntimeError("W5 support/value construction drifted")
    return _apply_mean_shift(values, labels, config), labels


def _shuffle_rows(matrix: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    order = rng.permutation(labels.size)
    return matrix[order].astype(np.float32, copy=False), labels[order].astype(np.int64, copy=False)


def generate_worlds(config: V24Q1Config, *, seed: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Generate V24 worlds without exposing labels to the fit/profile stages."""

    config.validate()
    rng = np.random.default_rng(int(seed))
    dependency, dependency_labels = _w4_dependency_only(config, rng, int(seed))

    global_null = _iid_amplitudes(_iid_support(config, rng), rng)
    # W0 has no label-conditioned generator branch. Fixed balanced evaluation
    # partitions make that fact explicit rather than looking like a shuffled-
    # label stress test.
    null_labels = _balanced_labels(config.n_samples, config.n_clusters)

    mean_only, mean_labels = _mean_only_world(config, rng)

    support_labels = _balanced_labels(config.n_samples, config.n_clusters)
    support_blocks = _class_weighted_support(config, rng)
    support_only = _iid_amplitudes(_expand_blocks(support_blocks, config.block_size), rng)

    marginal_only, marginal_labels = _marginal_only_world(config, rng)

    mixed, mixed_labels = _w5_mixed(config, rng, int(seed) + 67)
    generated = {
        "W0_global_null": (global_null, null_labels),
        "W1_mean_only": (mean_only, mean_labels),
        "W2_support_only": (support_only, support_labels),
        "W3_marginal_only": (marginal_only, marginal_labels),
        "W4_dependency_only": (dependency, dependency_labels),
        "W5_mixed_realistic": (mixed, mixed_labels),
    }
    return {name: _shuffle_rows(matrix, labels, rng) for name, (matrix, labels) in generated.items()}


def write_panel(output_root: str | Path, config: V24Q1Config, *, seed: int) -> dict[str, object]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for world, (matrix, labels) in generate_worlds(config, seed=seed).items():
        world_root = root / world / f"seed{seed}"
        world_root.mkdir(parents=True, exist_ok=True)
        matrix_path = world_root / "matrix_only.npz"
        labels_path = world_root / "labels_true.npy"
        np.savez_compressed(matrix_path, X=matrix)
        np.save(labels_path, labels)
        records.append(
            {
                "world": world,
                "dataset_id": f"synthetic__{world}__seed{seed}",
                "matrix_path": str(matrix_path.resolve()),
                "labels_path": str(labels_path.resolve()),
                "n_samples": int(matrix.shape[0]),
                "n_features": int(matrix.shape[1]),
                "zero_fraction": float(np.mean(matrix == 0.0)),
                "labels_used_during_generation": True,
                "labels_accessible_during_fit": False,
                "labels_accessible_during_profile": False,
            }
        )
    manifest = {
        "manifest_id": f"{config.protocol_id}__seed{seed}",
        "generation_config": asdict(config),
        "selection_uses_results": False,
        "records": records,
    }
    (root / f"manifest_seed{seed}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate V24-Q1 corrected synthetic worlds")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-samples", type=int, default=3000)
    parser.add_argument("--n-features", type=int, default=1000)
    parser.add_argument("--n-clusters", type=int, default=6)
    parser.add_argument("--zero-fraction", type=float, default=0.90)
    parser.add_argument("--block-size", type=int, default=20)
    parser.add_argument("--active-blocks-per-sample", type=int, default=5)
    args = parser.parse_args()
    config = V24Q1Config(
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_clusters=args.n_clusters,
        zero_fraction=args.zero_fraction,
        block_size=args.block_size,
        active_blocks_per_sample=args.active_blocks_per_sample,
    )
    manifest = write_panel(args.output_root, config, seed=args.seed)
    print(json.dumps({"manifest_id": manifest["manifest_id"], "records": len(manifest["records"])}, indent=2))


if __name__ == "__main__":
    main()
