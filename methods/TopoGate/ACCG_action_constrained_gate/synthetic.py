from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


WORLD_NAMES = (
    "W0_matched_null",
    "W1_isolated_corruption",
    "W2_rare_coherent_signal",
    "W3_coherent_nuisance",
    "W4_observational_alias",
    "W5_joint_interaction",
)
GENERATOR_FAMILIES = ("lognormal_sparse", "count_sparse", "gamma_sparse")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class SyntheticConfig:
    protocol_id: str = "accg_synthetic_contract_v1"
    n_samples: int = 1200
    n_features: int = 200
    n_clusters: int = 4
    rare_cluster_fraction: float = 0.10
    zero_fraction: float = 0.80
    module_size: int = 10
    cluster_effect: float = 1.25
    isolated_effect: float = 3.0
    nuisance_effect: float = 1.5
    corruption_fraction: float = 0.04
    shortcut_auc_ceiling: float = 0.60
    families: tuple[str, ...] = GENERATOR_FAMILIES

    def validate(self) -> None:
        if self.n_samples <= 0 or self.n_features <= 0 or self.n_clusters < 2:
            raise ValueError("invalid synthetic dimensions")
        if not 0.0 < self.rare_cluster_fraction < 0.5:
            raise ValueError("rare_cluster_fraction must be in (0, 0.5)")
        if not 0.0 < self.zero_fraction < 1.0:
            raise ValueError("zero_fraction must be in (0, 1)")
        if self.module_size < 2 or self.n_features < self.module_size * (self.n_clusters + 3):
            raise ValueError("n_features is too small for the frozen module layout")
        if min(self.cluster_effect, self.isolated_effect, self.nuisance_effect) <= 0.0:
            raise ValueError("synthetic effects must be positive")
        if not 0.0 < self.corruption_fraction < 0.5:
            raise ValueError("corruption_fraction must be in (0, 0.5)")
        if not 0.5 < self.shortcut_auc_ceiling < 1.0:
            raise ValueError("shortcut_auc_ceiling must be in (0.5, 1)")
        if not self.families or set(self.families) - set(GENERATOR_FAMILIES):
            raise ValueError(f"families must be drawn from {GENERATOR_FAMILIES}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SyntheticWorld:
    name: str
    family: str
    X: np.ndarray
    labels: np.ndarray
    alternative_labels: np.ndarray | None
    clean_reference: np.ndarray
    repair_mask: np.ndarray
    protect_mask: np.ndarray
    nuisance_mask: np.ndarray
    module_ids: np.ndarray
    metadata: dict[str, Any]


def _labels(config: SyntheticConfig, rng: np.random.Generator) -> np.ndarray:
    rare = max(2, int(round(config.n_samples * config.rare_cluster_fraction)))
    common_total = config.n_samples - rare
    common_clusters = config.n_clusters - 1
    base = common_total // common_clusters
    sizes = [base] * common_clusters
    for index in range(common_total - base * common_clusters):
        sizes[index] += 1
    sizes.append(rare)
    labels = np.concatenate([np.full(size, cluster, dtype=np.int64) for cluster, size in enumerate(sizes)])
    return labels[rng.permutation(labels.size)]


def _module_ids(config: SyntheticConfig) -> np.ndarray:
    modules = np.full(config.n_features, -1, dtype=np.int64)
    for module in range(config.n_features // config.module_size):
        start = module * config.module_size
        modules[start : start + config.module_size] = module
    return modules


def _shared_support(config: SyntheticConfig, rng: np.random.Generator) -> np.ndarray:
    active = max(2, int(round(config.n_samples * (1.0 - config.zero_fraction))))
    support = np.zeros((config.n_samples, config.n_features), dtype=np.bool_)
    # Keep support identical within each declared module. This preserves exact
    # support matching across worlds while making the W5 pair a real sparse
    # feature relation rather than a dense-latent relation erased by zeros.
    for start in range(0, config.n_features, config.module_size):
        stop = min(start + config.module_size, config.n_features)
        rows = rng.choice(config.n_samples, size=active, replace=False)
        support[np.ix_(rows, np.arange(start, stop, dtype=np.int64))] = True
    return support


def _target_values(config: SyntheticConfig, family: str, support: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    targets: list[np.ndarray] = []
    for feature in range(config.n_features):
        count = int(support[:, feature].sum())
        if family == "lognormal_sparse":
            values = rng.lognormal(mean=0.0, sigma=0.65, size=count)
        elif family == "count_sparse":
            values = 1.0 + rng.negative_binomial(n=3, p=0.55, size=count)
        elif family == "gamma_sparse":
            values = rng.gamma(shape=2.0, scale=1.0, size=count)
        else:
            raise ValueError(f"unsupported generator family: {family}")
        targets.append(np.sort(np.asarray(values, dtype=np.float32)))
    return targets


def _rank_match(latent: np.ndarray, support: np.ndarray, targets: list[np.ndarray]) -> np.ndarray:
    output = np.zeros(latent.shape, dtype=np.float32)
    for feature in range(latent.shape[1]):
        rows = np.flatnonzero(support[:, feature])
        order = np.argsort(latent[rows, feature], kind="mergesort")
        output[rows[order], feature] = targets[feature]
    return output


def _base_latent(config: SyntheticConfig, labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    latent = rng.normal(size=(config.n_samples, config.n_features))
    shared_modules = config.n_clusters
    for cluster in range(config.n_clusters):
        rows = labels == cluster
        start = cluster * config.module_size
        factor = rng.normal(loc=config.cluster_effect, scale=0.35, size=int(rows.sum()))
        latent[np.ix_(rows, np.arange(start, start + config.module_size))] += factor[:, None]
    # Shared weak modules ensure the feature graph is nontrivial in every world.
    for module in range(shared_modules, config.n_features // config.module_size):
        start = module * config.module_size
        factor = rng.normal(scale=0.35, size=config.n_samples)
        latent[:, start : start + config.module_size] += factor[:, None]
    return latent


def _world_latent(
    name: str,
    base: np.ndarray,
    support: np.ndarray,
    labels: np.ndarray,
    config: SyntheticConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, dict[str, Any]]:
    latent = base.copy()
    repair = np.zeros(latent.shape, dtype=np.bool_)
    protect = np.zeros(latent.shape, dtype=np.bool_)
    nuisance = np.zeros(latent.shape, dtype=np.bool_)
    alternative: np.ndarray | None = None
    metadata: dict[str, Any] = {}
    rare_cluster = config.n_clusters - 1
    special_start = config.n_clusters * config.module_size
    module = np.arange(special_start, special_start + config.module_size)
    if name == "W0_matched_null":
        metadata["oracle_action"] = "none"
    elif name == "W1_isolated_corruption":
        active = np.flatnonzero(support.reshape(-1))
        count = min(active.size, max(1, int(round(config.corruption_fraction * latent.size))))
        flat = rng.choice(active, size=count, replace=False)
        rows, cols = np.unravel_index(flat, latent.shape)
        latent[rows, cols] += rng.choice(np.asarray([-1.0, 1.0]), size=count) * config.isolated_effect
        repair[rows, cols] = True
        metadata["oracle_action"] = "repair_isolated_coordinates"
    elif name == "W2_rare_coherent_signal":
        rows = labels == rare_cluster
        factor = rng.normal(loc=config.cluster_effect * 1.4, scale=0.15, size=int(rows.sum()))
        latent[np.ix_(rows, module)] += factor[:, None]
        protect[np.ix_(rows, module)] = True
        protect &= support
        metadata["oracle_action"] = "preserve_rare_module"
    elif name == "W3_coherent_nuisance":
        nuisance_group = rng.integers(0, 2, size=config.n_samples)
        factor = (2 * nuisance_group - 1).astype(np.float64) * config.nuisance_effect
        latent[:, module] += factor[:, None]
        nuisance[:, module] = True
        nuisance &= support
        metadata.update({"oracle_action": "boundary_only", "nuisance_group_balance": float(nuisance_group.mean())})
    elif name == "W4_observational_alias":
        permutation = rng.permutation(config.n_samples)
        alternative = labels[permutation]
        metadata["oracle_action"] = "identifiability_boundary"
    elif name == "W5_joint_interaction":
        pair_module = module
        interaction_pair = pair_module[:2]
        factor = rng.normal(scale=1.2, size=config.n_samples)
        latent[:, pair_module] = factor[:, None] + rng.normal(scale=0.03, size=(config.n_samples, pair_module.size))
        protect[:, pair_module] = True
        protect &= support
        metadata.update(
            {
                "oracle_action": "evaluate_joint_not_singleton",
                "interaction_module": pair_module.tolist(),
                "interaction_pair": interaction_pair.tolist(),
                "construction": "same-donor replacement of a coherent pair can preserve joint residual while singleton replacement breaks it",
            }
        )
    else:
        raise ValueError(f"unsupported world: {name}")
    return latent, repair, protect, nuisance, alternative, metadata


def generate_worlds(config: SyntheticConfig, *, family: str, seed: int) -> dict[str, SyntheticWorld]:
    """Generate marginal/support-matched ACCG worlds for one family and seed."""

    config.validate()
    if family not in config.families:
        raise ValueError(f"family {family!r} is not enabled")
    shared_rng = np.random.default_rng(int(seed) + 100)
    labels = _labels(config, shared_rng)
    support = _shared_support(config, shared_rng)
    targets = _target_values(config, family, support, shared_rng)
    base = _base_latent(config, labels, shared_rng)
    clean_reference = _rank_match(base, support, targets)
    modules = _module_ids(config)
    worlds: dict[str, SyntheticWorld] = {}
    for world_index, name in enumerate(WORLD_NAMES):
        rng = np.random.default_rng(int(seed) + 10_007 * (world_index + 1))
        latent, repair, protect, nuisance, alternative, metadata = _world_latent(
            name, base, support, labels, config, rng
        )
        matrix = _rank_match(latent, support, targets)
        worlds[name] = SyntheticWorld(
            name=name,
            family=family,
            X=matrix,
            labels=labels.copy(),
            alternative_labels=None if alternative is None else alternative.copy(),
            clean_reference=clean_reference.copy(),
            repair_mask=repair,
            protect_mask=protect,
            nuisance_mask=nuisance,
            module_ids=modules.copy(),
            metadata={
                **metadata,
                "seed": int(seed),
                "family": family,
                "support_hash_shared_within_family_seed": True,
                "feature_marginals_rank_matched": True,
                "labels_used_by_generator_only": True,
                "labels_available_to_selector": False,
            },
        )
    return worlds


def write_panel(output_root: str | Path, config: SyntheticConfig, *, seeds: tuple[int, ...]) -> dict[str, Any]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for family in config.families:
        for seed in seeds:
            for world in generate_worlds(config, family=family, seed=int(seed)).values():
                out = root / family / world.name / f"seed{seed}"
                out.mkdir(parents=True, exist_ok=True)
                matrix_path = out / "matrix_only.npz"
                labels_path = out / "labels_true.npy"
                oracle_path = out / "oracle_masks.npz"
                np.savez_compressed(matrix_path, X=world.X)
                np.save(labels_path, world.labels)
                if world.alternative_labels is not None:
                    np.save(out / "labels_alternative.npy", world.alternative_labels)
                np.savez_compressed(
                    oracle_path,
                    clean_reference=world.clean_reference,
                    repair_mask=world.repair_mask,
                    protect_mask=world.protect_mask,
                    nuisance_mask=world.nuisance_mask,
                    module_ids=world.module_ids,
                )
                (out / "metadata.json").write_text(
                    json.dumps(world.metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
                )
                records.append(
                    {
                        "dataset_id": f"accg_synth__{family}__{world.name}__seed{seed}",
                        "family": family,
                        "world": world.name,
                        "seed": int(seed),
                        "matrix_path": str(matrix_path.resolve()),
                        "matrix_sha256": _sha256_file(matrix_path),
                        "labels_path": str(labels_path.resolve()),
                        "labels_sha256": _sha256_file(labels_path),
                        "oracle_path": str(oracle_path.resolve()),
                        "oracle_sha256": _sha256_file(oracle_path),
                        "metadata_path": str((out / "metadata.json").resolve()),
                        "metadata_sha256": _sha256_file(out / "metadata.json"),
                        "alternative_labels_path": str((out / "labels_alternative.npy").resolve())
                        if world.alternative_labels is not None
                        else None,
                        "alternative_labels_sha256": _sha256_file(out / "labels_alternative.npy")
                        if world.alternative_labels is not None
                        else None,
                        "n_samples": int(world.X.shape[0]),
                        "n_features": int(world.X.shape[1]),
                        "zero_fraction": float(np.mean(world.X == 0.0)),
                        "status": "generated_not_run",
                    }
                )
    manifest = {
        "manifest_id": "accg_synthetic_w0_w5_v1",
        "protocol_id": config.protocol_id,
        "config": config.to_dict(),
        "seeds": [int(seed) for seed in seeds],
        "worlds": list(WORLD_NAMES),
        "families": list(config.families),
        "labels_used_during_fit": False,
        "formal_training_started": False,
        "records": records,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return manifest


def w5_pair_counterexample() -> tuple[np.ndarray, np.ndarray]:
    """Return anchor/donor rows where singleton replacements break a coherent pair."""

    anchor = np.asarray([[-1.0, -1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    donor = anchor[::-1].copy()
    return anchor, donor
