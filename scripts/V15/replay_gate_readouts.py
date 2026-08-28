#!/usr/bin/env python3
"""Replay V15 gate readouts from one completed checkpoint.

The script intentionally never retrains a model. It consumes the detached
clean self branch, every single-edge counterfactual branch, the registered
utility, and the validity mask saved by ``fit_v15``. Labels, when present, are
used only for post-fit benchmark metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)


VARIANTS = (
    "self_only",
    "direct_counterfactual",
    "abstaining_top1",
    "union_uniform",
    "forced_topk",
    "shuffled_utility",
)
UTILITY_SOURCES = (
    "stored",
    "ema_clean",
    "ema_augmented",
    "raw_aligned",
    "consensus",
    "mean_views",
    "lcb_views",
    "quality_selected",
)
CANDIDATE_SCOPES = ("all", "both_views", "raw_supported", "latent_supported")


def _normalise_probabilities(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = np.clip(values, 1e-12, None)
    return values / values.sum(axis=-1, keepdims=True).clip(1e-12, None)


def counterfactual_delta_numpy(
    reference: np.ndarray,
    q_self: np.ndarray,
    q_edge: np.ndarray,
    *,
    valid: np.ndarray | None = None,
    min_gain: float = 0.0,
    clip: float | None = None,
) -> np.ndarray:
    """Exact clean-output utility used by the trainer's detached readout.

    With the default arguments this is the raw KL improvement retained for
    backwards-compatible diagnostics.  Supplying ``valid`` and ``min_gain``
    additionally matches ``clean_output_utility``: zero-preserving robust
    scaling followed by the null-margin threshold and optional clipping.
    """
    reference = _normalise_probabilities(reference)
    q_self = _normalise_probabilities(q_self)
    q_edge = _normalise_probabilities(q_edge)
    if reference.shape != q_self.shape or q_edge.shape[0] != q_self.shape[0]:
        raise ValueError("reference/self/edge assignments have incompatible shapes")
    if q_edge.shape[2] != q_self.shape[1]:
        raise ValueError("edge assignments have the wrong cluster dimension")
    self_kl = np.sum(reference * (np.log(reference) - np.log(q_self)), axis=1)
    edge_kl = np.sum(
        reference[:, None, :] * (np.log(reference[:, None, :]) - np.log(q_edge)),
        axis=2,
    )
    semantic = self_kl[:, None] - edge_kl
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != semantic.shape:
            raise ValueError("valid must match the utility shape")
        selected = np.abs(semantic[valid])
        selected = selected[selected > 1e-8]
        if selected.size:
            scale = max(1e-3, 1.4826 * float(np.median(selected)))
        else:
            scale = 1.0
        semantic = semantic / scale - float(min_gain)
        if clip is not None:
            semantic = np.clip(semantic, -float(clip), float(clip))
        semantic = np.where(valid, semantic, -float(clip) if clip is not None else semantic)
    return semantic.astype(np.float32)


def view_quality_numpy(probabilities: np.ndarray, neighbor_indices: np.ndarray) -> dict[str, float]:
    """Label-free local quality used to select one teacher reference view."""
    probabilities = _normalise_probabilities(probabilities)
    neighbors = np.asarray(neighbor_indices, dtype=np.int64)
    if probabilities.ndim != 2 or neighbors.ndim != 2 or probabilities.shape[0] != neighbors.shape[0]:
        raise ValueError("probabilities and neighbor_indices must have compatible shapes")
    labels = probabilities.argmax(axis=1)
    valid = (neighbors >= 0) & (neighbors < probabilities.shape[0])
    if not np.any(valid):
        return {"local_agreement": 0.0, "confidence": 0.0, "effective_clusters": 0.0, "score": -1.0}
    rows = np.broadcast_to(np.arange(probabilities.shape[0])[:, None], neighbors.shape)
    local_agreement = float(np.mean((labels[rows[valid]] == labels[neighbors[valid]]).astype(np.float32)))
    ordered = np.sort(probabilities, axis=1)
    confidence = float(np.mean(ordered[:, -1] - ordered[:, -2]))
    marginal = probabilities.mean(axis=0)
    marginal_entropy = float(-(marginal * np.log(np.clip(marginal, 1e-8, None))).sum())
    effective_clusters = float(np.exp(marginal_entropy))
    # Reject views that are locally coherent only because they activate a
    # small subset of the requested clusters. Keep the same threshold as the
    # training-time quality_auto selector.
    collapse_penalty = (
        1.0 if effective_clusters < max(1.5, 0.6 * probabilities.shape[1]) else 0.0
    )
    return {
        "local_agreement": local_agreement,
        "confidence": confidence,
        "effective_clusters": effective_clusters,
        "score": local_agreement + 0.05 * confidence - collapse_penalty,
    }


def select_quality_reference(
    references: dict[str, np.ndarray], neighbor_indices: np.ndarray
) -> tuple[str, dict[str, dict[str, float]]]:
    scores = {name: view_quality_numpy(values, neighbor_indices) for name, values in references.items()}
    selected = max(scores, key=lambda name: scores[name]["score"])
    return selected, scores


def candidate_scope_mask(
    candidate_features: np.ndarray,
    valid: np.ndarray,
    scope: str,
) -> np.ndarray:
    """Restrict replay to a graph-source ablation without rebuilding candidates."""
    features = np.asarray(candidate_features)
    valid = np.asarray(valid, dtype=bool)
    if features.ndim != 3 or features.shape[:2] != valid.shape or features.shape[2] < 3:
        raise ValueError("candidate features must have shape [N, M, >=3]")
    source = features[:, :, 2]
    if scope == "all":
        source_mask = np.ones_like(valid)
    elif scope == "both_views":
        source_mask = np.abs(source) < 0.5
    elif scope == "raw_supported":
        source_mask = source <= 0.5
    elif scope == "latent_supported":
        source_mask = source >= -0.5
    else:
        raise ValueError(f"unknown candidate scope: {scope}")
    return valid & source_mask


def load_utility_sources(
    run_dir: Path,
    q_self: np.ndarray,
    q_edge: np.ndarray,
    stored: np.ndarray,
    requested: Iterable[str],
    neighbor_indices: np.ndarray | None = None,
    utility_valid: np.ndarray | None = None,
    utility_min_gain: float = 0.0,
    utility_clip: float | None = None,
) -> dict[str, np.ndarray]:
    requested = tuple(requested)
    unknown = sorted(set(requested).difference(UTILITY_SOURCES))
    if unknown:
        raise ValueError("unknown utility sources: " + ", ".join(unknown))
    output: dict[str, np.ndarray] = {}
    if "stored" in requested:
        output["stored"] = np.asarray(stored, dtype=np.float32)
    reference_files = {
        "ema_clean": "teacher_probabilities_clean.npy",
        "ema_augmented": "teacher_probabilities_augmented.npy",
        "raw_aligned": "teacher_probabilities_raw_aligned.npy",
        "consensus": "teacher_probabilities_reference.npy",
    }
    required_references = {
        name
        for name in reference_files
        if name in requested
        or "mean_views" in requested
        or "lcb_views" in requested
        or "quality_selected" in requested
    }
    deltas: dict[str, np.ndarray] = {}
    references: dict[str, np.ndarray] = {}
    for name in required_references:
        path = run_dir / reference_files[name]
        if not path.exists():
            raise ValueError(f"missing reference assignment artifact: {path}")
        references[name] = np.load(path, allow_pickle=False)
        deltas[name] = counterfactual_delta_numpy(
            references[name],
            q_self,
            q_edge,
            valid=utility_valid,
            min_gain=utility_min_gain,
            clip=utility_clip,
        )
        if name in requested:
            output[name] = deltas[name]
    if "mean_views" in requested or "lcb_views" in requested:
        view_stack = np.stack(
            [deltas["ema_clean"], deltas["ema_augmented"], deltas["raw_aligned"]],
            axis=0,
        )
        if "mean_views" in requested:
            output["mean_views"] = view_stack.mean(axis=0).astype(np.float32)
        if "lcb_views" in requested:
            output["lcb_views"] = (view_stack.mean(axis=0) - view_stack.std(axis=0)).astype(
                np.float32
            )
    if "quality_selected" in requested:
        if neighbor_indices is None:
            raise ValueError("quality_selected utility requires candidate neighbor indices")
        selected, _ = select_quality_reference(
            {name: references[name] for name in ("ema_clean", "ema_augmented", "raw_aligned")},
            neighbor_indices,
        )
        output["quality_selected"] = deltas[selected]
    return {name: output[name] for name in requested}


def _sparsemax_row(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-values, kind="stable")
    sorted_values = values[order]
    cssv = np.cumsum(sorted_values) - 1.0
    positions = np.arange(1, sorted_values.size + 1, dtype=np.float64)
    support = sorted_values - cssv / positions > 0.0
    if not np.any(support):
        output = np.zeros_like(values)
        output[int(np.argmax(values))] = 1.0
        return output
    rho = int(np.flatnonzero(support)[-1]) + 1
    tau = float(cssv[rho - 1] / rho)
    return np.maximum(values - tau, 0.0)


def abstaining_sparsemax_numpy(scores: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Match V15's null-plus-positive sparsemax contract in NumPy."""
    scores = np.asarray(scores, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if scores.ndim != 2 or valid.shape != scores.shape:
        raise ValueError("scores and valid must both have shape [N, M]")
    output = np.zeros((scores.shape[0], scores.shape[1] + 1), dtype=np.float64)
    for row in range(scores.shape[0]):
        active = valid[row] & (scores[row] > 0.0)
        if not np.any(active):
            output[row, 0] = 1.0
            continue
        projected = _sparsemax_row(np.concatenate(([0.0], scores[row, active])))
        output[row, 0] = projected[0]
        output[row, 1:][active] = projected[1:]
    return output


def abstaining_top1_numpy(scores: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Choose one positive edge or the exact null branch."""
    scores = np.asarray(scores, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    output = np.zeros((scores.shape[0], scores.shape[1] + 1), dtype=np.float64)
    for row in range(scores.shape[0]):
        active = np.flatnonzero(valid[row] & (scores[row] > 0.0))
        if active.size == 0:
            output[row, 0] = 1.0
            continue
        selected = int(active[np.argmax(scores[row, active])])
        output[row, selected + 1] = 1.0
    return output


def uniform_numpy(valid: np.ndarray) -> np.ndarray:
    valid = np.asarray(valid, dtype=bool)
    counts = valid.sum(axis=1)
    output = np.zeros((valid.shape[0], valid.shape[1] + 1), dtype=np.float64)
    nonempty = counts > 0
    output[nonempty, 1:] = valid[nonempty] / counts[nonempty, None]
    output[~nonempty, 0] = 1.0
    return output


def forced_topk_numpy(scores: np.ndarray, valid: np.ndarray, topk: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    output = np.zeros((scores.shape[0], scores.shape[1] + 1), dtype=np.float64)
    width = min(max(int(topk), 0), scores.shape[1])
    for row in range(scores.shape[0]):
        active = np.flatnonzero(valid[row])
        if active.size == 0 or width == 0:
            output[row, 0] = 1.0
            continue
        order = active[np.argsort(-scores[row, active], kind="stable")[:width]]
        output[row, order + 1] = 1.0 / float(order.size)
    return output


def shuffle_scores_numpy(scores: np.ndarray, valid: np.ndarray, seed: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    shuffled = scores.copy()
    rng = np.random.default_rng(int(seed))
    for row in range(scores.shape[0]):
        positions = np.flatnonzero(valid[row])
        if positions.size > 1:
            shuffled[row, positions] = scores[row, rng.permutation(positions)]
    return shuffled


def make_gate_distribution(
    utility: np.ndarray,
    valid: np.ndarray,
    variant: str,
    *,
    seed: int = 4502,
    forced_topk: int = 2,
) -> np.ndarray:
    if variant == "self_only":
        output = np.zeros((utility.shape[0], utility.shape[1] + 1), dtype=np.float64)
        output[:, 0] = 1.0
        return output
    if variant == "union_uniform":
        return uniform_numpy(valid)
    if variant == "forced_topk":
        return forced_topk_numpy(utility, valid, forced_topk)
    if variant == "shuffled_utility":
        shuffled = shuffle_scores_numpy(utility, valid, seed)
        return abstaining_sparsemax_numpy(shuffled, valid)
    if variant == "abstaining_top1":
        return abstaining_top1_numpy(utility, valid)
    if variant == "direct_counterfactual":
        return abstaining_sparsemax_numpy(utility, valid)
    raise ValueError(f"unknown replay variant: {variant}")


def apply_assignment_readout(
    q_self: np.ndarray,
    q_edge: np.ndarray,
    pi: np.ndarray,
    z_self: np.ndarray,
    z_edge: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    q_self = _normalise_probabilities(q_self)
    q_edge = _normalise_probabilities(q_edge)
    pi = np.asarray(pi, dtype=np.float64)
    if np.any(pi < -1e-8) or not np.isfinite(pi).all():
        raise ValueError("gate probabilities must be finite and non-negative")
    pi = np.clip(pi, 0.0, None)
    pi = pi / pi.sum(axis=1, keepdims=True).clip(1e-12, None)
    if q_edge.ndim != 3 or pi.shape != (q_self.shape[0], q_edge.shape[1] + 1):
        raise ValueError("assignment readout arrays have incompatible shapes")
    probabilities = pi[:, :1] * q_self + np.sum(pi[:, 1:, None] * q_edge, axis=1)
    probabilities = _normalise_probabilities(probabilities)
    embedding = pi[:, :1] * z_self + np.sum(pi[:, 1:, None] * z_edge, axis=1)
    return embedding.astype(np.float32), probabilities.astype(np.float32)


def _metric_payload(
    embedding: np.ndarray,
    probabilities: np.ndarray,
    labels: np.ndarray | None,
    utility: np.ndarray,
    pi: np.ndarray,
) -> dict[str, Any]:
    prediction = probabilities.argmax(axis=1).astype(np.int64)
    edge_mass = pi[:, 1:].sum(axis=1)
    entropy = -(pi[:, 1:] * np.log(np.clip(pi[:, 1:], 1e-12, None))).sum(axis=1)
    selected_values = np.max(np.where(pi[:, 1:] > 0.0, utility, -np.inf), axis=1)
    selected_values = np.where(np.isfinite(selected_values), selected_values, 0.0)
    payload: dict[str, Any] = {
        "prediction_count": int(prediction.size),
        "null_mass": float(np.mean(pi[:, 0])),
        "edge_mass": float(np.mean(edge_mass)),
        "edge_positive_rate": float(np.mean(np.max(utility, axis=1) > 0.0)),
        "selected_edge_rate": float(np.mean(edge_mass > 0.0)),
        "effective_edges": float(np.mean(np.exp(entropy))),
        "selected_utility_mean": float(np.mean(selected_values)),
        "prediction": prediction,
    }
    if labels is not None:
        labels = np.asarray(labels).reshape(-1)
        payload.update(
            {
                "ari": float(adjusted_rand_score(labels, prediction)),
                "nmi": float(normalized_mutual_info_score(labels, prediction)),
                "ami": float(adjusted_mutual_info_score(labels, prediction)),
            }
        )
        unique = np.unique(prediction)
        payload["silhouette"] = (
            float(silhouette_score(embedding, prediction))
            if 1 < unique.size < len(prediction)
            else None
        )
    return payload


def replay_run(
    run_dir: str | Path,
    output_dir: str | Path,
    variants: Iterable[str] = VARIANTS,
    *,
    utility_sources: Iterable[str] = ("stored",),
    candidate_scope: str = "all",
    seed: int = 4502,
    forced_topk: int = 2,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(run_dir / "gate_diagnostics.npz", allow_pickle=False) as data:
        required = {
            "final_q_self",
            "final_q_edge",
            "final_edge_embedding",
            "final_embedding_self",
            "final_utility_hat",
            "final_gate_valid",
            "candidate_indices",
            "candidate_features",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(
                "checkpoint does not contain replay certificate arrays: " + ", ".join(missing)
            )
        q_self = np.asarray(data["final_q_self"], dtype=np.float32)
        q_edge = np.asarray(data["final_q_edge"], dtype=np.float32)
        candidate_indices = np.asarray(data["candidate_indices"], dtype=np.int64)
        candidate_features = np.asarray(data["candidate_features"], dtype=np.float32)
        z_edge = np.asarray(data["final_edge_embedding"], dtype=np.float32)
        z_self = np.asarray(data["final_embedding_self"], dtype=np.float32)
        utility = np.asarray(data["final_utility_hat"], dtype=np.float32)
        valid = np.asarray(data["final_gate_valid"], dtype=bool)
    if q_self.ndim != 2 or q_edge.ndim != 3 or z_edge.ndim != 3:
        raise ValueError("q_self, q_edge, and edge embeddings have invalid dimensions")
    if q_edge.shape[:2] != utility.shape or utility.shape != valid.shape:
        raise ValueError("utility/edge assignment/validity arrays have incompatible shapes")
    if z_edge.shape[:2] != utility.shape or z_self.shape[0] != q_self.shape[0]:
        raise ValueError("embedding arrays have incompatible shapes")
    valid = candidate_scope_mask(candidate_features, valid, candidate_scope)
    scoped_neighbors = np.where(valid, candidate_indices, -1)
    labels_path = run_dir / "labels_true.npy"
    labels = np.load(labels_path, allow_pickle=False) if labels_path.exists() else None
    config_path = run_dir / "resolved_config.json"
    resolved_config: dict[str, Any] = {}
    if config_path.exists():
        loaded_config = json.loads(config_path.read_text(encoding="utf-8"))
        if isinstance(loaded_config, dict):
            resolved_config = loaded_config
    utility_min_gain = float(resolved_config.get("utility_min_gain", 0.0))
    utility_clip_value = resolved_config.get("utility_clip", None)
    utility_clip = None if utility_clip_value is None else float(utility_clip_value)
    resolved_variants = tuple(variants)
    resolved_sources = tuple(utility_sources)
    utilities = load_utility_sources(
        run_dir,
        q_self,
        q_edge,
        utility,
        resolved_sources,
        neighbor_indices=scoped_neighbors,
        utility_valid=valid,
        utility_min_gain=utility_min_gain,
        utility_clip=utility_clip,
    )
    utility_metadata: dict[str, Any] = {}
    if "quality_selected" in resolved_sources:
        reference_arrays = {
            name: np.load(run_dir / filename, allow_pickle=False)
            for name, filename in {
                "ema_clean": "teacher_probabilities_clean.npy",
                "ema_augmented": "teacher_probabilities_augmented.npy",
                "raw_aligned": "teacher_probabilities_raw_aligned.npy",
            }.items()
        }
        selected, scores = select_quality_reference(reference_arrays, scoped_neighbors)
        utility_metadata["quality_selected_reference"] = selected
        utility_metadata["quality_reference_scores"] = scores
    results: dict[str, Any] = {}
    for utility_source, active_utility in utilities.items():
        for variant in resolved_variants:
            pi = make_gate_distribution(
                active_utility,
                valid,
                variant,
                seed=seed,
                forced_topk=forced_topk,
            )
            embedding, probabilities = apply_assignment_readout(q_self, q_edge, pi, z_self, z_edge)
            metrics = _metric_payload(embedding, probabilities, labels, active_utility, pi)
            metrics.pop("prediction", None)
            metrics["utility_source"] = utility_source
            metrics["gate_variant"] = variant
            variant_dir = output_dir / utility_source / variant
            variant_dir.mkdir(parents=True, exist_ok=True)
            np.save(variant_dir / "predictions.npy", probabilities.argmax(axis=1).astype(np.int64))
            np.save(variant_dir / "cluster_probabilities.npy", probabilities)
            np.save(variant_dir / "embedding_final.npy", embedding)
            np.save(variant_dir / "gate_probabilities.npy", pi.astype(np.float32))
            (variant_dir / "metrics.json").write_text(
                json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
            )
            results[f"{utility_source}__{variant}"] = metrics
    summary = {
        "schema_version": "V15-gate-replay-1",
        "run_dir": str(run_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "same_checkpoint": True,
        "labels_used_for_fit": False,
        "replay_seed": int(seed),
        "forced_topk": int(forced_topk),
        "candidate_scope": candidate_scope,
        "variants": list(resolved_variants),
        "utility_sources": list(resolved_sources),
        "utility_metadata": utility_metadata,
        "metrics": results,
    }
    (output_dir / "replay_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", action="append", choices=VARIANTS, default=None)
    parser.add_argument(
        "--utility-source",
        action="append",
        choices=UTILITY_SOURCES,
        default=None,
    )
    parser.add_argument("--seed", type=int, default=4502)
    parser.add_argument("--forced-topk", type=int, default=2)
    parser.add_argument("--candidate-scope", choices=CANDIDATE_SCOPES, default="all")
    args = parser.parse_args()
    summary = replay_run(
        args.run_dir,
        args.output_dir,
        args.variant or VARIANTS,
        utility_sources=args.utility_source or ("stored",),
        candidate_scope=args.candidate_scope,
        seed=args.seed,
        forced_topk=args.forced_topk,
    )
    print(
        json.dumps(
            {
                "variants": summary["variants"],
                "utility_sources": summary["utility_sources"],
                "candidate_scope": summary["candidate_scope"],
                "output_dir": summary["output_dir"],
            }
        )
    )


if __name__ == "__main__":
    main()
