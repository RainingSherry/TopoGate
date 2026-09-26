"""Feature-only, restartable single-task runner for frozen causal interventions.

Run as a standalone script from the independent experiment directory. The
historical ``v0rg_tunable_multihop`` package must be on PYTHONPATH.
"""
from __future__ import annotations

import argparse
import copy
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch

from v0rg_tunable_multihop.config import V0_RGConfig
from v0rg_tunable_multihop.input_adapter import load_npz_matrix_only
from v0rg_tunable_multihop.tuning import SplitPreprocessor
from v0rg_tunable_multihop import hop_sampling, mixing, trainer

try:
    from .intervention_plan import ARMS, transform_plan
except ImportError:
    from intervention_plan import ARMS, transform_plan


PROTOCOL = "multihop_mechanism_step5_frozen_feature_only_v1"
TRACE_DTYPE = np.dtype([("anchor", "<i4"), ("source", "<i4"),
                        ("original_hop", "u1"), ("actual_hop", "u1"),
                        ("weight", "<f4"), ("slot_probability", "<f4")])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _atomic_numpy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value)
    temporary.replace(path)


def _atomic_npz(path: Path, **values: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **values)
    temporary.replace(path)


def _atomic_torch(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _atomic_copy(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    temporary = target.with_suffix(target.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(target)


def _source_hashes() -> dict[str, str]:
    source_dir = Path(trainer.__file__).resolve().parent
    return {path.name: sha256(path) for path in sorted(source_dir.glob("*.py"))}


def _read_object(value: Any, field: str) -> dict:
    result = json.loads(Path(value).read_text(encoding="utf-8")) if isinstance(value, str) else value
    if not isinstance(result, dict):
        raise ValueError(f"{field} must be a JSON object or a path to one")
    return result


def _validate_task(task: dict) -> tuple[dict, dict, V0_RGConfig]:
    required = {"dataset", "seed", "arm", "locked_record", "tuning_manifest",
                "input_kind", "feature_limit"}
    missing = required - set(task)
    if missing:
        raise ValueError(f"missing task fields: {sorted(missing)}")
    if task["arm"] not in ARMS:
        raise ValueError("unrecognized arm")
    if task.get("norm_match_reference", False) not in (False, True):
        raise ValueError("norm_match_reference must be boolean")
    if task.get("norm_match_reference", False) and task["arm"] not in {
        "replace_far_le4", "far_shuffle"
    }:
        raise ValueError("norm matching is fixed to replace_far_le4/far_shuffle diagnostics")
    if int(task["seed"]) not in {42, 123, 7}:
        raise ValueError("mechanism task seed must be one of 42, 123, 7")
    locked = _read_object(task["locked_record"], "locked_record")
    manifest = _read_object(task["tuning_manifest"], "tuning_manifest")
    if locked.get("status") != "completed" or int(locked.get("seed", -1)) != int(task["seed"]):
        raise ValueError("locked record is incomplete or from another seed")
    if locked.get("test_labels_loaded") is not False:
        raise ValueError("locked validation record accessed test labels")
    if manifest.get("test_label_path_exposed") is not False or any(
        key in manifest for key in ("test_labels_path", "test_label_path", "full_labels_path")
    ):
        raise ValueError("fitting manifest exposes final or full labels")
    if manifest.get("K_provenance") != "benchmark_oracle_from_y":
        raise ValueError("K provenance must be explicit benchmark oracle")
    if locked.get("feature_sha256") != manifest.get("feature_sha256"):
        raise ValueError("locked record and manifest feature hashes disagree")
    config = V0_RGConfig(**locked["config"])
    if not str(config.hop_variant).startswith("H8_"):
        raise ValueError("reference config must be a frozen H8 variant")
    if int(config.epochs) != 80 or config.variant != "rg_full":
        raise ValueError("frozen intervention requires 80-epoch rg_full config")
    if task["input_kind"] not in {"raw_count", "log1p_expression", "general"}:
        raise ValueError("invalid input_kind")
    if int(task["feature_limit"]) <= 0:
        raise ValueError("feature_limit must be positive")
    return locked, manifest, config


class TraceCollector:
    def __init__(self, path: Path, n_samples: int):
        self.path = path
        self.tmp = path.with_suffix(path.suffix + ".tmp")
        self.handle = self.tmp.open("wb")
        self.buffer: list[np.ndarray] = []
        self.n_samples = n_samples
        self.draws = 0
        self.anchor_displacement_sum = np.zeros(n_samples, dtype=np.float64)
        self.anchor_displacement_count = np.zeros(n_samples, dtype=np.int64)
        self.original_hop_mass = defaultdict(float)
        self.actual_hop_mass = defaultdict(float)
        self.original_hop_draws = defaultdict(int)
        self.actual_hop_draws = defaultdict(int)
        self.original_hops: tuple[np.ndarray, ...] | None = None
        self.plan = None
        self.norm_reference_sum = 0.0
        self.norm_intervention_pre_sum = 0.0
        self.norm_intervention_post_sum = 0.0
        self.norm_unmatched_count = 0
        self.norm_scale_gt_one_count = 0
        self.norm_observations = 0

    def record(self, anchor: int, nodes: np.ndarray, original: np.ndarray,
               actual: np.ndarray, weights: np.ndarray, probs: np.ndarray) -> None:
        block = np.empty(len(nodes), dtype=TRACE_DTYPE)
        block["anchor"] = anchor
        block["source"] = nodes
        block["original_hop"] = original
        block["actual_hop"] = actual
        block["weight"] = weights
        block["slot_probability"] = probs
        self.buffer.append(block)
        self.draws += len(nodes)
        for hop in np.unique(original):
            mask = original == hop
            self.original_hop_mass[int(hop)] += float(weights[mask].sum())
            self.original_hop_draws[int(hop)] += int(mask.sum())
        for hop in np.unique(actual):
            mask = actual == hop
            self.actual_hop_mass[int(hop)] += float(weights[mask].sum())
            self.actual_hop_draws[int(hop)] += int(mask.sum())

    def flush_batch(self) -> None:
        if self.buffer:
            np.concatenate(self.buffer).tofile(self.handle)
            self.buffer.clear()

    def displacement(self, batch_indices: np.ndarray, mixed: torch.Tensor, anchor: torch.Tensor) -> None:
        values = torch.linalg.vector_norm(mixed - anchor, dim=1).detach().cpu().numpy()
        np.add.at(self.anchor_displacement_sum, batch_indices, values)
        np.add.at(self.anchor_displacement_count, batch_indices, 1)

    def finish(self) -> dict:
        self.flush_batch()
        self.handle.close()
        self.tmp.replace(self.path)
        rows = int(self.anchor_displacement_count.sum())
        _atomic_npz(self.path.parent / "displacement_by_anchor.npz",
                    sum=self.anchor_displacement_sum,
                    count=self.anchor_displacement_count,
                    mean=np.divide(self.anchor_displacement_sum, self.anchor_displacement_count,
                                   out=np.zeros(self.n_samples, dtype=np.float64),
                                   where=self.anchor_displacement_count > 0))
        result = {"trace_file": self.path.name, "trace_dtype": TRACE_DTYPE.descr,
                  "trace_sha256": sha256(self.path), "draws": self.draws,
                  "anchor_training_occurrences": rows,
                  "original_hop_weight_mass": dict(self.original_hop_mass),
                  "actual_hop_weight_mass": dict(self.actual_hop_mass),
                  "original_hop_draws": dict(self.original_hop_draws),
                  "actual_hop_draws": dict(self.actual_hop_draws),
                  "mean_actual_displacement": float(self.anchor_displacement_sum.sum() / max(1, rows)),
                  "norm_match": {
                      "enabled": bool(getattr(self, "norm_match_reference", False)),
                      "observations": self.norm_observations,
                      "reference_norm_mean": self.norm_reference_sum / max(1, self.norm_observations),
                      "intervention_pre_norm_mean": self.norm_intervention_pre_sum / max(1, self.norm_observations),
                      "intervention_post_norm_mean": self.norm_intervention_post_sum / max(1, self.norm_observations),
                      "unmatched_zero_direction_count": self.norm_unmatched_count,
                      "scale_gt_one_count": self.norm_scale_gt_one_count,
                  }}
        return result

    def abort(self) -> None:
        self.handle.close()
        self.tmp.unlink(missing_ok=True)


def _trace_plan_estimator(collector: TraceCollector, original):
    """Numerically identical sampler with one write per actual draw."""
    def traced(data, batch_indices, plan, rng, estimator):
        if estimator not in {"current", "uniform_sample", "full"}:
            raise ValueError("unknown neighbor estimator")
        output = np.empty((len(batch_indices), data.shape[1]), dtype=np.float32)
        mass = defaultdict(float); draw_hits = defaultdict(int); total_draws = 0
        known_hops = sorted({int(hop) for row in plan.hop_labels for hop in np.unique(row)})
        per_sample_mass = {hop: np.zeros(len(batch_indices), dtype=np.float64) for hop in known_hops}
        for position, source in enumerate(batch_indices):
            source = int(source)
            nodes = plan.indices[source]
            probabilities = plan.probs[source]
            labels = plan.hop_labels[source]
            original_labels = collector.original_hops[source]
            if estimator == "full":
                output[position] = np.sum(data[nodes] * probabilities[:, None], axis=0)
                for hop in known_hops:
                    contribution = float(probabilities[labels == hop].sum())
                    mass[hop] += contribution; per_sample_mass[hop][position] = contribution
                collector.record(source, nodes, original_labels, labels, probabilities, probabilities)
                continue
            choices = rng.choice(nodes.size, size=plan.draw_count, replace=True, p=probabilities)
            selected_probability = probabilities[choices]
            weights = (selected_probability / max(float(selected_probability.sum()), 1e-12)
                       if estimator == "current"
                       else np.full(plan.draw_count, 1.0 / float(plan.draw_count), dtype=np.float64))
            output[position] = np.sum(data[nodes[choices]] * weights[:, None], axis=0)
            selected_hops = labels[choices]
            collector.record(source, nodes[choices], original_labels[choices],
                             selected_hops, weights, selected_probability)
            for hop in known_hops:
                mask = selected_hops == hop
                contribution = float(weights[mask].sum())
                mass[hop] += contribution; per_sample_mass[hop][position] = contribution
                draw_hits[hop] += int(mask.sum())
            total_draws += int(plan.draw_count)
        collector.flush_batch()
        normalizer = max(1, len(batch_indices))
        return output, {
            "actual_sampling_mass": {str(hop): float(mass[hop] / normalizer) for hop in known_hops},
            "draw_fraction": {str(hop): (None if estimator == "full" else
                                      float(draw_hits[hop] / max(1, total_draws))) for hop in known_hops},
            "_per_sample_actual_sampling_mass": {str(hop): values for hop, values in per_sample_mass.items()},
        }
    return traced


def _clone_numpy_generator(rng: np.random.Generator) -> np.random.Generator:
    clone = np.random.Generator(type(rng.bit_generator)())
    clone.bit_generator.state = copy.deepcopy(rng.bit_generator.state)
    return clone


def _match_displacement_norm(anchor: np.ndarray, mixed: np.ndarray,
                             reference_mixed: np.ndarray) -> tuple[np.ndarray, dict]:
    delta = mixed - anchor
    reference_delta = reference_mixed - anchor
    before = np.linalg.norm(delta, axis=1)
    target = np.linalg.norm(reference_delta, axis=1)
    scale = np.divide(target, before, out=np.zeros_like(target), where=before > 1e-12)
    adjusted = anchor + scale[:, None] * delta
    after = np.linalg.norm(adjusted - anchor, axis=1)
    unmatched = (before <= 1e-12) & (target > 1e-12)
    return adjusted.astype(np.float32, copy=False), {
        "reference_sum": float(target.sum()), "before_sum": float(before.sum()),
        "after_sum": float(after.sum()), "unmatched": int(unmatched.sum()),
        "scale_gt_one": int(np.count_nonzero(scale > 1.0 + 1e-6)),
        "count": int(len(target)),
    }


@contextmanager
def _patched_training(arm: str, seed: int, feature_nonzero: np.ndarray,
                      collector: TraceCollector):
    old_build = trainer.build_exact_hop_plan
    old_sampler = hop_sampling.estimate_neighbors_from_plan
    old_make_plan = trainer.make_pseudo_batch_from_plan
    old_make_h1 = trainer.make_pseudo_batch
    old_h1_estimator = mixing.estimate_neighbors
    old_model = trainer.WeightedAutoEncoder
    pseudo_target = {"value": None}

    def build(*args, **kwargs):
        reference = old_build(*args, **kwargs)
        collector.reference_plan = reference
        graph, edge_weights = args[:2]
        config = getattr(collector, "config")
        plan, audit = transform_plan(reference, graph, edge_weights, config, arm,
                                     seed, feature_nonzero)
        collector.plan = plan
        collector.original_hops = audit["original_hops"]
        collector.plan_audit = {key: value for key, value in audit.items() if key != "original_hops"}
        return plan

    def make_plan(*args, **kwargs):
        copied_rng = (_clone_numpy_generator(kwargs["rng"])
                      if getattr(collector, "norm_match_reference", False) else None)
        mixed, weight, info = old_make_plan(*args, **kwargs)
        if copied_rng is not None:
            reference_mean, _usage = old_sampler(
                kwargs["data_np"], kwargs["batch_indices"], collector.reference_plan,
                copied_rng, kwargs["neighbor_estimator"])
            anchors = kwargs["data_np"][kwargs["batch_indices"]]
            gate = np.asarray(kwargs["node_gate"][kwargs["batch_indices"]], dtype=np.float32)
            mix_gate = np.clip(gate * float(kwargs["mix_amplitude_multiplier"]), 0.0, 1.0)
            reference_mixed = ((1.0 - mix_gate[:, None]) * anchors +
                               mix_gate[:, None] * reference_mean)
            adjusted, norm_audit = _match_displacement_norm(
                anchors, mixed.detach().cpu().numpy(), reference_mixed)
            mixed = torch.as_tensor(adjusted, dtype=mixed.dtype, device=mixed.device).detach()
            info["mean_actual_displacement"] = float(norm_audit["after_sum"] / max(1, norm_audit["count"]))
            collector.norm_reference_sum += norm_audit["reference_sum"]
            collector.norm_intervention_pre_sum += norm_audit["before_sum"]
            collector.norm_intervention_post_sum += norm_audit["after_sum"]
            collector.norm_unmatched_count += norm_audit["unmatched"]
            collector.norm_scale_gt_one_count += norm_audit["scale_gt_one"]
            collector.norm_observations += norm_audit["count"]
        collector.displacement(kwargs["batch_indices"], mixed, kwargs["batch_x"])
        pseudo_target["value"] = mixed
        return mixed, weight, info

    def h1_estimate(data, batch_indices, graph, edge_weights, sampled_count, rng,
                    estimator="current"):
        if estimator == "full":
            return old_h1_estimator(data, batch_indices, graph, edge_weights,
                                    sampled_count, rng, estimator)
        sampled = np.empty((len(batch_indices), sampled_count), dtype=np.int64)
        weights = np.empty((len(batch_indices), sampled_count), dtype=np.float32)
        for position, sample in enumerate(batch_indices):
            row = graph.indices[sample]
            probs = edge_weights[sample]
            choices = rng.choice(row.shape[0], size=sampled_count, replace=True,
                                 p=probs / np.clip(probs.sum(), 1e-12, None))
            sampled[position] = row[choices]
            selected = probs[choices].astype(np.float32, copy=False)
            weights[position] = (selected / max(float(selected.sum()), 1e-12)
                                 if estimator == "current" else 1.0 / sampled_count)
            ones = np.ones(sampled_count, dtype=np.uint8)
            collector.record(int(sample), sampled[position], ones, ones,
                             weights[position], selected)
        collector.flush_batch()
        return np.sum(data[sampled] * weights[:, :, None], axis=1).astype(np.float32)

    def make_h1(*args, **kwargs):
        mixed, weight, info = old_make_h1(*args, **kwargs)
        collector.displacement(kwargs["batch_indices"], mixed, kwargs["batch_x"])
        return mixed, weight, info

    class MixedTargetAutoEncoder(old_model):
        def loss_mask_weighted(self, x, target, mask, sample_weight=None):
            if sample_weight is not None:
                if pseudo_target["value"] is None:
                    raise RuntimeError("pseudo target missing at loss call")
                target = pseudo_target["value"]
                pseudo_target["value"] = None
            return super().loss_mask_weighted(x, target, mask, sample_weight=sample_weight)

    try:
        if arm == "h1_matched_amplitude":
            mixing.estimate_neighbors = h1_estimate
            trainer.make_pseudo_batch = make_h1
        else:
            trainer.build_exact_hop_plan = build
            hop_sampling.estimate_neighbors_from_plan = _trace_plan_estimator(collector, old_sampler)
            trainer.make_pseudo_batch_from_plan = make_plan
            if arm == "mixed_target":
                trainer.WeightedAutoEncoder = MixedTargetAutoEncoder
        yield
    finally:
        trainer.build_exact_hop_plan = old_build
        hop_sampling.estimate_neighbors_from_plan = old_sampler
        trainer.make_pseudo_batch_from_plan = old_make_plan
        trainer.make_pseudo_batch = old_make_h1
        mixing.estimate_neighbors = old_h1_estimator
        trainer.WeightedAutoEncoder = old_model


def run_task(task_path: str | Path, output_dir: str | Path, device: str) -> dict:
    task = json.loads(Path(task_path).read_text(encoding="utf-8"))
    locked, manifest, reference_config = _validate_task(task)
    arm = "reference" if task["arm"] == "rescue100" else task["arm"]
    seed = int(task["seed"])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source_before = _source_hashes()
    feature_path = Path(manifest["feature_path"])
    if sha256(feature_path) != manifest["feature_sha256"]:
        raise ValueError("feature-only matrix differs from locked manifest")
    loaded = load_npz_matrix_only(feature_path)
    raw_X = loaded.X
    feature_nonzero = (np.diff(raw_X.tocsr().indptr) if sp.issparse(raw_X)
                       else np.count_nonzero(raw_X, axis=1))
    prep = SplitPreprocessor(task["input_kind"], int(task["feature_limit"])).fit(raw_X)
    if prep.fingerprint != locked.get("preprocessing_hash"):
        raise ValueError("preprocessing fingerprint differs from locked selected record")
    X_all = prep.transform(raw_X)
    config = reference_config
    if arm == "h1_matched_amplitude":
        config = replace(reference_config, hop_variant="H1",
                         high_hop_probability_multiplier=1.0)
    historical_reuse = arm == "reference" and isinstance(task["locked_record"], str)
    if historical_reuse:
        source_dir = Path(task["locked_record"]).parent
        with np.load(source_dir / f"predictions_seed_{seed}.npz", allow_pickle=False) as packed:
            predictions = np.asarray(packed["predictions"], dtype=np.int64)
        with np.load(source_dir / f"embedding_seed_{seed}.npz", allow_pickle=False) as packed:
            embedding = np.asarray(packed["embedding"], dtype=np.float32)
        _atomic_copy(source_dir / f"model_seed_{seed}.pt", output / f"model_seed_{seed}.pt")
        _atomic_copy(source_dir / f"predictions_seed_{seed}.npz", output / f"predictions_seed_{seed}.npz")
        _atomic_copy(source_dir / f"embedding_seed_{seed}.npz", output / f"embedding_seed_{seed}.npz")
        trace = {"available": False,
                 "reason": "historical selected run did not persist actual per-step sampled donors",
                 "expected_exposure_analysis": "step2_reconstruction_only"}
        diagnostics = {"hop_sampling": json.loads((source_dir / f"hop_sampling_seed_{seed}.json").read_text(encoding="utf-8"))}
    else:
        collector = TraceCollector(output / f"source_trace_seed_{seed}.bin", X_all.shape[0])
        collector.config = config
        collector.norm_match_reference = bool(task.get("norm_match_reference", False))
        try:
            with _patched_training(arm, seed, feature_nonzero, collector):
                predictions, embedding, diagnostics = trainer.fit_predict(
                    X_all, fit_X=X_all, n_clusters=int(manifest["n_clusters"]),
                    config=config, seed=seed, device=device)
        except BaseException:
            collector.abort()
            raise
        trace = collector.finish()
    source_after = _source_hashes()
    if source_after != source_before:
        raise RuntimeError("historical package source hash changed during intervention")
    if predictions is None or len(predictions) != X_all.shape[0]:
        raise RuntimeError("KMeans readout did not produce all-sample predictions")
    if not np.isfinite(embedding).all():
        raise ValueError("nonfinite embedding")
    _atomic_numpy(output / f"predictions_seed_{seed}.npy", np.asarray(predictions, dtype=np.int64))
    _atomic_numpy(output / f"embedding_seed_{seed}.npy", np.asarray(embedding, dtype=np.float32))
    if not historical_reuse:
        _atomic_npz(output / f"predictions_seed_{seed}.npz", predictions=np.asarray(predictions, dtype=np.int64))
        _atomic_npz(output / f"embedding_seed_{seed}.npz", embedding=np.asarray(embedding, dtype=np.float32))
        _atomic_torch(output / f"model_seed_{seed}.pt", diagnostics["model_state_dict"])
    _atomic_npz(output / f"preprocessor_seed_{seed}.npz",
                features=np.asarray(prep.features, dtype=np.int64),
                mean=np.asarray(prep.scaler.mean_, dtype=np.float64),
                scale=np.asarray(prep.scaler.scale_, dtype=np.float64))
    if diagnostics.get("cluster_centers") is not None:
        _atomic_numpy(output / f"cluster_centers_seed_{seed}.npy",
                      np.asarray(diagnostics["cluster_centers"], dtype=np.float32))
    if not historical_reuse:
        _atomic_json(output / f"training_history_seed_{seed}.json", diagnostics["training_history"])
    _atomic_json(output / f"hop_sampling_seed_{seed}.json", diagnostics["hop_sampling"])
    _atomic_json(output / f"source_trace_audit_seed_{seed}.json", trace)
    _atomic_json(output / f"preprocessor_seed_{seed}.json",
                 {"fingerprint": prep.fingerprint, "input_kind": task["input_kind"],
                  "feature_limit": int(task["feature_limit"]), "selection": prep.selection,
                  "width": int(prep.width)})
    record = {
        "protocol_id": PROTOCOL, "status": "completed", "dataset": task["dataset"],
        "seed": seed, "arm": arm, "requested_arm": task["arm"],
        "norm_match_reference": bool(task.get("norm_match_reference", False)),
        "task_sha256": sha256(Path(task_path)),
        "config_sha256": _canonical_sha256(asdict(config)),
        "locked_config_sha256": _canonical_sha256(locked["config"]),
        "config": asdict(config), "reference_config": asdict(reference_config),
        "locked_record_path": task["locked_record"] if isinstance(task["locked_record"], str) else None,
        "tuning_manifest_path": task["tuning_manifest"] if isinstance(task["tuning_manifest"], str) else None,
        "feature_sha256": manifest["feature_sha256"],
        "preprocessing_hash": prep.fingerprint,
        "n_clusters": int(manifest["n_clusters"]),
        "K_provenance": manifest["K_provenance"],
        "evaluation_mode": "transductive_feature_only",
        "graph_scope": "all_samples_unlabeled",
        "labels_used_during_fit": False, "labels_used_for_selection": False,
        "test_labels_loaded": False, "full_x_labels_loaded": False,
        "posthoc_full_x_labels_loaded": False,
        "n_samples": int(X_all.shape[0]), "n_features": int(X_all.shape[1]),
        "source_hashes_before": source_before, "source_hashes_after": source_after,
        "training_trace": trace,
        "historical_reference_reused": historical_reuse,
        "plan_audit": {} if historical_reuse else getattr(collector, "plan_audit", {}),
        "hop_sampling": diagnostics["hop_sampling"],
        "artifact_sha256": {name: sha256(output / name) for name in (
            [f"predictions_seed_{seed}.npy", f"embedding_seed_{seed}.npy",
             f"predictions_seed_{seed}.npz", f"embedding_seed_{seed}.npz",
             f"model_seed_{seed}.pt", f"preprocessor_seed_{seed}.npz"] +
            ([] if historical_reuse else [f"source_trace_seed_{seed}.bin", "displacement_by_anchor.npz"]))},
    }
    _atomic_json(output / f"record_seed_{seed}.json", record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="run one frozen feature-only causal intervention")
    parser.add_argument("--task-json", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run_task(args.task_json, args.out, args.device)


if __name__ == "__main__":
    main()
