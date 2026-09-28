"""Feature-only single-task runner for the 5-arm source-prior x recovery-target design.

Subcommands
-----------
``train``      one frozen (dataset, arm, seed) task; formal output.
``refcheck``   CPU equivalence checks on the real dataset with a short epoch
               override; outputs are flagged smoke=True and never count as formal.
``sources``    training-free shell-prior statistics (alpha 0.1/0.2/0.5) on the
               exact candidate shells of each new seed's frozen graph.

The historical ``v0rg_tunable_multihop`` package is imported unmodified.  All
changes are process-local monkeypatches restored in ``finally``:

* H1 arm: ``mixing.estimate_neighbors`` replaced by a numerically identical
  traced copy (same rng.choice calls in the same order).
* multihop arms: ``trainer.build_exact_hop_plan`` wraps the original and (for
  geometric arms) reshapes shell masses; ``hop_sampling.estimate_neighbors_from_plan``
  replaced by a numerically identical traced copy.
* mixed arms: ``trainer.WeightedAutoEncoder`` subclass whose pseudo loss call
  (the only call with ``sample_weight``) reconstructs the mixed input x' instead
  of the anchor.  The real-input branch is untouched.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

EXP = Path(__file__).resolve().parent
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))
from common import (ARM_SPECS, TRAIN_REQUIRED_ARTIFACTS, atomic_json, atomic_npy, atomic_npz,  # noqa: E402
                    atomic_torch, canonical_sha256, imported_core_hashes, json_safe, read_json, sha256)
import factorial_plan as fp  # noqa: E402

PROTOCOL = "source_coverage_20260925_factorial_v1"
TRACE_DTYPE = np.dtype([("epoch", "<u2"), ("anchor", "<i4"), ("source", "<i4"), ("hop", "u1"),
                        ("weight", "<f4"), ("slot_probability", "<f4")])
FORBIDDEN_TASK_KEYS = ("labels_path", "label_path", "test_labels", "validation_labels_path",
                       "full_labels", "posthoc_label")


# ---------------------------------------------------------------- task checks

def load_task(task_path: Path) -> dict:
    task = read_json(task_path)
    for key in task:
        if any(bad in key for bad in FORBIDDEN_TASK_KEYS):
            raise ValueError(f"fitting task exposes a label field: {key}")
    required = {"id", "dataset", "arm", "seed", "config", "reference_multihop_config", "prior",
                "target", "feature_path", "feature_sha256", "n_clusters", "K_provenance",
                "input_kind", "feature_limit", "expected_preprocessing_hash"}
    missing = required - set(task)
    if missing:
        raise ValueError(f"task missing fields {sorted(missing)}")
    if task["arm"] not in ARM_SPECS:
        raise ValueError(f"unknown arm {task['arm']}")
    spec = ARM_SPECS[task["arm"]]
    if task["target"] != spec["target"] or task["prior"]["kind"] != spec["prior"]:
        raise ValueError("task prior/target disagree with the arm specification")
    if task["K_provenance"] != "benchmark_oracle_from_y":
        raise ValueError("K provenance must be the explicit benchmark oracle")
    from v0rg_tunable_multihop.config import V0_RGConfig

    config = V0_RGConfig(**task["config"])
    reference = V0_RGConfig(**task["reference_multihop_config"])
    if fp.resolve_ok(reference.hop_variant) is False:
        raise ValueError("reference config must be an exact-shell multihop variant")
    allowed = {"hop_variant", "high_hop_probability_multiplier"} if task["arm"] == "H1_anchor" else set()
    diff = set(fp.config_difference(asdict(config), asdict(reference)))
    if not diff <= allowed:
        raise ValueError(f"arm config differs from locked config beyond hop fields: {sorted(diff)}")
    if task["arm"] == "H1_anchor" and config.hop_variant != "H1":
        raise ValueError("H1 arm must use hop_variant H1")
    if spec["prior"] == "flat_q" and float(task["prior"]["q"]) != float(reference.high_hop_probability_multiplier):
        raise ValueError("flat prior q must equal the locked q")
    if spec["prior"] == "geometric" and float(task["prior"]["alpha"]) != fp.GEOMETRIC_ALPHA:
        raise ValueError("geometric arm alpha is frozen at 0.2")
    return task


def load_features(task: dict):
    from v0rg_tunable_multihop.input_adapter import load_npz_matrix_only
    from v0rg_tunable_multihop.tuning import SplitPreprocessor

    feature_path = Path(task["feature_path"])
    if sha256(feature_path) != task["feature_sha256"]:
        raise ValueError("feature-only matrix hash differs from task")
    loaded = load_npz_matrix_only(feature_path)
    if loaded.labels is not None or loaded.profile.get("labels_accessed") is not False:
        raise RuntimeError("feature loader touched labels")
    prep = SplitPreprocessor(task["input_kind"], int(task["feature_limit"])).fit(loaded.X)
    if prep.fingerprint != task["expected_preprocessing_hash"]:
        raise ValueError("preprocessing fingerprint differs from the locked historical record")
    return prep, prep.transform(loaded.X)


# ---------------------------------------------------------------- tracing

class TraceCollector:
    """Full per-draw trace (binary) plus exact running aggregates."""

    def __init__(self, path: Path | None, n_samples: int, max_hop: int = 8):
        self.path = path
        self.tmp = None if path is None else path.with_name(f".{path.name}.{os.getpid()}.tmp")
        self.handle = None if path is None else self.tmp.open("wb")
        self.buffer: list[np.ndarray] = []
        self.n = n_samples
        self.max_hop = max_hop
        self.epoch = 0
        self.draws = 0
        self.occurrences = 0
        self.hop_weight_mass = np.zeros(max_hop + 1, dtype=np.float64)
        self.hop_draws = np.zeros(max_hop + 1, dtype=np.int64)
        self.hop_gate_mass = np.zeros(max_hop + 1, dtype=np.float64)
        self.anchor_far_mass = np.zeros(n_samples, dtype=np.float64)
        self.anchor_occ = np.zeros(n_samples, dtype=np.int64)
        self.disp_sum = np.zeros(n_samples, dtype=np.float64)
        self.disp_sq = np.zeros(n_samples, dtype=np.float64)
        self.disp_zero = 0
        self.disp_hist_edges = np.concatenate([[0.0], np.geomspace(1e-4, 1e3, 70)])
        self.disp_hist = np.zeros(len(self.disp_hist_edges), dtype=np.int64)
        self.gate = None

    def record(self, anchor: int, nodes: np.ndarray, hops: np.ndarray, weights: np.ndarray,
               probs: np.ndarray) -> None:
        if self.handle is not None:
            block = np.empty(len(nodes), dtype=TRACE_DTYPE)
            block["epoch"] = self.epoch
            block["anchor"] = anchor
            block["source"] = nodes
            block["hop"] = hops
            block["weight"] = weights
            block["slot_probability"] = probs
            self.buffer.append(block)
        self.draws += len(nodes)
        self.occurrences += 1
        self.anchor_occ[anchor] += 1
        hops = np.asarray(hops, dtype=np.int64)
        np.add.at(self.hop_weight_mass, hops, weights)
        np.add.at(self.hop_draws, hops, 1)
        g = 0.0 if self.gate is None else float(self.gate[anchor])
        np.add.at(self.hop_gate_mass, hops, g * np.asarray(weights, dtype=np.float64))
        self.anchor_far_mass[anchor] += float(np.asarray(weights)[hops >= 2].sum())

    def flush(self) -> None:
        if self.buffer and self.handle is not None:
            np.concatenate(self.buffer).tofile(self.handle)
        self.buffer.clear()

    def displacement(self, batch_indices: np.ndarray, mixed, anchor) -> None:
        import torch

        values = torch.linalg.vector_norm(mixed.detach() - anchor.detach(), dim=1).cpu().numpy()
        values = values.astype(np.float64)
        np.add.at(self.disp_sum, batch_indices, values)
        np.add.at(self.disp_sq, batch_indices, values * values)
        self.disp_zero += int(np.count_nonzero(values <= 1e-12))
        bins = np.clip(np.searchsorted(self.disp_hist_edges, values, side="right") - 1, 0,
                       len(self.disp_hist) - 1)
        np.add.at(self.disp_hist, bins, 1)

    def finish(self, out: Path | None) -> dict:
        self.flush()
        if self.handle is not None:
            self.handle.close()
            self.tmp.replace(self.path)
        count = np.maximum(self.anchor_occ, 1)
        if out is not None:
            atomic_npz(out / "displacement_by_anchor.npz", sum=self.disp_sum, sumsq=self.disp_sq,
                       count=self.anchor_occ, mean=self.disp_sum / count,
                       far_mass_mean=self.anchor_far_mass / count,
                       hist_edges=self.disp_hist_edges, hist_counts=self.disp_hist)
        total_w = max(float(self.hop_weight_mass.sum()), 1e-300)
        occ = max(1, self.occurrences)
        return {"trace_file": None if self.path is None else self.path.name,
                "trace_dtype": [list(x) for x in TRACE_DTYPE.descr],
                "trace_complete": self.path is not None,
                "draws": int(self.draws), "anchor_training_occurrences": int(self.occurrences),
                "actual_hop_weight_mass_per_occurrence": {
                    str(h): float(self.hop_weight_mass[h] / occ) for h in range(1, self.max_hop + 1)},
                "actual_hop_weight_fraction": {
                    str(h): float(self.hop_weight_mass[h] / total_w) for h in range(1, self.max_hop + 1)},
                "actual_hop_draw_fraction": {
                    str(h): float(self.hop_draws[h] / max(1, self.draws)) for h in range(1, self.max_hop + 1)},
                "gate_weighted_hop_mass_per_occurrence": {
                    str(h): float(self.hop_gate_mass[h] / occ) for h in range(1, self.max_hop + 1)},
                "mean_actual_displacement": float(self.disp_sum.sum() / occ),
                "zero_displacement_occurrences": int(self.disp_zero)}

    def abort(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.tmp.unlink(missing_ok=True)


def traced_plan_estimator(collector: TraceCollector):
    """Numerically identical copy of hop_sampling.estimate_neighbors_from_plan with tracing."""
    def estimate(data, batch_indices, plan, rng, estimator):
        if estimator not in {"current", "uniform_sample", "full"}:
            raise ValueError("unknown neighbor estimator")
        output = np.empty((len(batch_indices), data.shape[1]), dtype=np.float32)
        mass = defaultdict(float)
        draw_hits = defaultdict(int)
        total_draws = 0
        known_hops = sorted({int(hop) for row in plan.hop_labels for hop in np.unique(row)})
        per_sample_mass = {hop: np.zeros(len(batch_indices), dtype=np.float64) for hop in known_hops}
        for position, source in enumerate(batch_indices):
            nodes = plan.indices[int(source)]
            probabilities = plan.probs[int(source)]
            labels = plan.hop_labels[int(source)]
            if estimator == "full":
                output[position] = np.sum(data[nodes] * probabilities[:, None], axis=0)
                for hop in known_hops:
                    contribution = float(probabilities[labels == hop].sum())
                    mass[hop] += contribution
                    per_sample_mass[hop][position] = contribution
                collector.record(int(source), nodes, labels, probabilities, probabilities)
                continue
            choices = rng.choice(nodes.size, size=plan.draw_count, replace=True, p=probabilities)
            selected_probability = probabilities[choices]
            weights = (
                selected_probability / max(float(selected_probability.sum()), 1e-12)
                if estimator == "current"
                else np.full(plan.draw_count, 1.0 / float(plan.draw_count), dtype=np.float64)
            )
            output[position] = np.sum(data[nodes[choices]] * weights[:, None], axis=0)
            selected_hops = labels[choices]
            collector.record(int(source), nodes[choices], selected_hops, weights, selected_probability)
            for hop in known_hops:
                mask = selected_hops == hop
                contribution = float(weights[mask].sum())
                mass[hop] += contribution
                per_sample_mass[hop][position] = contribution
                draw_hits[hop] += int(mask.sum())
            total_draws += int(plan.draw_count)
        collector.flush()
        normalizer = max(1, len(batch_indices))
        return output, {
            "actual_sampling_mass": {str(hop): float(mass[hop] / normalizer) for hop in known_hops},
            "draw_fraction": {
                str(hop): (None if estimator == "full" else float(draw_hits[hop] / max(1, total_draws)))
                for hop in known_hops
            },
            "_per_sample_actual_sampling_mass": {str(hop): values for hop, values in per_sample_mass.items()},
        }
    return estimate


def traced_h1_estimator(collector: TraceCollector):
    """Numerically identical copy of mixing.estimate_neighbors with tracing."""
    def estimate(data, batch_indices, graph, edge_weights, sampled_count, rng, estimator="current"):
        if estimator not in {"current", "uniform_sample", "full"}:
            raise ValueError("unknown neighbor estimator")
        if estimator == "full":
            for i in batch_indices:
                ones = np.ones(graph.indices.shape[1], dtype=np.uint8)
                collector.record(int(i), graph.indices[i], ones, edge_weights[i], edge_weights[i])
            collector.flush()
            return np.asarray([np.sum(data[graph.indices[i]] * edge_weights[i, :, None], axis=0)
                               for i in batch_indices], dtype=np.float32)
        sampled = np.empty((len(batch_indices), sampled_count), dtype=np.int64)
        weights = np.empty((len(batch_indices), sampled_count), dtype=np.float32)
        ones = np.ones(sampled_count, dtype=np.uint8)
        for position, sample in enumerate(batch_indices):
            row = graph.indices[sample]
            probs = edge_weights[sample]
            choices = rng.choice(row.shape[0], size=sampled_count, replace=True,
                                 p=probs / np.clip(probs.sum(), 1e-12, None))
            sampled[position] = row[choices]
            selected = probs[choices].astype(np.float32, copy=False)
            weights[position] = (selected / max(float(selected.sum()), 1e-12)
                                 if estimator == "current" else 1.0 / sampled_count)
            collector.record(int(sample), sampled[position], ones, weights[position], selected)
        collector.flush()
        return np.sum(data[sampled] * weights[:, :, None], axis=1).astype(np.float32)
    return estimate


@contextmanager
def patched_training(arm: str, prior: dict, collector: TraceCollector | None, captured: dict):
    """Process-local patches; ``arm=None`` means no patch at all (pure reference)."""
    from v0rg_tunable_multihop import hop_sampling, mixing, trainer

    saved = {"build": trainer.build_exact_hop_plan, "plan_est": hop_sampling.estimate_neighbors_from_plan,
             "h1_est": mixing.estimate_neighbors, "make_plan": trainer.make_pseudo_batch_from_plan,
             "make_h1": trainer.make_pseudo_batch, "model": trainer.WeightedAutoEncoder,
             "gate": trainer.compute_node_gate}
    spec = ARM_SPECS[arm] if arm is not None else None
    pseudo = {"value": None}
    captured.setdefault("target_calls", {"real": 0, "mixed": 0})

    def build(graph, edge_weights, hop_variant, mix_neighbors, q=1.0, *a, **k):
        reference = saved["build"](graph, edge_weights, hop_variant, mix_neighbors, q, *a, **k)
        captured["reference_plan"] = reference
        captured["graph_indices"] = np.asarray(graph.indices)
        captured["edge_weights"] = np.asarray(edge_weights)
        if spec["prior"] == "geometric":
            plan = fp.reshape_plan(reference, "geometric", float(prior["q"]), float(prior["alpha"]))
            fp.verify_same_candidates(reference, plan)
        else:
            plan = reference
        captured["plan"] = plan
        return plan

    def gate(*a, **k):
        result = saved["gate"](*a, **k)
        if collector is not None:
            collector.gate = np.clip(np.asarray(result[0], dtype=np.float64) *
                                     collector.amplitude, 0.0, 1.0)
        return result

    def make_plan(*a, **k):
        mixed, weight, info = saved["make_plan"](*a, **k)
        if collector is not None:
            collector.displacement(k["batch_indices"], mixed, k["batch_x"])
        pseudo["value"] = mixed
        return mixed, weight, info

    def make_h1(*a, **k):
        mixed, weight, info = saved["make_h1"](*a, **k)
        if collector is not None:
            collector.displacement(k["batch_indices"], mixed, k["batch_x"])
        pseudo["value"] = mixed
        return mixed, weight, info

    class MixedTargetAutoEncoder(saved["model"]):
        def loss_mask_weighted(self, x, target, mask, sample_weight=None):
            if sample_weight is not None:
                if pseudo["value"] is None:
                    raise RuntimeError("mixed target missing at pseudo loss call")
                target = pseudo["value"]
                pseudo["value"] = None
                captured["target_calls"]["mixed"] += 1
            elif self.training:
                captured["target_calls"]["real"] += 1
            return super().loss_mask_weighted(x, target, mask, sample_weight=sample_weight)

    try:
        if spec is not None:
            trainer.compute_node_gate = gate
            trainer.make_pseudo_batch = make_h1
            trainer.make_pseudo_batch_from_plan = make_plan
            if spec["candidates"] == "H1":
                if collector is not None:
                    mixing.estimate_neighbors = traced_h1_estimator(collector)
            else:
                trainer.build_exact_hop_plan = build
                if collector is not None:
                    hop_sampling.estimate_neighbors_from_plan = traced_plan_estimator(collector)
            if spec["target"] == "mixed":
                trainer.WeightedAutoEncoder = MixedTargetAutoEncoder
        yield
    finally:
        trainer.build_exact_hop_plan = saved["build"]
        hop_sampling.estimate_neighbors_from_plan = saved["plan_est"]
        mixing.estimate_neighbors = saved["h1_est"]
        trainer.make_pseudo_batch_from_plan = saved["make_plan"]
        trainer.make_pseudo_batch = saved["make_h1"]
        trainer.WeightedAutoEncoder = saved["model"]
        trainer.compute_node_gate = saved["gate"]


def fit_arm(X_all: np.ndarray, config_dict: dict, arm: str | None, prior: dict, seed: int, device: str,
            n_clusters: int, trace_path: Path | None, out: Path | None) -> dict:
    """Run trainer.fit_predict with the arm's patches.  ``arm=None`` -> unpatched reference."""
    from torch.utils.data import RandomSampler
    from v0rg_tunable_multihop import trainer
    from v0rg_tunable_multihop.config import V0_RGConfig

    config = V0_RGConfig(**config_dict)
    collector = None if arm is None else TraceCollector(trace_path, X_all.shape[0])
    if collector is not None:
        collector.amplitude = float(config.mix_amplitude_multiplier)
    captured: dict = {}
    loader_original = trainer.DataLoader
    epoch_state = {"n": 0}

    class EpochLoader(loader_original):
        # Only the training loader shuffles (RandomSampler); embedding extraction is
        # sequential.  Counting iterations consumes no random numbers.
        def __iter__(self):
            if collector is not None and isinstance(self.sampler, RandomSampler):
                epoch_state["n"] += 1
                collector.epoch = epoch_state["n"]
            return super().__iter__()

    try:
        with patched_training(arm, prior, collector, captured):
            if collector is not None:
                trainer.DataLoader = EpochLoader
            try:
                predictions, embedding, diagnostics = trainer.fit_predict(
                    X_all, fit_X=X_all, n_clusters=int(n_clusters), config=config, seed=int(seed), device=device)
            finally:
                trainer.DataLoader = loader_original
    except BaseException:
        if collector is not None:
            collector.abort()
        raise
    trace = collector.finish(out) if collector is not None else None
    return {"predictions": predictions, "embedding": embedding, "diagnostics": diagnostics,
            "captured": captured, "trace": trace, "epochs_traced": epoch_state["n"]}


# ---------------------------------------------------------------- train

def configuration_selection_scope(task: dict, validation_lock: Path | None = None) -> str:
    """Identify the frozen configuration's provenance, separate from label-free fitting."""
    lock = validation_lock or EXP / "outputs" / "p0" / "selection_lock_v2.json"
    supplied = task.get("selection_lock_sha256")
    if supplied is None:  # standalone synthetic fit tests have no selection step
        return "unspecified_synthetic"
    if lock.is_file() and supplied == sha256(lock):
        return "historical_validation_only"
    if supplied == canonical_sha256(task["reference_multihop_config"]):
        return "fixed_exploratory_policy"
    raise ValueError("unrecognized configuration selection lock; refusing ambiguous provenance")

def train(args) -> None:
    task_path = Path(args.task_json).resolve()
    task = load_task(task_path)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for stale in ("record.json",):
        if (out / stale).exists():
            raise RuntimeError("output directory already holds a record; the wrapper archives attempts")
    source_before = imported_core_hashes()
    prep, X_all = load_features(task)
    started = time.time()
    result = fit_arm(X_all, task["config"], task["arm"], task["prior"], int(task["seed"]), args.device,
                     int(task["n_clusters"]), out / "source_trace.bin", out)
    predictions, embedding, diagnostics = result["predictions"], result["embedding"], result["diagnostics"]
    source_after = imported_core_hashes()
    if source_after != source_before:
        raise RuntimeError("historical source changed during training")
    config = task["config"]
    if result["epochs_traced"] != int(config["epochs"]):
        raise RuntimeError(f"epoch counter {result['epochs_traced']} != {config['epochs']}")
    if predictions is None or len(predictions) != X_all.shape[0] or not np.isfinite(embedding).all():
        raise RuntimeError("missing or non-finite outputs")
    spec = ARM_SPECS[task["arm"]]
    calls = result["captured"]["target_calls"]
    if spec["target"] == "mixed" and (calls["mixed"] == 0 or calls["mixed"] != calls["real"]):
        raise RuntimeError(f"mixed target not applied on every pseudo loss: {calls}")
    graph_indices = np.asarray(diagnostics["neighbor_indices"])
    edge_weights = np.asarray(diagnostics["edge_weights"])
    if spec["candidates"] == "H1":
        arrays = fp.h1_plan_arrays(graph_indices, edge_weights,
                                   max(1, min(int(config["mix_neighbors"]), graph_indices.shape[1])))
        plan_audit = {"plan": "legacy_one_hop", **fp.plan_fingerprints(arrays)}
    else:
        plan = result["captured"]["plan"]
        reference = result["captured"]["reference_plan"]
        arrays = fp.plan_arrays(plan)
        ref_arrays = fp.plan_arrays(reference)
        plan_audit = {"plan": spec["prior"], **fp.plan_fingerprints(arrays),
                      "reference_flat": fp.plan_fingerprints(ref_arrays),
                      "same_candidates_as_flat": fp.verify_same_candidates(reference, plan),
                      "expected_shell_mass": fp.expected_shell_mass(arrays),
                      "reference_expected_shell_mass": fp.expected_shell_mass(ref_arrays),
                      "profile": json_safe(plan.profile)}
    atomic_npz(out / "plan.npz", **arrays)
    atomic_npz(out / "graph.npz", indices=graph_indices, edge_weights=edge_weights,
               node_gate=np.asarray(diagnostics["node_gate"]),
               similarity=np.asarray(diagnostics["neighbor_similarity"]))
    atomic_json(out / "plan_audit.json", json_safe(plan_audit))
    atomic_npy(out / "embedding.npy", np.asarray(embedding, dtype=np.float32))
    atomic_npy(out / "predictions.npy", np.asarray(predictions, dtype=np.int64))
    atomic_npy(out / "cluster_centers.npy", np.asarray(diagnostics["cluster_centers"], dtype=np.float32))
    atomic_torch(out / "model.pt", diagnostics["model_state_dict"])
    atomic_npz(out / "preprocessor.npz", features=np.asarray(prep.features, dtype=np.int64),
               mean=np.asarray(prep.scaler.mean_, dtype=np.float64),
               scale=np.asarray(prep.scaler.scale_, dtype=np.float64))
    atomic_json(out / "preprocessor.json", {"fingerprint": prep.fingerprint, "input_kind": task["input_kind"],
                                            "feature_limit": int(task["feature_limit"]),
                                            "selection": json_safe(prep.selection), "width": int(prep.width)})
    atomic_json(out / "training_history.json", json_safe(diagnostics["training_history"]))
    atomic_json(out / "hop_sampling.json", json_safe(diagnostics["hop_sampling"]))
    missing = [n for n in TRAIN_REQUIRED_ARTIFACTS if not (out / n).is_file()]
    if missing:
        raise RuntimeError(f"required artifacts missing: {missing}")
    import torch

    # The fit itself never reads labels.  Main-case configuration selection did
    # use historical validation labels; the exploratory transfer policy did not.
    selection_scope = configuration_selection_scope(task)
    selected_with_validation = selection_scope == "historical_validation_only"
    record = {
        "protocol_id": PROTOCOL, "status": "completed", "smoke": False, "epochs_override": None,
        "task_id": task["id"], "task_sha256": sha256(task_path), "dataset": task["dataset"],
        "arm": task["arm"], "arm_spec": spec, "seed": int(task["seed"]), "prior": task["prior"],
        "config": config, "config_sha256": canonical_sha256(config),
        "reference_multihop_config_sha256": canonical_sha256(task["reference_multihop_config"]),
        "feature_sha256": task["feature_sha256"], "preprocessing_hash": prep.fingerprint,
        "n_samples": int(X_all.shape[0]), "n_features": int(X_all.shape[1]),
        "n_clusters": int(task["n_clusters"]), "K_provenance": task["K_provenance"],
        "evaluation_mode": "transductive_feature_only", "graph_scope": "all_samples_unlabeled",
        "labels_used_during_fit": False,
        "labels_used_for_selection": bool(selected_with_validation),
        "configuration_selection_scope": selection_scope,
        "test_labels_loaded": False, "full_labels_loaded": False, "validation_labels_loaded": False,
        "target_calls": calls, "training_trace": result["trace"], "plan_audit_sha256": sha256(out / "plan_audit.json"),
        "source_hashes_before": source_before, "source_hashes_after": source_after,
        "runner_sha256": sha256(Path(__file__).resolve()), "plan_module_sha256": sha256(EXP / "factorial_plan.py"),
        "common_sha256": sha256(EXP / "common.py"),
        "device": args.device, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "wall_seconds": time.time() - started,
        "artifact_sha256": {name: sha256(out / name) for name in TRAIN_REQUIRED_ARTIFACTS},
    }
    atomic_json(out / "record.json", json_safe(record))


# ---------------------------------------------------------------- refcheck (smoke, CPU)

def refcheck(args) -> None:
    """Equivalence of wrappers vs. the unmodified trainer on the real matrix, short epochs."""
    tasks = [load_task(Path(p).resolve()) for p in args.task_json]
    by_arm = {t["arm"]: t for t in tasks}
    if set(by_arm) != set(ARM_SPECS) or len({t["seed"] for t in tasks}) != 1 or len({t["dataset"] for t in tasks}) != 1:
        raise ValueError("refcheck needs the five arms of one dataset and one seed")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    epochs = int(args.epochs)
    source_before = imported_core_hashes()
    prep, X_all = load_features(tasks[0])
    seed = int(tasks[0]["seed"])
    k = int(tasks[0]["n_clusters"])
    checks, details = {}, {}

    def short(cfg):
        c = dict(cfg)
        c["epochs"] = epochs
        return c

    runs = {}
    for name, arm in (("H1_unpatched", None), ("H1_anchor", "H1_anchor"),
                      ("flat_unpatched", None), ("flat_anchor", "flat_anchor"),
                      ("geometric_anchor", "geometric_anchor"), ("flat_mixed", "flat_mixed"),
                      ("geometric_mixed", "geometric_mixed")):
        source_arm = "H1_anchor" if name.startswith("H1") else (arm or "flat_anchor")
        t = by_arm[source_arm]
        runs[name] = fit_arm(X_all, short(t["config"]), arm, t["prior"], seed, "cpu", k,
                             out / f"trace_{name}.bin" if arm else None, None)
    for ref, new in (("H1_unpatched", "H1_anchor"), ("flat_unpatched", "flat_anchor")):
        a, b = runs[ref], runs[new]
        same_emb = bool(np.array_equal(a["embedding"], b["embedding"]))
        same_pred = bool(np.array_equal(a["predictions"], b["predictions"]))
        same_state = all(bool((a["diagnostics"]["model_state_dict"][key] ==
                               b["diagnostics"]["model_state_dict"][key]).all())
                         for key in a["diagnostics"]["model_state_dict"])
        checks[f"{new}_bitwise_equals_unpatched"] = same_emb and same_pred and same_state
        details[new] = {"embedding_equal": same_emb, "predictions_equal": same_pred, "state_equal": same_state,
                        "max_abs_embedding_diff": float(np.max(np.abs(a["embedding"] - b["embedding"])))}
    # Frozen-graph reconstruction used by analyses must equal the trainer's own graph.
    frozen = fp.build_frozen_graph(X_all, fp_config(by_arm["flat_anchor"]["config"]), seed)
    diag = runs["flat_unpatched"]["diagnostics"]
    checks["frozen_graph_matches_trainer"] = bool(
        np.array_equal(frozen["graph"].indices, diag["neighbor_indices"]) and
        np.array_equal(frozen["edge_weights"], diag["edge_weights"]) and
        np.array_equal(frozen["node_gate"], diag["node_gate"]))
    flat_plan = runs["flat_anchor"]["captured"]["plan"]
    for name in ("geometric_anchor", "flat_mixed", "geometric_mixed"):
        plan = runs[name]["captured"]["plan"]
        details[f"{name}_candidates"] = fp.verify_same_candidates(flat_plan, plan)
        checks[f"{name}_same_candidates_as_flat"] = True
    geo = runs["geometric_anchor"]["captured"]["plan"]
    q = float(by_arm["flat_anchor"]["prior"]["q"])
    max_norm_err, max_prior_err = 0.0, 0.0
    for probs, labels in zip(geo.probs, geo.hop_labels):
        max_norm_err = max(max_norm_err, abs(float(probs.sum()) - 1.0))
        available, mass, _ = fp.row_decomposition(np.asarray(probs, dtype=np.float64), labels)
        target = fp.shell_masses(available, "geometric", q=q, alpha=fp.GEOMETRIC_ALPHA)
        max_prior_err = max(max_prior_err, float(np.max(np.abs(target - [mass[int(h)] for h in available]))))
    checks["geometric_normalized_over_nonempty_shells"] = max_norm_err < 1e-9 and max_prior_err < 1e-9
    details["geometric"] = {"max_row_sum_error": max_norm_err, "max_shell_prior_error": max_prior_err}
    # Mixed arms: pseudo target is the mixed input; real branch untouched.
    for name in ("flat_mixed", "geometric_mixed"):
        calls = runs[name]["captured"]["target_calls"]
        checks[f"{name}_mixed_target_every_pseudo_call"] = calls["mixed"] > 0 and calls["mixed"] == calls["real"]
        details[f"{name}_target_calls"] = calls
    checks["mixed_changes_training"] = not np.array_equal(runs["flat_mixed"]["embedding"],
                                                          runs["flat_anchor"]["embedding"])
    for name, run in runs.items():
        if run["trace"] is not None:
            trace = run["trace"]
            checks[f"{name}_trace_covers_all_occurrences"] = (
                trace["anchor_training_occurrences"] == X_all.shape[0] * epochs and run["epochs_traced"] == epochs)
            details[f"{name}_trace"] = trace
    checks["core_source_unchanged"] = imported_core_hashes() == source_before
    checks["labels_not_loaded"] = True  # only load_npz_matrix_only is used in this process
    summary = {"status": "completed", "smoke": True, "counts_as_formal": False,
               "epochs_override": epochs, "dataset": tasks[0]["dataset"], "seed": seed,
               "all_checks_passed": all(bool(v) for v in checks.values()),
               "checks": checks, "details": json_safe(details),
               "runner_sha256": sha256(Path(__file__).resolve()),
               "task_sha256": {t["arm"]: sha256(Path(p).resolve()) for t, p in zip(tasks, args.task_json)}}
    atomic_json(out / "summary.json", json_safe(summary))
    if not summary["all_checks_passed"]:
        raise SystemExit("reference checks FAILED; see summary.json")


def fp_config(config_dict: dict):
    from v0rg_tunable_multihop.config import V0_RGConfig

    return V0_RGConfig(**config_dict)


# ---------------------------------------------------------------- source-only alpha sensitivity

def sources(args) -> None:
    from v0rg_tunable_multihop.hop_sampling import build_exact_hop_plan

    tasks = [load_task(Path(p).resolve()) for p in args.task_json]
    ref_tasks = [t for t in tasks if t["arm"] == "flat_anchor"]
    if not ref_tasks:
        raise ValueError("sources needs flat_anchor tasks")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prep, X_all = load_features(ref_tasks[0])
    rows = {}
    for t in sorted(ref_tasks, key=lambda x: x["seed"]):
        config = fp_config(t["config"])
        frozen = fp.build_frozen_graph(X_all, config, int(t["seed"]))
        reference = build_exact_hop_plan(frozen["graph"], frozen["edge_weights"], config.hop_variant,
                                         config.mix_neighbors, config.high_hop_probability_multiplier)
        rows[str(t["seed"])] = fp.source_prior_statistics(reference, float(t["prior"]["q"]))
    atomic_json(out / "source_prior_sensitivity.json", json_safe(rows))
    atomic_json(out / "summary.json", {"status": "completed", "nature": "training-free prior statistics",
                                       "alphas": list(fp.SENSITIVITY_ALPHAS),
                                       "output_sha256": {"source_prior_sensitivity.json":
                                                         sha256(out / "source_prior_sensitivity.json")}})


def main() -> None:
    parser = argparse.ArgumentParser(description="source-coverage factorial runner")
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--task-json", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--device", default="cuda:0")
    r = sub.add_parser("refcheck")
    r.add_argument("--task-json", nargs=5, required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--epochs", type=int, default=2)
    s = sub.add_parser("sources")
    s.add_argument("--task-json", nargs="+", required=True)
    s.add_argument("--out", required=True)
    args = parser.parse_args()
    {"train": train, "refcheck": refcheck, "sources": sources}[args.command](args)


if __name__ == "__main__":
    main()
