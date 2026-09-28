"""Finite-difference direction probe for matched H1 / multihop V0_RG checkpoints.

Feature-only and label-free. Two completed runs (``--reference-record`` must be
H1, ``--record`` a multihop variant) are restored strictly from their own
config, fitted preprocessor and ``model_seed_{seed}.pt``. Equivalence of both
restorations is proven (preprocessing fingerprint, saved-array equality, saved
embedding reproduction, rebuilt hop-plan profile) before any probe value is
computed. Both models then receive the exact same anchors and directions.

The historical ``v0rg_tunable_multihop`` package and ``methods.NeighborMix_scMAE``
must be importable (``--pythonpath`` may add roots). Pure numeric helpers at the
top of this file do not import the package so that they can be unit tested.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import socket
import sys
import time
from typing import Any

import numpy as np


PROTOCOL = "source_coverage_direction_probe_v1"
LAMBDAS = (0.0, 0.25, 0.5, 1.0)
MAX_ANCHORS = 512
ZERO_NORM_EPS = 1e-8
# Only path-policy fields may differ between the matched H1 and multihop runs.
CONFIG_DIFF_ALLOWED = frozenset({"hop_variant", "high_hop_probability_multiplier"})
# Same dtype as step5_interventions.TRACE_DTYPE (per-draw training trace).
TRACE_DTYPE = np.dtype([("anchor", "<i4"), ("source", "<i4"),
                        ("original_hop", "u1"), ("actual_hop", "u1"),
                        ("weight", "<f4"), ("slot_probability", "<f4")])
# Static fields of FixedCandidatePlan.profile (hop_sampling.build_exact_hop_plan);
# usage fields added after training are excluded.
STATIC_PLAN_KEYS = ("variant", "construction", "path_score", "base_neighbor_k", "max_hop",
                    "shell_width_multiplier", "high_hop_probability_multiplier",
                    "fixed_draw_count", "mean_candidate_count", "max_candidate_count",
                    "shell_mean_selected_count", "shell_nonempty_fraction",
                    "path_mass_retained_after_topk")
STATIC_H1_KEYS = ("variant", "construction", "base_neighbor_k", "fixed_draw_count")
# Keys that fix the candidate shells but not their probabilities (geometric-prior arms).
CANDIDATE_IDENTITY_KEYS = ("base_neighbor_k", "max_hop", "shell_width_multiplier", "fixed_draw_count",
                           "mean_candidate_count", "max_candidate_count", "shell_mean_selected_count",
                           "shell_nonempty_fraction")
FAMILIES = ("h1_graph", "plan_all", "plan_far", "trace")


class ProbeBlocked(RuntimeError):
    """A required artifact or equivalence check is missing; no result is produced."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


# ---------------------------------------------------------------- pure helpers

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def global_embedding_scale(embedding: np.ndarray) -> float:
    """RMS radius sqrt(mean_i ||z_i - mean z||^2); translation invariant, 1-homogeneous."""
    values = np.asarray(embedding, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("embedding must be 2-D with at least two rows")
    centered = values - values.mean(axis=0, keepdims=True)
    scale = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("global embedding scale must be finite and positive")
    return scale


def sample_anchors(n_samples: int, max_anchors: int, seed: int) -> np.ndarray:
    count = min(int(max_anchors), int(n_samples))
    if count <= 0:
        raise ValueError("no anchors available")
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(int(n_samples), size=count, replace=False)).astype(np.int64)


def direction_norms(directions: np.ndarray, eps: float = ZERO_NORM_EPS) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(np.asarray(directions, dtype=np.float64), axis=1)
    if not np.all(np.isfinite(norms)):
        raise ValueError("non-finite direction")
    return norms, norms > float(eps)


def unit_random_directions(n_rows: int, width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    raw = rng.standard_normal((int(n_rows), int(width)))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError("degenerate random draw")
    return raw / norms


def matched_random(unit: np.ndarray, source: np.ndarray,
                   eps: float = ZERO_NORM_EPS) -> tuple[np.ndarray, np.ndarray]:
    """Scale shared unit random rows to the per-row norm of the source directions.

    Rows whose source direction has zero norm are zeroed and marked invalid, so
    the random baseline is excluded exactly where the source is excluded.
    """
    if unit.shape != source.shape:
        raise ValueError("random and source directions must have identical shapes")
    norms, valid = direction_norms(source, eps)
    matched = unit * np.where(valid, norms, 0.0)[:, None]
    return matched.astype(np.float32), valid


def strict_restore(model, state: dict) -> dict:
    """Load ``state`` only if keys, shapes and dtypes match exactly and values are finite."""
    import torch

    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ProbeBlocked("checkpoint_not_tensor_state_dict", "model file must hold a flat tensor state_dict")
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    shape = sorted(key for key in set(expected) & set(state)
                   if tuple(expected[key].shape) != tuple(state[key].shape))
    dtype = sorted(key for key in set(expected) & set(state)
                   if expected[key].dtype != state[key].dtype)
    nonfinite = sorted(key for key, value in state.items()
                       if value.is_floating_point() and not bool(torch.isfinite(value).all()))
    problems = {"missing": missing, "unexpected": unexpected, "shape_mismatch": shape,
                "dtype_mismatch": dtype, "nonfinite": nonfinite}
    problems = {key: value for key, value in problems.items() if value}
    if problems:
        raise ProbeBlocked("checkpoint_mismatch", json.dumps(problems))
    model.load_state_dict(state, strict=True)
    restored = model.state_dict()
    if any(not torch.equal(restored[key].cpu(), state[key].cpu()) for key in state):
        raise ProbeBlocked("checkpoint_restore_not_exact", "restored tensors differ from checkpoint")
    return {"n_tensors": len(state),
            "n_parameters": int(sum(int(v.numel()) for v in state.values())),
            "state_sha256": hashlib.sha256(b"".join(
                key.encode() + state[key].cpu().contiguous().numpy().tobytes()
                for key in sorted(state))).hexdigest()}


def assert_eval_mode(model) -> None:
    import torch

    if model.training:
        raise ProbeBlocked("model_in_training_mode", "probe requires model.eval()")
    active = [name for name, module in model.named_modules()
              if isinstance(module, torch.nn.Dropout) and module.training]
    if active:
        raise ProbeBlocked("dropout_active", ",".join(active))


def probe_responses(model, anchors_x: np.ndarray, directions: np.ndarray,
                    lambdas=LAMBDAS, batch_size: int = 256, device="cpu") -> dict:
    """Mask-free finite differences of ``model.forward_mask`` along each direction.

    No scMAE corruption is applied: the input is exactly x + lambda * d. The
    decoder still receives the mask-predictor logits of that input because that
    is the model's forward graph. Returns arrays of shape (len(lambdas), n):
    encoder_response = ||z(x+ld) - z(x)||, and
    decoder_anchor_mse_delta = mean((r(x+ld) - x)^2) - mean((r(x) - x)^2).
    """
    import torch

    model.eval()
    assert_eval_mode(model)
    if anchors_x.shape != directions.shape:
        raise ValueError("anchors and directions must have identical shapes")
    n = int(anchors_x.shape[0])
    response = np.zeros((len(lambdas), n), dtype=np.float64)
    mse_delta = np.zeros((len(lambdas), n), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, n, int(batch_size)):
            stop = min(n, start + int(batch_size))
            x = torch.as_tensor(anchors_x[start:stop], dtype=torch.float32, device=device)
            d = torch.as_tensor(directions[start:stop], dtype=torch.float32, device=device)
            base_latent, _, base_rec = model.forward_mask(x)
            base_mse = ((base_rec - x) ** 2).mean(dim=1)
            for position, value in enumerate(lambdas):
                latent, _, rec = model.forward_mask(x + float(value) * d)
                response[position, start:stop] = torch.linalg.vector_norm(
                    latent - base_latent, dim=1).double().cpu().numpy()
                mse_delta[position, start:stop] = (((rec - x) ** 2).mean(dim=1)
                                                   - base_mse).double().cpu().numpy()
    if not (np.all(np.isfinite(response)) and np.all(np.isfinite(mse_delta))):
        raise ProbeBlocked("nonfinite_probe_output", "forward produced non-finite values")
    for position, value in enumerate(lambdas):
        if float(value) == 0.0 and (np.any(response[position] != 0.0) or np.any(mse_delta[position] != 0.0)):
            raise ProbeBlocked("nondeterministic_forward", "lambda=0 response is not exactly zero")
    return {"encoder_response": response, "decoder_anchor_mse_delta": mse_delta}


def config_difference(first: dict, second: dict) -> dict:
    keys = sorted(set(first) | set(second))
    return {key: [first.get(key), second.get(key)] for key in keys if first.get(key) != second.get(key)}


def compare_profile(saved: dict, rebuilt: dict, keys, rtol: float = 1e-7) -> list[str]:
    """Return mismatching keys between a saved hop profile and the rebuilt one."""
    def same(a, b) -> bool:
        if isinstance(a, dict) and isinstance(b, dict):
            return set(a) == set(b) and all(same(a[k], b[k]) for k in a)
        if isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str) or isinstance(b, str):
            return a == b
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return math.isclose(float(a), float(b), rel_tol=rtol, abs_tol=1e-12)
        return a == b
    return [key for key in keys if key not in saved or key not in rebuilt or not same(saved[key], rebuilt[key])]


def summarize(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"n": 0, "mean": None, "median": None, "fraction_positive": None}
    return {"n": int(values.size), "mean": float(values.mean()), "median": float(np.median(values)),
            "fraction_positive": float(np.mean(values > 0.0))}


# ----------------------------------------------------------- artifact loading

def _require(path: Path, what: str) -> Path:
    if not path.is_file():
        raise ProbeBlocked("missing_artifact", f"{what}: {path}")
    return path


def _read_json(path: Path, what: str) -> dict:
    value = json.loads(_require(path, what).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProbeBlocked("invalid_artifact", f"{what} is not a JSON object: {path}")
    return value


def _import_core():
    try:
        from v0rg_tunable_multihop import graph, hop_sampling, mixing, trainer
        from v0rg_tunable_multihop.config import V0_RGConfig
        from v0rg_tunable_multihop.input_adapter import load_npz_matrix_only
        from v0rg_tunable_multihop.model import WeightedAutoEncoder
        from v0rg_tunable_multihop.tuning import SplitPreprocessor
    except ImportError as error:
        raise ProbeBlocked("core_package_unimportable", repr(error)) from error
    return {"graph": graph, "hop_sampling": hop_sampling, "mixing": mixing, "trainer": trainer,
            "V0_RGConfig": V0_RGConfig, "load_npz_matrix_only": load_npz_matrix_only,
            "WeightedAutoEncoder": WeightedAutoEncoder, "SplitPreprocessor": SplitPreprocessor}


def _core_source_hashes(core) -> dict[str, str]:
    root = Path(core["trainer"].__file__).resolve().parent
    return {path.name: sha256_file(path) for path in sorted(root.glob("*.py"))}


def load_run_metadata(record_path: Path, core) -> dict:
    record = _read_json(record_path, "record")
    if record.get("status") != "completed":
        raise ProbeBlocked("record_not_completed", str(record_path))
    for flag in ("labels_used_during_fit", "test_labels_loaded"):
        if flag in record and record[flag] is not False:
            raise ProbeBlocked("record_label_flag", f"{flag}={record[flag]!r} in {record_path}")
    if record.get("labels_used_during_fit") is not False:
        raise ProbeBlocked("record_label_flag", f"labels_used_during_fit is not recorded False: {record_path}")
    if "config" not in record or "seed" not in record:
        raise ProbeBlocked("record_missing_field", f"config/seed absent: {record_path}")
    fields = set(core["V0_RGConfig"].__dataclass_fields__)
    unknown = sorted(set(record["config"]) - fields)
    if unknown:
        raise ProbeBlocked("config_unknown_keys", f"{unknown} in {record_path}")
    config = core["V0_RGConfig"](**record["config"])
    if config.variant != "rg_full":
        raise ProbeBlocked("config_not_rg_full", str(record_path))
    seed = int(record["seed"])
    folder = record_path.parent
    def artifact(stem, suffix):
        generic = folder / f"{stem}.{suffix}"
        return generic if generic.is_file() else folder / f"{stem}_seed_{seed}.{suffix}"
    embedding_npy = artifact("embedding", "npy")
    embedding_npz = artifact("embedding", "npz")
    if not embedding_npy.is_file() and not embedding_npz.is_file():
        raise ProbeBlocked("missing_artifact", f"saved embedding: {embedding_npy} or {embedding_npz}")
    artifacts = {
        "model": _require(artifact("model", "pt"), "model checkpoint"),
        "preprocessor_npz": _require(artifact("preprocessor", "npz"), "fitted preprocessor arrays"),
        "preprocessor_json": _require(artifact("preprocessor", "json"), "fitted preprocessor metadata"),
        "embedding": embedding_npy if embedding_npy.is_file() else embedding_npz,
        "hop_sampling": _require(artifact("hop_sampling", "json"), "hop sampling profile"),
    }
    listed = record.get("artifact_sha256") or {}
    verified = {}
    for name, digest in listed.items():
        path = _require(folder / name, f"artifact listed in record ({name})")
        actual = sha256_file(path)
        if actual != digest:
            raise ProbeBlocked("artifact_hash_mismatch", f"{path}: {actual} != {digest}")
        verified[name] = actual
    prep_meta = _read_json(artifacts["preprocessor_json"], "fitted preprocessor metadata")
    for key in ("input_kind", "feature_limit", "fingerprint"):
        if key not in prep_meta:
            raise ProbeBlocked("preprocessor_metadata_incomplete", f"{key} absent: {artifacts['preprocessor_json']}")
    if prep_meta["fingerprint"] != record.get("preprocessing_hash"):
        raise ProbeBlocked("preprocessing_hash_mismatch", f"record vs preprocessor json: {record_path}")
    return {"path": record_path, "record": record, "config": config, "seed": seed,
            "artifacts": artifacts, "artifact_hashes_verified": verified, "prep_meta": prep_meta,
            "hop_profile": _read_json(artifacts["hop_sampling"], "hop sampling profile")}


def fit_preprocessor(features_path: Path, run: dict, core) -> tuple[np.ndarray, Any, dict]:
    """Refit the label-free SplitPreprocessor and prove it equals the saved one."""
    meta = run["prep_meta"]
    target_sum = float(meta.get("target_sum", 10_000.0))
    loaded = core["load_npz_matrix_only"](features_path)
    prep = core["SplitPreprocessor"](meta["input_kind"], int(meta["feature_limit"]), target_sum).fit(loaded.X)
    if prep.fingerprint != meta["fingerprint"]:
        raise ProbeBlocked("preprocessor_refit_mismatch", f"refit fingerprint != saved for {run['path']}")
    with np.load(run["artifacts"]["preprocessor_npz"], allow_pickle=False) as saved:
        for key, value in (("features", prep.features), ("mean", prep.scaler.mean_), ("scale", prep.scaler.scale_)):
            if key not in saved.files:
                raise ProbeBlocked("preprocessor_array_missing", f"{key} in {run['artifacts']['preprocessor_npz']}")
            if not np.array_equal(np.asarray(saved[key]), np.asarray(value)):
                raise ProbeBlocked("preprocessor_array_mismatch", f"{key} differs from refit")
    X_all = prep.transform(loaded.X)
    return X_all, prep, {"input_kind": meta["input_kind"], "feature_limit": int(meta["feature_limit"]),
                         "target_sum": target_sum, "fingerprint": prep.fingerprint,
                         "preprocessor_npz_sha256": sha256_file(run["artifacts"]["preprocessor_npz"]),
                         "transformed_shape": list(X_all.shape)}


def restore_model(run: dict, X_all: np.ndarray, core, device, embedding_rtol: float,
                  embedding_atol: float) -> tuple[Any, np.ndarray, dict]:
    import torch

    config = run["config"]
    model = core["WeightedAutoEncoder"](
        num_genes=int(X_all.shape[1]), hidden_size=config.hidden_size, dropout=config.dropout,
        masked_data_weight=config.masked_data_weight, mask_loss_weight=config.mask_loss_weight)
    try:
        state = torch.load(run["artifacts"]["model"], map_location="cpu", weights_only=True)
    except TypeError as error:
        raise ProbeBlocked("unsafe_checkpoint_loader", "torch.load lacks weights_only; refusing pickle load") from error
    restore = strict_restore(model, state)
    model = model.to(device).eval()
    assert_eval_mode(model)
    reproduced = core["trainer"]._extract_embedding(model, X_all, config.batch_size, device)
    path = run["artifacts"]["embedding"]
    if path.suffix == ".npy":
        saved = np.load(path, allow_pickle=False)
    else:
        with np.load(path, allow_pickle=False) as packed:
            saved = np.asarray(packed["embedding"])
    if saved.shape != reproduced.shape:
        raise ProbeBlocked("embedding_shape_mismatch", f"{saved.shape} != {reproduced.shape}")
    max_abs = float(np.max(np.abs(saved.astype(np.float64) - reproduced.astype(np.float64))))
    if not np.allclose(saved, reproduced, rtol=embedding_rtol, atol=embedding_atol):
        raise ProbeBlocked("embedding_not_reproduced", f"max_abs_diff={max_abs} for {run['path']}")
    restore.update({"model_file_sha256": sha256_file(run["artifacts"]["model"]),
                    "embedding_file_sha256": sha256_file(path),
                    "embedding_reproduction_max_abs_diff": max_abs,
                    "embedding_rtol": embedding_rtol, "embedding_atol": embedding_atol,
                    "global_embedding_scale": global_embedding_scale(reproduced)})
    return model, reproduced, restore


def rebuild_graph(X_all: np.ndarray, config, seed: int, core) -> dict:
    """Recompute the trainer's frozen graph exactly as ``trainer.fit_predict`` (fit_X = X)."""
    graph_mod, mixing = core["graph"], core["mixing"]
    graph = graph_mod.build_pca_knn_graph(
        X_all, k=min(int(config.neighbor_k), int(X_all.shape[0]) - 1),
        pca_dim=config.knn_pca_dim, tau=config.tau, seed=int(seed))
    _, edge_weights, edge_summary = graph_mod.compute_edge_reliability(
        graph, mode=config.edge_reliability_mode, gamma_sim=config.gamma_sim,
        gamma_mutual=config.gamma_mutual, gamma_snn=config.gamma_snn, gamma_distance=config.gamma_distance)
    gate_min, gate_max = config.gate_bounds()
    node_gate, _, gate_summary = mixing.compute_node_gate(
        graph, edge_weights=edge_weights, gate_mode="topology", gate_min=gate_min, gate_max=gate_max,
        beta_mutual=config.beta_mutual, beta_snn=config.beta_snn, beta_perturb=config.beta_perturb,
        beta_uncertainty=config.beta_uncertainty, uncertainty=None)
    return {"graph": graph, "edge_weights": edge_weights, "node_gate": node_gate,
            "edge_summary": edge_summary, "gate_summary": gate_summary}


def rebuild_plan(frozen: dict, config, core):
    if config.hop_variant in {"Kwide2", "Kwide3"}:
        raise ProbeBlocked("unsupported_hop_variant", "Kwide plans are outside this probe")
    return core["hop_sampling"].build_exact_hop_plan(
        frozen["graph"], frozen["edge_weights"], config.hop_variant, config.mix_neighbors,
        config.high_hop_probability_multiplier)


def verify_graph_against_saved(run: dict, frozen: dict, plan, keys=None) -> dict:
    folder, seed = run["path"].parent, run["seed"]
    if plan is None:
        rebuilt = {"variant": "H1", "construction": "legacy_one_hop_passthrough",
                   "base_neighbor_k": int(frozen["graph"].indices.shape[1]),
                   "fixed_draw_count": max(1, min(int(run["config"].mix_neighbors),
                                                  int(frozen["graph"].indices.shape[1])))}
        keys = STATIC_H1_KEYS if keys is None else keys
    else:
        rebuilt, keys = plan.profile, (STATIC_PLAN_KEYS if keys is None else keys)
    mismatched = compare_profile(run["hop_profile"], rebuilt, keys)
    if mismatched:
        raise ProbeBlocked("hop_plan_reconstruction_mismatch", f"{run['path']}: {mismatched}")
    checks = {"hop_profile_keys_verified": list(keys)}
    summary_path = folder / f"graph_gate_summary_seed_{seed}.json"
    if summary_path.is_file():
        saved = _read_json(summary_path, "graph gate summary")
        numeric = [(name, key) for name in ("edge_summary", "gate_summary") for key, value in
                   (saved.get(name) or {}).items() if isinstance(value, (int, float)) and not isinstance(value, bool)]
        bad = [f"{name}.{key}" for name, key in numeric
               if not math.isclose(float(saved[name][key]), float(frozen[name].get(key, math.nan)),
                                   rel_tol=1e-6, abs_tol=1e-9)]
        if bad:
            raise ProbeBlocked("graph_summary_mismatch", f"{summary_path}: {bad}")
        checks["graph_gate_summary_verified"] = [f"{n}.{k}" for n, k in numeric]
    else:
        checks["graph_gate_summary_verified"] = "absent_for_this_record_format"
    return checks


# -------------------------------------------------------------- directions

def draw_plan_donors(nodes_rows, prob_rows, hop_rows, anchors: np.ndarray, seed: int,
                     min_hop: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """One seeded Monte Carlo donor per anchor from candidate rows restricted to hop >= min_hop."""
    rng = np.random.default_rng(int(seed))
    donors = np.full(anchors.size, -1, dtype=np.int64)
    hops = np.zeros(anchors.size, dtype=np.int64)
    for position, anchor in enumerate(anchors):
        nodes = np.asarray(nodes_rows[int(anchor)], dtype=np.int64)
        probs = np.asarray(prob_rows[int(anchor)], dtype=np.float64)
        labels = np.asarray(hop_rows[int(anchor)], dtype=np.int64)
        keep = labels >= int(min_hop)
        if not keep.any() or probs[keep].sum() <= 0.0:
            continue
        p = probs[keep] / probs[keep].sum()
        choice = int(rng.choice(int(keep.sum()), p=p))
        donors[position] = nodes[keep][choice]
        hops[position] = labels[keep][choice]
    return donors, hops


def trace_donors(trace_path: Path, anchors: np.ndarray, seed: int,
                 chunk_rows: int = 4_000_000, trace_dtype=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pick one seeded real training draw per anchor from a saved step5-format trace."""
    dtype = TRACE_DTYPE if trace_dtype is None else np.dtype([tuple(x) for x in trace_dtype])
    hop_field = "actual_hop" if "actual_hop" in dtype.names else "hop"
    if not {"anchor", "source", hop_field}.issubset(dtype.names):
        raise ProbeBlocked("trace_schema_invalid", str(dtype))
    compact_dtype = np.dtype([("anchor", "<i4"), ("source", "<i4"), ("actual_hop", "u1")])
    size = trace_path.stat().st_size
    if size % dtype.itemsize:
        raise ProbeBlocked("trace_size_invalid", str(trace_path))
    table = np.memmap(trace_path, dtype=dtype, mode="r")
    anchor_set = np.asarray(anchors, dtype=np.int64)
    collected = []
    for start in range(0, table.shape[0], chunk_rows):
        block = np.asarray(table[start:start + chunk_rows])
        hit = np.isin(block["anchor"].astype(np.int64), anchor_set)
        if hit.any():
            compact = np.empty(int(hit.sum()), dtype=compact_dtype)
            compact["anchor"] = block["anchor"][hit]
            compact["source"] = block["source"][hit]
            compact["actual_hop"] = block[hop_field][hit]
            collected.append(compact)
    rows = np.concatenate(collected) if collected else np.empty(0, dtype=compact_dtype)
    rng = np.random.default_rng(int(seed))
    donors = np.full(anchors.size, -1, dtype=np.int64)
    hops = np.zeros(anchors.size, dtype=np.int64)
    counts = np.zeros(anchors.size, dtype=np.int64)
    order = np.argsort(rows["anchor"], kind="stable")
    rows = rows[order]
    starts = np.searchsorted(rows["anchor"], anchors, side="left")
    stops = np.searchsorted(rows["anchor"], anchors, side="right")
    for position in range(anchors.size):
        counts[position] = int(stops[position] - starts[position])
        if counts[position]:
            pick = int(starts[position] + rng.integers(counts[position]))
            donors[position] = int(rows["source"][pick])
            hops[position] = int(rows["actual_hop"][pick])
    return donors, hops, counts


def family_directions(X_all: np.ndarray, anchors: np.ndarray, donors: np.ndarray) -> np.ndarray:
    directions = np.zeros((anchors.size, X_all.shape[1]), dtype=np.float32)
    have = donors >= 0
    directions[have] = X_all[donors[have]] - X_all[anchors[have]]
    return directions


# ------------------------------------------------------------------ driver

def run_probe(args) -> dict:
    import torch

    for root in args.pythonpath or []:
        if root not in sys.path:
            sys.path.insert(0, root)
    core = _import_core()
    sources_before = _core_source_hashes(core)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ProbeBlocked("cuda_unavailable", args.device)

    record = load_run_metadata(Path(args.record).resolve(), core)
    reference = load_run_metadata(Path(args.reference_record).resolve(), core)
    if reference["config"].hop_variant != "H1":
        raise ProbeBlocked("reference_not_h1", reference["config"].hop_variant)
    if record["config"].hop_variant == "H1":
        raise ProbeBlocked("record_not_multihop", "record must use a multihop hop_variant")
    if record["seed"] != reference["seed"]:
        raise ProbeBlocked("seed_mismatch", f"{record['seed']} != {reference['seed']}")
    difference = config_difference(asdict(reference["config"]), asdict(record["config"]))
    disallowed = sorted(set(difference) - CONFIG_DIFF_ALLOWED - {"protocol_id"})
    if disallowed:
        raise ProbeBlocked("config_not_matched", json.dumps({k: difference[k] for k in disallowed}, default=str))
    if record["record"].get("preprocessing_hash") != reference["record"].get("preprocessing_hash"):
        raise ProbeBlocked("preprocessing_not_matched", "record and reference preprocessing hashes differ")

    features_path = Path(args.features).resolve()
    feature_sha = sha256_file(_require(features_path, "feature-only NPZ"))
    feature_checks = {}
    for role, run in (("record", record), ("reference", reference)):
        expected = run["record"].get("feature_sha256")
        if expected is None:
            feature_checks[role] = "not_in_record_verified_by_preprocessing_fingerprint"
        elif expected != feature_sha:
            raise ProbeBlocked("feature_hash_mismatch", f"{role}: {expected} != {feature_sha}")
        else:
            feature_checks[role] = "sha256_equal"

    X_all, _prep, prep_info = fit_preprocessor(features_path, record, core)
    if reference["prep_meta"]["fingerprint"] != prep_info["fingerprint"]:
        raise ProbeBlocked("preprocessing_not_matched", "reference preprocessor fingerprint differs")
    if sha256_file(reference["artifacts"]["preprocessor_npz"]) != prep_info["preprocessor_npz_sha256"]:
        # Same fingerprint implies same arrays; still verify the reference file itself.
        fit_preprocessor(features_path, reference, core)

    models = {}
    for role, run in (("reference_h1", reference), ("record_multihop", record)):
        model, embedding, restore = restore_model(run, X_all, core, device,
                                                  args.embedding_rtol, args.embedding_atol)
        models[role] = {"model": model, "embedding": embedding, "restore": restore, "run": run}

    frozen = rebuild_graph(X_all, record["config"], record["seed"], core)
    plan = rebuild_plan(frozen, record["config"], core)
    arm = str(record["record"].get("arm", ""))
    # A plan is reconstructible only if the arm did not transform shell weights and
    # the complete static profile equals the rebuilt one. Transformed arms must still
    # match on candidate identity, and their source directions come from the trace.
    full_profile_match = not compare_profile(record["hop_profile"], plan.profile, STATIC_PLAN_KEYS)
    plan_reconstructible = full_profile_match and "geometric" not in arm
    graph_checks = {"record": verify_graph_against_saved(
                        record, frozen, plan, None if plan_reconstructible else CANDIDATE_IDENTITY_KEYS),
                    "reference": verify_graph_against_saved(reference, frozen, None)}

    anchors = sample_anchors(X_all.shape[0], args.max_anchors, args.probe_seed)
    graph = frozen["graph"]
    one_hop_labels = [np.ones(graph.indices.shape[1], dtype=np.int64)] * X_all.shape[0]
    donor_sets = {}
    donor_sets["h1_graph"] = draw_plan_donors(graph.indices, frozen["edge_weights"], one_hop_labels,
                                              anchors, args.probe_seed + 11)
    family_status = {"h1_graph": "monte_carlo_from_reconstructed_graph_edge_weights"}
    if plan_reconstructible:
        donor_sets["plan_all"] = draw_plan_donors(plan.indices, plan.probs, plan.hop_labels,
                                                  anchors, args.probe_seed + 13)
        donor_sets["plan_far"] = draw_plan_donors(plan.indices, plan.probs, plan.hop_labels,
                                                  anchors, args.probe_seed + 17, min_hop=2)
        family_status["plan_all"] = "monte_carlo_from_reconstructed_exact_hop_plan"
        family_status["plan_far"] = "monte_carlo_from_reconstructed_exact_hop_plan_hops_ge_2_renormalized"
    else:
        family_status["plan_all"] = family_status["plan_far"] = (
            "blocked_record_plan_was_transformed_use_saved_trace")
    trace_counts = None
    trace_info = record["record"].get("training_trace") or {}
    trace_file = trace_info.get("trace_file") if isinstance(trace_info, dict) else None
    if trace_file and (record["path"].parent / trace_file).is_file():
        trace_path = record["path"].parent / trace_file
        if trace_info.get("trace_sha256") and sha256_file(trace_path) != trace_info["trace_sha256"]:
            raise ProbeBlocked("trace_hash_mismatch", str(trace_path))
        donors, hops, trace_counts = trace_donors(trace_path, anchors, args.probe_seed + 19,
                                                trace_dtype=trace_info.get("trace_dtype"))
        donor_sets["trace"] = (donors, hops)
        family_status["trace"] = "saved_actual_training_draws"
    else:
        family_status["trace"] = "unavailable_no_saved_trace_for_record"
    if not plan_reconstructible and "trace" not in donor_sets:
        raise ProbeBlocked("no_actual_source_directions",
                           "record plan was transformed and no saved trace exists")

    unit = unit_random_directions(anchors.size, X_all.shape[1], args.probe_seed + 23)
    input_scale = global_embedding_scale(X_all)
    anchor_x = X_all[anchors]
    rows_out, summary, exclusions = [], {}, {}
    for family, (donors, hops) in donor_sets.items():
        source = family_directions(X_all, anchors, donors)
        norms, valid = direction_norms(source)
        random_dirs, random_valid = matched_random(unit, source)
        assert np.array_equal(valid, random_valid)
        exclusions[family] = {"n_anchors": int(anchors.size),
                              "no_donor": int(np.sum(donors < 0)),
                              "zero_norm_direction": int(np.sum((donors >= 0) & ~valid)),
                              "evaluated": int(valid.sum())}
        index = np.flatnonzero(valid)
        results = {}
        for role, entry in models.items():
            gate = np.clip(frozen["node_gate"][anchors] * float(entry["run"]["config"].mix_amplitude_multiplier), 0.0, 1.0)
            scale = entry["restore"]["global_embedding_scale"]
            for kind, dirs in (("source", source), ("random_norm_matched", random_dirs)):
                out = probe_responses(entry["model"], anchor_x[index], dirs[index], LAMBDAS,
                                      args.batch_size, device)
                results[(role, kind)] = out
                for li, lam in enumerate(LAMBDAS):
                    normalized = out["encoder_response"][li] / scale
                    summary[f"{role}|{family}|{kind}|lambda={lam}"] = {
                        "encoder_response_normalized": summarize(normalized),
                        "decoder_anchor_mse_delta": summarize(out["decoder_anchor_mse_delta"][li])}
                    for j, row in enumerate(index):
                        rows_out.append({
                            "model_role": role, "family": family, "kind": kind,
                            "anchor": int(anchors[row]), "donor": int(donors[row]),
                            "donor_hop": int(hops[row]), "lambda": lam,
                            "input_direction_norm": float(norms[row]),
                            "input_direction_norm_over_input_scale": float(norms[row] / input_scale),
                            "perturbation_norm": float(lam * norms[row]),
                            "encoder_response": float(out["encoder_response"][li, j]),
                            "encoder_response_normalized": float(normalized[j]),
                            "decoder_anchor_mse_delta": float(out["decoder_anchor_mse_delta"][li, j]),
                            "training_mix_gate": float(gate[row]),
                            "trace_rows_for_anchor": (None if family != "trace" or trace_counts is None
                                                      else int(trace_counts[row]))})
        for kind in ("source", "random_norm_matched"):
            for li, lam in enumerate(LAMBDAS):
                ref, rec = results[("reference_h1", kind)], results[("record_multihop", kind)]
                ref_scale = models["reference_h1"]["restore"]["global_embedding_scale"]
                rec_scale = models["record_multihop"]["restore"]["global_embedding_scale"]
                summary[f"paired_record_minus_reference|{family}|{kind}|lambda={lam}"] = {
                    "encoder_response_normalized": summarize(
                        rec["encoder_response"][li] / rec_scale - ref["encoder_response"][li] / ref_scale),
                    "decoder_anchor_mse_delta": summarize(
                        rec["decoder_anchor_mse_delta"][li] - ref["decoder_anchor_mse_delta"][li])}
        for role in models:
            scale = models[role]["restore"]["global_embedding_scale"]
            for li, lam in enumerate(LAMBDAS):
                src, rnd = results[(role, "source")], results[(role, "random_norm_matched")]
                summary[f"source_minus_random|{role}|{family}|lambda={lam}"] = {
                    "encoder_response_normalized": summarize(
                        (src["encoder_response"][li] - rnd["encoder_response"][li]) / scale),
                    "decoder_anchor_mse_delta": summarize(
                        src["decoder_anchor_mse_delta"][li] - rnd["decoder_anchor_mse_delta"][li])}

    sources_after = _core_source_hashes(core)
    if sources_after != sources_before:
        raise ProbeBlocked("core_source_changed", "package source hashes changed during probe")
    return {
        "protocol_id": PROTOCOL, "status": "completed",
        "label_reads": False, "labels_loaded": False, "labels_used": False,
        "record_path": str(record["path"]), "reference_record_path": str(reference["path"]),
        "features_path": str(features_path), "feature_sha256": feature_sha, "feature_hash_checks": feature_checks,
        "seed": record["seed"], "record_arm": arm or None,
        "config_difference": difference,
        "record_config_sha256": canonical_sha256(asdict(record["config"])),
        "reference_config_sha256": canonical_sha256(asdict(reference["config"])),
        "preprocessing": prep_info,
        "models": {role: {**entry["restore"], "hop_variant": entry["run"]["config"].hop_variant,
                          "artifact_hashes_verified": entry["run"]["artifact_hashes_verified"]}
                   for role, entry in models.items()},
        "graph": {"provenance": "recomputed_deterministically_from_preprocessed_features_with_record_config",
                  "checks": graph_checks, "plan_profile": plan.profile,
                  "record_plan_reconstructible": plan_reconstructible},
        "direction_families": family_status,
        "direction_notation": "d = x_donor - x_anchor in preprocessed model-input space; "
                              "random = seeded Gaussian rescaled to ||d|| per anchor (same vectors for both models); "
                              "MC = Monte Carlo draw from reconstructed weights, not a historical training draw",
        "exclusions": exclusions,
        "anchors": {"count": int(anchors.size), "max": int(args.max_anchors), "seed": int(args.probe_seed),
                    "selection": "uniform_without_replacement_feature_only",
                    "sha256": hashlib.sha256(anchors.astype("<i8").tobytes()).hexdigest()},
        "lambdas": list(LAMBDAS),
        "input_global_scale": input_scale,
        "evaluation": {"mask_noise_applied": False, "model_training_mode": False, "dropout_active": False,
                       "forward": "WeightedAutoEncoder.forward_mask on uncorrupted x + lambda*d; decoder "
                                  "receives mask-predictor logits of that input",
                       "decoder_target": "the anchor's preprocessed model input, not a denoised truth",
                       "encoder_scale": "per-model RMS radius of its reproduced all-sample embedding",
                       "no_gene_program_inference": True},
        "summary": summary,
        "_rows": rows_out,
        "core_source_hashes": sources_before,
        "environment": {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
                        "platform": platform.platform(), "host": socket.gethostname(),
                        "device": str(device), "script_sha256": sha256_file(Path(__file__))},
    }


def _write_outputs(out: Path, result: dict) -> None:
    rows = result.pop("_rows")
    columns = list(rows[0]) if rows else []
    temporary = out / "probe_rows.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(out / "probe_rows.csv")
    result["rows_file"] = "probe_rows.csv"
    result["rows_sha256"] = sha256_file(out / "probe_rows.csv")
    temporary = out / "probe_summary.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(out / "probe_summary.json")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="matched H1/multihop finite-difference direction probe")
    parser.add_argument("--record", required=True, help="multihop record_seed_{seed}.json")
    parser.add_argument("--reference-record", required=True, help="matched H1 record_seed_{seed}.json")
    parser.add_argument("--features", required=True, help="feature-only NPZ used for fitting")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-anchors", type=int, default=MAX_ANCHORS)
    parser.add_argument("--probe-seed", type=int, default=20260925)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--embedding-rtol", type=float, default=1e-4)
    parser.add_argument("--embedding-atol", type=float, default=1e-4)
    parser.add_argument("--pythonpath", action="append", help="extra import root (repeatable)")
    args = parser.parse_args(argv)
    if args.max_anchors > MAX_ANCHORS or args.max_anchors <= 0:
        parser.error(f"--max-anchors must be in [1, {MAX_ANCHORS}]")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "probe_summary.json").exists():
        parser.error(f"refusing to overwrite {out / 'probe_summary.json'}")
    started = time.time()
    try:
        result = run_probe(args)
    except ProbeBlocked as blocked:
        (out / "probe_blocked.json").write_text(json.dumps({
            "protocol_id": PROTOCOL, "status": "blocked", "code": blocked.code, "detail": blocked.detail,
            "label_reads": False, "record": args.record, "reference_record": args.reference_record},
            indent=2), encoding="utf-8")
        print(f"BLOCKED {blocked.code}: {blocked.detail}", file=sys.stderr)
        return 2
    result["elapsed_seconds"] = time.time() - started
    _write_outputs(out, result)
    print(f"completed: {out / 'probe_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
