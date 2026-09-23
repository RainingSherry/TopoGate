"""Frozen observation and full-derived direction probes. Never trains a model."""
from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from common import (load_config, load_case, load_encoder, replay_clean, encode, corrupt,
                    atomic_json, array_hash, write_npz, condition_slug)
from metrics import (displacement, distribution_metrics, normalized_rows, graph_comparison,
                     representation_metrics, knn_self_excluded, rms_or_none,
                     applied_direction_family, subset_prediction_metrics)


def signature(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def graph_context(case, seed):
    """Replay original graph and independently retain its clean-fitted PCA map."""
    from methods.TopoGate.V0_RG.graph import build_pca_knn_graph
    x, cfg = case["x"], case["cfg"]
    scope = case["summary"].get("fit_scope")
    metadata = case["checkpoint"] / "preprocessor.json"
    if scope is None and metadata.is_file():
        scope = json.loads(metadata.read_text()).get("fit_scope")
    if scope not in {"all_samples_unlabeled", "all_samples", "transductive_all"}:
        raise ValueError(f"Cannot reproduce graph fit scope {scope!r}; this probe requires audited all-row checkpoints")
    graph = build_pca_knn_graph(x, k=cfg.neighbor_k, pca_dim=cfg.knn_pca_dim, tau=cfg.tau, seed=seed)
    dim = max(1, min(int(cfg.knn_pca_dim), x.shape[1], len(x) - 1))
    pca = PCA(n_components=dim, random_state=seed) if dim < min(x.shape) else None
    projected = pca.fit_transform(x) if pca is not None else x
    projected = normalized_rows(projected)
    error = float(np.max(np.abs(projected - graph.embedding)))
    if not np.allclose(projected, graph.embedding, atol=2e-5, rtol=2e-5):
        raise ValueError(f"Clean PCA graph replay mismatch: max absolute error {error}")
    profile_path = case["checkpoint"] / "graph_gate_summary.json"
    profile_audit = {"status": "saved_profile_unavailable", "comparisons": {}}
    if profile_path.is_file():
        saved = json.loads(profile_path.read_text()).get("graph", {})
        comparisons = {}
        for key, value in graph.profile.items():
            if key in saved and isinstance(value, (float, int)) and not isinstance(value, bool):
                comparisons[key] = {"saved": saved[key], "replayed": value,
                                    "matches": bool(np.isclose(value, saved[key], rtol=2e-4, atol=2e-5))}
        # scMAE has an intentionally empty saved training graph; its offline graph is diagnostic only.
        enabled = cfg.variant == "rg_full"
        matches = all(v["matches"] for v in comparisons.values())
        profile_audit = {"status": "matched" if matches else ("mismatch" if enabled else "training_graph_disabled"), "comparisons": comparisons}
        if enabled and not matches:
            raise ValueError(f"Replayed training graph profile differs from saved graph: {comparisons}")
    # Use the identical transform operation on clean and noisy diagnostic inputs.
    # Keep fit_transform (the historical graph operation) separately audited above.
    clean_transform = normalized_rows(pca.transform(x) if pca is not None else x)
    transform_error = float(np.max(np.abs(clean_transform - projected)))
    diagnostic_nn = knn_self_excluded(clean_transform, cfg.neighbor_k)
    self_rows = np.any(graph.indices == np.arange(len(x))[:, None], axis=1)
    audit = {"status": "replayed", "fit_scope": scope, "pca_embedding_abs_max_error": error,
             "saved_profile": profile_audit, "clean_fit_transform_vs_transform_abs_max_error": transform_error,
             "original_self_neighbor_rows": int(self_rows.sum()),
             "original_vs_self_excluded_rows_differ": int(np.sum(np.any(graph.indices != diagnostic_nn, axis=1))),
             "note": "Original graph replay is unmodified; diagnostics use the same clean-fitted transform on clean/noisy rows and explicitly exclude self, including ties."}
    return graph, pca, clean_transform, audit


def measure(case, model, clean_z, noisy_x, noisy_raw, config, device, graph_state):
    t = case["test_indices"]
    noisy_z = encode(case, model, noisy_x, device)
    rep, prediction, arrays = representation_metrics(clean_z, noisy_z, case["centers"], case["labels"], t, case["saved_predictions"])
    reference_predictions = case["ref"].nearest_centroid(noisy_z, case["centers"])
    if not np.array_equal(arrays["noisy_predictions"], reference_predictions[t]):
        raise AssertionError("Probe noisy predictions differ from audited reference.nearest_centroid")
    prediction["reference_prediction_match"] = True
    raw_stats, raw_delta = displacement(case["raw"][t], noisy_raw[t]) if noisy_raw is not None else ({"status": "not_applicable_model_input_direction", "normalized_displacement": None, "changed_fraction": None}, np.full(len(t), np.nan))
    model_stats, model_delta = displacement(case["x"][t], noisy_x[t])
    distribution = distribution_metrics(case["x"][t], noisy_x[t], config.get("max_mmd_rows", 512),
                                        config.get("max_correlation_features", 128), config.get("diagnostic_seed", 9102026))
    graph, pca, clean_projection, audit = graph_state
    noisy_projection = normalized_rows(pca.transform(noisy_x) if pca is not None else noisy_x)
    graph_metrics, graph_arrays = graph_comparison(clean_projection, noisy_projection, case["labels"], t, case["cfg"].neighbor_k)
    rep.update(graph=graph_metrics, graph_replay=audit)
    arrays.update(graph_arrays, raw_normalized_displacement=raw_delta, model_normalized_displacement=model_delta)
    return {"raw": raw_stats, "model": model_stats, **distribution,
            "distribution_space": "frozen_model_input_on_saved_test_rows",
            "raw_original_dtype": case.get("raw_original_dtype"), "raw_cast_fraction": case.get("raw_cast_fraction")}, rep, prediction, arrays


def make_directions(full, graph, seed):
    from methods.TopoGate.V0_RG.graph import compute_edge_reliability, build_random_neighbors, build_far_neighbors
    from methods.TopoGate.V0_RG.mixing import compute_node_gate, estimate_neighbors
    cfg, x = full["cfg"], full["x"]
    _, weights, _ = compute_edge_reliability(graph, cfg.edge_reliability_mode if cfg.edge_modulation_enabled else "none",
                                            cfg.gamma_sim, cfg.gamma_mutual, cfg.gamma_snn, cfg.gamma_distance)
    lo, hi = cfg.gate_bounds()
    gate, _, _ = compute_node_gate(graph, weights, "topology", lo, hi, cfg.beta_mutual,
                                   cfg.beta_snn, cfg.beta_perturb, cfg.beta_uncertainty, None)
    rng = np.random.default_rng(seed)
    # Follow official sampling, including legacy stream consumption; this is a new diagnostic draw,
    # not a claim to replay a particular epoch/minibatch's augmentation.
    count = max(1, min(cfg.mix_neighbors, len(x) - 1))
    build_random_neighbors(len(x), count, rng, exclude=graph.indices)
    build_far_neighbors(graph.embedding, count, rng)
    neighbors = estimate_neighbors(x, np.arange(len(x)), graph, weights,
                                   min(cfg.mix_neighbors, graph.indices.shape[1]), rng, cfg.neighbor_estimator)
    amplitude = np.full_like(gate, cfg.fixed_mix_strength) if cfg.mix_strength_mode == "fixed_train_mean" else gate
    native = amplitude[:, None] * (neighbors - x)
    target = np.linalg.norm(native.astype(float), axis=1)
    random = rng.standard_normal(x.shape).astype(np.float32)
    shuffled = native[rng.permutation(len(x))].copy()
    result = {}
    for name, vector in (("native", native), ("random", random), ("shuffled", shuffled)):
        norm = np.linalg.norm(vector.astype(float), axis=1)
        valid = (target > 0) & (norm > 0)
        matched = np.zeros_like(vector)
        matched[valid] = vector[valid] * (target[valid] / norm[valid])[:, None]
        result[name] = (matched, valid, target)
    return result, gate


def save_task(config, args, case, model, clean_z, clean_audit, graph_state, kind, severity, noise_seed,
              noisy_raw, noisy_x, noise_meta, extra=None, extra_arrays=None):
    root = Path(config["output_root"]) / args.dataset / args.stage / args.variant / f"model_seed_{args.model_seed}" / condition_slug(kind, severity, noise_seed)
    path = root / "result.json"
    provenance = {**case["provenance"], **noise_meta}
    fingerprint = signature({"provenance": provenance, "stage": args.stage, "corruption": kind,
                             "severity": severity, "noise_seed": noise_seed, "direction": extra})
    if path.is_file():
        existing = json.loads(path.read_text())
        if existing.get("status") == "completed" and existing.get("task_signature") == fingerprint:
            print(json.dumps({"status": "skipped_matching", "path": str(path)}), flush=True)
            return
        if existing.get("status") == "completed":
            atomic_json(root / "signature_mismatch.json", {"status": "failed", "expected": fingerprint, "existing": existing.get("task_signature")})
            raise ValueError(f"Refusing to overwrite completed result with different task signature: {path}")
    base = {"protocol": config["protocol"], "dataset": args.dataset, "variant": args.variant,
            "model_seed": args.model_seed, "stage": args.stage, "corruption": kind, "severity": severity,
            "noise_seed": noise_seed, "provenance": provenance, "clean_replay": clean_audit,
            "task_signature": fingerprint, "seed_role": config.get("seed_role"), "training_performed": False}
    try:
        inp, rep, pred, arrays = measure(case, model, clean_z, noisy_x, noisy_raw, config, args.device, graph_state)
        if extra_arrays:
            arrays.update(extra_arrays)
        if extra:
            base["direction"] = extra
            common_valid = arrays["direction_common_valid"].astype(bool)
            rep["normalized_drift_common_valid"] = (rms_or_none(arrays["normalized_drift"][common_valid])
                if np.any(common_valid) and np.all(np.isfinite(arrays["normalized_drift"][common_valid])) else None)
            common_prediction = subset_prediction_metrics(arrays["labels"], arrays["clean_predictions"], arrays["noisy_predictions"], common_valid)
            pred["common_valid"] = common_prediction
            pred["all_test"] = {"n": len(common_valid), **{key: pred[key] for key in ("clean_ari", "noisy_ari", "ari_drop", "ari_status", "flip_rate")}}
            for key in ("clean_ari", "noisy_ari", "ari_drop", "flip_rate"):
                pred[key + "_common_valid"] = common_prediction[key]
            rep["all_test"] = {"n": len(common_valid), "normalized_drift": rep["normalized_drift"]}
            rep["common_valid"] = {"n": int(common_valid.sum()), "normalized_drift": rep["normalized_drift_common_valid"],
                "normalization": "RMS_displacement_on_common_rows_over_all_test_clean_RMS_centered_scale",
                "clean_rms_centered_scale": rep["clean_rms_centered_scale"]}
        write_npz(root / "samples.npz", **arrays)
        atomic_json(path, {**base, "status": "completed", "input": inp, "representation": rep, "prediction": pred,
                           "inference": "frozen_encoder_and_saved_clean_centroids; graph_is_offline_diagnostic_only"})
        print(json.dumps({"status": "completed", "path": str(path)}), flush=True)
    except Exception as exc:
        atomic_json(path, {**base, "status": "failed", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        raise


def record_failure(config, args, kind, severity, seed, exc):
    failure = {"status": "failed", "stage": args.stage, "dataset": args.dataset, "variant": args.variant,
               "model_seed": args.model_seed, "corruption": kind, "severity": severity, "noise_seed": seed,
               "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
    directory = Path(config["output_root"]) / args.dataset / args.stage / args.variant / f"model_seed_{args.model_seed}"
    if kind != "startup":
        directory /= condition_slug(kind, severity, seed)
    atomic_json(directory / ("startup_failure.json" if kind == "startup" else "failure.json"), failure)
    print(json.dumps({key: value for key, value in failure.items() if key != "traceback"}), flush=True)
    return failure


def run(args, config):
    failures = []
    case = load_case(config, args.dataset, args.model_seed, args.variant)
    model = load_encoder(case, args.device)
    clean_z, clean_audit = replay_clean(case, model, args.device)
    context = graph_context(case, args.model_seed)
    if args.stage == "observe":
        conditions = config["conditions"] if not args.condition else [dict(corruption=x.split(":")[0], severity=float(x.split(":")[1])) for x in args.condition]
        for condition in conditions:
            for noise_seed in args.noise_seeds or config["noise_seeds"]:
                kind, severity = condition["corruption"], float(condition["severity"])
                try:
                    raw, x, metadata = corrupt(case, kind, severity, noise_seed)
                    save_task(config, args, case, model, clean_z, clean_audit, context, kind, severity, noise_seed, raw, x, metadata)
                except Exception as exc:
                    failures.append(record_failure(config, args, kind, severity, noise_seed, exc))
    else:
        full = case if args.variant == "full" else load_case(config, args.dataset, args.model_seed, "full")
        if not np.array_equal(case["x"], full["x"]) or not np.array_equal(case["test_indices"], full["test_indices"]):
            raise ValueError("Direction comparisons require identical full-derived clean input coordinates and test rows")
        full_context = context if args.variant == "full" else graph_context(full, args.model_seed)
        directions, gate = make_directions(full, full_context[0], config["direction_seed"])
        t = case["test_indices"]
        for multiplier in config["direction_multipliers"]:
            try:
                applied, common_valid = applied_direction_family(full["x"], directions, multiplier, rtol=1e-4, atol=1e-8)
            except Exception as exc:
                for name in directions:
                    failures.append(record_failure(config, args, f"direction_{name}", float(multiplier), config["direction_seed"], exc))
                continue
            for name, (delta, nominal_valid, target) in directions.items():
                try:
                    noisy_x = applied[name]["noisy"]
                    norm = np.linalg.norm(delta.astype(float), axis=1)
                    applied_norm, valid = applied[name]["norm"], applied[name]["valid"]
                    extra = {"source_variant": "full", "space": "frozen_model_input_only", "direction": name,
                             "source_checkpoint": str(full["checkpoint"]), "source_model_sha256": full["provenance"].get("model_sha256"),
                             "source_config_sha256": full["provenance"].get("checkpoint_config_sha256"),
                             "common_input_sha256": array_hash(full["x"]), "direction_sha256": array_hash(delta),
                             "valid_count": int(valid[t].sum()), "common_valid_count": int(common_valid[t].sum()),
                             "test_count": len(t), "excluded_test_count": int((~common_valid[t]).sum()),
                             "nominal_valid_count": int(nominal_valid[t].sum()),
                             "actual_validation_excluded_count": int(np.sum(nominal_valid[t] & ~valid[t])),
                             "norm_tolerance_rtol": 1e-4, "norm_tolerance_atol": 1e-8,
                             "common_valid_mask_sha256": array_hash(common_valid[t]),
                             "equal_amplitude_status": "verified_on_common_valid_rows" if np.any(common_valid[t]) else "no_valid_comparable_rows",
                             "max_error_all_test": float(np.max(applied[name]["error"][t])),
                             "max_error_common_valid": float(np.max(applied[name]["error"][t][common_valid[t]])) if np.any(common_valid[t]) else None,
                             "norm_match_error": float(np.max(np.abs(norm[valid] - target[valid]))) if np.any(valid) else None,
                             "applied_norm_match_error": float(np.max(np.abs(applied_norm[valid] - multiplier * target[valid]))) if np.any(valid) else None,
                             "sampling": "official_estimate_neighbors_with_full_config; fresh_label_free_diagnostic_draw",
                             "zero_direction_policy": "common_mask_requires_all_families_actual_float32_displacements_nonzero_and_within_norm_tolerance"}
                    metadata = {"raw_input_sha256": None, "model_input_sha256": array_hash(noisy_x)}
                    extra_arrays = {"direction_norm": applied_norm[t], "direction_nominal_norm": norm[t] * multiplier,
                                    "target_direction_norm": target[t] * multiplier, "direction_applied_nonzero": applied_norm[t] > 0,
                                    "direction_valid": valid[t], "direction_common_valid": common_valid[t], "full_gate": gate[t]}
                    save_task(config, args, case, model, clean_z, clean_audit, context, f"direction_{name}", float(multiplier),
                              config["direction_seed"], None, noisy_x, metadata, extra, extra_arrays)
                except Exception as exc:
                    failures.append(record_failure(config, args, f"direction_{name}", float(multiplier), config["direction_seed"], exc))
    summary = {"status": "failed" if failures else "completed", "dataset": args.dataset, "stage": args.stage,
               "variant": args.variant, "model_seed": args.model_seed, "failure_count": len(failures), "failures": failures}
    root = Path(config["output_root"]) / args.dataset / args.stage / args.variant / f"model_seed_{args.model_seed}"
    atomic_json(root / "execution_summary.json", summary)
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--variant", default="full")
    parser.add_argument("--stage", choices=("observe", "directions"), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--noise-seeds", type=int, nargs="+")
    parser.add_argument("--condition", action="append", help="e.g. replacement:0.2; repeat for multiple conditions")
    args = parser.parse_args()
    config = {"output_root": str(Path(args.config).resolve().parent / "startup_failures")}
    try:
        config = load_config(args.config)
        return run(args, config)
    except Exception as exc:
        record_failure(config, args, "startup", None, None, exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
