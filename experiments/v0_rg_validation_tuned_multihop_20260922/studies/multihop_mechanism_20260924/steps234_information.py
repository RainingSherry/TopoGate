"""Frozen, label-free mechanism audits for multihop V0_RG (steps 2--4).

This is an analysis of the *sampling distribution*, not a recovered training
trace.  It never reads benchmark label sidecars or changes training artifacts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold


INPUT_KIND = {"Tosches": "raw_count", "Baron": "raw_count", "fabert": "general"}
FEATURE_LIMIT = 2000
SEEDS = (42, 123, 7)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_core(core_parent: Path):
    if not (core_parent / "v0rg_tunable_multihop" / "hop_sampling.py").exists():
        raise FileNotFoundError(f"core package missing under {core_parent}")
    sys.path.insert(0, str(core_parent.resolve()))
    for ancestor in core_parent.resolve().parents:
        if (ancestor / "methods" / "NeighborMix_scMAE" / "model.py").exists():
            sys.path.insert(0, str(ancestor))
            break
    from v0rg_tunable_multihop.config import V0_RGConfig
    from v0rg_tunable_multihop.graph import build_pca_knn_graph, compute_edge_reliability
    from v0rg_tunable_multihop.hop_sampling import (
        _transition_matrix, _directed_exact_shells, _rank_shell, resolve_path_variant,
    )
    from v0rg_tunable_multihop.input_adapter import load_npz_matrix_only
    from v0rg_tunable_multihop.mixing import compute_node_gate
    from v0rg_tunable_multihop.tuning import SplitPreprocessor
    return locals()


def frozen_row(graph, edge_weights: np.ndarray, source: int, variant: str,
               q: float, core: dict, transition=None) -> dict:
    """Exact per-source slice of build_exact_hop_plan, avoiding all-N plans."""
    depth, width = core["resolve_path_variant"](variant)
    k = graph.indices.shape[1]
    if transition is None:
        transition = core["_transition_matrix"](graph, edge_weights)
    shells = core["_directed_exact_shells"](graph.indices, source, depth)
    chosen = {1: (np.asarray(graph.indices[source], dtype=np.int64),
                  np.asarray(edge_weights[source], dtype=np.float64), 1.0)}
    walk = transition.getrow(source)
    for hop in range(2, depth + 1):
        walk = (walk @ transition).tocsr()
        nodes = shells[hop]
        scores = np.asarray(walk[:, nodes].toarray()).ravel() if nodes.size else np.empty(0)
        picked, mass, retained = core["_rank_shell"](nodes, scores, k * width)
        chosen[hop] = picked, mass, retained
    available = [h for h in chosen if chosen[h][0].size]
    multipliers = np.asarray([1.0 if h == 1 else float(q) for h in available])
    multipliers /= multipliers.sum()
    indices, probs, hops = [], [], []
    for h, shell_mass in zip(available, multipliers):
        nodes, scores, _ = chosen[h]
        indices.append(nodes)
        probs.append(shell_mass * scores / scores.sum())
        hops.append(np.full(nodes.size, h, dtype=np.int8))
    return {
        "indices": np.concatenate(indices), "probs": np.concatenate(probs),
        "hops": np.concatenate(hops), "chosen": chosen,
        "available": available,
    }


def mc_decompose(X: np.ndarray, source: int, row: dict, gate: float,
                 draws: int, repeats: int, rng: np.random.Generator,
                 estimator: str = "current", max_hop: int | None = None) -> dict:
    """Repeatedly apply the historical sampler; vector pieces sum to the mix."""
    nodes, p, hops = row["indices"], row["probs"], row["hops"]
    depth = int(hops.max()) if max_hop is None else int(max_hop)
    if depth < int(hops.max()):
        raise ValueError("max_hop is less than an observed candidate hop")
    d = X.shape[1]
    mass = np.zeros((repeats, depth), dtype=np.float64)
    vectors = np.zeros((repeats, depth, d), dtype=np.float32)
    endpoint_coefficients = np.zeros(nodes.size, dtype=np.float64)
    endpoint_hits = np.zeros(nodes.size, dtype=np.int64)
    for rep in range(repeats):
        if estimator == "full":
            selected = np.arange(nodes.size)
            weights = p
        else:
            selected = rng.choice(nodes.size, size=draws, replace=True, p=p)
            selected_p = p[selected]
            weights = (selected_p / selected_p.sum() if estimator == "current"
                       else np.full(draws, 1.0 / draws))
        np.add.at(endpoint_coefficients, selected, weights)
        np.add.at(endpoint_hits, selected, 1)
        selected_hops = hops[selected]
        for h in range(1, depth + 1):
            mask = selected_hops == h
            if mask.any():
                mass[rep, h - 1] = float(weights[mask].sum())
                vectors[rep, h - 1] = gate * np.sum(
                    weights[mask, None] * (X[nodes[selected[mask]]] - X[source]),
                    axis=0,
                )
    mean_vectors = vectors.mean(axis=0)
    total = vectors.sum(axis=1)
    norm_total = np.linalg.norm(total, axis=1)
    return {
        "sampling_mass": mass.mean(axis=0),
        "vectors": mean_vectors,
        "vector_norm": np.linalg.norm(mean_vectors, axis=1),
        "mean_draw_vector_norm": np.linalg.norm(vectors, axis=2).mean(axis=0),
        "mean_total_norm": float(norm_total.mean()),
        "mean_total_vector_norm": float(np.linalg.norm(total.mean(axis=0))),
        "mean_total_vector": total.mean(axis=0),
        "mass_repeats": mass,
        "total_norm_repeats": norm_total,
        "endpoint_coefficients": endpoint_coefficients / repeats,
        "endpoint_hits": endpoint_hits,
    }


def deterministic_anchors(X: np.ndarray, limit: int, seed: int) -> np.ndarray:
    """X-only systematic subsample after a seeded permutation."""
    if limit < 1:
        raise ValueError("max-anchors must be positive")
    n = X.shape[0]
    rng = np.random.default_rng(seed + 827)
    return np.sort(rng.choice(n, size=min(n, limit), replace=False))


def max_product_path(graph, edge_weights, source: int, endpoint: int, hop: int):
    """One representative maximum-product walk, not attribution of P^h."""
    states = {source: 0.0}
    parents = []
    for _ in range(hop):
        next_states, pred = {}, {}
        for node, logp in states.items():
            for neighbor, weight in zip(graph.indices[node], edge_weights[node]):
                neighbor = int(neighbor)
                candidate = logp + float(np.log(max(float(weight), 1e-300)))
                if candidate > next_states.get(neighbor, -np.inf):
                    next_states[neighbor] = candidate
                    pred[neighbor] = node
        states = next_states
        parents.append(pred)
    if endpoint not in states:
        raise ValueError("endpoint unreachable at requested hop")
    path = [endpoint]
    for pred in reversed(parents):
        path.append(pred[path[-1]])
    return list(reversed(path)), float(np.exp(states[endpoint]))


def graph_and_gate(X: np.ndarray, cfg, seed: int, core: dict):
    graph = core["build_pca_knn_graph"](
        X, min(cfg.neighbor_k, len(X) - 1), cfg.knn_pca_dim, cfg.tau, seed,
    )
    _, weights, _ = core["compute_edge_reliability"](
        graph, cfg.edge_reliability_mode, cfg.gamma_sim, cfg.gamma_mutual,
        cfg.gamma_snn, cfg.gamma_distance,
    )
    gate_min, gate_max = cfg.gate_bounds()
    gates, _, _ = core["compute_node_gate"](
        graph, weights, "constant" if gate_min == gate_max else "topology",
        gate_min, gate_max, cfg.beta_mutual, cfg.beta_snn, cfg.beta_perturb,
        cfg.beta_uncertainty,
    )
    return graph, weights, gates


def audit_flow(X, graph, edge_weights, gates, cfg, anchors, repeats, seed, core):
    transition = core["_transition_matrix"](graph, edge_weights)
    draws = max(1, min(cfg.mix_neighbors, graph.indices.shape[1]))
    depth, _ = core["resolve_path_variant"](cfg.hop_variant)
    rng = np.random.default_rng(seed + 62024)
    anchor_rows, hop_rows, endpoint_rows, sparse_rows = [], [], [], []
    all_vectors, all_mass = [], []
    for source in anchors:
        source = int(source)
        row = frozen_row(graph, edge_weights, source, cfg.hop_variant,
                         cfg.high_hop_probability_multiplier, core, transition)
        mix_gate = float(np.clip(gates[source] * cfg.mix_amplitude_multiplier, 0, 1))
        result = mc_decompose(X, source, row, mix_gate, draws, repeats, rng,
                              cfg.neighbor_estimator, max_hop=depth)
        candidate_mass = np.bincount(row["hops"], weights=row["probs"],
                                      minlength=9)[1:9]
        mean_vectors = result["vectors"]
        sum_piece_norms = float(np.linalg.norm(mean_vectors, axis=1).sum())
        anchor_rows.append({
            "anchor_index": source,
            "gate": float(gates[source]), "mix_gate": mix_gate,
            "candidate_count": int(row["indices"].size),
            "far_candidate_count": int((row["hops"] >= 2).sum()),
            "far_candidate_probability": float(row["probs"][row["hops"] >= 2].sum()),
            "far_actual_sampling_mass": float(result["sampling_mass"][1:].sum()),
            "mean_displacement_norm": result["mean_total_norm"],
            "expected_displacement_norm": result["mean_total_vector_norm"],
            "direction_retention": (result["mean_total_vector_norm"] / sum_piece_norms
                                    if sum_piece_norms else 0.0),
            "anchor_input_norm": float(np.linalg.norm(X[source])),
            "anchor_processed_nonzero_fraction": float(np.count_nonzero(X[source]) / X.shape[1]),
        })
        for hop in range(1, 9):
            nodes, _, retained = row["chosen"][hop]
            hop_rows.append({
                "anchor_index": source, "hop": hop,
                "candidate_count": int(nodes.size),
                "candidate_probability": float(candidate_mass[hop - 1]),
                "actual_sampling_mass_mc": float(result["sampling_mass"][hop - 1]),
                "path_score_retained": float(retained) if nodes.size else "",
                "expected_contribution_norm": float(result["vector_norm"][hop - 1]),
                "mean_draw_contribution_norm": float(result["mean_draw_vector_norm"][hop - 1]),
            })
        for position, (endpoint, hop, probability) in enumerate(zip(
                row["indices"], row["hops"], row["probs"])):
            endpoint_rows.append({
                "anchor_index": source, "source_index": int(endpoint), "hop": int(hop),
                "candidate_probability": float(probability),
                "mc_actual_weight": float(result["endpoint_coefficients"][position]),
                "mc_draw_hits": int(result["endpoint_hits"][position]),
                "mc_repeats": repeats, "draws_per_repeat": draws,
                "mix_gate": mix_gate,
                "expected_displacement_coefficient":
                    float(mix_gate * result["endpoint_coefficients"][position]),
            })
        all_vectors.append(mean_vectors)
        all_mass.append(result["mass_repeats"])
        sparse_rows.append(row)
    return (anchor_rows, hop_rows, endpoint_rows, sparse_rows,
            np.asarray(all_vectors), np.asarray(all_mass))


def attach_raw_covariates(anchor_rows: list[dict], raw, input_kind: str) -> None:
    """Expose true source-matrix counts, separate from standardized X."""
    indices = np.asarray([row["anchor_index"] for row in anchor_rows], dtype=np.int64)
    if sp.issparse(raw):
        subset = sp.csr_matrix(raw[indices], copy=True)
        subset.sum_duplicates()
        subset.eliminate_zeros()
        nonzero = np.diff(subset.indptr)
        totals = np.asarray(subset.sum(axis=1)).ravel()
    else:
        subset = np.asarray(raw)[indices]
        nonzero = np.count_nonzero(subset, axis=1)
        totals = np.sum(subset, axis=1)
    for position, row in enumerate(anchor_rows):
        row["raw_nonzero_count"] = int(nonzero[position])
        row["raw_nonzero_fraction"] = float(nonzero[position] / raw.shape[1])
        row["raw_row_sum"] = float(totals[position])
        row["raw_library_size"] = (float(totals[position])
                                   if input_kind == "raw_count" else "")


def witness_rows(graph, weights, rows, anchor_rows, limit: int = 3):
    """Pick central far-mass anchors without labels and show representative paths."""
    order = np.argsort([r["far_actual_sampling_mass"] for r in anchor_rows])
    chosen_positions = np.unique(np.linspace(0, len(order) - 1, min(limit, len(order)),
                                             dtype=int))
    output = []
    for rank, position in enumerate(chosen_positions):
        index = int(order[position])
        source = int(anchor_rows[index]["anchor_index"])
        row = rows[index]
        for hop in (1, 5, 6, 7, 8):
            nodes, scores, _ = row["chosen"][hop]
            if not nodes.size:
                continue
            for endpoint, score in zip(nodes[: min(3, nodes.size)], scores[:3]):
                path, max_path_weight = max_product_path(
                    graph, weights, source, int(endpoint), hop,
                )
                output.append({
                    "witness_rank": rank, "anchor_index": source, "hop": hop,
                    "endpoint_index": int(endpoint), "path_score_P_power": float(score),
                    "one_max_product_path": "-".join(map(str, path)),
                    "one_path_weight": max_path_weight,
                    "path_is_attribution": False,
                })
    return output


def feature_block_probe(X: np.ndarray, cfg, anchors, seed: int, core: dict,
                        *, blocks: int = 4, targets_per_block: int = 32,
                        ridge_alpha: float = 10.0) -> tuple[list[dict], dict]:
    """Target-excluded graph, equal-dimensional ridge inputs, graph-region CV.

    This is an exploratory anonymous-feature probe. Target blocks are chosen
    by a fixed seeded permutation, independent of labels or performance.
    """
    if X.shape[1] < blocks * 2:
        raise ValueError("insufficient features for independent probe blocks")
    rng = np.random.default_rng(seed + 9024)
    feature_order = rng.permutation(X.shape[1])
    block_width = min(targets_per_block, X.shape[1] // blocks)
    rows = []
    audits = []
    for block in range(blocks):
        targets = np.sort(feature_order[block * block_width:(block + 1) * block_width])
        predictors = np.setdiff1d(np.arange(X.shape[1]), targets, assume_unique=True)
        Z = X[:, predictors]
        # The probe graph and paths see only predictors. The reference graph
        # used for step 2 remains the original frozen all-feature graph.
        graph, weights, _ = graph_and_gate(Z, cfg, seed, core)
        transition = core["_transition_matrix"](graph, weights)
        local = np.zeros((len(anchors), Z.shape[1]), dtype=np.float32)
        far = np.zeros_like(local)
        high_far = np.zeros_like(local)
        high_far_available = np.zeros(len(anchors), dtype=bool)
        for pos, source in enumerate(anchors):
            row = frozen_row(graph, weights, int(source), cfg.hop_variant,
                             cfg.high_hop_probability_multiplier, core, transition)
            h = row["hops"]
            p = row["probs"]
            near_mask, far_mask = h == 1, h >= 2
            high_mask = h >= 5
            local[pos] = np.sum(Z[row["indices"][near_mask]] *
                                (p[near_mask] / p[near_mask].sum())[:, None], axis=0)
            if far_mask.any():
                far[pos] = np.sum(Z[row["indices"][far_mask]] *
                                       (p[far_mask] / p[far_mask].sum())[:, None], axis=0)
            else:
                far[pos] = local[pos]
            if high_mask.any():
                high_far_available[pos] = True
                high_far[pos] = np.sum(Z[row["indices"][high_mask]] *
                    (p[high_mask] / p[high_mask].sum())[:, None], axis=0)
            else:
                high_far[pos] = local[pos]
        # Both arms have exactly 2 * n_predictors columns and one alpha.
        baseline = np.concatenate([Z[anchors], local], axis=1)
        augmented = np.concatenate([Z[anchors], 0.5 * (local + far)], axis=1)
        high_augmented = np.concatenate([Z[anchors], 0.5 * (local + high_far)], axis=1)
        target = X[anchors][:, targets]
        n_groups = min(4, len(anchors) // 8)
        if n_groups < 2:
            raise ValueError("at least 16 probe anchors required")
        groups = KMeans(n_clusters=n_groups, random_state=seed + block,
                        n_init=10).fit_predict(graph.embedding[anchors])
        # KMeans can produce imbalanced groups, but every fold is a held-out
        # graph region; both arms receive identical folds and target columns.
        folds = GroupKFold(n_splits=n_groups)
        for fold, (train, test) in enumerate(folds.split(baseline, target, groups)):
            for arm, source in (("one_hop", baseline),
                                ("one_hop_plus_far", augmented),
                                ("one_hop_plus_h5h8", high_augmented)):
                model = Ridge(alpha=ridge_alpha).fit(source[train], target[train])
                prediction = model.predict(source[test])
                error = float(mean_squared_error(target[test], prediction))
                rows.append({
                    "block": block, "fold": fold, "arm": arm,
                    "mse": error, "n_train": len(train), "n_test": len(test),
                    "n_predictors": int(len(predictors)), "n_targets": int(len(targets)),
                    "input_dimension": int(source.shape[1]), "ridge_alpha": ridge_alpha,
                    "test_anchor_indices": ";".join(map(str, anchors[test])),
                    "target_feature_indices_processed": ";".join(map(str, targets)),
                })
        audits.append({"block": block, "target_feature_indices_processed": targets.tolist(),
                       "predictor_feature_count": int(predictors.size),
                       "probe_graph_target_columns_excluded": True,
                       "high_hop_available_fraction": float(high_far_available.mean()),
                       "graph_regions": np.bincount(groups, minlength=n_groups).tolist()})
    return rows, {"blocks": audits, "target_selection": "seeded_permutation_before_outcomes",
                  "folds": "GroupKFold_on_predictor_only_graph_PCA_KMeans_regions",
                  "both_arms_input_dimension": "2_times_predictor_feature_count",
                  "augmented_context": "half_onehop_plus_half_far_expected_distribution",
                  "high_hop_context": "half_onehop_plus_half_hops5to8_else_onehop",
                  "estimator_note": "probe uses candidate distribution means, not current-estimator Monte Carlo"}


def checkpoint_sensitivity(X, anchors, vectors, lock_row, selected_record,
                           seed: int, device_name: str, core: dict):
    """Fixed encoder response to equal-norm near/far mean perturbations.

    No model is trained here. This response statistic is not causal evidence
    about what would happen if an altered source were used during training.
    """
    import torch
    from v0rg_tunable_multihop.model import WeightedAutoEncoder

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device unavailable: {device_name}")
    anchor_X = X[anchors]
    near = vectors[:, 0, :]
    far = vectors[:, 1:, :].sum(axis=1)
    high = vectors[:, 4:, :].sum(axis=1)
    target_norm = np.linalg.norm(vectors.sum(axis=1), axis=1)
    def matched(delta):
        norm = np.linalg.norm(delta, axis=1)
        scale = np.divide(target_norm, norm, out=np.zeros_like(norm), where=norm > 1e-12)
        return anchor_X + delta * scale[:, None], norm > 1e-12
    inputs = {"one_hop": matched(near), "far_all": matched(far),
              "far_h5h8": matched(high)}
    outcomes = []
    paths = {"H8_selected": Path(lock_row["selected_records"][str(seed)]),
             "H1_locked": Path(lock_row["h1_records"][str(seed)])}
    for model_name, record_path in paths.items():
        record = selected_record if model_name == "H8_selected" else json.loads(
            record_path.read_text(encoding="utf-8"))
        if (record.get("status") != "completed" or record.get("seed") != seed or
                record.get("feature_sha256") != selected_record.get("feature_sha256") or
                record.get("preprocessing_hash") != selected_record.get("preprocessing_hash")):
            raise ValueError(f"checkpoint provenance mismatch: {model_name}")
        cfg = core["V0_RGConfig"](**record["config"])
        state_path = record_path.parent / f"model_seed_{seed}.pt"
        if not state_path.exists():
            raise FileNotFoundError(state_path)
        state_sha = sha256(state_path)
        model = WeightedAutoEncoder(
            num_genes=X.shape[1], hidden_size=cfg.hidden_size, dropout=cfg.dropout,
            masked_data_weight=cfg.masked_data_weight,
            mask_loss_weight=cfg.mask_loss_weight,
        ).to(device)
        state = torch.load(state_path, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        def encode(values):
            output = []
            with torch.no_grad():
                for start in range(0, len(values), 256):
                    batch = torch.as_tensor(values[start:start + 256], dtype=torch.float32,
                                            device=device)
                    output.append(model.feature(batch).cpu().numpy())
            return np.concatenate(output)
        clean = encode(anchor_X)
        clean_norm = np.linalg.norm(clean, axis=1)
        for arm, (input_values, available) in inputs.items():
            response = encode(input_values)
            displacement = np.linalg.norm(response - clean, axis=1)
            cosine = np.sum(response * clean, axis=1) / np.maximum(
                np.linalg.norm(response, axis=1) * clean_norm, 1e-12)
            for pos, source in enumerate(anchors):
                outcomes.append({
                    "anchor_index": int(source), "encoder": model_name, "arm": arm,
                    "source_available": bool(available[pos]),
                    "input_displacement_norm_matched": float(target_norm[pos]) if available[pos] else 0.0,
                    "clean_embedding_norm": float(clean_norm[pos]),
                    "embedding_shift_norm": float(displacement[pos]),
                    "embedding_shift_relative_to_clean": float(displacement[pos] / max(clean_norm[pos], 1e-12)),
                    "embedding_cosine_distance": float(1.0 - np.clip(cosine[pos], -1, 1)),
                    "checkpoint_sha256": state_sha,
                })
    return outcomes


def run(args) -> None:
    core = load_core(args.core)
    lock_path = Path(args.lock)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "LOCKED_BEFORE_TEST":
        raise ValueError("selection lock lacks LOCKED_BEFORE_TEST status")
    row = next((x for x in lock["rows"] if x["dataset"] == args.dataset), None)
    if row is None or args.dataset not in INPUT_KIND or args.seed not in SEEDS:
        raise ValueError("dataset/seed outside prespecified mechanism cases")
    record_path = Path(row["selected_records"][str(args.seed)])
    record = json.loads(record_path.read_text(encoding="utf-8"))
    manifest_path = Path(row["tuning_manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if any(key in manifest for key in ("test_labels_path", "test_label_path")):
        raise ValueError("tuning manifest exposes test labels")
    if manifest.get("feature_sha256") != row["feature_sha256"] or \
       record.get("feature_sha256") != row["feature_sha256"] or \
       record.get("seed") != args.seed or record.get("status") != "completed":
        raise ValueError("lock/manifest/record identity mismatch")
    cfg = core["V0_RGConfig"](**record["config"])
    if cfg.hop_variant != row["selected_config_id"].split("_q", 1)[0]:
        raise ValueError("selected hop configuration mismatch")
    if cfg.neighbor_estimator != "current":
        raise ValueError("this mechanism audit requires the locked current estimator")
    feature_path = Path(manifest["feature_path"])
    if sha256(feature_path) != row["feature_sha256"]:
        raise ValueError("feature-only input hash mismatch")
    loaded = core["load_npz_matrix_only"](feature_path)
    raw = loaded.X
    prep = core["SplitPreprocessor"](INPUT_KIND[args.dataset], FEATURE_LIMIT).fit(raw)
    if prep.fingerprint != record["preprocessing_hash"]:
        raise ValueError("preprocessing fingerprint does not match selected record")
    X = prep.transform(raw)
    if X.shape[0] != row["n_samples"]:
        raise ValueError("sample count differs from lock")
    out = Path(args.out) / args.dataset / f"seed_{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    stage_status = {"step2_flow": "pending", "step3_metadata": "pending",
                    "step4_probe": "pending", "step4_checkpoint_sensitivity": "pending"}
    def checkpoint(status: str) -> None:
        write_json(out / "steps234_summary.json", {
            "status": status, "dataset": args.dataset, "seed": args.seed,
            "stages": stage_status, "benchmark_labels_loaded": False,
            "historical_training_trace_recovered": False,
        })
    checkpoint("partial")
    anchors = deterministic_anchors(X, args.max_anchors, args.seed)
    graph, edge_weights, gates = graph_and_gate(X, cfg, args.seed, core)
    anchor_rows, hop_rows, endpoint_rows, frozen_rows, vectors, mass = audit_flow(
        X, graph, edge_weights, gates, cfg, anchors, args.mc_repeats, args.seed, core,
    )
    attach_raw_covariates(anchor_rows, raw, INPUT_KIND[args.dataset])
    write_csv(out / "step2_anchors.csv", anchor_rows)
    write_csv(out / "step2_hops.csv", hop_rows)
    write_csv(out / "step2_endpoints.csv", endpoint_rows)
    write_csv(out / "step3_path_witnesses.csv", witness_rows(
        graph, edge_weights, frozen_rows, anchor_rows,
    ))
    with (out / "step2_vectors.npz").open("wb") as handle:
        np.savez_compressed(handle, anchors=anchors, expected_hop_vectors=vectors,
                            sampling_mass_repeats=mass,
                            original_pca_coordinates=graph.embedding[anchors, :2],
                            selected_feature_indices=prep.features)
    stage_status["step2_flow"] = "completed"
    checkpoint("partial")
    # The source NPZ is feature-only and contains no trusted feature/cell/donor
    # identities. Biological source H5AD index strings are placeholders.
    metadata = {
        "feature_identity_verified": False, "cell_identity_verified": False,
        "donor_identity_verified": False,
        "feature_semantics": "anonymous_processed_columns",
        "path_witness_semantics": "one_possible_max_product_shortest_path_not_P_power_attribution",
        "raw_matrix_shape": list(raw.shape),
        "raw_zero_fraction": float(1.0 - raw.nnz / (raw.shape[0] * raw.shape[1]))
        if hasattr(raw, "nnz") else float(np.mean(np.asarray(raw) == 0)),
        "raw_row_sum_quantiles": np.quantile(np.asarray(raw.sum(axis=1)).ravel(),
                                             [0, .25, .5, .75, 1]).tolist(),
        "preprocessing_selected_feature_indices_saved": True,
    }
    write_json(out / "step3_metadata_availability.json", metadata)
    stage_status["step3_metadata"] = "completed"
    checkpoint("partial")
    probe_rows, probe_audit = feature_block_probe(
        X, cfg, anchors, args.seed, core, blocks=args.probe_blocks,
        targets_per_block=args.probe_targets,
    )
    write_csv(out / "step4_probe_folds.csv", probe_rows)
    write_json(out / "step4_probe_audit.json", probe_audit)
    stage_status["step4_probe"] = "completed"
    checkpoint("partial")
    try:
        sensitivity = checkpoint_sensitivity(
            X, anchors, vectors, row, record, args.seed, args.device, core,
        )
        write_csv(out / "step4_checkpoint_sensitivity.csv", sensitivity)
        stage_status["step4_checkpoint_sensitivity"] = "completed"
    except (FileNotFoundError, ImportError, RuntimeError) as exc:
        stage_status["step4_checkpoint_sensitivity"] = "unavailable"
        write_json(out / "step4_checkpoint_sensitivity_unavailable.json", {
            "error_type": type(exc).__name__, "error": str(exc),
            "meaning": "optional fixed-checkpoint diagnostic unavailable; main steps complete",
        })
    write_json(out / "provenance.json", {
        "status": "completed", "dataset": args.dataset, "seed": args.seed,
        "selection_lock_sha256": sha256(lock_path), "record_path": str(record_path),
        "record_sha256": sha256(record_path), "tuning_manifest_path": str(manifest_path),
        "feature_sha256": sha256(feature_path), "preprocessing_hash": prep.fingerprint,
        "config": record["config"], "feature_only": True,
        "benchmark_labels_loaded": False,
        "historical_training_trace_recovered": False,
        "sampling_distribution_reconstructed": True,
        "n_anchors": len(anchors), "max_anchors": args.max_anchors,
        "anchor_rule": "seeded_uniform_X_only", "mc_repeats": args.mc_repeats,
        "probe_blocks": args.probe_blocks, "probe_targets_per_block": args.probe_targets,
        "core_path": str(args.core.resolve()),
        "cpu_cost_caps": {"anchors": args.max_anchors, "mc_repeats": args.mc_repeats,
                          "probe_blocks": args.probe_blocks,
                          "probe_targets_per_block": args.probe_targets,
                          "probe_graph_builds": args.probe_blocks + 1},
    })
    checkpoint("completed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--core", type=Path, required=True,
                        help="parent directory containing v0rg_tunable_multihop")
    parser.add_argument("--dataset", choices=tuple(INPUT_KIND), required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--device", default="cpu", help="reserved; graph/probe use CPU")
    parser.add_argument("--max-anchors", type=int, default=512)
    parser.add_argument("--mc-repeats", type=int, default=16)
    parser.add_argument("--probe-blocks", type=int, default=4)
    parser.add_argument("--probe-targets", type=int, default=32)
    args = parser.parse_args()
    if args.mc_repeats < 2 or args.probe_blocks < 2 or args.probe_targets < 1:
        parser.error("mc-repeats>=2, probe-blocks>=2, probe-targets>=1 required")
    try:
        run(args)
    except Exception as exc:
        failure_path = args.out / args.dataset / f"seed_{args.seed}" / "steps234_summary.json"
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        existing = json.loads(failure_path.read_text(encoding="utf-8")) if failure_path.exists() else {}
        existing.update({"status": "partial", "dataset": args.dataset, "seed": args.seed,
                         "error_type": type(exc).__name__, "error": str(exc)})
        existing.setdefault("stages", {"step2_flow": "pending", "step3_metadata": "pending",
                                       "step4_probe": "pending",
                                       "step4_checkpoint_sensitivity": "pending"})
        write_json(failure_path, existing)
        raise


if __name__ == "__main__":
    main()
