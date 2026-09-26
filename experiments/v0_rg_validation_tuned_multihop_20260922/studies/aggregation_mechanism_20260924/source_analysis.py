"""Post hoc aggregation and fragment analysis of the frozen step 2 sources.

The graph, fragments, anchors, endpoints and MC weights are label free. Original
full-X labels are loaded only after they have been fixed, for descriptive joins.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import scipy.sparse as sp

DATASETS = ("Tosches", "Baron", "fabert")
SEEDS = (42, 123, 7)
FRAGMENT_SEED = 20260924
FRAGMENT_RESOLUTION = 1.0
HOP_GROUPS = ("H1", "H2_4", "H5_8")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if fields is None:
        if not rows:
            raise ValueError(f"cannot infer fields for empty CSV: {path}")
        fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fragment_ids(indices: np.ndarray, weights: np.ndarray, n: int,
                 resolution: float = FRAGMENT_RESOLUTION,
                 seed: int = FRAGMENT_SEED) -> tuple[np.ndarray, nx.Graph]:
    """Louvain on a symmetric union graph; reciprocal weights are averaged."""
    if indices.shape != weights.shape or indices.shape[0] != n:
        raise ValueError("graph indices/weights shape mismatch")
    directed: dict[tuple[int, int], float] = {}
    for u in range(n):
        for v, w in zip(indices[u], weights[u]):
            v = int(v)
            if v != u and 0 <= v < n and np.isfinite(w) and w > 0:
                directed[u, v] = float(w)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for (u, v), w in directed.items():
        if u < v:
            graph.add_edge(u, v, weight=(w + directed[v, u]) / 2 if (v, u) in directed else w)
        elif (v, u) not in directed:
            graph.add_edge(u, v, weight=w)
    groups = nx.community.louvain_communities(graph, weight="weight",
                                               resolution=resolution, seed=seed)
    # Louvain IDs have no meaning; sort by the minimum original row index.
    groups = sorted(groups, key=min)
    ids = np.full(n, -1, dtype=np.int32)
    for group_id, members in enumerate(groups):
        ids[list(members)] = group_id
    if np.any(ids < 0):
        raise AssertionError("unassigned fragment member")
    return ids, graph


def embedding_scale(embedding: np.ndarray) -> float:
    """RMS pairwise Euclidean distance, computed from the global variance."""
    embedding = np.asarray(embedding, dtype=np.float64)
    if embedding.ndim != 2 or not np.isfinite(embedding).all():
        raise ValueError("embedding must be a finite 2D array")
    scale = float(np.sqrt(2 * np.var(embedding, axis=0).sum()))
    if scale <= 0:
        raise ValueError("embedding has zero global scale")
    return scale


def raw_statistics(raw) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    matrix = sp.csr_matrix(raw, dtype=np.float64, copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    nonzero = np.diff(matrix.indptr).astype(np.int64)
    row_sum = np.asarray(matrix.sum(axis=1)).ravel()
    l2 = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    return matrix, nonzero, row_sum, l2


def feature_pairs(raw: sp.csr_matrix, anchor: int, sources: np.ndarray,
                  nonzero: np.ndarray, l2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Only selected anchor-source pairs are multiplied; no N by N matrix."""
    subset = raw[sources]
    anchor_row = raw.getrow(anchor)
    dot = (subset @ anchor_row.T).toarray().ravel()
    denom = l2[sources] * l2[anchor]
    cosine = np.divide(dot, denom, out=np.zeros_like(dot), where=denom > 0)
    binary = subset.copy()
    binary.data[:] = 1
    anchor_binary = anchor_row.copy()
    anchor_binary.data[:] = 1
    intersection = (binary @ anchor_binary.T).toarray().ravel()
    union = nonzero[sources] + nonzero[anchor] - intersection
    overlap = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union > 0)
    return cosine, overlap


def hop_group(hop: int) -> str:
    if hop == 1:
        return "H1"
    if 2 <= hop <= 4:
        return "H2_4"
    if 5 <= hop <= 8:
        return "H5_8"
    raise ValueError(f"unexpected hop: {hop}")


def verify_frozen_candidates(endpoint_rows: list[dict], anchors: list[int],
                             graph, edge_weights: np.ndarray, cfg, prior, core: dict) -> int:
    """Hard-stop unless every saved endpoint is the reconstructed frozen row."""
    transition = core["_transition_matrix"](graph, edge_weights)
    by_anchor: dict[int, list[dict]] = defaultdict(list)
    for item in endpoint_rows:
        by_anchor[int(item["anchor_index"])].append(item)
    if set(by_anchor) != set(anchors):
        raise ValueError("saved candidate anchor inventory differs from selected anchors")
    checked = 0
    for anchor in anchors:
        actual = prior.frozen_row(graph, edge_weights, anchor, cfg.hop_variant,
                                  cfg.high_hop_probability_multiplier, core, transition)
        saved = by_anchor[anchor]
        if len(saved) != len(actual["indices"]):
            raise ValueError(f"candidate count mismatch for anchor {anchor}")
        old_nodes = np.fromiter((int(r["source_index"]) for r in saved), dtype=np.int64)
        old_hops = np.fromiter((int(r["hop"]) for r in saved), dtype=np.int64)
        old_prob = np.fromiter((float(r["candidate_probability"]) for r in saved), dtype=np.float64)
        if not np.array_equal(old_nodes, actual["indices"]):
            raise ValueError(f"candidate endpoint IDs/order mismatch for anchor {anchor}")
        if not np.array_equal(old_hops, actual["hops"]):
            raise ValueError(f"candidate hop/order mismatch for anchor {anchor}")
        if not np.allclose(old_prob, actual["probs"], rtol=1e-6, atol=1e-8):
            raise ValueError(f"candidate probability mismatch for anchor {anchor}")
        checked += len(saved)
    return checked


def summarize_anchor_pairs(anchor: int, rows: list[dict], y: np.ndarray,
                           fragments: np.ndarray, class_fragment_counts: dict[int, int]) -> list[dict]:
    """Complete per-anchor group accounting, including zero sampled mass."""
    result = []
    for group in HOP_GROUPS:
        subset = [r for r in rows if r["hop_group"] == group]
        mass = sum(r["mc_actual_weight"] for r in subset)
        candidate_mass = sum(r["candidate_probability"] for r in subset)
        cross_same = [r for r in subset if r["same_class"] and not r["same_fragment"]]
        sampled_fragments = {int(r["source_fragment"]) for r in cross_same if r["mc_actual_weight"] > 0}
        candidate_fragments = {int(r["source_fragment"]) for r in cross_same}
        global_possible = max(0, class_fragment_counts[int(y[anchor])] - 1)
        result.append({
            "anchor_index": anchor, "anchor_class_id": int(y[anchor]),
            "anchor_fragment": int(fragments[anchor]), "hop_group": group,
            "candidate_count": len(subset), "candidate_mass": candidate_mass,
            "mc_actual_mass": mass,
            "same_class_cross_fragment_candidate_mass": sum(r["candidate_probability"] for r in cross_same),
            "same_class_cross_fragment_mc_mass": sum(r["mc_actual_weight"] for r in cross_same),
            "same_class_cross_fragment_fraction_of_group_mass":
                sum(r["mc_actual_weight"] for r in cross_same) / mass if mass > 0 else None,
            "same_class_cross_fragment_candidate_fragments": len(candidate_fragments),
            "same_class_cross_fragment_sampled_fragments": len(sampled_fragments),
            "same_class_cross_fragment_coverage_of_candidate":
                len(sampled_fragments) / len(candidate_fragments) if candidate_fragments else None,
            "same_class_cross_fragment_coverage_of_global":
                len(sampled_fragments) / global_possible if global_possible else None,
        })
    return result


def analyze_sources(endpoint_rows: list[dict], anchors: list[int], y: np.ndarray,
                    fragments: np.ndarray, raw, h1: np.ndarray, h8: np.ndarray,
                    input_kind: str, smoke: bool = False) -> tuple[list[dict], list[dict], list[dict]]:
    """Join all frozen endpoints; selection and weights are never recomputed."""
    raw, nonzero, row_sum, l2 = raw_statistics(raw)
    scale1, scale8 = embedding_scale(h1), embedding_scale(h8)
    class_fragment_counts = {}
    for class_id in np.unique(y):
        class_fragment_counts[int(class_id)] = len(np.unique(fragments[y == class_id]))
    by_anchor: dict[int, list[dict]] = defaultdict(list)
    for row in endpoint_rows:
        by_anchor[int(row["anchor_index"])].append(row)
    if set(by_anchor) != set(anchors):
        raise ValueError("frozen endpoint and anchor sets differ")
    pair_rows, anchor_rows = [], []
    for anchor in anchors:
        previous = by_anchor[anchor]
        sources = np.asarray([int(r["source_index"]) for r in previous], dtype=np.int64)
        if np.any((sources < 0) | (sources >= len(y))):
            raise ValueError("source index outside matrix")
        cosine, overlap = feature_pairs(raw, anchor, sources, nonzero, l2)
        distance1 = np.linalg.norm(h1[sources] - h1[anchor], axis=1) / scale1
        distance8 = np.linalg.norm(h8[sources] - h8[anchor], axis=1) / scale8
        local = []
        for pos, old in enumerate(previous):
            source = int(sources[pos])
            p = float(old["candidate_probability"])
            w = float(old["mc_actual_weight"])
            if p < 0 or w < 0:
                raise ValueError("negative frozen source weight")
            local.append({
                "anchor_index": anchor, "source_index": source, "hop": int(old["hop"]),
                "hop_group": hop_group(int(old["hop"])),
                "candidate_probability": p, "mc_actual_weight": w,
                "mc_draw_hits": int(old["mc_draw_hits"]),
                "anchor_class_id": int(y[anchor]), "source_class_id": int(y[source]),
                "same_class": int(y[anchor] == y[source]),
                "anchor_fragment": int(fragments[anchor]), "source_fragment": int(fragments[source]),
                "same_fragment": int(fragments[anchor] == fragments[source]),
                "h1_distance_global_rms": float(distance1[pos]),
                "h8_distance_global_rms": float(distance8[pos]),
                "h8_minus_h1_normalized_distance": float(distance8[pos] - distance1[pos]),
                "raw_feature_cosine": float(cosine[pos]),
                "raw_nonzero_jaccard": float(overlap[pos]),
                "anchor_raw_nonzero": int(nonzero[anchor]), "source_raw_nonzero": int(nonzero[source]),
                "anchor_raw_row_sum": float(row_sum[anchor]), "source_raw_row_sum": float(row_sum[source]),
                "anchor_library_size": float(row_sum[anchor]) if input_kind == "raw_count" else None,
                "source_library_size": float(row_sum[source]) if input_kind == "raw_count" else None,
            })
        if not np.isclose(sum(r["candidate_probability"] for r in local), 1, atol=1e-5):
            raise ValueError(f"candidate probabilities do not sum to one for anchor {anchor}")
        if not np.isclose(sum(r["mc_actual_weight"] for r in local), 1, atol=1e-5):
            raise ValueError(f"MC weights do not sum to one for anchor {anchor}")
        pair_rows.extend(local)
        anchor_rows.extend(summarize_anchor_pairs(anchor, local, y, fragments, class_fragment_counts))
    # Cross-fragment distances are endpoint-pair descriptive distributions; each
    # source appears once per candidate row, not once per graph path.
    summary = []
    for group in HOP_GROUPS:
        for relation in ("same_class_cross_fragment", "cross_class_cross_fragment",
                         "same_class_same_fragment", "cross_class_same_fragment"):
            same_class = relation.startswith("same_class")
            same_fragment = relation.endswith("same_fragment")
            subset = [r for r in pair_rows if r["hop_group"] == group
                      and bool(r["same_class"]) == same_class
                      and bool(r["same_fragment"]) == same_fragment]
            weights = np.asarray([r["mc_actual_weight"] for r in subset], dtype=float)
            def average(key):
                values = np.asarray([r[key] for r in subset], dtype=float)
                return float(np.average(values, weights=weights)) if weights.sum() > 0 else None
            summary.append({
                "hop_group": group, "relation": relation, "candidate_pair_count": len(subset),
                "mc_mass": float(weights.sum()),
                "h1_distance_global_rms_mc_weighted": average("h1_distance_global_rms"),
                "h8_distance_global_rms_mc_weighted": average("h8_distance_global_rms"),
                "h8_minus_h1_normalized_distance_mc_weighted": average("h8_minus_h1_normalized_distance"),
                "raw_feature_cosine_mc_weighted": average("raw_feature_cosine"),
                "raw_nonzero_jaccard_mc_weighted": average("raw_nonzero_jaccard"),
                "source_raw_nonzero_mc_weighted": average("source_raw_nonzero"),
                "source_library_size_mc_weighted": average("source_library_size") if input_kind == "raw_count" else None,
            })
    return pair_rows, anchor_rows, summary


def selected_illustrations(anchor_records: list[dict]) -> list[dict]:
    """Choose lower/median/upper far-mass exemplars with no labels or outcomes."""
    ordered = sorted(anchor_records, key=lambda r: (float(r["far_actual_sampling_mass"]), int(r["anchor_index"])))
    positions = sorted(set(int(round(q * (len(ordered) - 1))) for q in (0.1, 0.5, 0.9)))
    return [{"quantile": q, "rank_index": p, "anchor_index": int(ordered[p]["anchor_index"]),
             "far_actual_sampling_mass": float(ordered[p]["far_actual_sampling_mass"])}
            for q, p in zip((0.1, 0.5, 0.9), positions)]


def observed_fragment_distances(pair_rows: list[dict], y: np.ndarray,
                                fragments: np.ndarray, h1: np.ndarray,
                                h8: np.ndarray) -> list[dict]:
    """Class-fragment centroid distances for fragment pairs in frozen links."""
    groups, inverse, counts = np.unique(np.column_stack((y, fragments)), axis=0,
                                        return_inverse=True, return_counts=True)
    centroids = []
    for embedding in (h1, h8):
        totals = np.zeros((len(groups), embedding.shape[1]), dtype=np.float64)
        np.add.at(totals, inverse, embedding)
        centroids.append(totals / counts[:, None])
    observed: dict[tuple[int, int], dict] = {}
    for row in pair_rows:
        if row["same_fragment"]:
            continue
        a, b = sorted((int(inverse[row["anchor_index"]]), int(inverse[row["source_index"]])))
        key = (a, b)
        if key not in observed:
            observed[key] = {"candidate_pair_count": 0, "candidate_mass": 0.0, "mc_mass": 0.0}
        observed[key]["candidate_pair_count"] += 1
        observed[key]["candidate_mass"] += row["candidate_probability"]
        observed[key]["mc_mass"] += row["mc_actual_weight"]
    scale1, scale8 = embedding_scale(h1), embedding_scale(h8)
    result = []
    for (a, b), acc in sorted(observed.items()):
        d1 = float(np.linalg.norm(centroids[0][a] - centroids[0][b]) / scale1)
        d8 = float(np.linalg.norm(centroids[1][a] - centroids[1][b]) / scale8)
        result.append({
            "class_id_a": int(groups[a, 0]), "fragment_id_a": int(groups[a, 1]),
            "class_fragment_size_a": int(counts[a]),
            "class_id_b": int(groups[b, 0]), "fragment_id_b": int(groups[b, 1]),
            "class_fragment_size_b": int(counts[b]),
            "same_class": int(groups[a, 0] == groups[b, 0]),
            "h1_centroid_distance_global_rms": d1,
            "h8_centroid_distance_global_rms": d8,
            "h8_minus_h1_normalized_centroid_distance": d8 - d1,
            **acc,
        })
    return result


def draw_illustrations(out: Path, original_layout: np.ndarray, pair_rows: list[dict],
                       selections: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    coords = np.asarray(original_layout, dtype=float)
    selected = {s["anchor_index"] for s in selections}
    endpoints = defaultdict(list)
    for row in pair_rows:
        if row["anchor_index"] in selected and row["mc_actual_weight"] > 0:
            endpoints[row["anchor_index"]].append(row)
    fig, axes = plt.subplots(len(selections), 2, figsize=(8.4, 3.2 * len(selections)), squeeze=False)
    low, high = np.percentile(coords, [0.5, 99.5], axis=0)
    for k, choice in enumerate(selections):
        anchor = choice["anchor_index"]
        rows = endpoints[anchor]
        for column, (key, title) in enumerate((("h1_distance_global_rms", "H1"),
                                               ("h8_distance_global_rms", "H8"))):
            ax = axes[k, column]
            ax.scatter(coords[:, 0], coords[:, 1], s=1, color="#c8c8c8", alpha=.28, rasterized=True)
            for group, color in (("H1", "#277da1"), ("H2_4", "#f9a03f"), ("H5_8", "#b5465d")):
                chosen = [r for r in rows if r["hop_group"] == group]
                ids = [r["source_index"] for r in chosen]
                if ids:
                    size = [8 + 35 * np.exp(-r[key]) for r in chosen]
                    ax.scatter(coords[ids, 0], coords[ids, 1], s=size, color=color,
                               alpha=.72, label=group, rasterized=True)
            ax.scatter(*coords[anchor], s=58, marker="*", color="black", zorder=5)
            weighted_distance = (sum(r[key] * r["mc_actual_weight"] for r in rows)
                                 / sum(r["mc_actual_weight"] for r in rows)) if rows else float("nan")
            ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]),
                   title=f"{title}: anchor {anchor}, weighted distance {weighted_distance:.2f}")
            ax.set_aspect("equal", adjustable="box")
            if k == 0 and column == 1:
                ax.legend(frameon=False, markerscale=1.5)
    fig.tight_layout()
    fig.savefig(out / "illustrative_sources.png", dpi=180)
    fig.savefig(out / "illustrative_sources.pdf")
    plt.close(fig)


def run(args) -> None:
    previous = Path(args.previous_root).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    prior = load_module(previous / "steps234_information.py", "frozen_steps234_information")
    outcome = load_module(previous / "step1_outcomes.py", "frozen_step1_outcomes")
    lock_path = previous / "inputs" / "selection_lock.json"
    background_path = previous / "inputs" / "validation_best_multihop_vs_h1_full_x_60.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "LOCKED_BEFORE_TEST":
        raise ValueError("selection lock is not frozen")
    selection = next(r for r in lock["rows"] if r["dataset"] == args.dataset)
    background = json.loads(background_path.read_text(encoding="utf-8"))
    source_row = next(r for r in background["rows"] if r["dataset"] == args.dataset)
    if source_row["multihop_config_id"] != selection["selected_config_id"]:
        raise ValueError("background and lock selected configs differ")
    record_path = Path(selection["selected_records"][str(args.seed)])
    h1_record_path = Path(selection["h1_records"][str(args.seed)])
    record = json.loads(record_path.read_text(encoding="utf-8"))
    h1_record = json.loads(h1_record_path.read_text(encoding="utf-8"))
    for item in (record, h1_record):
        if (item.get("status") != "completed" or item.get("seed") != args.seed
                or item.get("feature_sha256") != selection["feature_sha256"]
                or item.get("labels_used_during_fit") is not False):
            raise ValueError("record failed frozen identity/fit audit")
    core = prior.load_core(previous.parent)
    manifest = json.loads(Path(selection["tuning_manifest_path"]).read_text(encoding="utf-8"))
    feature_path = Path(manifest["feature_path"])
    if (manifest.get("feature_sha256") != selection["feature_sha256"]
            or prior.sha256(feature_path) != selection["feature_sha256"]):
        raise ValueError("feature-only matrix hash mismatch")
    raw = core["load_npz_matrix_only"](feature_path).X
    prep = core["SplitPreprocessor"](prior.INPUT_KIND[args.dataset], prior.FEATURE_LIMIT).fit(raw)
    if prep.fingerprint != record["preprocessing_hash"] or prep.fingerprint != h1_record["preprocessing_hash"]:
        raise ValueError("preprocessor fingerprint mismatch")
    X = prep.transform(raw)
    cfg = core["V0_RGConfig"](**record["config"])
    graph, edge_weights, _ = prior.graph_and_gate(X, cfg, args.seed, core)
    prior_output = previous / "outputs" / "steps234" / args.dataset / f"seed_{args.seed}"
    provenance = json.loads((prior_output / "provenance.json").read_text(encoding="utf-8"))
    if (provenance.get("status") != "completed" or provenance.get("dataset") != args.dataset
            or provenance.get("seed") != args.seed
            or provenance.get("feature_sha256") != selection["feature_sha256"]
            or provenance.get("preprocessing_hash") != prep.fingerprint
            or provenance.get("selection_lock_sha256") != digest(lock_path)
            or provenance.get("record_sha256") != digest(record_path)
            or provenance.get("config") != record["config"]):
        raise ValueError("prior step 2 provenance mismatch")
    frozen_anchors = read_csv(prior_output / "step2_anchors.csv")
    anchors = [int(r["anchor_index"]) for r in frozen_anchors]
    if len(anchors) != len(set(anchors)) or len(anchors) != provenance["n_anchors"]:
        raise ValueError("prior anchor inventory mismatch")
    endpoint_rows = read_csv(prior_output / "step2_endpoints.csv")
    if args.smoke:
        anchors = anchors[:16]
        endpoint_rows = [r for r in endpoint_rows if int(r["anchor_index"]) in set(anchors)]
        frozen_anchors = [r for r in frozen_anchors if int(r["anchor_index"]) in set(anchors)]
    checked = verify_frozen_candidates(endpoint_rows, anchors, graph, edge_weights,
                                       cfg, prior, core)
    fragments, symmetric_graph = fragment_ids(graph.indices, edge_weights, len(X))
    np.save(out / "fragment_ids.npy", fragments)
    positions = np.asarray(graph.embedding[:, :2], dtype=float)
    if positions.shape[1] != 2:
        raise ValueError("original graph PCA layout has fewer than two coordinates")
    np.save(out / "original_layout.npy", positions.astype(np.float32))
    layout = []
    for fid in np.unique(fragments):
        members = fragments == fid
        centroid = positions[members].mean(axis=0)
        layout.append({"fragment_id": int(fid), "size": int(members.sum()),
                       "pca1_centroid": float(centroid[0]),
                       "pca2_centroid": float(centroid[1]) if len(centroid) > 1 else 0.0})
    write_csv(out / "fragment_layout.csv", layout)
    with np.load(record_path.with_name(f"embedding_seed_{args.seed}.npz"), allow_pickle=False) as z:
        h8 = np.asarray(z["embedding"], dtype=np.float32)
    with np.load(h1_record_path.with_name(f"embedding_seed_{args.seed}.npz"), allow_pickle=False) as z:
        h1 = np.asarray(z["embedding"], dtype=np.float32)
    if h1.shape[0] != len(X) or h8.shape[0] != len(X):
        raise ValueError("embedding row count does not match feature matrix")
    y, _ = outcome.load_labels(Path(source_row["source_path"]))
    if len(y) != len(X):
        raise ValueError("full-X labels do not align with feature rows")
    np.save(out / "labels_true.npy", y.astype(np.int64))
    pairs, per_anchor, pair_summary = analyze_sources(
        endpoint_rows, anchors, y, fragments, raw, h1, h8, prior.INPUT_KIND[args.dataset], args.smoke)
    write_csv(out / "anchor_source_links.csv", pairs)
    write_csv(out / "source_per_anchor.csv", per_anchor)
    write_csv(out / "source_pair_summary.csv", pair_summary)
    fragment_pairs = observed_fragment_distances(pairs, y, fragments, h1, h8)
    write_csv(out / "fragment_pair_distances.csv", fragment_pairs,
              fields=["class_id_a", "fragment_id_a", "class_fragment_size_a",
                      "class_id_b", "fragment_id_b", "class_fragment_size_b",
                      "same_class", "h1_centroid_distance_global_rms",
                      "h8_centroid_distance_global_rms",
                      "h8_minus_h1_normalized_centroid_distance",
                      "candidate_pair_count", "candidate_mass", "mc_mass"])
    write_csv(out / "illustrative_selection.csv", selected_illustrations(frozen_anchors))
    draw_illustrations(out, positions, pairs, selected_illustrations(frozen_anchors))
    summary = {
        "status": "completed", "dataset": args.dataset, "seed": args.seed,
        "smoke": bool(args.smoke), "posthoc_full_x_labels_loaded": True,
        "labels_used_for_model_fit": False, "labels_used_for_graph_or_fragments": False,
        "fragment_method": "networkx_louvain_on_symmetrized_union_of_locked_directed_pca_knn_graph",
        "fragment_seed": FRAGMENT_SEED, "fragment_resolution": FRAGMENT_RESOLUTION,
        "symmetrization": "single directed edge retains its weight; reciprocal directed weights averaged",
        "n_samples": len(X), "n_fragments": len(layout), "n_symmetric_graph_edges": symmetric_graph.number_of_edges(),
        "n_anchors": len(anchors), "n_source_candidate_pairs": len(pairs),
        "frozen_candidate_rows_recomputed_and_verified": checked,
        "n_observed_cross_fragment_centroid_pairs": len(fragment_pairs),
        "h1_global_embedding_rms_pair_distance": embedding_scale(h1),
        "h8_global_embedding_rms_pair_distance": embedding_scale(h8),
        "raw_row_sum_semantics": "library_size" if prior.INPUT_KIND[args.dataset] == "raw_count" else "row_sum_not_library_size",
        "pair_distance_semantics": "frozen_anchor_candidate_endpoint_pairs; cross-fragment within/cross-class posthoc strata",
        "fragment_distance_semantics": "full-X anonymous class-by-fragment centroids, only pairs linked by frozen candidates; descriptive posthoc",
        "source_selection": "all_frozen_step2_anchors_and_all_candidate_endpoints" if not args.smoke else "first_16_frozen_step2_anchors_for_smoke",
        "illustration_selection": "10th/50th/90th percentile ranks of feature-only far_actual_sampling_mass, tie by anchor index",
        "shared_figure_layout": "original feature-only graph PCA coordinates, identical axes and nodes in H1/H8 panels; marker size encodes exp(-normalized embedding distance)",
        "kwide_comparator": "unavailable_not_reconstructed",
        "previous_root": str(previous),
        "sha256": {"selection_lock": digest(lock_path), "background": digest(background_path),
                   "step2_anchors": digest(prior_output / "step2_anchors.csv"),
                   "step2_endpoints": digest(prior_output / "step2_endpoints.csv"),
                   "h1_record": digest(h1_record_path), "h8_record": digest(record_path),
                   "feature_only": digest(feature_path)},
    }
    write_json(out / "summary.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument("--previous-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path,
                        help="exact output directory for this dataset and seed")
    parser.add_argument("--smoke", action="store_true", help="first 16 frozen anchors; use a separate output directory")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
