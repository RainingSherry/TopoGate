from __future__ import annotations

"""Build a label-free dataset geometry/topology audit for TopoGate variants.

The feature extractor reads only the ``x`` array from each NPZ.  Labels from
the NPZ are never loaded.  Outcome columns are joined from persisted result
tables after feature extraction so this script cannot use labels to choose a
graph, a threshold, or a model variant.

The TDA-like features are deliberately narrow: 0-dimensional component
persistence on a fixed sparse kNN Vietoris-Rips 1-skeleton, plus cycle rank of
thresholded graph 1-skeletons.  They are diagnostics and detached-prior
candidates, not a claim that the existing TopoGate graph is persistent
homology or that the sparse skeleton is a full VR complex.
"""

import csv
import json
import math
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, normalize


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "result"
OUT = RESULT / "analysis"
K_GRAPH = 5
K_TDA = 15
RANDOM_STATE = 20260803
MAX_FEATURE_ELEMENTS = 80_000_000
MAX_ANALYSIS_SAMPLES = 4_000
MAX_ANALYSIS_FEATURES = 512


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def number(value: Any) -> float | None:
    if value in (None, "", "null", "None"):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def std(values: list[float]) -> float | None:
    return float(np.std(values, ddof=1)) if len(values) > 1 else (0.0 if values else None)


def normalize_name(name: str) -> str:
    return "".join(ch.lower() for ch in name if ch.isalnum())


def manifest_sources() -> dict[str, dict[str, Any]]:
    sources: dict[str, dict[str, Any]] = {}
    for manifest_path in (
        ROOT / "datasets/AHDPC/MANIFEST.json",
        ROOT / "datasets/AHDPC_related_advantage/MANIFEST.json",
    ):
        data = read_json(manifest_path)
        datasets = data.get("datasets", {})
        if isinstance(datasets, dict):
            entries = [dict(meta, dataset=name) for name, meta in datasets.items()]
        else:
            entries = datasets
        for item in entries:
            dataset = str(item.get("dataset", ""))
            processed = item.get("processed") or {}
            path = ROOT / "datasets/AHDPC" / str(processed.get("path", ""))
            if not path.is_file():
                path = ROOT / str(item.get("path", ""))
            if path.is_file():
                sources[normalize_name(dataset.removesuffix(".npz"))] = {
                    "dataset": dataset.removesuffix(".npz"),
                    "path": path,
                    "category": item.get("category", "related_manifest"),
                    "manifest": str(manifest_path.relative_to(ROOT)),
                    "n_clusters_metadata": item.get("n_clusters", processed.get("clusters")),
                }
    return sources


def result_sources(sources: dict[str, dict[str, Any]], needed_keys: set[str]) -> None:
    for path in sorted((ROOT / "datasets").glob("*.npz")):
        key = normalize_name(path.stem)
        if key not in needed_keys:
            continue
        sources.setdefault(
            key,
            {
                "dataset": path.stem,
                "path": path,
                "category": "result_dataset_pool",
                "manifest": "dataset_root_scan",
                "n_clusters_metadata": None,
            },
        )


def load_x(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if "x" not in data.files:
            raise ValueError(f"{path} has no x array")
        x = np.asarray(data["x"], dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
        raise ValueError(f"invalid x shape {x.shape} in {path}")
    x = np.where(np.isfinite(x), x, np.nan)
    medians = np.nanmedian(x, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    missing = ~np.isfinite(x)
    if missing.any():
        rows, cols = np.where(missing)
        x[rows, cols] = medians[cols]
    return x


def x_shape(path: Path) -> tuple[int, int]:
    """Read the compressed NPZ header without materialising the feature matrix."""
    with zipfile.ZipFile(path) as archive:
        with archive.open("x.npy") as handle:
            version = np.lib.format.read_magic(handle)
            if version == (1, 0):
                shape, _, _ = np.lib.format.read_array_header_1_0(handle)
            elif version == (2, 0):
                shape, _, _ = np.lib.format.read_array_header_2_0(handle)
            elif version == (3, 0):
                shape, _, _ = np.lib.format.read_array_header_2_0(handle)
            else:
                raise ValueError(f"unsupported x.npy header version {version}")
    if len(shape) != 2:
        raise ValueError(f"invalid x shape {shape} in {path}")
    return int(shape[0]), int(shape[1])


def analysis_embedding(x: np.ndarray) -> np.ndarray:
    z = StandardScaler().fit_transform(x).astype(np.float32)
    dim = min(50, z.shape[1], z.shape[0] - 1)
    if dim < z.shape[1]:
        z = PCA(
            n_components=max(1, dim),
            svd_solver="randomized",
            random_state=RANDOM_STATE,
        ).fit_transform(z)
    return normalize(np.nan_to_num(z), axis=1).astype(np.float32)


def union_find_components(n: int, edges: list[tuple[float, int, int]]) -> tuple[list[float], int]:
    parent = np.arange(n, dtype=np.int64)
    size = np.ones(n, dtype=np.int64)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    merges: list[float] = []
    components = n
    for distance, left, right in sorted(edges):
        root_left = find(left)
        root_right = find(right)
        if root_left == root_right:
            continue
        if size[root_left] < size[root_right]:
            root_left, root_right = root_right, root_left
        parent[root_right] = root_left
        size[root_left] += size[root_right]
        components -= 1
        merges.append(float(distance))
    return merges, components


def graph_features(embedding: np.ndarray) -> dict[str, float | int]:
    n = embedding.shape[0]
    k_graph = min(K_GRAPH, n - 1)
    k_tda = min(K_TDA, n - 1)
    nn = NearestNeighbors(n_neighbors=k_tda + 1, metric="cosine", n_jobs=1).fit(embedding)
    distances, indices = nn.kneighbors(embedding)
    distances = distances[:, 1:].astype(np.float64)
    indices = indices[:, 1:].astype(np.int64)
    graph_distances = distances[:, :k_graph]
    graph_indices = indices[:, :k_graph]
    neighbor_sets = [set(row.tolist()) for row in graph_indices]

    mutual = []
    snn = []
    edges: dict[tuple[int, int], float] = {}
    for i in range(n):
        set_i = neighbor_sets[i]
        for pos, j in enumerate(graph_indices[i]):
            j_int = int(j)
            mutual.append(float(i in neighbor_sets[j_int]))
            union = set_i.union(neighbor_sets[j_int])
            snn.append(len(set_i.intersection(neighbor_sets[j_int])) / max(1, len(union)))
        for pos, j in enumerate(indices[i]):
            a, b = sorted((i, int(j)))
            edges[(a, b)] = min(edges.get((a, b), float("inf")), float(distances[i, pos]))

    edge_list = [(distance, a, b) for (a, b), distance in edges.items()]
    merges, sparse_components = union_find_components(n, edge_list)
    edge_matrix = csr_matrix(
        (
            [distance for distance, _, _ in edge_list] * 2,
            (
                [a for _, a, _ in edge_list] + [b for _, _, b in edge_list],
                [b for _, a, b in edge_list] + [a for _, a, b in edge_list],
            ),
        ),
        shape=(n, n),
    )
    component_count, labels = connected_components(edge_matrix, directed=False)
    component_sizes = np.bincount(labels, minlength=component_count)
    largest_fraction = float(component_sizes.max() / n)

    # The sparse MST is a compact component-persistence representation of the
    # same fixed edge filtration.  It is not a full dense VR computation.
    mst = minimum_spanning_tree(edge_matrix)
    mst_values = mst.data.astype(np.float64)
    nearest_scale = float(np.median(distances[:, 0]))
    threshold_values = [nearest_scale, float(np.median(distances)), float(np.quantile(distances, 0.9))]
    cycle_ranks = []
    threshold_components = []
    for threshold in threshold_values:
        selected = [(a, b) for distance, a, b in edge_list if distance <= threshold]
        selected_matrix = csr_matrix(
            (
                [1.0 for _ in selected] * 2,
                (
                    [a for a, _ in selected] + [b for _, b in selected],
                    [b for _, b in selected] + [a for a, _ in selected],
                ),
            ),
            shape=(n, n),
        )
        c_count, _ = connected_components(selected_matrix, directed=False)
        cycle_ranks.append(float(len(selected) - n + c_count))
        threshold_components.append(int(c_count))

    merge_values = np.asarray(merges, dtype=np.float64)
    finite_mst = mst_values if mst_values.size else merge_values
    scale = max(nearest_scale, 1e-8)
    normalized_merges = merge_values / scale if merge_values.size else merge_values
    total = float(normalized_merges.sum()) if normalized_merges.size else 0.0
    tail_count = max(1, int(math.ceil(normalized_merges.size * 0.1))) if normalized_merges.size else 1
    tail_share = float(np.sort(normalized_merges)[-tail_count:].sum() / max(total, 1e-8)) if normalized_merges.size else 0.0
    effective = np.exp(-graph_distances / max(float(np.median(graph_distances)), 1e-8))

    return {
        "analysis_pca_dim": int(embedding.shape[1]),
        "mean_knn_cosine_distance": float(np.mean(graph_distances)),
        "p95_knn_cosine_distance": float(np.quantile(graph_distances, 0.95)),
        "cv_knn_cosine_distance": float(np.std(graph_distances) / max(np.mean(graph_distances), 1e-8)),
        "median_1nn_cosine_distance": nearest_scale,
        "mean_mutual_ratio": float(np.mean(mutual)),
        "mean_snn": float(np.mean(snn)),
        "sparse_graph_components": int(component_count),
        "sparse_graph_largest_component_fraction": largest_fraction,
        "sparse_graph_edge_count": int(len(edge_list)),
        "sparse_graph_cycle_rank": float(len(edge_list) - n + sparse_components),
        "tda_h0_merge_count": int(merge_values.size),
        "tda_h0_sparse_components": int(sparse_components),
        "tda_h0_total_persistence_norm": total,
        "tda_h0_q50_death_norm": float(np.quantile(normalized_merges, 0.50)) if normalized_merges.size else None,
        "tda_h0_q90_death_norm": float(np.quantile(normalized_merges, 0.90)) if normalized_merges.size else None,
        "tda_h0_tail10_share": tail_share,
        "tda_h0_mst_edge_count": int(finite_mst.size),
        "cycle_rank_at_1nn_scale": cycle_ranks[0],
        "cycle_rank_at_median_knn_scale": cycle_ranks[1],
        "cycle_rank_at_p90_knn_scale": cycle_ranks[2],
        "components_at_1nn_scale": threshold_components[0],
        "components_at_median_knn_scale": threshold_components[1],
        "components_at_p90_knn_scale": threshold_components[2],
        "effective_neighbor_proxy": float(np.mean(effective)),
    }


def build_outcomes() -> dict[str, dict[str, Any]]:
    outcomes: dict[str, dict[str, Any]] = defaultdict(dict)
    paper_path = RESULT / "v9_results_2026-08-02_paper_preprocess/comparison_by_dataset.csv"
    for row in read_csv(paper_path):
        key = normalize_name(row["dataset"])
        outcomes[key].update(
            {
                "v9_ari_vs_ahdpc": number(row.get("v9_minus_ahdpc_ari")),
                "v9_ari_vs_hdpc": number(row.get("v9_minus_hdpc_ari")),
                "v9_paper_protocol": True,
            }
        )
    ablation = read_csv(RESULT / "v9_results_2026-08-02_advantage_ablation/ablation_runs.csv")
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in ablation:
        if row.get("status") == "completed" and number(row.get("ari")) is not None:
            grouped[(normalize_name(row["dataset"]), row["variant"])].append(float(row["ari"]))
    for key in sorted({dataset for dataset, _ in grouped}):
        full = mean(grouped.get((key, "v9_full"), []))
        nomix = mean(grouped.get((key, "v9_nomix"), []))
        random = mean(grouped.get((key, "v9_random"), []))
        static = mean(grouped.get((key, "v9_static"), []))
        outcomes[key].update(
            {
                "v9_full_ari": full,
                "v9_nomix_ari": nomix,
                "v9_random_ari": random,
                "v9_static_ari": static,
                "v9_full_nomix_ari": full - nomix if full is not None and nomix is not None else None,
                "v9_full_random_ari": full - random if full is not None and random is not None else None,
                "v9_full_static_ari": full - static if full is not None and static is not None else None,
            }
        )
    evidence = read_csv(OUT / "cross_version_evidence_2026-08-03.csv")
    for row in evidence:
        key = normalize_name(row.get("dataset", ""))
        version = row.get("version", "")
        delta = number(row.get("ari_vs_control"))
        expected_full = {
            "V11": "V11_full",
            "V12": "v12_full",
            "V13": "v13_full",
            "V14": "v14_full",
            "StaticGate": "static_gate_full",
        }.get(version)
        if (
            delta is not None
            and expected_full is not None
            and row.get("variant") == expected_full
        ):
            outcomes[key][f"{version.lower()}_full_nomix_ari"] = delta
    return outcomes


def build_features(sources: dict[str, dict[str, Any]], outcomes: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in sorted(sources):
        source = sources[key]
        print(f"processing {source['dataset']}", flush=True)
        original_shape = x_shape(source["path"])
        base_row: dict[str, Any] = {
            "dataset": source["dataset"],
            "dataset_key": key,
            "category": source["category"],
            "source_path": str(source["path"].relative_to(ROOT)),
            "manifest": source["manifest"],
            "n": original_shape[0],
            "d": original_shape[1],
            "log_nd": float(math.log10(max(1, original_shape[0] * original_shape[1]))),
            "metadata_k": source.get("n_clusters_metadata"),
        }
        if original_shape[0] * original_shape[1] > MAX_FEATURE_ELEMENTS:
            rows.append(
                {
                    **base_row,
                    "feature_error": (
                        f"skipped_shape_cap: {original_shape[0]}x{original_shape[1]} "
                        f"> {MAX_FEATURE_ELEMENTS} elements"
                    ),
                    **outcomes.get(key, {}),
                }
            )
            continue
        try:
            x = load_x(source["path"])
            analysis_sampled = False
            analysis_feature_sampled = False
            if x.shape[1] > MAX_ANALYSIS_FEATURES:
                rng = np.random.default_rng(RANDOM_STATE + x.shape[1])
                selected_features = np.sort(
                    rng.choice(x.shape[1], size=MAX_ANALYSIS_FEATURES, replace=False)
                )
                x = x[:, selected_features]
                analysis_feature_sampled = True
            if x.shape[0] > MAX_ANALYSIS_SAMPLES:
                rng = np.random.default_rng(RANDOM_STATE)
                selected = np.sort(rng.choice(x.shape[0], size=MAX_ANALYSIS_SAMPLES, replace=False))
                x = x[selected]
                analysis_sampled = True
            embedding = analysis_embedding(x)
            features = graph_features(embedding)
            row: dict[str, Any] = {
                **base_row,
                "analysis_n": int(x.shape[0]),
                "analysis_d": int(x.shape[1]),
                "analysis_sampled": analysis_sampled,
                "analysis_feature_sampled": analysis_feature_sampled,
            }
            row.update(features)
            row.update(outcomes.get(key, {}))
            rows.append(row)
        except Exception as exc:  # record a missing feature instead of aborting all datasets
            rows.append(
                {
                    **base_row,
                    "feature_error": f"{type(exc).__name__}: {exc}",
                    **outcomes.get(key, {}),
                }
            )
    return rows


def correlations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    feature_names = [
        key
        for key in rows[0]
        if key.startswith(("mean_", "p95_", "cv_", "median_", "sparse_", "tda_", "cycle_", "components_", "effective_"))
    ]
    outcome_names = [
        "v9_ari_vs_ahdpc",
        "v9_ari_vs_hdpc",
        "v9_full_nomix_ari",
        "v11_full_nomix_ari",
        "v12_full_nomix_ari",
        "v13_full_nomix_ari",
        "v14_full_nomix_ari",
        "staticgate_full_nomix_ari",
    ]
    result: list[dict[str, Any]] = []
    for feature in feature_names:
        for outcome in outcome_names:
            pairs = [
                (number(row.get(feature)), number(row.get(outcome)))
                for row in rows
                if number(row.get(feature)) is not None and number(row.get(outcome)) is not None
            ]
            if len(pairs) < 4:
                continue
            x, y = zip(*pairs)
            if np.std(x) == 0.0 or np.std(y) == 0.0:
                continue
            rho, p_value = spearmanr(x, y)
            result.append(
                {
                    "outcome": outcome,
                    "feature": feature,
                    "n": len(pairs),
                    "spearman_rho": float(rho),
                    "p_value_exploratory": float(p_value),
                }
            )
    return sorted(result, key=lambda row: (row["outcome"], -abs(row["spearman_rho"])))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(rows: list[dict[str, Any]], corr: list[dict[str, Any]]) -> None:
    outcome_names = [
        "v9_full_nomix_ari",
        "v11_full_nomix_ari",
        "v12_full_nomix_ari",
        "v13_full_nomix_ari",
        "v14_full_nomix_ari",
        "staticgate_full_nomix_ari",
    ]
    lines = [
        "# TopoGate 优势/劣势数据集与真正 TDA 特征审计",
        "",
        "生成时间：2026-08-03。特征计算只读取 NPZ 中的 `x`；脚本没有加载 `y`，也没有用标签选择图、阈值、尺度或 variant。结果列在特征提取后从持久化 CSV 连接，仅用于事后解释。",
        "",
        "## 研究边界",
        "",
        "本报告把现有 `PCA/kNN`、mutual/SNN 和动态图称为有限图结构，不把它们称为 persistent homology。`tda_h0_*` 是固定稀疏 kNN Vietoris–Rips 1-skeleton 上的 0 维 component persistence 摘要；`cycle_rank_*` 是阈值图 1-skeleton 的循环秩。它们是可审计的 TDA/拓扑诊断候选，但不是完整 dense VR complex 的 H1 persistence diagram。",
        "",
        "## 版本结果边界",
        "",
        "- V9 paper-preprocess：相对 AHDPC 只有 `spect_heart`、`balance_scale`、`landsat` 正差值；Full-NoMix 仅在 7 个相关数据集上配对。",
        "- V11 minimum：5 datasets × Full/NoMix × 3 seeds；宏观 head ARI 差值接近零且为负。",
        "- V12/V13：同一批 12 个扩展数据集；两者 Full-NoMix 均未形成正向稳定证据。",
        "- V14：5 个代表性数据集；机制路径可运行，但配对 ARI 增益未显著。",
        "- StaticGate：15 个数据集的历史消融只作机制方向参考，不与 V9/V11 输入协议混合。",
        "",
        "## 特征计算覆盖与协议",
        "",
        f"本轮共有 {len(rows)} 个结果相关数据集；{sum(not row.get('feature_error') for row in rows)} 个完成无标签特征计算，{sum(bool(row.get('feature_error')) for row in rows)} 个因矩阵元素上限跳过。计算协议为标准化后 PCA 上限 50、graph `k={K_GRAPH}`、TDA skeleton `k={K_TDA}`；超过 {MAX_ANALYSIS_SAMPLES} 个样本或 {MAX_ANALYSIS_FEATURES} 个特征时使用固定随机子集。`analysis_sampled` 和 `analysis_feature_sampled` 在 CSV 中显式记录。",
        "",
        "`Campbell` 与 `hrvatin_filtered` 被跳过；V11 的 `Mouse_retina`、`enron` 和 StaticGate 的部分高维数据使用采样结果。采样特征只用于生成候选假设，不足以替代完整矩阵的预注册验证。",
        "",
        "## 如何解读优势与劣势",
        "",
        "当前最完整的拓扑正例是 `balance_scale`：V9 Full 同时高于 NoMix 和 AHDPC，且 3/3 seeds 方向一致。`spect_heart` 的 V9 相对 AHDPC 优势在 NoMix 下仍存在，因此不能归因于 topology mixing；`landsat` 差值很小且 random 邻居均值略高，也不能证明 reliability 权重有效。",
        "",
        "## 各版本正负集合",
        "",
        "下列集合按同一批次内的 Full−NoMix 或 V9−baseline 结果列出；`positive/negative` 只描述方向，不代表显著性。",
        "",
    ]
    set_outcomes = [
        ("V9 vs AHDPC", "v9_ari_vs_ahdpc"),
        ("V9 vs HDPC", "v9_ari_vs_hdpc"),
        ("V9 Full−NoMix", "v9_full_nomix_ari"),
        ("V11 Full−NoMix", "v11_full_nomix_ari"),
        ("V12 Full−NoMix", "v12_full_nomix_ari"),
        ("V13 Full−NoMix", "v13_full_nomix_ari"),
        ("V14 Full−NoMix", "v14_full_nomix_ari"),
        ("StaticGate Full−NoMix", "staticgate_full_nomix_ari"),
    ]
    for label, outcome in set_outcomes:
        values = [
            (row["dataset"], number(row.get(outcome)))
            for row in rows
            if not row.get("feature_error") and number(row.get(outcome)) is not None
        ]
        positive = sorted(
            (item for item in values if item[1] > 1e-12),
            key=lambda item: -item[1],
        )
        negative = sorted(
            (item for item in values if item[1] < -1e-12),
            key=lambda item: item[1],
        )
        pos_text = ", ".join(f"`{dataset}` ({value:+.4f})" for dataset, value in positive) or "无"
        neg_text = ", ".join(f"`{dataset}` ({value:+.4f})" for dataset, value in negative) or "无"
        lines.append(f"- **{label}** (`n={len(values)}`)：正向 {pos_text}；负向 {neg_text}。")
    lines.extend(
        [
            "",
            "V12/V13/V14 的结果说明继续增加 risk、assignment residual 或 strict minimum 约束并没有把局部拓扑信号稳定转化为聚类收益。更合理的下一步是先验证 TDA 诊断是否在优势/劣势分界上提供独立信息，再决定是否进入训练梯度路径。",
            "",
            "## 特征与版本增益的统计边界",
            "",
            "下面的 Spearman 结果使用本报告的标准化/PCA/采样协议，是小样本探索性关联，不是选择配置或证明因果；不得以其 p 值直接支持论文性能结论。若某版本只有 5 个数据集，相关系数只用于生成下一轮预注册候选，不用于宣称普遍规律。它不能直接替换既有 `geometry_features_no_label.csv` 的历史标准化协议。",
            "",
            "| outcome | strongest exploratory feature | n | rho | p |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for outcome in outcome_names:
        candidates = [row for row in corr if row["outcome"] == outcome]
        if candidates:
            best = candidates[0]
            lines.append(
                f"| `{outcome}` | `{best['feature']}` | {best['n']} | {best['spearman_rho']:.3f} | {best['p_value_exploratory']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## 推荐的真正拓扑 pilot",
            "",
            "1. 先固定输入标准化、PCA 上限、kNN `k` 和 filtration scale；用 sparse H0 persistence 生成 detached edge/node prior，不参与当前主模型反向传播。",
            "2. 在同一批数据上至少比较原 V11、NoMix、random prior、fixed-filtration prior 和 TDA prior；每个 variant 使用同一 `[42, 123, 7]`，不按标签挑选阈值。",
            "3. 若 TDA prior 只改变 gate coverage 而不改善 paired ARI/NMI，保留为诊断；若只在 `balance_scale` 等少数几何区间有效，报告为条件机制，不晋升为普遍主方法。",
            "4. H1/persistence image/Mapper 只有在引入经过验证的 TDA 库、固定 subsampling 和复杂度上限后才进入第二阶段；不能用当前 graph cycle-rank proxy 替代 H1 persistence。",
            "",
            "## 可复核产物",
            "",
            "- `result/analysis/topogate_dataset_features_2026-08-03.csv`：标签隔离的特征与事后结果连接表。",
            "- `result/analysis/topogate_feature_version_correlations_2026-08-03.csv`：探索性 Spearman 表。",
            "- `scripts/analysis/build_topogate_dataset_feature_audit.py`：可重跑脚本。",
        ]
    )
    (OUT / "topogate_advantage_feature_audit_2026-08-03.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    sources = manifest_sources()
    outcomes = build_outcomes()
    result_sources(sources, set(outcomes))
    rows = build_features(sources, outcomes)
    corr = correlations(rows) if rows else []
    write_csv(OUT / "topogate_dataset_features_2026-08-03.csv", rows)
    write_csv(OUT / "topogate_feature_version_correlations_2026-08-03.csv", corr)
    write_report(rows, corr)
    print(f"wrote {len(rows)} dataset feature rows and {len(corr)} exploratory correlations")


if __name__ == "__main__":
    main()
