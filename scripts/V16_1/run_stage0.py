from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from methods.TopoGate.V16_1_predictive_graph_gate.gate import cross_fitted_predictive_support
from methods.TopoGate.V16_1_predictive_graph_gate.graph import build_candidate_graph, candidate_recurrence, consensus_graph
from methods.TopoGate.V16_1_predictive_graph_gate.sparse import (
    DenseNPZReference,
    assess_count_domain,
    dense_reference_profile,
    load_npz_matrix,
    repeated_splits,
    summarize_split_views,
)
from scripts.V16_1.dataset_registry import load_registry, resolve_metadata


DEFAULT_DATASETS = [
    "Campbell.npz",
    "Mouse_retina.npz",
    "Baron Human.npz",
    "Quake_Smart-seq2_Lung.npz",
    "hrvatin.npz",
    "hrvatin_filtered.npz",
    "fbis.wc.npz",
    "tr45.wc.npz",
]


def audit(
    path: Path,
    k: int,
    repeats: int,
    seed: int,
    *,
    input_policy: str = "expanded_count",
    registry: dict[str, dict] | None = None,
) -> dict:
    dataset_name = path.stem
    metadata = resolve_metadata(dataset_name, registry)
    semantics = metadata.get("count_semantics")
    source = metadata.get("semantics_source")
    if not path.exists():
        return {
            "dataset": dataset_name,
            "path": str(path),
            "status": "missing_dataset",
            "theory_domain": "theory_domain_not_supported",
            "domain_reasons": ["missing_dataset"],
            "source_metadata": metadata,
            "count_semantics_declared": semantics,
            "count_semantics_source": source,
            "input_policy": input_policy,
        }
    X, input_storage = load_npz_matrix(path)
    if isinstance(X, DenseNPZReference):
        profile = dense_reference_profile(
            X,
            count_semantics=semantics,
            semantics_source=source,
            input_policy=input_policy,
        )
        return {
            "dataset": dataset_name,
            "path": str(path),
            "status": "theory_domain_not_supported",
            "source_metadata": metadata,
            **profile,
        }
    matrix, profile = assess_count_domain(
        X,
        count_semantics=semantics,
        semantics_source=source,
        storage_override=input_storage,
        input_policy=input_policy,
    )
    if profile["theory_domain"] != "candidate":
        return {
            "dataset": dataset_name,
            "path": str(path),
            "status": "theory_domain_not_supported",
            "source_metadata": metadata,
            **profile,
        }
    views = repeated_splits(matrix, 0.5, repeats, seed)
    split_profile = summarize_split_views(views)
    profile["count_split"] = split_profile
    if not split_profile["has_nonempty_heldout"]:
        return {
            "dataset": dataset_name,
            "path": str(path),
            "status": "theory_domain_not_supported",
            "source_metadata": metadata,
            **profile,
            "domain_reasons": list(profile.get("domain_reasons", [])) + ["heldout_view_empty"],
            "theory_domain": "theory_domain_not_supported",
        }
    graphs = [build_candidate_graph(view_a, k) for view_a, _ in views]
    consensus = consensus_graph(graphs, k=k, min_repeats=2)
    support, support_profile = cross_fitted_predictive_support(
        views,
        consensus.indices,
        consensus.valid,
    )
    return {
        "dataset": dataset_name,
        "path": str(path),
        "status": "stage0_candidate",
        "source_metadata": metadata,
        **profile,
        "candidate_recurrence": candidate_recurrence(graphs),
        "candidate_profile": consensus.profile,
        "support_profile": support_profile,
        "support_non_degenerate": bool(support_profile["positive_support_row_rate"] > 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="V16.1 unlabeled sparse-count Stage 0 audit")
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--output", default="/tmp/v16_1_stage0.json")
    parser.add_argument("--registry", default=None, help="count registry; defaults to the bundled local registry")
    parser.add_argument("--input-policy", choices=("strict_legacy", "expanded_count"), default="expanded_count")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.k != 20 or args.repeats != 3:
        parser.error("V16.1 Stage 0 fixes k=20 and repeats=3")
    registry = load_registry(args.registry)
    rows = [
        audit(
            Path(args.data_root) / name,
            args.k,
            args.repeats,
            args.seed,
            input_policy=args.input_policy,
            registry=registry,
        )
        for name in args.datasets
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
