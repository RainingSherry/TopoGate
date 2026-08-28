from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from methods.TopoGate.V16_predictive_graph_gate.graph import build_candidate_graph, candidate_recurrence
from methods.TopoGate.V16_predictive_graph_gate.gate import predictive_support
from methods.TopoGate.V16_predictive_graph_gate.sparse import assess_count_domain, load_npz_matrix, repeated_splits


DEFAULT_DATASETS = [
    "Campbell.npz",
    "Mouse_retina.npz",
    "Baron Human.npz",
    "Quake_Smart-seq2_Lung.npz",
    "fbis.wc.npz",
    "tr45.wc.npz",
]


def audit(path: Path, k: int, repeats: int, seed: int) -> dict:
    X, input_storage = load_npz_matrix(path)
    matrix, profile = assess_count_domain(X, storage_override=input_storage)
    if profile["theory_domain"] != "candidate":
        return {"dataset": path.stem, "path": str(path), **profile}
    views = repeated_splits(matrix, 0.5, repeats, seed)
    graphs = [build_candidate_graph(view[0], k) for view in views]
    support, support_profile = predictive_support([view[1] for view in views], graphs[0].indices, graphs[0].valid)
    return {
        "dataset": path.stem,
        "path": str(path),
        **profile,
        "candidate_recurrence": candidate_recurrence(graphs),
        "candidate_profile": graphs[0].profile,
        "support_profile": support_profile,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="V16 unlabeled sparse-count Stage 0 audit")
    parser.add_argument("--data_root", default="datasets")
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--output", default="/tmp/v16_stage0.json")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = []
    for name in args.datasets:
        path = Path(args.data_root) / name
        rows.append(audit(path, args.k, args.repeats, args.seed))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
