from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.TopoGate.V17_topology_native.candidate import build_candidate_union
from methods.TopoGate.V17_topology_native.config import V17Config, load_config
from methods.TopoGate.V17_topology_native.input_adapter import (
    build_projection_views,
    load_sparse_npz,
    prepare_input,
)


def run_stage0(
    matrix_path: str | Path,
    output_dir: str | Path,
    *,
    config: V17Config,
    count_semantics: str | None,
) -> dict:
    path = Path(matrix_path)
    if path.suffix != ".npz":
        raise ValueError("Stage-0 currently accepts a sparse NPZ bundle")
    matrix = load_sparse_npz(str(path))
    if count_semantics == "raw_count" and config.input_mode != "count":
        raise ValueError("raw_count audit requires input_mode=count")
    prepared = prepare_input(matrix, input_mode=config.input_mode)
    projections = build_projection_views(
        prepared,
        n_views=config.projection_views,
        projection_dim=config.projection_dim,
        density=config.projection_density,
        seed=config.seed,
    )
    candidates = build_candidate_union(
        projections.values,
        k_per_view=config.candidate_k,
        union_k=config.candidate_union_k,
        block_size=config.candidate_block_size,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "candidate_graph.npz",
        indices=candidates.indices,
        similarity=candidates.similarity,
        valid=candidates.valid,
        view_count=candidates.view_count,
    )
    summary = {
        "stage": "V17_stage0_candidate_graph",
        "status": "candidate_graph_audit",
        "source_path": str(path.resolve()),
        "count_semantics": count_semantics,
        "labels_used_during_fit": False,
        "K_used": False,
        "hashes_computed": False,
        "input": prepared.profile,
        "projection": projections.profile,
        "candidate": candidates.profile,
        "output_files": {"candidate_graph": "candidate_graph.npz", "summary": "summary.json"},
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="TopoGate V17 label-free candidate graph audit")
    parser.add_argument("--matrix-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-mode", choices=["auto", "count", "nonnegative", "continuous"], default="count")
    parser.add_argument("--count-semantics", default=None)
    args = parser.parse_args()
    config = load_config(args.config, {"seed": int(args.seed), "input_mode": args.input_mode})
    summary = run_stage0(args.matrix_path, args.output_dir, config=config, count_semantics=args.count_semantics)
    print(json.dumps(summary, ensure_ascii=True))


if __name__ == "__main__":
    main()
