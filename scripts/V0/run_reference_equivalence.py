#!/usr/bin/env python
"""Run a small, exact PlantNet-vs-V0 F/T equivalence check.

The check deliberately compares the arrays consumed by training: graph
construction, reliability/gate statistics, pseudo views, and row-swap noise.
It also runs a two-epoch label-free V0 fit for both parameterizations.  The
output is a compact engineering record, not a clustering-performance result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
PLANTNET_ROOT = Path("/home/luolie/biopipeline/dimension-reduction/plantnet")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PLANTNET_ROOT) not in sys.path:
    sys.path.insert(0, str(PLANTNET_ROOT))

from methods.TopoGate.V0.config import V0Config
from methods.TopoGate.V0.corruption import apply_scmae_noise, compute_node_gate, make_pseudo_batch
from methods.TopoGate.V0.graph import build_pca_knn_graph, compute_edge_reliability
from methods.TopoGate.V0.trainer import fit_predict


def _data() -> np.ndarray:
    return np.asarray(np.random.default_rng(902).normal(size=(24, 11)), dtype=np.float32)


def _exact(name: str, left: np.ndarray, right: np.ndarray) -> dict:
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return {
        "name": name,
        "shape": list(left.shape),
        "exact_equal": bool(np.array_equal(left, right)),
        "max_abs_difference": float(np.max(np.abs(delta))) if delta.size else 0.0,
    }


def _reference_modules():
    if not PLANTNET_ROOT.is_dir():
        raise FileNotFoundError(f"PlantNet checkout is not available: {PLANTNET_ROOT}")
    try:
        from experimental_retired_models.NeighborMix_scMAE import run as fixed
        from experimental_retired_models.RG_NeighborMix_scMAE import mixing as topology_mixing
        from experimental_retired_models.RG_NeighborMix_scMAE import neighbor_graph as topology_graph
    except Exception as exc:
        raise RuntimeError("PlantNet reference modules could not be imported") from exc
    return fixed, topology_graph, topology_mixing


def run(output_dir: Path) -> dict:
    fixed, topology_graph, topology_mixing = _reference_modules()
    data = _data()
    batch_indices = np.array([0, 2, 8, 19], dtype=np.int64)
    batch = torch.as_tensor(data[batch_indices])
    comparisons: list[dict] = []

    old_indices, old_probs, _ = fixed.build_knn_distribution(data, 5, 7, 0.2, 42)
    new_fixed_graph = build_pca_knn_graph(data, 5, 7, 0.2, 42)
    comparisons.extend(
        [
            _exact("fixed.neighbor_indices", old_indices, new_fixed_graph.indices),
            _exact("fixed.neighbor_probs", old_probs, new_fixed_graph.probs),
        ]
    )
    fixed_gate, _fixed_weight, _ = compute_node_gate(
        new_fixed_graph, parameterization="fixed", alpha=0.9
    )
    old_fixed_view = fixed.sample_mix(
        data,
        batch_indices,
        batch,
        0.9,
        4,
        np.random.default_rng(777),
        old_indices,
        old_probs,
    )
    new_fixed_view, new_fixed_weight, _ = make_pseudo_batch(
        data,
        batch_indices,
        batch,
        parameterization="fixed",
        graph=new_fixed_graph,
        edge_weights=new_fixed_graph.probs,
        node_gate=fixed_gate,
        mix_neighbors=4,
        alpha=0.9,
        rng=np.random.default_rng(777),
        neighbor_estimator="current",
        legacy_plantnet=True,
    )
    comparisons.append(_exact("fixed.pseudo_view", old_fixed_view.numpy(), new_fixed_view.numpy()))
    comparisons.append(
        _exact(
            "fixed.pseudo_sample_weight",
            np.ones(batch_indices.size, dtype=np.float32),
            new_fixed_weight.numpy(),
        )
    )

    old_topology_graph = topology_graph.build_pca_knn_graph(data, 5, 7, 0.2, 42)
    new_topology_graph = build_pca_knn_graph(data, 5, 7, 0.2, 42)
    for field in ("indices", "probs", "similarity", "distance", "mutual", "snn"):
        comparisons.append(
            _exact(
                f"topology.graph.{field}",
                getattr(old_topology_graph, field),
                getattr(new_topology_graph, field),
            )
        )
    old_reliability, old_weights, _ = topology_graph.compute_edge_reliability(
        old_topology_graph, "sim_mutual_snn_distance", 1.0, 1.0, 1.0, 1.0
    )
    new_reliability, new_weights, _ = compute_edge_reliability(
        new_topology_graph, "sim_mutual_snn_distance", 1.0, 1.0, 1.0, 1.0
    )
    comparisons.extend(
        [
            _exact("topology.edge_reliability", old_reliability, new_reliability),
            _exact("topology.edge_weights", old_weights, new_weights),
        ]
    )
    old_gate, _old_weight, _ = topology_mixing.compute_node_gate(
        old_topology_graph, old_weights, "topology", 0.0, 0.15, 1.0, 1.0, 2.0, 1.0
    )
    new_gate, _new_weight, _ = compute_node_gate(
        new_topology_graph,
        parameterization="topology",
        gate_min=0.0,
        gate_max=0.15,
        beta_mutual=1.0,
        beta_snn=1.0,
        beta_perturb=2.0,
        beta_uncertainty=1.0,
    )
    comparisons.append(_exact("topology.node_gate", old_gate, new_gate))
    old_topology_view, old_topology_weight, _ = topology_mixing.make_pseudo_batch(
        data,
        batch_indices,
        batch,
        "reliability",
        old_topology_graph,
        old_weights,
        old_gate,
        4,
        np.random.default_rng(777),
        neighbor_estimator="current",
    )
    new_topology_view, new_topology_weight, _ = make_pseudo_batch(
        data,
        batch_indices,
        batch,
        parameterization="topology",
        graph=new_topology_graph,
        edge_weights=new_weights,
        node_gate=new_gate,
        mix_neighbors=4,
        alpha=0.9,
        rng=np.random.default_rng(777),
        neighbor_estimator="current",
        legacy_plantnet=True,
    )
    comparisons.extend(
        [
            _exact("topology.pseudo_view", old_topology_view.numpy(), new_topology_view.numpy()),
            _exact(
                "topology.pseudo_sample_weight",
                old_topology_weight.numpy(),
                new_topology_weight.numpy(),
            ),
        ]
    )

    values = torch.as_tensor(data[:7, :6])
    torch.manual_seed(31415)
    old_corrupted, old_mask = fixed.apply_scmae_noise(values, 0.4)
    torch.manual_seed(31415)
    new_corrupted, new_mask = apply_scmae_noise(values, 0.4, legacy_plantnet=True)
    comparisons.extend(
        [
            _exact("legacy.row_swap.values", old_corrupted.numpy(), new_corrupted.numpy()),
            _exact("legacy.row_swap.mask", old_mask.numpy(), new_mask.numpy()),
        ]
    )

    fit_checks = {}
    for parameterization in ("fixed", "topology"):
        config = V0Config(
            parameterization=parameterization,
            hidden_size=8,
            epochs=2,
            batch_size=8,
            neighbor_k=5,
            mix_neighbors=4,
            knn_pca_dim=7,
            n_top_features=0,
            kmeans_n_init=2,
        )
        predictions, embedding, diagnostics = fit_predict(
            data, n_clusters=3, config=config, seed=42, device="cpu"
        )
        fit_checks[parameterization] = {
            "embedding_shape": list(embedding.shape),
            "prediction_shape": list(predictions.shape) if predictions is not None else None,
            "finite_embedding": bool(np.all(np.isfinite(embedding))),
            "labels_used_during_fit": diagnostics["core_summary"]["labels_used_during_fit"],
        }

    payload = {
        "status": "completed" if all(item["exact_equal"] for item in comparisons) else "failed",
        "protocol": "topogate_v0_plantnet_reference_equivalence_smoke_v1",
        "data": {"n_samples": int(data.shape[0]), "n_features": int(data.shape[1]), "seed": 902},
        "comparisons": comparisons,
        "fit_checks": fit_checks,
        "scope": "engineering equivalence only; no clustering-performance claim",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "result/v0/reference_equivalence_smoke_20260902"),
    )
    args = parser.parse_args()
    payload = run(Path(args.output_dir))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
