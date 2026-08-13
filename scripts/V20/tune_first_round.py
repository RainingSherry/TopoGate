#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.TopoGate.V20_topology_conditioned_adv_mask.config import load_config
from methods.TopoGate.V20_topology_conditioned_adv_mask.graph import build_svd_knn_graph, compute_topology_statistics
from methods.TopoGate.V20_topology_conditioned_adv_mask.input_adapter import load_npz_matrix_only, prepare_dual_input
from methods.TopoGate.V20_topology_conditioned_adv_mask.model import cyclic_donor, random_topk_mask, straight_through_topk
from methods.TopoGate.V20_topology_conditioned_adv_mask.trainer import fit_full


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, default=str), encoding="utf-8")


def evaluate_random_proxy(model: torch.nn.Module, X_eval: np.ndarray, *, config: Any, seed: int, device: torch.device) -> dict[str, float]:
    model.eval()
    rng = torch.Generator(device=device).manual_seed(int(seed) + 901)
    rec_values: list[float] = []
    cosine_values: list[float] = []
    k_mask = max(1, min(X_eval.shape[1], int(round(config.mask_ratio * X_eval.shape[1]))))
    with torch.no_grad():
        for start in range(0, X_eval.shape[0], config.batch_size):
            batch = torch.as_tensor(X_eval[start : start + config.batch_size], dtype=torch.float32, device=device)
            mask = random_topk_mask((batch.shape[0], batch.shape[1]), k_mask, device=device, generator=rng)
            donor = cyclic_donor(batch, generator=rng)
            corrupted = batch + mask * (donor - batch)
            _latent_corrupt, _mask_logits, reconstruction = model.forward_mask(corrupted)
            raw = functional.mse_loss(reconstruction, batch, reduction="none")
            weights = mask * config.masked_data_weight + (1.0 - mask) * (1.0 - config.masked_data_weight)
            rec_values.append(float(((1.0 - config.mask_loss_weight) * (raw * weights).mean()).cpu()))
            latent_clean = model.encoder(batch)
            latent_corrupt = model.encoder(corrupted)
            cosine_values.append(float(functional.cosine_similarity(latent_clean, latent_corrupt, dim=1).mean().cpu()))
    return {"heldout_random_reconstruction_loss": float(np.mean(rec_values)), "heldout_latent_cosine": float(np.mean(cosine_values))}


def run_candidate(candidate_id: str, base: Any, prepared: Any, fit_idx: np.ndarray, eval_idx: np.ndarray, output: Path, seed: int) -> dict[str, Any]:
    candidate = replace(base, gate_lr=float(candidate_id.split("__")[0].replace("lr", "")), tau_ste=float(candidate_id.split("__")[1].replace("tau", "")), epochs=4, warmup_epochs=2)
    candidate_dir = output / candidate_id
    embedding, diagnostics = fit_full(
        prepared.X_model[fit_idx],
        prepared.X_graph[fit_idx],
        config=candidate,
        seed=seed,
        device=torch.device("cpu"),
        stats_cache_dir=candidate_dir / "cache",
    )
    proxy = evaluate_random_proxy(diagnostics["model"], prepared.X_model[eval_idx], config=candidate, seed=seed, device=torch.device("cpu"))
    score = float(proxy["heldout_random_reconstruction_loss"] + 0.1 * (1.0 - proxy["heldout_latent_cosine"]))
    record = {"candidate_id": candidate_id, "config": candidate.to_dict(), "proxy": proxy, "proxy_score_lower_is_better": score, "labels_accessed": False, "y_key_read": False, "n_clusters_used": None, "status": "completed"}
    _write(candidate_dir / "tuning_record.json", record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="V20 first-round X-only Full tuning")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--input-protocol", choices=("clubench_bridge", "shared_text"), required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "methods/TopoGate/V20_topology_conditioned_adv_mask/configs/v20_full.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    base = load_config(args.config)
    matrix = load_npz_matrix_only(args.data)
    prepared = prepare_dual_input(matrix, dataset_name=args.dataset_name, input_protocol=args.input_protocol)
    split = int(round(0.8 * prepared.X_model.shape[0]))
    fit_idx = np.arange(split, dtype=np.int64)
    eval_idx = np.arange(split, prepared.X_model.shape[0], dtype=np.int64)
    candidates = ("lr0.0005__tau0.5", "lr0.0005__tau1.0", "lr0.001__tau0.5", "lr0.001__tau1.0")
    records = [run_candidate(candidate_id, base, prepared, fit_idx, eval_idx, args.output_dir, args.seed) for candidate_id in candidates]
    selected = min(records, key=lambda row: float(row["proxy_score_lower_is_better"]))
    _write(args.output_dir / "stage_spec.json", {"protocol_id": "v20_topology_conditioned_adv_mask_tuning_v1", "dataset": args.dataset_name, "seed": args.seed, "split": {"fit_rows": int(fit_idx.size), "evaluation_rows": int(eval_idx.size), "selection_uses_labels": False}, "candidates": list(candidates), "selection_target": "heldout_random_reconstruction_plus_latent_stability", "expected_runs": len(candidates), "completed_runs": len(records)})
    _write(args.output_dir / "selected_config.json", {"selection_status": "completed", "selected_candidate_id": selected["candidate_id"], "selected_config": selected["config"], "selection_proxy": selected["proxy"], "labels_accessed": False, "y_key_read": False, "n_clusters_used": None})


if __name__ == "__main__":
    main()
