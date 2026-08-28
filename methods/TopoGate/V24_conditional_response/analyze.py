from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import V24Q1Config
from .controls import build_marginal_controls
from .evaluation import bootstrap_conditional_delta, conditional_pair_utility, crossfit_residual_response


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def analyze_response(
    matrix: np.ndarray,
    fingerprints: dict[str, np.ndarray],
    *,
    labels: np.ndarray,
    config: V24Q1Config,
    seed: int,
    bootstrap_replicates: int,
    bootstrap_workers: int = 1,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Outer-only V24 conditional utility analysis; fit/profile remain label-free."""

    config.validate()
    required = ("clean_embedding", "cycle_repair_standardized", "mask_dictionary", "donor_offsets")
    missing = [name for name in required if name not in fingerprints]
    if missing:
        raise ValueError(f"fingerprints artifact lacks required arrays: {missing}")
    controls = build_marginal_controls(
        matrix,
        fingerprints["mask_dictionary"],
        fingerprints["donor_offsets"],
        standardized_clip=config.marginal_standardized_clip,
        relative_scale_floor=config.marginal_relative_scale_floor,
    )
    state = np.asarray(fingerprints["clean_embedding"], dtype=np.float32)
    response = np.asarray(fingerprints["cycle_repair_standardized"], dtype=np.float32)
    residual = crossfit_residual_response(
        state,
        controls.support,
        controls.marginal,
        response,
        n_splits=config.outer_folds,
        seed=seed,
        alpha=config.ridge_alpha,
    )
    utility = conditional_pair_utility(
        state,
        controls.support,
        controls.marginal,
        response,
        labels=labels,
        outer_folds=config.outer_folds,
        inner_folds=config.inner_folds,
        seed=seed,
        alpha=config.ridge_alpha,
        pair_count_per_fold=config.pair_count_per_fold,
    )
    bootstrap = (
        bootstrap_conditional_delta(
            state,
            controls.support,
            controls.marginal,
            response,
            labels=labels,
            outer_folds=config.outer_folds,
            inner_folds=config.inner_folds,
            seed=seed,
            alpha=config.ridge_alpha,
            pair_count_per_fold=config.pair_count_per_fold,
            replicates=bootstrap_replicates,
            workers=bootstrap_workers,
        )
        if bootstrap_replicates > 0
        else np.empty(0, dtype=np.float32)
    )
    summary: dict[str, object] = {
        "status": "completed",
        "protocol_id": config.protocol_id,
        "stage": "outer_conditional_analysis",
        "seed": int(seed),
        "scientific_question": "conditional_incremental_utility_after_observed_state_support_marginal_controls",
        "claims_not_supported": ["statistical_independence", "causality", "functional_redundancy"],
        "labels_accessible_during_fit": False,
        "labels_accessible_during_profile": False,
        "labels_accessible_during_analysis": True,
        "K_accessible_during_fit": False,
        "K_accessible_during_profile": False,
        "support_cycle_raw_pearson": None,
        "support_cycle_raw_pearson_status": "unavailable_without_a_V23_support_raw_fingerprint",
        "controls": controls.diagnostics,
        "residualizer": residual.diagnostics,
        "conditional_pair_utility": {
            "base_auc": utility.base_auc,
            "plus_auc": utility.plus_auc,
            "delta_auc": utility.delta_auc,
            "fold_delta_auc": utility.fold_deltas.tolist(),
            "diagnostics": utility.residual_diagnostics,
        },
        "bootstrap": {
            "replicates_requested": int(bootstrap_replicates),
            "replicates_completed": int(bootstrap.size),
            "workers": int(bootstrap_workers),
            "ci95_low": float(np.quantile(bootstrap, 0.025)) if bootstrap.size else None,
            "ci95_high": float(np.quantile(bootstrap, 0.975)) if bootstrap.size else None,
            "full_pipeline_refit": bool(bootstrap_replicates > 0),
            "scheme": "poisson_weighted_sample_bootstrap_fixed_outer_folds",
            "physical_row_duplication": False,
            "outer_train_test_original_sample_disjoint": True,
        },
    }
    arrays = {
        "C_residual": residual.residual,
        "C_prediction": residual.prediction,
        "C_residual_r2_by_intervention": residual.r2_by_intervention,
        "S_support": controls.support,
        "M_marginal": controls.marginal,
        "pair_targets": utility.records["targets"],
        "pair_base_scores": utility.records["base_scores"],
        "pair_plus_scores": utility.records["plus_scores"],
        "pair_indices": utility.records["pairs"],
        "pair_outer_fold": utility.records["pair_fold"],
        "bootstrap_delta_auc": bootstrap,
    }
    return summary, arrays


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V24-Q1 outer conditional response analysis")
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--fingerprints", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True, help="outer evaluation labels only")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-replicates", type=int, default=0)
    parser.add_argument("--bootstrap-workers", type=int, default=1)
    args = parser.parse_args()
    with np.load(args.matrix, allow_pickle=False) as loaded:
        matrix = np.asarray(loaded["X"], dtype=np.float32)
    with np.load(args.fingerprints, allow_pickle=False) as loaded:
        fingerprints = {name: np.asarray(loaded[name]) for name in loaded.files}
    labels = np.asarray(np.load(args.labels, allow_pickle=False), dtype=np.int64)
    config = V24Q1Config()
    summary, arrays = analyze_response(
        matrix,
        fingerprints,
        labels=labels,
        config=config,
        seed=args.seed,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_workers=args.bootstrap_workers,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "conditional_response.npz", **arrays)
    _write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
