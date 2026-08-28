from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.TopoGate.V23_cycle_response.config import V23Config
from methods.TopoGate.V23_cycle_response.data import fit_semantic_preprocessor
from methods.TopoGate.V23_cycle_response.masks import build_mask_dictionary
from methods.TopoGate.V23_cycle_response.profiling import profile_fingerprints
from methods.TopoGate.V23_cycle_response.training import fit_backbone
from methods.TopoGate.V24_conditional_response.analyze import analyze_response
from methods.TopoGate.V24_conditional_response.config import V24Q1Config
from methods.TopoGate.V24_conditional_response.contracts import audit_world
from methods.TopoGate.V24_conditional_response.synthetic import generate_worlds


def _v24_smoke_config() -> V24Q1Config:
    return V24Q1Config(
        n_samples=120,
        n_features=40,
        n_clusters=3,
        zero_fraction=0.50,
        block_size=4,
        active_blocks_per_sample=5,
        fingerprint_masks=3,
        fingerprint_mask_ratio=0.25,
        dependency_separation_min=0.05,
        outer_folds=3,
        inner_folds=2,
        pair_count_per_fold=20,
        bootstrap_replicates=3,
        calibration_replicates=2,
    )


def _v23_smoke_config() -> V23Config:
    return V23Config(
        protocol_id="v24_q1_engineering_smoke_only",
        feature_cap=40,
        hidden_size=8,
        epochs=2,
        batch_size=32,
        fingerprint_masks=3,
        fingerprint_mask_ratio=0.25,
        latent_linear_epochs=2,
        lowrank_rank=8,
        profile_batch_size=64,
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a CPU-only V24-Q1 engineering smoke")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    v24_config = _v24_smoke_config()
    v23_config = _v23_smoke_config()
    matrix, labels = generate_worlds(v24_config, seed=args.seed)["W4_dependency_only"]
    contract = audit_world(
        matrix,
        labels,
        world="W4_dependency_only",
        config=v24_config,
        seed=args.seed,
        run_classifiers=False,
    )
    if not contract.valid:
        raise RuntimeError("engineering smoke synthetic contract failed")

    # Labels remain outside the V23 fit/profile calls. They enter only the
    # final V24 outer evaluation, as they would in the formal protocol.
    prepared = fit_semantic_preprocessor(matrix, input_protocol="clubench_bridge", feature_cap=v23_config.feature_cap)
    device = torch.device("cpu")
    fitted = fit_backbone(prepared, config=v23_config, seed=args.seed, device=device)
    dictionary = build_mask_dictionary(
        n_samples=prepared.semantic.shape[0],
        n_features=prepared.semantic.shape[1],
        n_masks=v23_config.fingerprint_masks,
        mask_ratio=v23_config.fingerprint_mask_ratio,
        mask_seed=1701,
        donor_seed=2903,
    )
    bundle = profile_fingerprints(
        prepared,
        model=fitted.model,
        linear_decoder=fitted.linear_decoder,
        mask_dictionary=dictionary,
        config=v23_config,
        seed=args.seed,
        corruption_mode="donor_swap",
        device=device,
    )
    summary, arrays = analyze_response(
        matrix,
        bundle.arrays,
        labels=labels,
        config=v24_config,
        seed=args.seed,
        bootstrap_replicates=v24_config.bootstrap_replicates,
    )

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "matrix_only.npz", X=matrix)
    np.save(output / "labels_true.npy", labels)
    np.savez_compressed(output / "fingerprints.npz", **bundle.arrays)
    np.savez_compressed(output / "conditional_response.npz", **arrays)
    _write_json(output / "contract.json", {"valid": contract.valid, "metrics": contract.metrics})
    _write_json(
        output / "summary.json",
        {
            "status": "completed",
            "evidence_level": "engineering_smoke_only",
            "formal_q1_eligible": False,
            "purpose": "verify V23 fit/profile to V24 conditional-analysis integration on CPU",
            "labels_accessible_during_fit": False,
            "labels_accessible_during_profile": False,
            "labels_accessible_during_analysis": True,
            "v23_config": v23_config.to_dict(),
            "v24_config": v24_config.to_dict(),
            "conditional_analysis": summary,
        },
    )
    print(json.dumps({"status": "completed", "evidence_level": "engineering_smoke_only"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
