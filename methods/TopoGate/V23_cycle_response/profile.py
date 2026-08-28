from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .config import V23Config
from .data import apply_semantic_preprocessor, file_sha256, load_matrix_only, load_preprocessor
from .masks import build_mask_dictionary


ALLOWED_PHYSICAL_GPUS = frozenset({1, 2, 3, 4, 5, 6})


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate label-isolated V23 frozen response fingerprints")
    parser.add_argument("--matrix", type=Path, required=True, help="matrix-only NPZ; labels are not accepted")
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mask-seed", type=int, default=1701)
    parser.add_argument("--donor-seed", type=int, default=2903)
    parser.add_argument("--corruption-mode", choices=("donor_swap", "zero"), default="donor_swap")
    parser.add_argument("--fingerprint-masks", type=int, default=None)
    parser.add_argument("--fingerprint-mask-ratio", type=float, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--gpu", type=int, default=None)
    return parser.parse_args()


def _configure_device(args: argparse.Namespace):
    if args.device == "cpu":
        if args.gpu is not None:
            raise ValueError("--gpu is invalid with --device cpu")
        import torch

        return torch.device("cpu")
    if args.gpu not in ALLOWED_PHYSICAL_GPUS:
        raise ValueError("CUDA requires explicit --gpu in 1..6; physical GPUs 0 and 7 are forbidden")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device("cuda:0")


def main() -> dict[str, Any]:
    args = _parse_args()
    device = _configure_device(args)
    import torch

    from .model import CycleAutoEncoder, LatentLinearDecoder
    from .profiling import profile_fingerprints

    checkpoint = torch.load(args.fit_dir / "checkpoint.pt", map_location=device, weights_only=True)
    config_payload = dict(checkpoint["config"])
    if args.fingerprint_masks is not None:
        config_payload["fingerprint_masks"] = int(args.fingerprint_masks)
    if args.fingerprint_mask_ratio is not None:
        config_payload["fingerprint_mask_ratio"] = float(args.fingerprint_mask_ratio)
    config = V23Config(**config_payload)
    config.validate()
    preprocessor = load_preprocessor(args.fit_dir / "preprocessor.npz")
    prepared = apply_semantic_preprocessor(load_matrix_only(args.matrix), preprocessor)
    model = CycleAutoEncoder(
        num_genes=prepared.model.shape[1],
        hidden_size=config.hidden_size,
        masked_data_weight=config.masked_data_weight,
        mask_loss_weight=config.mask_loss_weight,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    linear_decoder = LatentLinearDecoder(config.hidden_size, prepared.model.shape[1]).to(device)
    linear_decoder.load_state_dict(checkpoint["linear_decoder"])
    mask_dictionary = build_mask_dictionary(
        n_samples=prepared.semantic.shape[0],
        n_features=prepared.semantic.shape[1],
        n_masks=config.fingerprint_masks,
        mask_ratio=config.fingerprint_mask_ratio,
        mask_seed=args.mask_seed,
        donor_seed=args.donor_seed,
    )
    bundle = profile_fingerprints(
        prepared,
        model=model,
        linear_decoder=linear_decoder,
        mask_dictionary=mask_dictionary,
        config=config,
        seed=int(checkpoint["seed"]),
        corruption_mode=args.corruption_mode,
        device=device,
    )
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "fingerprints.npz", **bundle.arrays)
    np.savez_compressed(
        output / "mask_dictionary.npz",
        masks=mask_dictionary.masks,
        donor_offsets=mask_dictionary.donor_offsets,
        mask_seed=np.asarray(mask_dictionary.mask_seed, dtype=np.int64),
        donor_seed=np.asarray(mask_dictionary.donor_seed, dtype=np.int64),
        mask_ratio=np.asarray(mask_dictionary.mask_ratio, dtype=np.float32),
    )
    mask_config = {
        "T": int(config.fingerprint_masks),
        "rho": float(config.fingerprint_mask_ratio),
        "mask_seed": int(args.mask_seed),
        "donor_seed": int(args.donor_seed),
        "corruption_mode": args.corruption_mode,
        "mask_dictionary_scope": "shared_across_all_samples",
        "corruption_space": "pre_centered_semantic",
        "effective_mask_space": "pre_centered_semantic",
        "primary_distance": "cosine",
    }
    _write_json(output / "mask_config.json", mask_config)
    _write_json(output / "mask_statistics.json", bundle.diagnostics)
    summary = {
        "status": "completed",
        "evidence_level": "engineering_smoke" if config.epochs < 10 else "experiment",
        "protocol_id": config.protocol_id,
        "stage": "profile",
        "matrix_path": str(args.matrix.resolve()),
        "matrix_sha256": file_sha256(args.matrix),
        "fit_dir": str(args.fit_dir.resolve()),
        "checkpoint_sha256": file_sha256(args.fit_dir / "checkpoint.pt"),
        "preprocessor_sha256": file_sha256(args.fit_dir / "preprocessor.npz"),
        "seed": int(checkpoint["seed"]),
        "physical_gpu": args.gpu,
        "primary_scientific_object": "cycle_repair_standardized",
        "secondary_mechanistic_object": "recovery_gain_standardized",
        "labels_accessible_during_profile": False,
        "K_accessible_during_profile": False,
        "diagnostics": bundle.diagnostics,
    }
    _write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    main()
