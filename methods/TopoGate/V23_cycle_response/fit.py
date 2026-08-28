from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config
from .data import file_sha256, fit_semantic_preprocessor, load_matrix_only, save_preprocessor


ALLOWED_PHYSICAL_GPUS = frozenset({1, 2, 3, 4, 5, 6})


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit one label-isolated V23 canonical scMAE backbone")
    parser.add_argument("--matrix", type=Path, required=True, help="matrix-only NPZ; labels are not accepted")
    parser.add_argument("--input-protocol", choices=("clubench_bridge", "shared_text", "scRNA_count"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--feature-cap", type=int, default=None)
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
    from .training import checkpoint_payload, fit_backbone

    overrides = {
        key: value
        for key, value in {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "feature_cap": args.feature_cap,
        }.items()
        if value is not None
    }
    config = load_config(args.config, overrides)
    matrix = load_matrix_only(args.matrix)
    prepared = fit_semantic_preprocessor(
        matrix,
        input_protocol=args.input_protocol,
        feature_cap=config.feature_cap,
    )
    result = fit_backbone(prepared, config=config, seed=args.seed, device=device)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    save_preprocessor(output / "preprocessor.npz", prepared.preprocessor)
    np.save(output / "embedding_clean.npy", result.clean_embedding)
    import torch

    torch.save(checkpoint_payload(result, config, args.seed), output / "checkpoint.pt")
    matrix_sha256 = file_sha256(args.matrix)
    _write_json(
        output / "resolved_config.json",
        config.to_dict()
        | {
            "seed": args.seed,
            "device": str(device),
            "physical_gpu": args.gpu,
            "matrix_path": str(args.matrix.resolve()),
            "matrix_sha256": matrix_sha256,
        },
    )
    _write_json(output / "preprocess_profile.json", prepared.profile)
    _write_json(output / "training_history.json", result.history)
    _write_json(output / "linear_decoder_history.json", result.linear_history)
    summary = {
        "status": "completed",
        "evidence_level": "engineering_smoke" if config.epochs < 10 else "experiment",
        "protocol_id": config.protocol_id,
        "stage": "fit",
        "seed": int(args.seed),
        "matrix_path": str(args.matrix.resolve()),
        "matrix_sha256": matrix_sha256,
        "physical_gpu": args.gpu,
        "n_samples": int(prepared.model.shape[0]),
        "n_features_original": int(prepared.profile["n_features_original"]),
        "n_features_selected": int(prepared.model.shape[1]),
        "labels_accessible_during_fit": False,
        "K_accessible_during_fit": False,
        "canonical_decoder": "scmae_mask_logit_assisted",
        "decoder_control": "frozen_encoder_latent_only_linear",
    }
    _write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    main()
