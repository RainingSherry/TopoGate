from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from sklearn.decomposition import PCA

from .config import V23Config
from .data import PreparedSemanticInput
from .masks import MaskDictionary, corrupt_semantic
from .model import CycleAutoEncoder, LatentLinearDecoder


@dataclass(frozen=True)
class FingerprintBundle:
    arrays: dict[str, np.ndarray]
    diagnostics: dict[str, object]


def _cosine_distance_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    numerator = np.sum(left * right, axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    similarity = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12)
    return np.asarray(1.0 - np.clip(similarity, -1.0, 1.0), dtype=np.float32)


@torch.no_grad()
def _apply_batches(
    values: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    function: Callable[[torch.Tensor], torch.Tensor],
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, values.shape[0], batch_size):
        batch = torch.as_tensor(values[start : start + batch_size], dtype=torch.float32, device=device)
        chunks.append(function(batch).detach().cpu().numpy())
    return np.nan_to_num(
        np.concatenate(chunks, axis=0).astype(np.float32, copy=False),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def robust_standardize(
    raw: np.ndarray,
    epsilon: float,
    clip: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    raw = np.asarray(raw, dtype=np.float32)
    median = np.median(raw, axis=0).astype(np.float32)
    mad = np.median(np.abs(raw - median[None, :]), axis=0).astype(np.float32)
    valid = mad > float(epsilon)
    standardized = np.zeros_like(raw, dtype=np.float32)
    standardized[:, valid] = (raw[:, valid] - median[None, valid]) / (1.4826 * mad[None, valid])
    standardized = np.nan_to_num(standardized, nan=0.0, posinf=0.0, neginf=0.0)
    clipped_fraction = float(np.mean(np.abs(standardized[:, valid]) > float(clip))) if bool(valid.any()) else 0.0
    np.clip(standardized, -float(clip), float(clip), out=standardized)
    return standardized, median, mad, valid, clipped_fraction


def _repair(
    clean_semantic: np.ndarray,
    reconstructed_model: np.ndarray,
    effective: np.ndarray,
    prepared: PreparedSemanticInput,
) -> np.ndarray:
    reconstructed_semantic = prepared.preprocessor.inverse_transform(reconstructed_model)
    repaired = clean_semantic.copy()
    repaired[effective] = reconstructed_semantic[effective]
    return prepared.preprocessor.transform(repaired)


def _fit_lowrank(model_view: np.ndarray, requested_rank: int, seed: int) -> np.ndarray:
    rank = max(1, min(int(requested_rank), model_view.shape[0] - 1, model_view.shape[1] - 1))
    pca = PCA(n_components=rank, svd_solver="randomized", random_state=int(seed))
    pca.fit(model_view)
    return np.asarray(pca.components_.T, dtype=np.float64)


def _lowrank_reconstruction(
    clean_model: np.ndarray,
    feature_mask: np.ndarray,
    basis: np.ndarray,
    ridge: float,
) -> np.ndarray:
    visible = ~feature_mask
    visible_basis = basis[visible]
    gram = visible_basis.T @ visible_basis + float(ridge) * np.eye(basis.shape[1], dtype=np.float64)
    coefficients = clean_model[:, visible].astype(np.float64) @ visible_basis @ np.linalg.inv(gram)
    return np.asarray(coefficients @ basis.T, dtype=np.float32)


def profile_fingerprints(
    prepared: PreparedSemanticInput,
    *,
    model: CycleAutoEncoder,
    linear_decoder: LatentLinearDecoder,
    mask_dictionary: MaskDictionary,
    config: V23Config,
    seed: int,
    corruption_mode: str,
    device: torch.device,
) -> FingerprintBundle:
    """Generate frozen response profiles. This function has no labels or K input."""

    mask_dictionary.validate(prepared.semantic.shape[0])
    if mask_dictionary.masks.shape[1] != prepared.semantic.shape[1]:
        raise ValueError("mask dictionary feature count differs from prepared input")
    model.eval()
    linear_decoder.eval()
    batch_size = int(config.profile_batch_size)
    clean_z = _apply_batches(prepared.model, batch_size=batch_size, device=device, function=model.encode)
    clean_reconstruction = _apply_batches(prepared.model, batch_size=batch_size, device=device, function=model.reconstruct)
    clean_full_z = _apply_batches(clean_reconstruction, batch_size=batch_size, device=device, function=model.encode)
    clean_full_drift = _cosine_distance_rows(clean_z, clean_full_z)

    torch.manual_seed(int(seed) + 100_003)
    if device.type == "cuda":
        torch.cuda.manual_seed(int(seed) + 100_003)
    untrained = CycleAutoEncoder(
        num_genes=prepared.model.shape[1],
        hidden_size=config.hidden_size,
        masked_data_weight=config.masked_data_weight,
        mask_loss_weight=config.mask_loss_weight,
    ).to(device)
    untrained.eval()
    untrained_clean_z = _apply_batches(prepared.model, batch_size=batch_size, device=device, function=untrained.encode)

    basis = _fit_lowrank(prepared.model, config.lowrank_rank, seed)
    n_samples = prepared.semantic.shape[0]
    n_masks = mask_dictionary.masks.shape[0]
    names = (
        "precycle_raw",
        "cycle_repair_raw",
        "recovery_gain_raw",
        "support_raw",
        "effective_mass_raw",
        "untrained_cycle_raw",
        "linear_cycle_raw",
        "lowrank_cycle_raw",
        "full_cycle_adjusted_raw",
    )
    arrays = {name: np.zeros((n_samples, n_masks), dtype=np.float32) for name in names}

    for mask_index, feature_mask in enumerate(mask_dictionary.masks):
        corrupted_semantic, effective = corrupt_semantic(
            prepared.semantic,
            feature_mask,
            donor_offset=int(mask_dictionary.donor_offsets[mask_index]),
            corruption_mode=corruption_mode,
        )
        corrupted_model = prepared.preprocessor.transform(corrupted_semantic)
        corrupted_z = _apply_batches(corrupted_model, batch_size=batch_size, device=device, function=model.encode)
        canonical_reconstruction = _apply_batches(
            corrupted_model,
            batch_size=batch_size,
            device=device,
            function=model.reconstruct,
        )
        repaired_model = _repair(prepared.semantic, canonical_reconstruction, effective, prepared)
        repaired_z = _apply_batches(repaired_model, batch_size=batch_size, device=device, function=model.encode)
        precycle = _cosine_distance_rows(clean_z, corrupted_z)
        cycle = _cosine_distance_rows(clean_z, repaired_z)
        arrays["precycle_raw"][:, mask_index] = precycle
        arrays["cycle_repair_raw"][:, mask_index] = cycle
        arrays["recovery_gain_raw"][:, mask_index] = precycle - cycle
        arrays["support_raw"][:, mask_index] = effective.mean(axis=1, dtype=np.float64).astype(np.float32)
        arrays["effective_mass_raw"][:, mask_index] = np.mean(
            np.abs(corrupted_semantic - prepared.semantic) * effective,
            axis=1,
            dtype=np.float64,
        ).astype(np.float32)

        full_z = _apply_batches(
            canonical_reconstruction,
            batch_size=batch_size,
            device=device,
            function=model.encode,
        )
        arrays["full_cycle_adjusted_raw"][:, mask_index] = _cosine_distance_rows(clean_z, full_z) - clean_full_drift

        untrained_reconstruction = _apply_batches(
            corrupted_model,
            batch_size=batch_size,
            device=device,
            function=untrained.reconstruct,
        )
        untrained_repaired_model = _repair(prepared.semantic, untrained_reconstruction, effective, prepared)
        untrained_repaired_z = _apply_batches(
            untrained_repaired_model,
            batch_size=batch_size,
            device=device,
            function=untrained.encode,
        )
        arrays["untrained_cycle_raw"][:, mask_index] = _cosine_distance_rows(
            untrained_clean_z,
            untrained_repaired_z,
        )

        linear_reconstruction = _apply_batches(
            corrupted_model,
            batch_size=batch_size,
            device=device,
            function=lambda batch: linear_decoder(model.encode(batch)),
        )
        linear_repaired_model = _repair(prepared.semantic, linear_reconstruction, effective, prepared)
        linear_repaired_z = _apply_batches(
            linear_repaired_model,
            batch_size=batch_size,
            device=device,
            function=model.encode,
        )
        arrays["linear_cycle_raw"][:, mask_index] = _cosine_distance_rows(clean_z, linear_repaired_z)

        lowrank_reconstruction = _lowrank_reconstruction(
            prepared.model,
            feature_mask,
            basis,
            config.lowrank_ridge,
        )
        lowrank_repaired_model = _repair(prepared.semantic, lowrank_reconstruction, effective, prepared)
        lowrank_repaired_z = _apply_batches(
            lowrank_repaired_model,
            batch_size=batch_size,
            device=device,
            function=model.encode,
        )
        arrays["lowrank_cycle_raw"][:, mask_index] = _cosine_distance_rows(clean_z, lowrank_repaired_z)

    standardization: dict[str, object] = {}
    for name in names:
        standardized_name = name.replace("_raw", "_standardized")
        standardized, median, mad, valid, clipped_fraction = robust_standardize(
            arrays[name],
            config.mad_epsilon,
            config.robust_clip,
        )
        arrays[standardized_name] = standardized
        arrays[f"{name}_column_median"] = median
        arrays[f"{name}_column_mad"] = mad
        arrays[f"{name}_valid_columns"] = valid.astype(np.bool_)
        standardization[name] = {
            "valid_columns": int(valid.sum()),
            "total_columns": int(valid.size),
            "raw_column_variance_mean": float(np.var(arrays[name], axis=0).mean()),
            "standardized_column_variance_mean": float(np.var(standardized, axis=0).mean()),
            "robust_clip": float(config.robust_clip),
            "clipped_fraction": clipped_fraction,
        }
    arrays["clean_embedding"] = clean_z
    arrays["clean_full_cycle_drift"] = clean_full_drift
    arrays["mask_dictionary"] = mask_dictionary.masks
    arrays["donor_offsets"] = mask_dictionary.donor_offsets
    diagnostics: dict[str, object] = {
        "primary_scientific_object": "cycle_repair_standardized",
        "secondary_recoverability_object": "recovery_gain_standardized",
        "primary_distance": "cosine",
        "corruption_mode": corruption_mode,
        "corruption_space": "pre_centered_semantic",
        "effective_mask_space": "pre_centered_semantic",
        "nominal_mask_ratio": float(mask_dictionary.masks.mean()),
        "effective_mask_ratio": float(arrays["support_raw"].mean()),
        "clean_cycle_drift_mean": float(clean_full_drift.mean()),
        "standardization": standardization,
        "labels_accessible_during_profile": False,
        "K_accessible_during_profile": False,
    }
    return FingerprintBundle(arrays=arrays, diagnostics=diagnostics)
