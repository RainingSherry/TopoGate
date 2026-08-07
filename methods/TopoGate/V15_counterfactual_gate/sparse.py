from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch


@dataclass
class PreparedInput:
    """Memory-aware representation used by the V15 trainer."""

    matrix: np.ndarray | sp.csr_matrix
    n_samples: int
    n_features: int
    sparse: bool
    profile: dict

    def get(self, indices: np.ndarray | torch.Tensor, device: torch.device) -> torch.Tensor:
        if isinstance(indices, torch.Tensor):
            indices_np = indices.detach().cpu().numpy().astype(np.int64, copy=False)
        else:
            indices_np = np.asarray(indices, dtype=np.int64)
        if self.sparse:
            values = self.matrix[indices_np].toarray()
        else:
            values = self.matrix[indices_np]
        values = np.asarray(values, dtype=np.float32)
        return torch.as_tensor(values, dtype=torch.float32, device=device)

    def full(self, device: torch.device) -> torch.Tensor:
        return self.get(np.arange(self.n_samples, dtype=np.int64), device)


def _safe_float32(X: np.ndarray) -> np.ndarray:
    values = np.asarray(X, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"X must be a 2D matrix, got {values.shape}")
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def prepare_input(
    X: np.ndarray | sp.spmatrix,
    sparse_zero_threshold: float = 0.5,
    sparse_transform: str = "log1p_row",
) -> PreparedInput:
    """Prepare dense or naturally sparse input without densifying sparse storage.

    A scipy sparse input stays sparse throughout preprocessing. Dense inputs are
    converted only once at this boundary; mini-batches are densified in
    :meth:`PreparedInput.get` when they enter the neural network.
    """
    if sparse_transform not in {"log1p_row", "tfidf_l2"}:
        raise ValueError("sparse_transform must be 'log1p_row' or 'tfidf_l2'")
    input_is_sparse = sp.issparse(X)
    if input_is_sparse:
        raw_csr = sp.csr_matrix(X, dtype=np.float32, copy=True)
        raw_csr.data = np.nan_to_num(raw_csr.data, nan=0.0, posinf=0.0, neginf=0.0)
        raw_csr.eliminate_zeros()
        n_samples, n_features = raw_csr.shape
        size = int(n_samples) * int(n_features)
        zero_fraction = float(1.0 - (raw_csr.nnz / size)) if size else 0.0
        nonnegative = bool(raw_csr.nnz == 0 or np.min(raw_csr.data) >= 0.0)
        values: np.ndarray | sp.csr_matrix = raw_csr
    else:
        values = _safe_float32(X)
        n_samples, n_features = values.shape
        zero_fraction = float(np.mean(values == 0.0)) if values.size else 0.0
        nonnegative = bool(np.min(values) >= 0.0) if values.size else True
    use_sparse = bool(input_is_sparse or (nonnegative and zero_fraction >= float(sparse_zero_threshold)))
    if use_sparse:
        csr = values if sp.issparse(values) else sp.csr_matrix(values, dtype=np.float32)
        csr = sp.csr_matrix(csr, dtype=np.float32, copy=True)
        if nonnegative:
            # Count/text-like inputs use the planned log1p-then-row-normalize
            # path. Negative sparse inputs remain sparse but are not clipped.
            csr.data = np.log1p(np.clip(csr.data, 0.0, None)).astype(np.float32)
        if sparse_transform == "tfidf_l2":
            document_frequency = np.asarray(csr.getnnz(axis=0)).ravel().astype(np.float32)
            idf = np.log((1.0 + float(n_samples)) / (1.0 + document_frequency)) + 1.0
            csr = csr.multiply(idf[None, :]).tocsr()
            row_norm = np.sqrt(np.asarray(csr.multiply(csr).sum(axis=1)).ravel()).astype(np.float32)
            inv = np.divide(1.0, row_norm, out=np.zeros_like(row_norm), where=row_norm > 0)
            csr = sp.diags(inv).dot(csr).tocsr()
            representation = "csr_tfidf_l2"
        else:
            row_sum = np.asarray(csr.sum(axis=1)).ravel().astype(np.float32)
            inv = np.divide(1.0, row_sum, out=np.zeros_like(row_sum), where=row_sum > 0)
            csr = sp.diags(inv).dot(csr).tocsr()
            representation = "csr_log1p_row_normalized"
        csr.eliminate_zeros()
        nnz = np.diff(csr.indptr)
        matrix: np.ndarray | sp.csr_matrix = csr
        profile = {
            "representation": representation,
            "raw_zero_fraction": zero_fraction,
            "raw_nonnegative": nonnegative,
            "input_storage": "sparse" if input_is_sparse else "dense",
            "nnz_mean": float(np.mean(nnz)) if nnz.size else 0.0,
            "nnz_median": float(np.median(nnz)) if nnz.size else 0.0,
            "nnz_p95": float(np.quantile(nnz, 0.95)) if nnz.size else 0.0,
        }
    else:
        # Dense embeddings are kept in float32. Standardization is intentionally
        # omitted here because graph cosine normalization already removes global
        # scale and some benchmark embeddings contain meaningful zero coordinates.
        matrix = values
        nnz = np.count_nonzero(values, axis=1)
        profile = {
            "representation": "dense_float32",
            "raw_zero_fraction": zero_fraction,
            "raw_nonnegative": nonnegative,
            "input_storage": "dense",
            "nnz_mean": float(np.mean(nnz)) if nnz.size else 0.0,
            "nnz_median": float(np.median(nnz)) if nnz.size else 0.0,
            "nnz_p95": float(np.quantile(nnz, 0.95)) if nnz.size else 0.0,
        }
    return PreparedInput(
        matrix=matrix,
        n_samples=int(values.shape[0]),
        n_features=int(values.shape[1]),
        sparse=use_sparse,
        profile=profile,
    )


def apply_mask(
    x: torch.Tensor,
    ratio: float,
    generator: torch.Generator | None = None,
    strategy: str = "zero",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Corrupt an anchor view and return coordinates whose values changed."""
    if strategy not in {"zero", "row_swap"}:
        raise ValueError("strategy must be 'zero' or 'row_swap'")
    if strategy == "row_swap":
        if x.shape[0] <= 1:
            return x.clone(), torch.zeros_like(x)
        permutation = torch.randperm(x.shape[0], device=x.device, generator=generator)
        replacement = x[permutation]
        selected = torch.rand(x.shape, dtype=x.dtype, device=x.device, generator=generator) < float(ratio)
        corrupted = torch.where(selected, replacement, x)
        return corrupted, corrupted.ne(x).to(dtype=x.dtype)
    observed = x.ne(0.0)
    random_values = torch.rand(x.shape, dtype=x.dtype, device=x.device, generator=generator)
    mask = (random_values < float(ratio)) & observed
    # A sparse row may have no Bernoulli-selected observed coordinate.  The
    # old fallback selected from every feature, which often marked a zero as
    # "masked" even though the input had not changed.  That silently made a
    # sampled-zero reconstruction term look like an observed MAE target.
    # Choose an actual observed coordinate instead; true all-zero rows remain
    # unmasked because there is no anchor evidence to hide.
    empty = mask.sum(dim=1).eq(0) & observed.any(dim=1)
    if torch.any(empty):
        for row in torch.where(empty)[0].tolist():
            observed_columns = torch.where(observed[row])[0]
            selected = torch.randint(
                observed_columns.numel(),
                (1,),
                device=x.device,
                generator=generator,
            )
            mask[row, observed_columns[selected]] = True
    corrupted = x.masked_fill(mask, 0.0)
    return corrupted, mask.to(dtype=x.dtype)


def sparse_reconstruction_per_sample(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    masked_weight: float,
    visible_weight: float,
    zero_weight: float,
    zero_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.shape != mask.shape:
        raise ValueError("prediction, target, and mask must have identical shapes")
    observed = target.ne(0.0).to(dtype=target.dtype)
    if zero_mask is None:
        zero_mask = (target == 0.0) & (mask == 0.0)
    if zero_mask.shape != target.shape:
        raise ValueError("zero_mask must match target shape")
    zero_mask = zero_mask.to(dtype=target.dtype) * (1.0 - observed) * (1.0 - mask)
    element = (prediction - target).square()
    weights = (
        mask * float(masked_weight)
        + (1.0 - mask) * observed * float(visible_weight)
        + zero_mask * float(zero_weight)
    )
    return (element * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1e-8)
