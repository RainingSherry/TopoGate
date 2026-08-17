from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import torch

from .feature_model import CrossFittedFeatureModel, FeatureFoldModel


def _torch_sparse(matrix: sp.spmatrix, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coo = matrix.tocoo()
    indices = torch.as_tensor(np.vstack((coo.row, coo.col)), dtype=torch.long, device=device)
    values = torch.as_tensor(coo.data, dtype=dtype, device=device)
    return torch.sparse_coo_tensor(indices, values, size=coo.shape, device=device, dtype=dtype).coalesce()


def _footprint_matrix(fold: FeatureFoldModel) -> sp.csr_matrix:
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    for feature, footprint in enumerate(fold.footprints):
        rows.append(np.full(footprint.size, feature, dtype=np.int64))
        cols.append(footprint)
    row = np.concatenate(rows) if rows else np.empty(0, dtype=np.int64)
    col = np.concatenate(cols) if cols else np.empty(0, dtype=np.int64)
    data = np.ones(row.size, dtype=np.float32)
    return sp.csr_matrix((data, (row, col)), shape=(fold.n_features, fold.n_features))


@dataclass
class TorchFeatureConstraint:
    model: CrossFittedFeatureModel
    _cache: dict[tuple[int, str, torch.dtype], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = field(default_factory=dict)

    def _fold_tensors(
        self,
        fold: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (int(fold), str(device), dtype)
        if key not in self._cache:
            fitted = self.model.folds[int(fold)]
            prediction = _torch_sparse(fitted.prediction, device, dtype)
            footprints = _torch_sparse(_footprint_matrix(fitted), device, dtype)
            scale = torch.as_tensor(fitted.residual_scale, dtype=dtype, device=device)
            self._cache[key] = (prediction, footprints, scale)
        return self._cache[key]

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        center = torch.as_tensor(self.model.transform.center, dtype=X.dtype, device=X.device)
        scale = torch.as_tensor(self.model.transform.scale, dtype=X.dtype, device=X.device)
        return torch.clamp((X - center[None, :]) / scale[None, :], -self.model.transform.clip, self.model.transform.clip)

    def joint_delta(
        self,
        clean: torch.Tensor,
        donor: torch.Tensor,
        mask_st: torch.Tensor,
        hard_mask: torch.Tensor,
        row_ids: np.ndarray,
    ) -> torch.Tensor:
        """Differentiate action values while keeping the declared hard footprint."""

        if clean.shape != donor.shape or clean.shape != mask_st.shape or clean.shape != hard_mask.shape:
            raise ValueError("joint energy tensors must have identical shapes")
        rows = np.asarray(row_ids, dtype=np.int64)
        if rows.size != clean.shape[0]:
            raise ValueError("row_ids must have one entry per batch row")
        clean_z = self.transform(clean)
        donor_z = self.transform(donor)
        action_z = clean_z + mask_st * (donor_z - clean_z)
        output = torch.zeros(clean.shape[0], dtype=clean.dtype, device=clean.device)
        fold_ids = self.model.fold_ids[rows]
        for fold in np.unique(fold_ids):
            local_np = np.flatnonzero(fold_ids == fold)
            local = torch.as_tensor(local_np, dtype=torch.long, device=clean.device)
            prediction, footprints, residual_scale = self._fold_tensors(int(fold), clean.device, clean.dtype)
            clean_local = clean_z.index_select(0, local)
            action_local = action_z.index_select(0, local)
            clean_prediction = torch.sparse.mm(prediction, clean_local.T).T
            action_prediction = torch.sparse.mm(prediction, action_local.T).T
            clean_residual = (clean_local - clean_prediction) / residual_scale[None, :]
            action_residual = (action_local - action_prediction) / residual_scale[None, :]
            hard_local = hard_mask.index_select(0, local).to(clean.dtype)
            footprint_score = torch.sparse.mm(footprints.T, hard_local.T).T
            footprint = (footprint_score > 0.0).to(clean.dtype)
            denominator = footprint.sum(dim=1).clamp_min(1.0)
            delta = ((action_residual.square() - clean_residual.square()) * footprint).sum(dim=1) / denominator
            output.index_copy_(0, local, delta)
        return output
