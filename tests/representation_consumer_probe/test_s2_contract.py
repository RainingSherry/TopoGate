from __future__ import annotations

import inspect

import numpy as np
import scipy.sparse as sp
import torch

from scripts.representation_consumer_probe.s2_simple_cut import (
    SimpleCutEncoder,
    simplecut_loss,
    train_simplecut,
)


def _small_graph() -> sp.csr_matrix:
    return sp.csr_matrix(
        np.array(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.8, 0.0, 0.0, 0.0],
                [0.0, 0.8, 0.0, 1.1, 0.0, 0.0],
                [0.0, 0.0, 1.1, 0.0, 0.7, 0.0],
                [0.0, 0.0, 0.0, 0.7, 0.0, 0.9],
                [0.0, 0.0, 0.0, 0.0, 0.9, 0.0],
            ],
            dtype=np.float32,
        )
    )


def test_simplecut_fit_contract_excludes_labels_and_k() -> None:
    signature = inspect.signature(train_simplecut)
    assert "labels" not in signature.parameters
    assert "n_clusters" not in signature.parameters
    assert SimpleCutEncoder(4).network[-1].out_features == 32


def test_simplecut_loss_is_finite_and_noncollapsed_on_tiny_graph() -> None:
    h0 = np.eye(6, 4, dtype=np.float32)
    graph = _small_graph()
    embedding, history, metadata = train_simplecut(
        h0,
        graph,
        seed=42,
        device=torch.device("cpu"),
        epochs=3,
    )
    assert embedding.shape == (6, 32)
    assert len(history) == 3
    assert all(np.isfinite(row["loss"]) for row in history)
    assert metadata["labels_vector_used_in_fit"] is False
    assert np.isfinite(embedding).all()
    dimension_std = np.std(embedding, axis=0)
    assert float(np.min(dimension_std)) > 0.0
    assert float(np.mean(dimension_std <= 1e-6)) < 1.0

    z = torch.from_numpy(embedding)
    indices = torch.tensor(np.vstack(graph.nonzero()), dtype=torch.long)
    values = torch.tensor(graph.data, dtype=torch.float32)
    sparse = torch.sparse_coo_tensor(indices, values, size=graph.shape).coalesce()
    degrees = torch.sparse.sum(sparse, dim=1).to_dense()
    total, parts = simplecut_loss(z, sparse, degrees)
    assert torch.isfinite(total)
    assert all(torch.isfinite(value).all() for value in parts.values())
