#!/usr/bin/env python3
"""Run the missing Olivetti t-SNE + HDPC reference used by the comparison."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.manifold import TSNE

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "baseline" / "AHDPC"))

from ahdpc import AHDPC, evaluate_clustering


DATA_PATH = REPO_ROOT / "datasets" / "AHDPC" / "processed" / "olivetti_faces.npz"
OUTPUT_DIR = REPO_ROOT / "result" / "v9_results_2026-08-02" / "olivetti_hdpc_reference"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    data = np.load(DATA_PATH)
    x = np.asarray(data["x"] if "x" in data.files else data["X"], dtype=np.float64)
    y = np.asarray(data["y"]).ravel()
    n_clusters = int(np.unique(y).size)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    embedding = TSNE(
        n_components=2,
        perplexity=30.0,
        max_iter=1000,
        init="pca",
        learning_rate="auto",
        random_state=42,
    ).fit_transform(x)
    model = AHDPC(
        n_clusters=n_clusters,
        epsilon=0.1,
        adaptive=False,
        adaptive_distance_rule="table_reproduction",
        store_distance_matrix=False,
    )
    prediction = model.fit_predict(embedding)
    elapsed = time.perf_counter() - started
    metrics = evaluate_clustering(y, prediction)

    np.save(OUTPUT_DIR / "tsne_embedding.npy", embedding)
    np.save(OUTPUT_DIR / "predictions.npy", prediction)
    np.save(OUTPUT_DIR / "labels_true.npy", y)
    summary = {
        "run_status": "completed",
        "dataset": "olivetti_faces",
        "variant": "hdpc_tsne_reference",
        "source_path": str(DATA_PATH.resolve()),
        "source_sha256": sha256_file(DATA_PATH),
        "n_samples": int(x.shape[0]),
        "n_features_raw": int(x.shape[1]),
        "n_clusters": n_clusters,
        "epsilon": 0.1,
        "adaptive": False,
        "tsne_perplexity": 30.0,
        "tsne_max_iter": 1000,
        "tsne_seed": 42,
        "labels_used_during_fit": False,
        "metrics": metrics,
        "elapsed_seconds": float(elapsed),
        "output_contract": {
            "embedding": "tsne_embedding.npy",
            "predictions": "predictions.npy",
            "labels_true": "labels_true.npy",
        },
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8"
    )
    print(" ".join(f"{key}={value:.4f}" for key, value in metrics.items()))
    print(f"Artifacts: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
