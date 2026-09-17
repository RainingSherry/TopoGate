from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, "/home/luolie/ToPoGate")

from methods.TopoGate.V0_RG.config import V0_RGConfig
from methods.TopoGate.V0_RG.run import clustering_metrics
from methods.TopoGate.V0_RG.trainer import fit_predict
from methods.TopoGate.V0_RG.tuning import SplitPreprocessor, split_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Young")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with h5py.File(args.data_root / f"{args.dataset}.h5", "r") as handle:
        raw_x = np.asarray(handle["X"][:], dtype=np.float32)
        raw_y = np.asarray(handle["Y"][:]).reshape(-1)
    labels = LabelEncoder().fit_transform(raw_y.astype(str)).astype(np.int64)
    test = split_rows(len(labels), 9102026)[2]
    selected = json.loads((args.old_root / args.dataset / "selected.json").read_text())
    config = V0_RGConfig(**selected["winner"]["config"])
    preprocessor = SplitPreprocessor("raw_count", 2000).fit(raw_x)
    x_all = preprocessor.transform(raw_x)

    runs = []
    embeddings = []
    histories = []
    for repeat in range(args.repeats):
        predictions, embedding, diagnostics = fit_predict(
            x_all,
            fit_X=x_all,
            n_clusters=len(np.unique(labels)),
            config=config,
            seed=args.seed,
            device="cuda:0",
        )
        embeddings.append(embedding)
        histories.append(diagnostics["training_history"])
        runs.append(
            {
                "repeat": repeat + 1,
                "metrics": clustering_metrics(labels[test], predictions[test]),
                "embedding_sum": float(np.sum(embedding, dtype=np.float64)),
            }
        )

    comparisons = []
    for index in range(1, len(runs)):
        history_delta = 0.0
        for key, first in histories[0].items():
            other = histories[index].get(key)
            if isinstance(first, list) and isinstance(other, list):
                history_delta = max(
                    history_delta,
                    float(np.max(np.abs(np.asarray(first) - np.asarray(other)))),
                )
        comparisons.append(
            {
                "repeat_a": 1,
                "repeat_b": index + 1,
                "embedding_max_abs_delta": float(
                    np.max(np.abs(embeddings[0] - embeddings[index]))
                ),
                "training_history_max_abs_delta": history_delta,
                "metric_abs_deltas": {
                    key: abs(runs[0]["metrics"][key] - runs[index]["metrics"][key])
                    for key in ("ari", "nmi", "acc")
                },
            }
        )

    payload = {
        "dataset": args.dataset,
        "seed": args.seed,
        "same_process_same_gpu_repeats": args.repeats,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "torch_deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "runs": runs,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
