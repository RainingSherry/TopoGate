"""Read-only audit of student_grade H4/H8 candidate and embedding identity."""
from __future__ import annotations

import json
import numpy as np

import depth_worker as d


def main():
    manifest = json.loads((d.HERE / "REUSE_MANIFEST.json").read_text())
    case = manifest["cases"]["student_grade"]
    loaded = d.load_npz_matrix_only(case["feature_path"])
    prep = d.SplitPreprocessor(case["input_kind"], case["feature_limit"]).fit(loaded.X)
    if prep.fingerprint != case["preprocessing_hash"]:
        raise ValueError("preprocessing mismatch")
    X = prep.transform(loaded.X)
    rows = []
    for seed in d.SEEDS:
        frozen = d.build_frozen_graph(X, d.materialize(case["expected_config_template"], "H1"), seed)
        cfg = d.materialize(case["expected_config_template"], "H4_B1")
        h4 = d.plan_arrays(d.build_exact_hop_plan(frozen["graph"], frozen["edge_weights"],
                                                    "H4_B1", cfg.mix_neighbors,
                                                    cfg.high_hop_probability_multiplier))
        h8_folder = d.HERE / f"outputs/train/student_grade__H8__seed{seed}/attempt_001"
        h8 = d.load_saved_plan(h8_folder / "plan.npz", len(X))
        reused = next(c for c in manifest["depth_cells"] if
                      c["dataset"] == "student_grade" and c["depth"] == "H4" and c["seed"] == seed)
        with np.load(reused["artifacts"][f"embedding_seed_{seed}.npz"], allow_pickle=False) as packed:
            h4_emb = packed["embedding"]
        h8_emb = np.load(h8_folder / "embedding.npy", allow_pickle=False)
        comparisons = {key: bool(np.array_equal(h4[key], h8[key]))
                       for key in ("offsets", "nodes", "probs", "hops", "draw_count")}
        max_probability_delta = float(np.max(np.abs(h4["probs"] - h8["probs"])))
        max_embedding_delta = float(np.max(np.abs(h4_emb - h8_emb)))
        rows.append({"seed": seed, **comparisons,
                     "h4_max_hop": int(h4["hops"].max()),
                     "h8_max_hop": int(h8["hops"].max()),
                     "max_probability_delta": max_probability_delta,
                     "embedding_exact_equal": bool(np.array_equal(h4_emb, h8_emb)),
                     "max_embedding_delta": max_embedding_delta})
    out = d.HERE / "outputs/E2_E6/student_h4_h8_saturation_audit.json"
    d.atomic_json(out, {"status": "completed", "rows": rows,
                        "interpretation": "H4/H8 candidate identity and saved-embedding comparison; no training"})
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
