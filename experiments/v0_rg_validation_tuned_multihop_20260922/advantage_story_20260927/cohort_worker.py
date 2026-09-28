# papers/advantage_story_20260927/cohort_worker.py
"""
Cohort E3 posthoc worker: pure readout analysis on ALL available 59 pairs.

Usage:
  python cohort_worker.py --dataset NAME --out OUT

Protocol:
  - Remote BASE=/data/luolie/ToPoGate/experiments/v0_rg_validation_tuned_multihop_20260922
  - Import source_coverage helpers explicitly
  - Lock=BASE/source_coverage_20260925/outputs/p0/selection_lock_v2.json
  - info=lock['datasets'][name]; reject if matched_config false
  - H1 key: h1_counterpart_of_selected_multihop; multi key: selected_multihop
  - Each per_seed has record_path, embedding_path, record_sha256, embedding_sha256
  - Verify input hashes
  - Load features_only using tuning_manifest_path
  - Use load_aligned_labels(info, X) to verify matrix row alignment
  - K=info.n_clusters with benchmark/oracle provenance
  - For each seed in [7,42,123]: load saved embedding key 'embedding'
  - KMeans states 0..9, n_init=20, original sklearn environment, 2 threads
  - Save labels/layout stub NPZ and frozen_readouts_10.npz
  - Call local compute_e3_analysis(...) for exact pair rates/macro/readout ARI
  - No encoder fit or selection; posthoc_full_x_labels_loaded=true, labels_used_for_model_fit=false
  - Save record.json last with full input+artifact hashes and frozen protocol
  - Failure raises nonzero rather than zero
  - Parent will aggregate 59 outputs afterwards
  - No E4 community computations in this worker
  - Do not confuse selected_h1 (independent tuning) with matched counterpart
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='2'
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# Thread control
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"

HERE = Path(__file__).resolve().parent
BASE = Path("/data/luolie/ToPoGate/experiments/v0_rg_validation_tuned_multihop_20260922")
SOURCE_DIR = BASE / "source_coverage_20260925"

if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)

def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def matched_config(info):
    mh = info.get("selected_multihop")
    h1 = info.get("h1_counterpart_of_selected_multihop")
    
    if not isinstance(h1, dict):
        return False, "missing_matched_H1"
    if mh.get("preprocessing_hash") != h1.get("preprocessing_hash"):
        return False, "preprocessing_hash_differs"
    
    diffs = sorted(k for k in set(mh["config"]) | set(h1["config"])
                   if mh["config"].get(k) != h1["config"].get(k))
    if set(diffs) - {"hop_variant", "high_hop_probability_multiplier"}:
        return False, "non_path_config_differs:" + ",".join(diffs)
    
    if h1["config"].get("hop_variant") != "H1":
        return False, "counterpart_is_not_H1"
    
    if {int(s) for s in mh["per_seed"]} != {7, 42, 123} or {int(s) for s in h1["per_seed"]} != {7, 42, 123}:
        return False, "incomplete_training_seeds"
    
    return True, "complete"

def load_aligned_labels(info, X):
    from v0rg_tunable_multihop.input_adapter import encode_labels, load_matrix
    import scipy.sparse as sp
    
    problems = []
    for source in info["posthoc_label_source_paths"]:
        try:
            loaded = load_matrix(source)
            if loaded.X.shape != X.shape:
                raise ValueError("matrix_shape_mismatch")
            delta = sp.csr_matrix(loaded.X) - sp.csr_matrix(X)
            if delta.nnz and float(np.max(np.abs(delta.data))) > 1e-6:
                raise ValueError("matrix_rows_or_values_mismatch")
            y, names = encode_labels(loaded.labels)
            y = np.asarray(y, dtype=np.int32).reshape(-1)
            if len(y) != X.shape[0]:
                raise ValueError("label_length_mismatch")
            return y, [str(x) for x in names], {
                "posthoc_label_source": source,
                "posthoc_label_source_sha256": sha256(source),
                "matrix_row_alignment_verified": True
            }
        except Exception as exc:
            problems.append(f"{source}: {type(exc).__name__}: {exc}")
    
    raise ValueError("no aligned posthoc labels: " + "; ".join(problems))

def compute_split_merge_rates(labels_true, predictions):
    labels_true = np.asarray(labels_true)
    predictions = np.asarray(predictions)
    _, predictions = np.unique(predictions, return_inverse=True)
    
    n = len(labels_true)
    if n == 0:
        return np.nan, np.nan, {}, {}
    
    k = int(labels_true.max()) + 1
    nc = int(predictions.max()) + 1
    
    tab = np.zeros((k, nc), dtype=np.int64)
    for c, p in zip(labels_true, predictions):
        tab[c, p] += 1
    
    class_counts = tab.sum(axis=1)
    cluster_counts = tab.sum(axis=0)
    
    same_class_pairs = (class_counts * (class_counts - 1)) / 2
    correct_pairs_per_class = ((tab * (tab - 1)) / 2).sum(axis=1)
    split_pairs_per_class = same_class_pairs - correct_pairs_per_class
    
    per_class_split = {}
    for c in range(k):
        denom = same_class_pairs[c]
        per_class_split[c] = (split_pairs_per_class[c] / denom) if denom > 0 else np.nan
    
    total_pairs = (n * (n - 1)) / 2
    different_class_pairs = total_pairs - same_class_pairs.sum()
    
    cluster_total_pairs = (cluster_counts * (cluster_counts - 1)) / 2
    correct_pairs_per_cluster = ((tab * (tab - 1)) / 2).sum(axis=0)
    merge_pairs_total = (cluster_total_pairs - correct_pairs_per_cluster).sum()
    
    per_class_merge = {}
    for c in range(k):
        c_in_cluster = tab[c, :]
        others_in_cluster = cluster_counts - c_in_cluster
        merge_pairs_c = (c_in_cluster * others_in_cluster).sum()
        denom = class_counts[c] * (n - class_counts[c])
        per_class_merge[c] = (merge_pairs_c / denom) if denom > 0 else np.nan
    
    split_rate = split_pairs_per_class.sum() / same_class_pairs.sum() if same_class_pairs.sum() > 0 else np.nan
    merge_rate = merge_pairs_total / different_class_pairs if different_class_pairs > 0 else np.nan
    
    return split_rate, merge_rate, per_class_split, per_class_merge

def compute_e3_analysis_data(y, names, frozen_readouts):
    seeds = frozen_readouts["seeds"]
    n = len(y)
    
    results = {
        "per_readout": [],
        "per_seed": [],
        "aggregate": []
    }
    
    aggregate = {"H1": {"split": [], "merge": []}, "multihop": {"split": [], "merge": []}}
    
    for seed in seeds:
        for arm in ["H1", "multihop"]:
            key = f"seed_{seed}_{arm}_predictions"
            if key not in frozen_readouts:
                raise ValueError(f"Missing {key}")
            
            preds = frozen_readouts[key]
            if preds.shape != (10, n):
                raise ValueError(f"Invalid shape for {key}: {preds.shape}")
            
            readout_splits = []
            readout_merges = []
            readout_aris = []
            readout_macro_splits = []
            readout_macro_merges = []
            
            for r_idx in range(10):
                pred = preds[r_idx]
                split_r, merge_r, per_class_split_r, per_class_merge_r = compute_split_merge_rates(y, pred)
                ari = adjusted_rand_score(y, pred)
                
                readout_splits.append(split_r)
                readout_merges.append(merge_r)
                readout_aris.append(ari)
                readout_macro_splits.append(np.nanmean(list(per_class_split_r.values())))
                readout_macro_merges.append(np.nanmean(list(per_class_merge_r.values())))
                
                results["per_readout"].append({
                    "arm": arm,
                    "seed": int(seed),
                    "readout_idx": r_idx,
                    "split_rate": float(split_r) if np.isfinite(split_r) else None,
                    "merge_rate": float(merge_r) if np.isfinite(merge_r) else None,
                    "ARI": float(ari),
                    "macro_split_rate": float(np.nanmean(list(per_class_split_r.values()))),
                    "macro_merge_rate": float(np.nanmean(list(per_class_merge_r.values())))
                })
            
            split_mean = np.mean(readout_splits)
            merge_mean = np.mean(readout_merges)
            ari_mean = np.mean(readout_aris)
            
            results["per_seed"].append({
                "arm": arm,
                "seed": int(seed),
                "split_rate_mean": float(split_mean),
                "merge_rate_mean": float(merge_mean),
                "ARI_mean": float(ari_mean),
                "macro_split_rate_mean": float(np.mean(readout_macro_splits)),
                "macro_merge_rate_mean": float(np.mean(readout_macro_merges))
            })
            
            aggregate[arm]["split"].append(split_mean)
            aggregate[arm]["merge"].append(merge_mean)
    
    for arm in ["H1", "multihop"]:
        results["aggregate"].append({
            "arm": arm,
            "split_rate_mean": float(np.mean(aggregate[arm]["split"])),
            "split_rate_std": float(np.std(aggregate[arm]["split"], ddof=1)) if len(aggregate[arm]["split"]) > 1 else None,
            "merge_rate_mean": float(np.mean(aggregate[arm]["merge"])),
            "merge_rate_std": float(np.std(aggregate[arm]["merge"], ddof=1)) if len(aggregate[arm]["merge"]) > 1 else None
        })
    
    return results

def main():
    threadpool_limits(limits=2)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    
    dataset_name = args.dataset
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    lock_path = SOURCE_DIR / "outputs/p0/selection_lock_v2.json"
    if not lock_path.exists():
        raise FileNotFoundError(f"Selection lock missing: {lock_path}")
    
    lock_data = read_json(lock_path)
    expected_lock=Path(__file__).with_name('COHORT_LOCK_SHA256.txt').read_text().strip()
    if sha256(lock_path)!=expected_lock:raise ValueError('Cohort selection lock changed')
    
    if dataset_name not in lock_data["datasets"]:
        raise KeyError(f"Dataset {dataset_name!r} not in selection lock")
    
    info = lock_data["datasets"][dataset_name]
    
    ok, reason = matched_config(info)
    if not ok:
        raise ValueError(f"No matched config for {dataset_name}: {reason}")
    
    # Load features
    from v0rg_tunable_multihop.input_adapter import load_npz_matrix_only
    
    manifest = read_json(info["tuning_manifest_path"])
    if sha256(manifest["feature_path"]) != info["feature_sha256"]:
        raise ValueError("Feature matrix hash mismatch")
    
    X = load_npz_matrix_only(manifest["feature_path"]).X
    if X.shape[0] != info["n_samples"]:
        raise ValueError("Feature row count mismatch")
    
    # Load labels posthoc
    y, names, alignment = load_aligned_labels(info, X)
    
    if len(names) != info["n_clusters"]:
        raise ValueError("Oracle K differs from aligned posthoc labels")
    
    K = info["n_clusters"]
    
    # Verify embeddings and run KMeans
    seeds = [7, 42, 123]
    frozen_dict = {"seeds": np.array(seeds, dtype=np.int32),'states':np.arange(10,dtype=np.int32),'n_init':np.array(20)}
    
    input_hashes = {
        "selection_lock_v2.json": sha256(lock_path),
        "feature_matrix": info["feature_sha256"],
        "tuning_manifest": sha256(info["tuning_manifest_path"])
    }
    
    for arm, field in [("H1", "h1_counterpart_of_selected_multihop"), ("multihop", "selected_multihop")]:
        selected = info[field]
        
        for seed in seeds:
            item = selected["per_seed"][str(seed)]
            
            record_path = Path(item["record_path"])
            embedding_path = Path(item["embedding_path"])
            
            if sha256(record_path) != item["record_sha256"]:
                raise ValueError(f"Record hash mismatch: {arm} seed {seed}")
            if sha256(embedding_path) != item["embedding_sha256"]:
                raise ValueError(f"Embedding hash mismatch: {arm} seed {seed}")
            
            input_hashes[f"{arm}_seed_{seed}_record"] = item["record_sha256"]
            input_hashes[f"{arm}_seed_{seed}_embedding"] = item["embedding_sha256"]
            
            with np.load(embedding_path, allow_pickle=False) as packed:
                arrays = [packed[k] for k in packed.files if packed[k].ndim == 2]
                if len(arrays) != 1:
                    raise ValueError(f"Expected exactly one 2D embedding: {embedding_path}")
                z = np.asarray(arrays[0], dtype=np.float32)
            
            if z.shape[0] != X.shape[0]:
                raise ValueError(f"Embedding row mismatch: {arm} seed {seed}")
            if not np.isfinite(z).all():
                raise ValueError(f"Non-finite embedding: {arm} seed {seed}")
            
            # KMeans readouts
            predictions = []
            for state in range(10):
                km = KMeans(n_clusters=K, random_state=state, n_init=20, max_iter=300)
                km.fit(z)
                predictions.append(km.labels_.astype(np.int32))
            
            predictions = np.stack(predictions, axis=0)
            frozen_dict[f"seed_{seed}_{arm}_predictions"] = predictions
    
    # Save frozen readouts
    atomic_npz(out_dir / "frozen_readouts_10.npz", **frozen_dict)
    
    # Save layout stub
    atomic_npz(
        out_dir / "fixed_input_layout.npz",
        labels_true=y.astype(np.int32),
        class_names=np.array(names, dtype=str)
    )
    
    # Compute E3 analysis
    e3_results = compute_e3_analysis_data(y, names, frozen_dict)
    for kind,rows in e3_results.items():
        pd.DataFrame(rows).assign(dataset=dataset_name).to_csv(out_dir/f'e3_{kind}.csv',index=False)
    
    # Save record
    artifact_sha = {
        "frozen_readouts_10.npz": sha256(out_dir / "frozen_readouts_10.npz"),
        "fixed_input_layout.npz": sha256(out_dir / "fixed_input_layout.npz")
    }
    for kind in e3_results:
        name=f'e3_{kind}.csv';artifact_sha[name]=sha256(out_dir/name)
    
    record = {
        "status": "completed",
        "dataset": dataset_name,
        "n_samples": int(X.shape[0]),
        "n_clusters": K,
        "K_provenance": info["K_provenance"],
        "posthoc_full_x_labels_loaded": True,
        "labels_used_for_model_fit": False,
        "new_encoder_training": False,
        "protocol": "posthoc_e3_readout_analysis_only",
        "seeds": seeds,
        "readout_states": list(range(10)),
        "kmeans_n_init": 20,
        "alignment": alignment,
        "input_sha256": input_hashes,
        "artifact_sha256": artifact_sha,
        "e3_summary": e3_results["aggregate"]
    }
    
    from common import json_safe
    atomic_json(out_dir / "record.json", json_safe(record))
    
    print(f"[cohort_worker] {dataset_name} completed: {out_dir}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
