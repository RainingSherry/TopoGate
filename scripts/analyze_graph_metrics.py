"""
Compute graph metrics for large/high-dimensional datasets that were skipped.
Uses very aggressive subsampling and simpler computations.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import normalized_mutual_info_score
from scipy import sparse
import os
import warnings
import gc
warnings.filterwarnings('ignore')

DATA_DIR = "/data/luolie/ToPoGate/datasets"
RESULT_DIR = "/home/luolie/ToPoGate/result/ablation"

# Datasets to analyze
SKIP_GRAPH = {'Campbell', 'Quake_Smart-seq2_Lung', 'hrvatin_filtered', 'enron', 
              'first-order-theorem-proving', 'Mouse_retina', 'reuters'}

# Also process datasets we already have for completeness
ALL_DS = {
    'iris': {'n': 150, 'd': 4, 'K': 3},
    'har': {'n': 735, 'd': 561, 'K': 6},
    'spambase': {'n': 4601, 'd': 57, 'K': 2},
    'cnae9': {'n': 1080, 'd': 856, 'K': 9},
    'breast_cancer_wisconsin_original': {'n': 683, 'd': 9, 'K': 2},
    'sms_spam_collection': {'n': 835, 'd': 500, 'K': 2},
    'mammographic_mass': {'n': 830, 'd': 5, 'K': 2},
    'enron': {'n': 9999, 'd': 4096, 'K': 2},
    'ISOLET': {'n': 7797, 'd': 617, 'K': 26},
    'Quake_Smart-seq2_Lung': {'n': 1676, 'd': 23341, 'K': 11},
    'Campbell': {'n': 9993, 'd': 26774, 'K': 14},
    'Mouse_retina': {'n': 8352, 'd': 6198, 'K': 5},
    'reuters': {'n': 6576, 'd': 4096, 'K': 3},
    'first-order-theorem-proving': {'n': 6118, 'd': 51, 'K': 6},
    'hrvatin_filtered': {'n': 48266, 'd': 25187, 'K': 8},
}

# Load existing metrics
df_existing = pd.read_csv(f"{RESULT_DIR}/dataset_characteristics.csv")

def compute_graph_metrics_aggressive(name, info):
    """Aggressive subsampling for large datasets."""
    path = os.path.join(DATA_DIR, f"{name}.npz")
    data = np.load(path)
    X = data['x'].astype(np.float32)
    y = data['y'].astype(np.int32)
    n, d = X.shape
    
    k = 5
    n_sub = 1500  # fixed subsample
    
    np.random.seed(42)
    idx = np.random.choice(n, min(n_sub, n), replace=False)
    X_sub = X[idx]
    y_sub = y[idx]
    
    del X, y
    gc.collect()
    
    # Normalize
    mean = X_sub.mean(axis=0)
    std = X_sub.std(axis=0) + 1e-8
    X_norm = (X_sub - mean) / std
    
    # For very high-dim data, first reduce with SVD
    if d > 500:
        n_svd = min(100, d, len(X_sub))
        svd = TruncatedSVD(n_components=n_svd, random_state=42)
        svd.fit(X_norm)
        X_for_knn = svd.transform(X_norm)
        del svd
    else:
        X_for_knn = X_norm
    
    del X_sub, X_norm, mean, std
    gc.collect()
    
    # KNN
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree', n_jobs=1).fit(X_for_knn)
    _, indices = nbrs.kneighbors(X_for_knn)
    indices = indices[:, 1:]
    
    del nbrs, X_for_knn
    gc.collect()
    
    # Same-label ratio
    n_s = len(y_sub)
    same = np.array([np.sum(y_sub[indices[i]] == y_sub[i]) / k for i in range(n_s)], dtype=np.float32)
    m = {
        'dataset': name,
        'knn_same_ratio': float(same.mean()),
        'knn_random_ratio': float(np.sum(np.bincount(y_sub, minlength=info['K']) / n_s ** 2)),
    }
    m['knn_excess'] = float((same.mean() - m['knn_random_ratio']) / (1 - m['knn_random_ratio'])) if m['knn_random_ratio'] < 1 else 0.0
    
    # Graph metrics
    rows = np.repeat(np.arange(n_s), k)
    cols = indices.flatten()
    adj = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_s, n_s))
    adj = adj + adj.T
    adj.data[:] = 1
    
    degrees = np.array(adj.sum(axis=1)).flatten()
    m['graph_avg_deg'] = float(degrees.mean())
    m['graph_density'] = float(adj.nnz / 2 / (n_s * (n_s - 1) / 2))
    m['graph_degree_std'] = float(degrees.std())
    
    del adj, degrees, indices, rows, cols, same
    gc.collect()
    
    return m

print("Computing graph metrics for all datasets...\n")

all_results = {}
for name in ALL_DS:
    print(f"Processing {name}...", end=" ", flush=True)
    try:
        r = compute_graph_metrics_aggressive(name, ALL_DS[name])
        all_results[name] = r
        print(f"knn_same={r['knn_same_ratio']:.3f}, knn_excess={r['knn_excess']:.3f}, "
              f"deg={r['graph_avg_deg']:.1f}, density={r['graph_density']:.6f}")
    except Exception as e:
        print(f"FAILED: {e}")
        all_results[name] = {'dataset': name, 'knn_same_ratio': np.nan, 
                           'knn_excess': np.nan, 'graph_avg_deg': np.nan,
                           'graph_density': np.nan, 'graph_degree_std': np.nan}

# Update existing metrics
for name, r in all_results.items():
    mask = df_existing['dataset'] == name
    if mask.any():
        for col in ['knn_same_ratio', 'knn_random_ratio', 'knn_excess', 
                    'graph_avg_deg', 'graph_density', 'graph_degree_std']:
            if col in r:
                df_existing.loc[mask, col] = r[col]

df_existing.to_csv(f"{RESULT_DIR}/dataset_characteristics_full.csv", index=False)

# Final correlation analysis
print("\n" + "=" * 80)
print("FINAL CORRELATION ANALYSIS (all datasets, updated)")
print("=" * 80)

# Merge with delta_ari
df_analysis = df_existing.merge(
    df_existing[['dataset']].assign(effect=df_existing['delta_ari'].apply(
        lambda x: 'HELPS' if x > 0.01 else 'HURTS' if x < -0.01 else 'NEUTRAL'
    )), on='dataset' if 'effect' in df_existing.columns else None
)

# Properly merge
df_analysis = df_existing[['dataset', 'n', 'd', 'd_over_n', 'K', 'balance_ratio', 
                            'cluster_cv', 'pc1_var', 'pc1_pc2_ratio', 'pc_95_n',
                            'knn_same_ratio', 'knn_random_ratio', 'knn_excess',
                            'graph_avg_deg', 'graph_density', 'delta_ari']].copy()

df_analysis['effect'] = df_analysis['delta_ari'].apply(
    lambda x: 'HELPS' if x > 0.01 else 'HURTS' if x < -0.01 else 'NEUTRAL'
)

# Separate groups
helps = df_analysis[df_analysis['effect'] == 'HELPS']
hurts = df_analysis[df_analysis['effect'] == 'HURTS']

print("\n--- Full dataset metrics ---")
key_cols = ['dataset', 'K', 'balance_ratio', 'pc1_pc2_ratio', 
            'knn_same_ratio', 'knn_excess', 'graph_avg_deg', 'graph_density', 'delta_ari']
print(df_analysis[key_cols].sort_values('delta_ari', ascending=False).to_string(index=False))

print("\n--- Group comparison ---")
print(f"\n{'Metric':<25} {'HELPS':>12} {'HURTS':>12} {'Diff':>10} {'r_with_delta':>12}")
print("-" * 75)

metric_cols = ['K', 'balance_ratio', 'cluster_cv', 'pc1_var', 'pc1_pc2_ratio',
               'knn_same_ratio', 'knn_excess', 'graph_avg_deg', 'graph_density']

for col in metric_cols:
    if col in helps.columns and col in hurts.columns:
        hv = helps[col].dropna()
        uv = hurts[col].dropna()
        if len(hv) > 0 and len(uv) > 0:
            # Correlation with delta_ari across all datasets
            valid = df_analysis[col].notna()
            r = df_analysis.loc[valid, col].corr(df_analysis.loc[valid, 'delta_ari']) if valid.sum() >= 3 else 0.0
            sig = "***" if abs(r) > 0.5 else "**" if abs(r) > 0.4 else "*" if abs(r) > 0.3 else ""
            print(f"{col:<25} {hv.mean():>12.4f} {uv.mean():>12.4f} {hv.mean()-uv.mean():>+10.4f} {r:>+12.3f} {sig}")

print("\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)
