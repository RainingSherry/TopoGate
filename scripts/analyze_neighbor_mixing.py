"""
Fast analysis of dataset characteristics for neighbor mixing.
Skip large/high-dim datasets for graph metrics, compute fast approximations.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score
from scipy import sparse
import os
import warnings
import gc
warnings.filterwarnings('ignore')

DATA_DIR = "/data/luolie/ToPoGate/datasets"
RESULT_DIR = "/home/luolie/ToPoGate/result/ablation"

# Load ablation
df = pd.read_csv(f"{RESULT_DIR}/merged_summary.csv")
df_full = df[df['variant'] == 'static_gate_full'].copy()
df_nomix = df[df['variant'] == 'static_gate_nomix'].copy()

merged = pd.merge(
    df_full[['dataset', 'ari', 'n_samples', 'n_features', 'n_clusters']],
    df_nomix[['dataset', 'ari']],
    on='dataset',
    suffixes=('_full', '_nomix')
)
merged['delta_ari'] = merged['ari_full'] - merged['ari_nomix']
merged = merged.sort_values('delta_ari')

# Add a flag for helps/hurts
merged['effect'] = merged['delta_ari'].apply(
    lambda x: 'HELPS' if x > 0.01 else 'HURTS' if x < -0.01 else 'NEUTRAL'
)

print("=" * 80)
print("ABLATION RESULTS")
print("=" * 80)
for _, row in merged.iterrows():
    print(f"  {row['dataset']:<40} Δ={row['delta_ari']:+.4f} [{row['effect']}]")

helps_ds = merged[merged['delta_ari'] > 0.01]['dataset'].tolist()
hurts_ds = merged[merged['delta_ari'] < -0.01]['dataset'].tolist()
neutral_ds = merged[merged['delta_ari'].abs() <= 0.01]['dataset'].tolist()

print(f"\nHELPS: {helps_ds}")
print(f"HURTS: {hurts_ds}")
print(f"NEUTRAL: {neutral_ds}")

# Skip large datasets for graph analysis
SKIP_GRAPH = {'Campbell', 'Quake_Smart-seq2_Lung', 'hrvatin_filtered', 'enron', 
              'first-order-theorem-proving', 'Mouse_retina', 'reuters'}

def compute_fast(name):
    """Fast computation with LITE mode for large datasets."""
    path = os.path.join(DATA_DIR, f"{name}.npz")
    if not os.path.exists(path):
        return None
    
    data = np.load(path)
    X = data['x'].astype(np.float32)
    y = data['y'].astype(np.int32)
    del data
    
    n, d = X.shape
    m = {
        'dataset': name, 'n': n, 'd': d, 'd_over_n': d/n
    }
    
    # === Basic cluster info ===
    unique, counts = np.unique(y, return_counts=True)
    m['K'] = len(unique)
    m['min_cluster'] = int(counts.min())
    m['max_cluster'] = int(counts.max())
    m['balance_ratio'] = counts.min() / counts.max()
    m['cluster_cv'] = float(counts.std() / counts.mean())
    m['cluster_entropy_norm'] = float(-np.sum((counts/counts.sum()) * np.log(counts/counts.sum())) / np.log(m['K']))
    
    # === PCA via SVD (faster than full PCA) ===
    try:
        X_std = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
        
        # Use SVD for dimensionality estimation
        n_comp = min(100, n, d)
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        svd.fit(X_std)
        
        var = svd.explained_variance_ratio_
        cumvar = np.cumsum(var)
        
        m['pc1_var'] = float(var[0])
        m['pc2_var'] = float(var[1]) if len(var) > 1 else 0.0
        m['pc1_pc2_ratio'] = float(var[0] / var[1]) if len(var) > 1 else 0.0
        m['pc_95_n'] = int(np.searchsorted(cumvar, 0.95)) + 1
        m['pc_90_n'] = int(np.searchsorted(cumvar, 0.90)) + 1
        m['cumvar_20'] = float(cumvar[19]) if len(cumvar) > 19 else float(cumvar[-1])
        m['svd_entropy'] = float(-np.sum(var * np.log(var + 1e-10)))
        
        del X_std, svd, var, cumvar
        gc.collect()
    except Exception as e:
        m['pc1_var'] = np.nan
        m['pc2_var'] = np.nan
        m['pc1_pc2_ratio'] = np.nan
        m['pc_95_n'] = np.nan
        m['pc_90_n'] = np.nan
    
    # === Graph metrics (skip for large datasets) ===
    if name in SKIP_GRAPH:
        m['knn_same_ratio'] = np.nan
        m['knn_random_ratio'] = np.nan
        m['knn_excess'] = np.nan
        m['knn_comp_nmi'] = np.nan
        m['graph_avg_deg'] = np.nan
        m['graph_density'] = np.nan
    else:
        try:
            k = 5
            
            # Subsample if n > 2000
            if n > 2000:
                np.random.seed(42)
                idx = np.random.choice(n, 2000, replace=False)
                X_sub = X[idx]
                y_sub = y[idx]
            else:
                X_sub = X
                y_sub = y
            
            # Normalize
            mean = X_sub.mean(axis=0)
            std = X_sub.std(axis=0) + 1e-8
            X_norm = (X_sub - mean) / std
            
            del X_sub, mean, std
            gc.collect()
            
            # KNN
            nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree', n_jobs=1).fit(X_norm)
            _, indices = nbrs.kneighbors(X_norm)
            indices = indices[:, 1:]  # remove self
            
            # Same-label ratio
            n_sub = len(y_sub)
            same_counts = np.array([np.sum(y_sub[indices[i]] == y_sub[i]) / k for i in range(n_sub)], dtype=np.float32)
            m['knn_same_ratio'] = float(same_counts.mean())
            
            # Random baseline
            uniq, cnt = np.unique(y_sub, return_counts=True)
            m['knn_random_ratio'] = float(np.sum((cnt / n_sub) ** 2))
            m['knn_excess'] = float((same_counts.mean() - m['knn_random_ratio']) / (1 - m['knn_random_ratio'])) if m['knn_random_ratio'] < 1 else 0.0
            
            # Build sparse KNN graph
            rows = np.repeat(np.arange(n_sub), k)
            cols = indices.flatten()
            
            adj = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_sub, n_sub))
            adj = adj + adj.T
            adj.data[:] = 1
            
            degrees = np.array(adj.sum(axis=1)).flatten()
            m['graph_avg_deg'] = float(degrees.mean())
            m['graph_density'] = float(adj.nnz / 2 / (n_sub * (n_sub - 1) / 2))
            
            # NMI with KNN components
            n_comp, comp_lab = sparse.csgraph.connected_components(adj, directed=False)
            m['knn_comp_nmi'] = float(normalized_mutual_info_score(y_sub, comp_lab))
            
            del adj, degrees, nbrs, X_norm, indices, rows, cols
            gc.collect()
        except Exception as e:
            m['knn_same_ratio'] = np.nan
            m['knn_random_ratio'] = np.nan
            m['knn_excess'] = np.nan
            m['knn_comp_nmi'] = np.nan
            m['graph_avg_deg'] = np.nan
            m['graph_density'] = np.nan
    
    del X, y
    gc.collect()
    
    return m

# Process all
results = []
for name in merged['dataset'].tolist():
    print(f"\nProcessing {name}...", end=" ", flush=True)
    r = compute_fast(name)
    if r:
        results.append(r)
        print(f"n={r['n']}, d={r['d']}, K={r['K']}, ΔARI={merged[merged['dataset']==name]['delta_ari'].values[0]:+.4f}")
        if not np.isnan(r.get('knn_same_ratio', np.nan)):
            print(f"  knn_same={r['knn_same_ratio']:.3f}, knn_excess={r['knn_excess']:.3f}, knn_nmi={r['knn_comp_nmi']:.3f}")
    else:
        print(f"FAILED")

df_metrics = pd.DataFrame(results)
df_metrics = df_metrics.merge(merged[['dataset', 'ari_full', 'ari_nomix', 'delta_ari', 'effect']], on='dataset')

# Save
df_metrics.to_csv(f"{RESULT_DIR}/dataset_characteristics.csv", index=False)
print(f"\nSaved to {RESULT_DIR}/dataset_characteristics.csv")

# === ANALYSIS ===
print("\n" + "=" * 80)
print("METRICS TABLE (sorted by ΔARI)")
print("=" * 80)

cols = ['dataset', 'n', 'd', 'K', 'balance_ratio', 'pc1_var', 
        'knn_same_ratio', 'knn_excess', 'knn_comp_nmi', 'delta_ari', 'effect']
print(df_metrics[cols].sort_values('delta_ari', ascending=False).to_string(index=False))

# === GROUP COMPARISON ===
helps_df = df_metrics[df_metrics['effect'] == 'HELPS']
hurts_df = df_metrics[df_metrics['effect'] == 'HURTS']

print("\n" + "=" * 80)
print("GROUP STATISTICS")
print("=" * 80)

stat_cols = ['n', 'd', 'd_over_n', 'K', 'balance_ratio', 'cluster_cv',
             'pc1_var', 'pc1_pc2_ratio', 'pc_95_n', 'pc_90_n',
             'knn_same_ratio', 'knn_random_ratio', 'knn_excess', 'knn_comp_nmi',
             'graph_avg_deg', 'graph_density']

print(f"\n{'Metric':<25} {'HELPS':>12} {'HURTS':>12} {'Diff':>10}")
print("-" * 60)
for col in stat_cols:
    if col in helps_df.columns and col in hurts_df.columns:
        h_vals = helps_df[col].dropna()
        hu_vals = hurts_df[col].dropna()
        if len(h_vals) > 0 and len(hu_vals) > 0:
            print(f"{col:<25} {h_vals.mean():>12.4f} {hu_vals.mean():>12.4f} {h_vals.mean()-hu_vals.mean():>+10.4f}")

# === CORRELATION ===
print("\n" + "=" * 80)
print("CORRELATIONS WITH ΔARI")
print("=" * 80)

corr_cols = ['n', 'd', 'd_over_n', 'K', 'balance_ratio', 'cluster_cv',
             'pc1_var', 'pc1_pc2_ratio', 'pc_95_n',
             'knn_same_ratio', 'knn_excess', 'knn_comp_nmi',
             'graph_avg_deg', 'graph_density']

for col in corr_cols:
    if col in df_metrics.columns:
        valid = df_metrics[col].notna()
        if valid.sum() >= 3:
            r = df_metrics.loc[valid, col].corr(df_metrics.loc[valid, 'delta_ari'])
            sig = "***" if abs(r) > 0.5 else "**" if abs(r) > 0.4 else "*" if abs(r) > 0.3 else ""
            print(f"  {col:<25}: r = {r:+.4f} {sig}")

# === HYPOTHESIS TESTS ===
print("\n" + "=" * 80)
print("HYPOTHESIS ANALYSIS")
print("=" * 80)

print("\n[H1] Neighbor mixing helps when KNN graph aligns with cluster structure")
h_knn = helps_df['knn_same_ratio'].dropna()
hu_knn = hurts_df['knn_same_ratio'].dropna()
print(f"    knn_same_ratio: HELPS={h_knn.mean():.3f}(n={len(h_knn)})  HURTS={hu_knn.mean():.3f}(n={len(hu_knn)})")

print("\n[H2] Neighbor mixing hurts when KNN crosses cluster boundaries")
h_exc = helps_df['knn_excess'].dropna()
hu_exc = hurts_df['knn_excess'].dropna()
print(f"    knn_excess:     HELPS={h_exc.mean():.3f}(n={len(h_exc)})  HURTS={hu_exc.mean():.3f}(n={len(hu_exc)})")

print("\n[H3] Number of clusters K")
print(f"    HELPS: {helps_df['K'].tolist()}")
print(f"    HURTS: {hurts_df['K'].tolist()}")

print("\n[H4] Cluster balance")
print(f"    HELPS: mean={helps_df['balance_ratio'].mean():.3f}, HURTS: mean={hurts_df['balance_ratio'].mean():.3f}")

print("\n[H5] Dimensionality ratio d/n")
print(f"    HELPS: mean={helps_df['d_over_n'].mean():.2f}, HURTS: mean={hurts_df['d_over_n'].mean():.2f}")

print("\n[H6] PCA structure (signal concentration)")
print(f"    HELPS pc1_var: mean={helps_df['pc1_var'].mean():.3f}, HURTS: mean={hurts_df['pc1_var'].mean():.3f}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

# What are the most distinguishing features?
print("\nKey distinguishing characteristics:")
print("=" * 60)

for col in corr_cols:
    if col in df_metrics.columns:
        valid = df_metrics[col].notna()
        if valid.sum() >= 3:
            r = df_metrics.loc[valid, col].corr(df_metrics.loc[valid, 'delta_ari'])
            if abs(r) > 0.3:
                direction = "POSITIVE" if r > 0 else "NEGATIVE"
                interpretation = ""
                if col == 'knn_same_ratio':
                    interpretation = "→ High same-label neighbors: mixing HELPS"
                elif col == 'knn_excess':
                    interpretation = "→ Strong graph-cluster alignment: mixing HELPS"
                elif col == 'balance_ratio':
                    interpretation = "→ Balanced clusters benefit from mixing"
                elif col == 'K':
                    interpretation = "→ More clusters: mixing may help less"
                elif col == 'd_over_n':
                    interpretation = "→ High-dim (d>>n): mixing may hurt"
                print(f"  {col}: r={r:+.3f} [{direction}] {interpretation}")
