#!/usr/bin/env python
# coding: utf-8
"""
E3/E4 Pair-error and Community Coassignment Analysis CLI

Purpose:
  Bounded deliverable for E3 (pair-error structure map) and E4 (community 
  coassignment cross-fragment integration). Posthoc analysis of frozen readouts 
  using no new training. Implements split/merge rate calculations, community-pair 
  coassignment tracking, and permutation-invariant aggregation.

Rationale:
  - Split rate = same-class different-pred unordered pairs / all same-class pairs
  - Merge rate = different-class same-pred pairs / all different-class pairs
  - Macro: class-wise average to avoid domination by large classes
  - No NxN allocation: iterate pairs or use sparse structures
  - Community coassignment: per same-class community pair, track H1/multihop 
    average coassignment across readouts then seeds
  - Metadata: SHA256 of inputs, posthoc_full_x_labels_loaded=true, 
    labels_used_for_model_fit=false

Inputs (per case):
  - fixed_input_layout.npz: keys 'labels_true', 'class_names', 'tsne2' (layout)
  - community_ids.npy: shape (n,), integer community assignments
  - frozen_readouts_10.npz: keys 'seeds' (array of seed values), 
    'seed_{s}_{arm}_predictions' where s in seeds, arm in ['H1', 'multihop'],
    shape (10, n) for 10 KMeans readouts

Outputs (to papers/advantage_story_20260927/outputs/{case}/):
  - e3_pair_errors_per_readout.csv: arm, seed, readout_idx, split_rate, merge_rate
  - e3_pair_errors_per_seed.csv: arm, seed, split_rate_mean, merge_rate_mean
  - e3_pair_errors_aggregate.csv: arm, split_rate_mean, split_rate_std, merge_rate_mean, merge_rate_std
  - e3_class_macro_stats.csv: arm, seed, class_id, class_name, n_samples, split_rate, merge_rate
  - e4_community_pair_coassignment.csv: class_id, comm_a, comm_b, support, H1_coassign_mean, multihop_coassign_mean, change
  - e4_community_pair_selected_display.csv: subset for plotting by size/support rules
  - metadata.json: input SHA256, run timestamp, posthoc/labels flags
  - e3_structure_map.png: scatter plot of (H1_split - multihop_split, H1_merge - multihop_merge)
  - e4_three_case_community_change.png: community relation change for selected pairs

Usage:
  python compute_e3_e4_analysis.py --case Baron
  python compute_e3_e4_analysis.py --case Quake_Smart-seq2_Lung --output-dir custom_out
  python compute_e3_e4_analysis.py --all-cases
"""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics import adjusted_rand_score

def checked_predictions(frozen, seed, arm, n):
    key = f'seed_{seed}_{arm}_predictions'
    if key not in frozen:
        raise ValueError(f'Missing paired readouts: {key}')
    p = frozen[key]
    if p.shape != (10, n) or not np.issubdtype(p.dtype, np.integer):
        raise ValueError(f'Invalid readout array {key}: {p.shape}/{p.dtype}')
    return p


# ============================================================================
#  Utility: SHA256 hash of file
# ============================================================================
def compute_file_sha256(path: Path) -> str:
    """Compute SHA256 hex digest of file at path."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


# ============================================================================
#  Core: Split and Merge Rate Computation
# ============================================================================
def compute_split_merge_rates_sparse(labels_true: np.ndarray, predictions: np.ndarray):
    """
    Compute split and merge rates without allocating NxN arrays.
    
    Split rate = (same-class different-pred pairs) / (all same-class pairs)
    Merge rate = (different-class same-pred pairs) / (all different-class pairs)
    
    Returns:
      split_rate: float
      merge_rate: float
      per_class_split: dict[int, float] mapping class_id to its split rate
      per_class_merge: dict[int, float] mapping class_id to its merge rate
    """
    labels_true = np.asarray(labels_true)
    predictions = np.asarray(predictions)
    if labels_true.ndim != 1 or predictions.shape != labels_true.shape:
        raise ValueError('Labels and predictions must be aligned one-dimensional vectors')
    _, predictions = np.unique(predictions, return_inverse=True)
    n = len(labels_true)
    if n == 0:
        return np.nan, np.nan, {}, {}
    
    k = int(labels_true.max()) + 1
    nc = int(predictions.max()) + 1
    
    # Contingency table: tab[c, p] = count of (true_class=c, pred_cluster=p)
    tab = np.zeros((k, nc), dtype=np.int64)
    for c, p in zip(labels_true, predictions):
        tab[c, p] += 1
    
    # Class sizes and cluster sizes
    class_counts = tab.sum(axis=1)  # shape (k,)
    cluster_counts = tab.sum(axis=0)  # shape (nc,)
    
    # Total same-class pairs and correct-cluster pairs per class
    same_class_pairs = (class_counts * (class_counts - 1)) / 2
    # Within each class c and cluster p: tab[c,p]*(tab[c,p]-1)/2 correct pairs
    correct_pairs_per_class = ((tab * (tab - 1)) / 2).sum(axis=1)  # shape (k,)
    split_pairs_per_class = same_class_pairs - correct_pairs_per_class
    
    # Per-class split rates (handle singletons: denominator zero)
    per_class_split = {}
    for c in range(k):
        denom = same_class_pairs[c]
        per_class_split[c] = (split_pairs_per_class[c] / denom) if denom > 0 else np.nan
    
    # Total different-class pairs: all pairs - sum of same-class pairs
    total_pairs = (n * (n - 1)) / 2
    different_class_pairs = total_pairs - same_class_pairs.sum()
    
    # Correct total pairs (same pred, same true): sum over all (c, p) of tab[c,p]*(tab[c,p]-1)/2
    correct_total = ((tab * (tab - 1)) / 2).sum()
    
    # Merge pairs (same pred, different true): within each cluster p, total pairs - correct
    # cluster_counts[p]*(cluster_counts[p]-1)/2 - sum_c tab[c,p]*(tab[c,p]-1)/2
    cluster_total_pairs = (cluster_counts * (cluster_counts - 1)) / 2
    # For each cluster, subtract the correct-class pairs
    correct_pairs_per_cluster = ((tab * (tab - 1)) / 2).sum(axis=0)  # shape (nc,)
    merge_pairs_total = (cluster_total_pairs - correct_pairs_per_cluster).sum()
    
    # Per-class merge rate: for class c, how many times class-c samples are merged with other classes
    # We compute: for class c, sum over all clusters p the number of (c, other_class) pairs in cluster p
    # = for each p: tab[c, p] * (cluster_counts[p] - tab[c, p])
    per_class_merge = {}
    for c in range(k):
        # How many samples of class c are in each cluster
        c_in_cluster = tab[c, :]  # shape (nc,)
        # Other samples in same cluster
        others_in_cluster = cluster_counts - c_in_cluster
        # Merge pairs involving class c in cluster p: c_in_cluster[p] * others_in_cluster[p]
        merge_pairs_c = (c_in_cluster * others_in_cluster).sum()
        # Denominator: all different-class pairs involving class c
        # = class_counts[c] * (n - class_counts[c])
        denom = class_counts[c] * (n - class_counts[c])
        per_class_merge[c] = (merge_pairs_c / denom) if denom > 0 else np.nan
    
    # Global rates
    split_rate = split_pairs_per_class.sum() / same_class_pairs.sum() if same_class_pairs.sum() > 0 else np.nan
    merge_rate = merge_pairs_total / different_class_pairs if different_class_pairs > 0 else np.nan
    
    return split_rate, merge_rate, per_class_split, per_class_merge


# ============================================================================
#  E3: Pair-error Analysis
# ============================================================================
def compute_e3_analysis(case_name: str, layout_path: Path, frozen_path: Path, output_dir: Path):
    """
    Compute E3 pair-error structure map for one case.
    
    Outputs:
      - e3_pair_errors_per_readout.csv
      - e3_pair_errors_per_seed.csv
      - e3_pair_errors_aggregate.csv
      - e3_class_macro_stats.csv
      - e3_structure_map.png
    """
    print(f"[E3] Processing {case_name}")
    
    # Load inputs
    layout = np.load(layout_path)
    labels_true = layout['labels_true']
    class_names = layout['class_names'] if 'class_names' in layout else [str(i) for i in range(int(labels_true.max())+1)]
    n = len(labels_true)
    k = int(labels_true.max()) + 1
    
    frozen = np.load(frozen_path)
    seeds = frozen['seeds']
    
    # Per-readout records
    per_readout_records = []
    # Per-seed records
    per_seed_records = []
    # Class macro stats
    class_macro_records = []
    
    # Aggregate: collect H1 and multihop split/merge across all readouts
    aggregate = {'H1': {'split': [], 'merge': []}, 'multihop': {'split': [], 'merge': []}}
    
    for seed in seeds:
        for arm in ['H1', 'multihop']:
            key = f'seed_{seed}_{arm}_predictions'
            preds = checked_predictions(frozen, seed, arm, n)
            
            num_readouts = preds.shape[0]
            
            # Per-readout statistics
            readout_splits = []
            readout_merges = []
            # Accumulate class stats across readouts for this seed
            class_stats_accum = defaultdict(lambda: {'split': [], 'merge': []})
            
            for r_idx in range(num_readouts):
                pred = preds[r_idx]
                split_r, merge_r, per_class_split_r, per_class_merge_r = compute_split_merge_rates_sparse(labels_true, pred)
                readout_splits.append(split_r)
                readout_merges.append(merge_r)
                
                per_readout_records.append({
                    'case': case_name,
                    'arm': arm,
                    'seed': int(seed),
                    'readout_idx': r_idx,
                    'split_rate': split_r,
                    'merge_rate': merge_r,
                    'ARI': adjusted_rand_score(labels_true, pred),
                    'macro_split_rate': np.nanmean(list(per_class_split_r.values())),
                    'macro_merge_rate': np.nanmean(list(per_class_merge_r.values()))
                })
                
                # Accumulate per-class
                for c in range(k):
                    class_stats_accum[c]['split'].append(per_class_split_r.get(c, 0.0))
                    class_stats_accum[c]['merge'].append(per_class_merge_r.get(c, 0.0))
            
            # Per-seed aggregation: mean over readouts
            split_mean = np.mean(readout_splits)
            merge_mean = np.mean(readout_merges)
            per_seed_records.append({
                'case': case_name,
                'arm': arm,
                'seed': int(seed),
                'split_rate_mean': split_mean,
                'merge_rate_mean': merge_mean
            })
            
            # Store for aggregate
            aggregate[arm]['split'].append(split_mean)
            aggregate[arm]['merge'].append(merge_mean)
            
            # Per-class macro: average over readouts
            for c in range(k):
                class_split_mean = np.mean(class_stats_accum[c]['split'])
                class_merge_mean = np.mean(class_stats_accum[c]['merge'])
                class_name = class_names[c] if c < len(class_names) else str(c)
                n_c = int((labels_true == c).sum())
                class_macro_records.append({
                    'case': case_name,
                    'arm': arm,
                    'seed': int(seed),
                    'class_id': c,
                    'class_name': class_name,
                    'n_samples': n_c,
                    'split_rate': class_split_mean,
                    'merge_rate': class_merge_mean
                })
    
    # Aggregate statistics
    aggregate_records = []
    for arm in ['H1', 'multihop']:
        aggregate_records.append({
            'case': case_name,
            'arm': arm,
            'split_rate_mean': np.mean(aggregate[arm]['split']) if aggregate[arm]['split'] else 0.0,
            'split_rate_std': np.std(aggregate[arm]['split'], ddof=1) if len(aggregate[arm]['split']) > 1 else np.nan,
            'merge_rate_mean': np.mean(aggregate[arm]['merge']) if aggregate[arm]['merge'] else 0.0,
            'merge_rate_std': np.std(aggregate[arm]['merge'], ddof=1) if len(aggregate[arm]['merge']) > 1 else np.nan,
            'uncertainty_unit': 'training_seed_mean_of_10_readouts'
        })
    
    # Write CSVs
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_readout_records).to_csv(output_dir / 'e3_pair_errors_per_readout.csv', index=False)
    pd.DataFrame(per_seed_records).to_csv(output_dir / 'e3_pair_errors_per_seed.csv', index=False)
    pd.DataFrame(aggregate_records).to_csv(output_dir / 'e3_pair_errors_aggregate.csv', index=False)
    pd.DataFrame(class_macro_records).to_csv(output_dir / 'e3_class_macro_stats.csv', index=False)
    
    # Plot E3 structure map: (H1_split - multihop_split, H1_merge - multihop_merge)
    # Aggregate by seed to get one point per seed
    per_seed_df = pd.DataFrame(per_seed_records)
    h1_df = per_seed_df[per_seed_df.arm == 'H1'].set_index('seed')
    mh_df = per_seed_df[per_seed_df.arm == 'multihop'].set_index('seed')
    
    if not h1_df.empty and not mh_df.empty:
        delta_split = h1_df['split_rate_mean'] - mh_df['split_rate_mean']
        delta_merge = h1_df['merge_rate_mean'] - mh_df['merge_rate_mean']
        
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(100*delta_split, 100*delta_merge, s=60, color='#B5465D', alpha=0.7, edgecolors='k', linewidths=0.5)
        ax.axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
        ax.axvline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
        ax.set_xlabel('H1 split rate − multihop split rate (pp)', fontsize=10)
        ax.set_ylabel('H1 merge rate − multihop merge rate (pp)', fontsize=10)
        ax.set_title(f'{case_name} E3 Structure Map\n(positive = multihop reduces error)', fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        for ext in ('png','pdf','svg'):
            fig.savefig(output_dir / f'e3_structure_map.{ext}', dpi=300, facecolor='white')
        plt.close(fig)
    
    print(f"[E3] {case_name} complete: {output_dir}")


# ============================================================================
#  E4: Community Coassignment Analysis
# ============================================================================
def compute_community_coassignment(labels_true, community_ids, predictions):
    """Exact contingency products, including within-community unordered pairs."""
    y=np.asarray(labels_true); g=np.asarray(community_ids); p=np.asarray(predictions)
    if y.shape != g.shape or y.shape != p.shape or y.ndim != 1:
        raise ValueError('Unaligned community inputs')
    _,p=np.unique(p,return_inverse=True); k=int(p.max())+1 if len(p) else 0
    out={}
    for c in np.unique(y):
        comms=np.unique(g[y==c])
        counts={int(a):np.bincount(p[(y==c)&(g==a)],minlength=k) for a in comms}
        for i,a in enumerate(comms):
            ca=counts[int(a)]; na=int(ca.sum())
            for b in comms[i:]:
                cb=counts[int(b)]; nb=int(cb.sum())
                den=na*(na-1)//2 if a==b else na*nb
                if den:
                    num=int((ca*(ca-1)).sum()//2) if a==b else int(ca@cb)
                    out[(int(c),int(a),int(b))]=num/den
    return out


def compute_e4_analysis(case_name, layout_path, frozen_path, community_path, output_dir):
    layout=np.load(layout_path); y=layout['labels_true']; g=np.load(community_path)
    frozen=np.load(frozen_path); seeds=frozen['seeds']
    if g.shape!=y.shape: raise ValueError('Unaligned community IDs')
    names=layout['class_names'] if 'class_names' in layout else np.unique(y).astype(str)
    rows=[]
    for seed in seeds:
        for arm in ('H1','multihop'):
            sums=defaultdict(list)
            for pred in checked_predictions(frozen,seed,arm,len(y)):
                for key,val in compute_community_coassignment(y,g,pred).items(): sums[key].append(val)
            for (c,a,b),vals in sums.items():
                na=int(((y==c)&(g==a)).sum()); nb=int(((y==c)&(g==b)).sum())
                rows.append(dict(case=case_name,seed=int(seed),arm=arm,class_id=c,
                    class_name=str(names[c]),comm_a=a,comm_b=b,n_a=na,n_b=nb,
                    pair_support=na*(na-1)//2 if a==b else na*nb,
                    readout_count=len(vals),coassignment=float(np.mean(vals))))
    output_dir.mkdir(parents=True,exist_ok=True)
    per=pd.DataFrame(rows); per.to_csv(output_dir/'e4_community_pair_per_seed.csv',index=False)
    records=[]
    if len(per):
        for (c,a,b),grp in per.groupby(['class_id','comm_a','comm_b']):
            h=grp[grp.arm=='H1'].set_index('seed'); m=grp[grp.arm=='multihop'].set_index('seed')
            if set(h.index)!=set(seeds) or set(m.index)!=set(seeds):raise ValueError('Incomplete community pair')
            delta=m.coassignment-h.coassignment; r=h.iloc[0]
            records.append(dict(case=case_name,class_id=int(c),class_name=r.class_name,
                comm_a=int(a),comm_b=int(b),n_a=int(r.n_a),n_b=int(r.n_b),support=int(r.pair_support),
                seed_count=len(seeds),readouts_per_seed=10,
                H1_coassign_mean=float(h.coassignment.mean()),multihop_coassign_mean=float(m.coassignment.mean()),
                change=float(delta.mean()),change_seed_sd=float(delta.std(ddof=1)),
                positive_seeds=int((delta>0).sum())))
    df=pd.DataFrame(records)
    df.to_csv(output_dir/'e4_community_pair_coassignment.csv',index=False)
    if df.empty:
        df.to_csv(output_dir/'e4_community_pair_selected_display.csv',index=False)
        return
    # Largest 18 nonempty class/community groups, selected solely by size.
    groups=sorted([(int(((y==c)&(g==a)).sum()),int(c),int(a))
                  for c in np.unique(y) for a in np.unique(g[y==c])],key=lambda t:(-t[0],t[1],t[2]))[:18]
    groups=sorted(groups,key=lambda t:(t[1],-t[0],t[2])); lookup={(c,a):i for i,(n,c,a) in enumerate(groups)}
    selected=df.apply(lambda r:(int(r.class_id),int(r.comm_a)) in lookup and
                      (int(r.class_id),int(r.comm_b)) in lookup,axis=1)
    df[selected].to_csv(output_dir/'e4_community_pair_selected_display.csv',index=False)
    from matplotlib.colors import LinearSegmentedColormap
    cm=LinearSegmentedColormap.from_list('change',['#797A8E','#F4F0EA','#B5465D']); cm.set_bad('#DADADA')
    seq=LinearSegmentedColormap.from_list('coassignment',['#F4F0EA','#E29DAB','#B5465D']);seq.set_bad('#DADADA')
    fig,axes=plt.subplots(1,3,figsize=(12,4.8))
    labels=[f'{c}:C{a} (n={n})' for n,c,a in groups]
    for ax,col,title in zip(axes,['H1_coassign_mean','multihop_coassign_mean','change'],['H1','Multi-hop','Multi-hop minus H1']):
        mat=np.full((len(groups),len(groups)),np.nan)
        for _,r in df[selected].iterrows():
            i=lookup[int(r.class_id),int(r.comm_a)]; j=lookup[int(r.class_id),int(r.comm_b)]
            mat[i,j]=mat[j,i]=r[col]
        delta=col=='change'; lim=max(.01,float(np.nanmax(np.abs(mat)))) if delta else 1
        im=ax.imshow(100*mat,cmap=cm if delta else seq,vmin=-100*lim if delta else 0,vmax=100*lim)
        ax.set_title(title);ax.set_xticks(range(len(groups)));ax.set_yticks(range(len(groups)))
        ax.set_xticklabels(labels,rotation=90,fontsize=6);ax.set_yticklabels(labels,fontsize=6)
        fig.colorbar(im,ax=ax,shrink=.7,label='Change (pp)' if delta else 'Coassignment (%)')
    fig.suptitle(case_name+' | size-selected class/community groups; gray = not a same-class pair')
    fig.tight_layout()
    for ext in ('png','pdf','svg'):fig.savefig(output_dir/f'e4_community_matrix.{ext}',dpi=300)
    plt.close(fig)


# ============================================================================
#  Metadata Generation
# ============================================================================
def generate_metadata(case_name: str, layout_path: Path, frozen_path: Path, community_path: Path, output_dir: Path):
    """Generate metadata.json with input SHA256 and flags."""
    metadata = {
        'case': case_name,
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'posthoc_full_x_labels_loaded': True,
        'labels_used_for_model_fit': False,
        'input_sha256': {
            'fixed_input_layout.npz': compute_file_sha256(layout_path),
            'frozen_readouts_10.npz': compute_file_sha256(frozen_path),
            'community_ids.npy': compute_file_sha256(community_path) if community_path.exists() else None
        },
        'note': 'Posthoc analysis of frozen readouts; no encoder training or selection performed in this script.'
    }
    
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)


# ============================================================================
#  Main CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='E3/E4 Pair-error and Community Coassignment Analysis CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--case', type=str, help='Case name (e.g., Baron, Quake_Smart-seq2_Lung, student_grade)')
    parser.add_argument('--all-cases', action='store_true', help='Process all three main cases')
    parser.add_argument('--input-dir', type=str, default='papers/visual_argument_rebuild_20260926/case_assets',
                        help='Directory containing case_assets subdirectories')
    parser.add_argument('--output-dir', type=str, default='papers/advantage_story_20260927/outputs',
                        help='Root output directory')
    parser.add_argument('--skip-e3', action='store_true', help='Skip E3 analysis')
    parser.add_argument('--skip-e4', action='store_true', help='Skip E4 analysis')
    
    args = parser.parse_args()
    
    if args.all_cases:
        cases = ['Baron', 'Quake_Smart-seq2_Lung', 'student_grade']
    elif args.case:
        cases = [args.case]
    else:
        parser.error('Must specify --case or --all-cases')
    
    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)
    
    failures=[]
    for case_name in cases:
        case_dir = input_root / case_name
        layout_path = case_dir / 'fixed_input_layout.npz'
        frozen_path = case_dir / 'frozen_readouts_10.npz'
        community_path = case_dir / 'community_ids.npy'
        
        if not layout_path.exists():
            failures.append(f"Missing {layout_path}")
            continue
        if not frozen_path.exists():
            failures.append(f"Missing {frozen_path}")
            continue
        
        case_output_dir = output_root / case_name
        
        manifest=json.loads((case_dir/'asset_manifest.json').read_text(encoding='utf-8'))
        for path in (layout_path,frozen_path,community_path):
            if not path.exists() or compute_file_sha256(path)!=manifest['files_sha256'][path.name]:
                raise ValueError(f'Frozen asset hash mismatch: {path}')
        before_errors=len(failures)
        
        # E3 analysis
        if not args.skip_e3:
            try:
                compute_e3_analysis(case_name, layout_path, frozen_path, case_output_dir)
            except Exception as e:
                failures.append(f"E3 {case_name}: {e}")
        
        # E4 analysis
        if not args.skip_e4 and community_path.exists():
            try:
                compute_e4_analysis(case_name, layout_path, frozen_path, community_path, case_output_dir)
            except Exception as e:
                failures.append(f"E4 {case_name}: {e}")
        elif not args.skip_e4:
            failures.append(f"Missing {community_path}")
        if len(failures)==before_errors:
            generate_metadata(case_name, layout_path, frozen_path, community_path, case_output_dir)
            (case_output_dir/'CASE_COMPLETED.json').write_text(json.dumps({
                'status':'completed', 'code_sha256':compute_file_sha256(Path(__file__)),
                'manifest_sha256':compute_file_sha256(case_dir/'asset_manifest.json'),
                'stages':[s for s,skip in [('E3',args.skip_e3),('E4',args.skip_e4)] if not skip],
                'posthoc_full_x_labels_loaded':True,'labels_used_for_model_fit':False},indent=2))
    
    if failures:
        raise RuntimeError("; ".join(failures))
    print("\nAll requested processing complete.")


if __name__ == '__main__':
    main()

