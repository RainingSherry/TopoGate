#!/usr/bin/env python3
"""
cohort_depth.py - Posthoc read-only summary of matched-depth cohort experiments.

Reads existing depth experiment records, matches against locked selected configs,
and generates heatmaps showing ARI performance deltas from H1 baseline across depths.
No model fitting, no selection, no validation/test scores used.
"""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


def compute_sha256(path: Path) -> str:
    """Compute SHA256 hash of file."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def load_config_class(base_path: Path):
    """Import V0_RGConfig from config.py."""
    import importlib.util
    config_py = base_path / "v0rg_tunable_multihop" / "config.py"
    spec = importlib.util.spec_from_file_location("v0rg_config", config_py)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.V0_RGConfig


def materialize_defaults(config_dict: Dict[str, Any], V0_RGConfig) -> Dict[str, Any]:
    """Materialize all defaults via dataclasses.asdict."""
    instance = V0_RGConfig(**config_dict)
    return asdict(instance)


def extract_depth_from_hop_variant(hop_variant: str) -> Optional[int]:
    """Extract depth from hop_variant string (e.g., 'H1' -> 1, 'H2' -> 2)."""
    if not hop_variant or not hop_variant.startswith('H'):
        return None
    try:
        return int(hop_variant.split('_')[0][1:])
    except (ValueError, IndexError):
        return None


def extract_bandwidth_suffix(hop_variant: str) -> Optional[str]:
    """Extract bandwidth suffix from hop_variant (e.g., 'H2_B200' -> 'B200')."""
    if '_B' in hop_variant:
        return hop_variant.split('_B', 1)[1]
    return None


def configs_match_except_hop(config1: Dict, config2: Dict, locked_hop_variant: str) -> bool:
    """Check if configs match except protocol_id and hop_variant.
    
    H1 always allowed. Other hop variants must have same B width suffix as locked.
    """
    c1 = config1.copy()
    c2 = config2.copy()
    if config1.get('hop_variant') == 'H1':
        # No higher-order shell exists in H1; q is inactive in its sampler.
        c1.pop('high_hop_probability_multiplier', None)
        c2.pop('high_hop_probability_multiplier', None)
    
    # Remove excluded keys
    for key in ['protocol_id', 'hop_variant']:
        c1.pop(key, None)
        c2.pop(key, None)
    
    if c1 != c2:
        return False
    
    # Check hop_variant bandwidth suffix constraint
    candidate_hop = config1.get('hop_variant', '')
    if candidate_hop == 'H1':
        return True  # H1 always allowed
    
    locked_suffix = extract_bandwidth_suffix(locked_hop_variant)
    candidate_suffix = extract_bandwidth_suffix(candidate_hop)
    
    return locked_suffix == candidate_suffix


def main():
    parser = argparse.ArgumentParser(description='Generate cohort depth analysis')
    parser.add_argument('--out', required=True, help='Output directory')
    args = parser.parse_args()
    
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Define paths
    base_path = Path("/data/luolie/ToPoGate/experiments/v0_rg_validation_tuned_multihop_20260922")
    here_path = base_path / "advantage_story_20260927"
    src_path = base_path / "source_coverage_20260925" / "outputs" / "p0"
    
    selection_lock_path = src_path / "selection_lock_v2.json"
    audit_path = src_path / "record_audit.csv"
    readout_path = Path("/tmp/claude_readout_all/readout_all.csv")
    cohort_lock_sha_path = here_path / "COHORT_LOCK_SHA256.txt"
    
    # Load config class
    V0_RGConfig = load_config_class(base_path)
    
    # Compute and save input hashes
    input_hashes = {
        'selection_lock_v2.json': compute_sha256(selection_lock_path),
        'record_audit.csv': compute_sha256(audit_path),
        'readout_all.csv': compute_sha256(readout_path)
    }
    
    # Verify lock SHA if exists
    if not cohort_lock_sha_path.exists(): raise FileNotFoundError(cohort_lock_sha_path)
    if cohort_lock_sha_path.exists():
        expected_sha = cohort_lock_sha_path.read_text().strip()
        if input_hashes['selection_lock_v2.json'] != expected_sha:
            print(f"ERROR: selection_lock SHA mismatch. Expected {expected_sha}, got {input_hashes['selection_lock_v2.json']}")
            sys.exit(1)
    
    with open(out_path / "input_hashes.json", 'w') as f:
        json.dump(input_hashes, f, indent=2)
    
    # Load data
    with open(selection_lock_path) as f:
        lock_data = json.load(f)
    
    audit_df = pd.read_csv(audit_path)
    readout_df = pd.read_csv(readout_path, usecols=['dataset', 'stage', 'config_id', 'seed', 'random_state', 'ari_full'])
    
    # Process datasets
    datasets = lock_data['datasets']
    matched_datasets = {k: v for k, v in datasets.items() 
                       if v.get('h1_counterpart_of_selected_multihop') is not None}
    
    print(f"Found {len(matched_datasets)} matched datasets")
    
    exclusions = []
    per_seed_records = []
    summary_records = []
    
    KNOWN_SEEDS = [7, 42, 123]
    KNOWN_STATES = list(range(10))
    
    for dataset_name in sorted(matched_datasets.keys()):
        dataset_info = matched_datasets[dataset_name]
        locked_config = dataset_info['selected_multihop']['config']
        locked_preprocessing = dataset_info['selected_multihop']['preprocessing_hash']
        locked_feature_sha = dataset_info.get('feature_sha256')
        locked_k = dataset_info.get('n_clusters')
        locked_hop_variant = locked_config.get('hop_variant', '')
        
        # Get eligible audit rows for this dataset
        dataset_audit = audit_df[audit_df['dataset'] == dataset_name].copy()
        
        # Group by depth
        depth_data = defaultdict(lambda: defaultdict(list))
        
        for _, row in dataset_audit.iterrows():
            if str(row['eligible']).lower() != 'true':
                continue
            
            record_path = Path(row['record_path'])
            if not record_path.exists():
                exclusions.append({
                    'dataset': dataset_name,
                    'record_path': str(record_path),
                    'reason': 'record_not_found'
                })
                continue
            
            # Verify SHA
            computed_sha = compute_sha256(record_path)
            if computed_sha != row['record_sha256']:
                exclusions.append({
                    'dataset': dataset_name,
                    'record_path': str(record_path),
                    'reason': 'sha256_mismatch'
                })
                continue
            
            # Load record
            with open(record_path) as f:
                record = json.load(f)
            
            if record.get('status') != 'completed':
                exclusions.append({
                    'dataset': dataset_name,
                    'record_path': str(record_path),
                    'reason': 'not_completed'
                })
                continue
            
            # Check feature hash and preprocessing
            if record.get('feature_sha256') != locked_feature_sha:
                exclusions.append({
                    'dataset': dataset_name,
                    'record_path': str(record_path),
                    'reason': 'feature_sha_mismatch'
                })
                continue
            
            if record.get('preprocessing_hash') != locked_preprocessing:
                exclusions.append({
                    'dataset': dataset_name,
                    'record_path': str(record_path),
                    'reason': 'preprocessing_hash_mismatch'
                })
                continue
            
            # Check K
            record_config = record.get('config', {})
            if record.get('n_clusters') != locked_k:
                exclusions.append({
                    'dataset': dataset_name,
                    'record_path': str(record_path),
                    'reason': 'k_mismatch'
                })
                continue
            
            # Materialize defaults
            materialized_record_config = materialize_defaults(record_config, V0_RGConfig)
            materialized_locked_config = materialize_defaults(locked_config, V0_RGConfig)
            
            # Check config match
            if not configs_match_except_hop(materialized_record_config, materialized_locked_config, locked_hop_variant):
                exclusions.append({
                    'dataset': dataset_name,
                    'record_path': str(record_path),
                    'reason': 'config_mismatch'
                })
                continue
            
            # Extract depth
            hop_variant = record_config.get('hop_variant', '')
            depth = extract_depth_from_hop_variant(hop_variant)
            if depth is None:
                exclusions.append({
                    'dataset': dataset_name,
                    'record_path': str(record_path),
                    'reason': 'invalid_hop_variant'
                })
                continue
            
            seed = record.get('seed')
            if seed not in KNOWN_SEEDS:
                continue
            
            # Add to depth data
            key = (row['stage'], row['config_id'], row['attempt'], str(record_path))
            depth_data[depth][seed].append(key)
        
        # Process each depth
        for depth in sorted(depth_data.keys()):
            seed_results = {}
            
            for seed in KNOWN_SEEDS:
                if seed not in depth_data[depth]:
                    continue
                
                candidates = sorted(depth_data[depth][seed])
                if len(candidates) == 0:
                    continue
                
                # First complete candidate in a deterministic metadata order.
                state_values = None
                for stage, config_id, attempt, record_path_str in candidates:
                    rr = readout_df[(readout_df['dataset']==dataset_name)&
                        (readout_df['stage']==stage)&(readout_df['config_id']==config_id)&
                        (readout_df['seed']==seed)]
                    values={}; valid=True
                    for state in KNOWN_STATES:
                        v=rr.loc[rr['random_state']==state,'ari_full'].unique()
                        if len(v)!=1 or not np.isfinite(v[0]): valid=False;break
                        values[state]=float(v[0])
                    if valid:
                        state_values=values
                        break
                    exclusions.append(dict(dataset=dataset_name,depth=depth,seed=seed,
                        record_path=record_path_str,reason='readout_states_incomplete_or_conflict'))
                if state_values is None: continue
                # Compute mean across 10 states
                mean_ari = np.mean(list(state_values.values()))
                seed_results[seed] = mean_ari
            
            # Check if we have all 3 seeds
            if len(seed_results) < 3:
                per_seed_records.append({
                    'dataset': dataset_name,
                    'depth': depth,
                    'n_seeds': len(seed_results),
                    **{f'seed_{s}_ari': seed_results.get(s, np.nan) for s in KNOWN_SEEDS}
                })
                continue
            
            # Compute summary statistics
            seed_values = [seed_results[s] for s in KNOWN_SEEDS]
            mean_ari = np.mean(seed_values)
            sd_ari = np.std(seed_values, ddof=1)
            
            per_seed_records.append({
                'dataset': dataset_name,
                'depth': depth,
                'n_seeds': 3,
                **{f'seed_{s}_ari': seed_results[s] for s in KNOWN_SEEDS}
            })
            
            summary_records.append({
                'dataset': dataset_name,
                'depth': depth,
                'mean_ari': mean_ari,
                'sd_ari': sd_ari,
                'n_seeds': 3
            })
    
    # Save paired seed differences, not pooled readout standard errors.
    h1={r['dataset']:r for r in per_seed_records if r['depth']==1}
    for r in per_seed_records:
        ref=h1.get(r['dataset'],{})
        for seed in KNOWN_SEEDS:
            r[f'seed_{seed}_delta']=r.get(f'seed_{seed}_ari',np.nan)-ref.get(f'seed_{seed}_ari',np.nan)
    for r in summary_records:
        p=next(x for x in per_seed_records if x['dataset']==r['dataset'] and x['depth']==r['depth'])
        ds=[p[f'seed_{seed}_delta'] for seed in KNOWN_SEEDS]
        r['delta_mean']=float(np.mean(ds));r['delta_seed_sd']=float(np.std(ds,ddof=1))
    if not summary_records: raise RuntimeError('No complete depth groups; inspect exclusions')
    # Save CSV outputs
    pd.DataFrame(per_seed_records).to_csv(out_path / "per_seed.csv", index=False)
    pd.DataFrame(summary_records).to_csv(out_path / "summary.csv", index=False)
    pd.DataFrame(exclusions).to_csv(out_path / "exclusions.csv", index=False)
    
    # Generate heatmap
    summary_df = pd.DataFrame(summary_records)
    
    # Pivot for heatmap (datasets x depths), computing delta from H1
    all_datasets = sorted(matched_datasets.keys())
    depths_present = sorted(summary_df['depth'].unique()) if len(summary_df) > 0 else []
    depths_for_plot = [d for d in depths_present if d != 1]  # Exclude H1
    
    heatmap_data = np.full((len(all_datasets), len(depths_for_plot)), np.nan)
    
    for i, dataset in enumerate(all_datasets):
        dataset_summary = summary_df[summary_df['dataset'] == dataset]
        
        # Get H1 baseline
        h1_rows = dataset_summary[dataset_summary['depth'] == 1]
        if len(h1_rows) == 0:
            continue
        h1_mean = h1_rows['mean_ari'].iloc[0]
        
        # Compute deltas for other depths
        for j, depth in enumerate(depths_for_plot):
            depth_rows = dataset_summary[dataset_summary['depth'] == depth]
            if len(depth_rows) > 0:
                depth_mean = depth_rows['mean_ari'].iloc[0]
                heatmap_data[i, j] = depth_mean - h1_mean
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(max(8, len(depths_for_plot) * 0.8), max(10, len(all_datasets) * 0.3)))
    
    # Determine symmetric bounds for colormap
    max_abs = np.nanmax(np.abs(heatmap_data)) if not np.all(np.isnan(heatmap_data)) else 0.1
    vmin, vmax = -max_abs, max_abs
    
    # Create colormap: rose negative, gray positive, midpoint beige
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#797A8E', '#F4F0EA', '#B5465D']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
    
    im = ax.imshow(heatmap_data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    
    # Add hatching for missing values
    for i in range(len(all_datasets)):
        for j in range(len(depths_for_plot)):
            if np.isnan(heatmap_data[i, j]):
                ax.add_patch(Rectangle((j-0.5, i-0.5), 1, 1, 
                                      fill=True, facecolor='white', 
                                      edgecolor='gray', hatch='///', linewidth=0))
    
    # Labels
    ax.set_xticks(range(len(depths_for_plot)))
    ax.set_xticklabels([f'H{d}' for d in depths_for_plot])
    ax.set_yticks(range(len(all_datasets)))
    ax.set_yticklabels(all_datasets, fontsize=6)
    ax.set_xlabel('Depth')
    ax.set_ylabel('Dataset')
    ax.set_title('ARI change at fixed active non-depth parameters; hatched = missing')
    
    plt.colorbar(im, ax=ax, label='ΔARI (depth minus H1)')
    plt.tight_layout()
    
    # Save figures
    plt.savefig(out_path / "cohort_depth_heatmap.png", dpi=300, bbox_inches='tight')
    plt.savefig(out_path / "cohort_depth_heatmap.pdf", bbox_inches='tight')
    plt.savefig(out_path / "cohort_depth_heatmap.svg", bbox_inches='tight')
    plt.close()
    
    # Save record.json
    record_json = {
        'status': 'completed',
        'scope': 'posthoc_existing_fullXreadouts_no_new_selection',
        'labels_used_for_model_fit': False,
        'new_training': False,
        'n_matched_datasets': len(matched_datasets),
        'n_exclusions': len(exclusions),
        'h1_equivalence': 'ignore inactive high_hop_probability_multiplier only for H1; all active parameters equal',
        'artifact_sha256': {
            'per_seed.csv': compute_sha256(out_path / "per_seed.csv"),
            'summary.csv': compute_sha256(out_path / "summary.csv"),
            'exclusions.csv': compute_sha256(out_path / "exclusions.csv"),
            'input_hashes.json': compute_sha256(out_path / "input_hashes.json")
        }
    }
    
    for ext in ('png','pdf','svg'):
        name=f'cohort_depth_heatmap.{ext}'; record_json['artifact_sha256'][name]=compute_sha256(out_path/name)
    record_json['posthoc_full_x_readouts_loaded']=True
    tmp=out_path/'record.tmp';tmp.write_text(json.dumps(record_json,indent=2));tmp.replace(out_path/'record.json')
    
    print(f"Completed. Processed {len(matched_datasets)} datasets, {len(exclusions)} exclusions.")
    print(f"Outputs saved to {out_path}")


if __name__ == '__main__':
    main()
