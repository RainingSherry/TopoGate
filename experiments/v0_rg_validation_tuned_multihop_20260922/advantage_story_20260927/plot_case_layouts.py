#!/usr/bin/env python3
"""
plot_case_layouts.py

Pure post-hoc visualization of fixed input layouts with cross-community 
co-assignment fractions. No model training or tuning.

Usage:
  python plot_case_layouts.py --out results/ [--community-root path/to/communities]
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import matplotlib.patches as mpatches

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent / "visual_argument_rebuild_20260926" / "case_assets"
DATASETS = ["Baron", "Quake_Smart-seq2_Lung", "student_grade"]


def load_case(dataset_name, community_root=None):
    """Load fixed layout, frozen readouts, and community assignments."""
    case_path = ASSETS / dataset_name
    
    # Load fixed input layout
    layout_data = np.load(case_path / "fixed_input_layout.npz")
    tsne2 = np.asarray(layout_data["tsne2"], dtype=np.float64)
    labels_true = np.asarray(layout_data["labels_true"], dtype=np.int32)
    class_names = layout_data["class_names"].astype(str)
    
    # Load frozen readouts
    readout_data = np.load(case_path / "frozen_readouts_10.npz")
    seeds = readout_data["seeds"]
    
    # Load community IDs
    community_path = case_path / "community_ids.npy"
    if not community_path.exists():
        if community_root is None:
            raise FileNotFoundError(f"No community_ids.npy in {case_path} and no --community-root provided")
        community_path = Path(community_root) / dataset_name / "community_ids.npy"
        if not community_path.exists():
            raise FileNotFoundError(f"Community IDs not found at {community_path}")
    
    communities = np.load(community_path).astype(np.int32)
    
    # Validate
    n = len(tsne2)
    assert tsne2.shape == (n, 2) and np.all(np.isfinite(tsne2))
    assert labels_true.shape == (n,) and len(class_names) > 0
    assert communities.shape == (n,)
    
    return {
        "tsne2": tsne2,
        "labels_true": labels_true,
        "class_names": class_names,
        "communities": communities,
        "readouts": readout_data,
        "seeds": seeds,
        "n": n
    }


def compute_cross_community_coassignment(labels_true, communities, predictions):
    """
    Compute per-anchor same-class CROSS-community co-assignment fraction.
    
    Returns: array[n] with values in [0,1] or NaN where denominator is zero.
    """
    _, y = np.unique(labels_true, return_inverse=True)
    _, c = np.unique(communities, return_inverse=True)
    _, z = np.unique(predictions, return_inverse=True)
    nc, nz = int(c.max()+1), int(z.max()+1)
    yc = y*nc+c
    yz = y*nz+z
    ycz = yc*nz+z
    den = np.bincount(y)[y]-np.bincount(yc)[yc]
    num = np.bincount(yz)[yz]-np.bincount(ycz)[ycz]
    out = np.full(len(y), np.nan)
    np.divide(num, den, out=out, where=den>0)
    return out


def aggregate_readouts(data):
    """Aggregate 10 readouts per seed, then 3 seeds for H1 and multihop."""
    seeds = data["seeds"]
    n = data["n"]
    
    h1_per_seed = []
    multi_per_seed = []
    
    for seed in seeds:
        h1_preds = []
        multi_preds = []
        for readout_idx in range(10):
            h1_key = f"seed_{seed}_H1_predictions"
            multi_key = f"seed_{seed}_multihop_predictions"
            
            h1_preds.append(data["readouts"][h1_key][readout_idx])
            multi_preds.append(data["readouts"][multi_key][readout_idx])
        
        # Average over 10 readouts
        h1_per_seed.append(np.array(h1_preds))
        multi_per_seed.append(np.array(multi_preds))
    
    h1_per_seed = np.array(h1_per_seed)  # [3, 10, n]
    multi_per_seed = np.array(multi_per_seed)
    
    return h1_per_seed, multi_per_seed


def compute_metrics_for_case(case_data):
    """Compute cross-community co-assignment for all readouts."""
    h1_per_seed, multi_per_seed = aggregate_readouts(case_data)
    
    n = case_data["n"]
    seeds = case_data["seeds"]
    
    # Storage: [3 seeds, 10 readouts, n]
    h1_scores = np.zeros((3, 10, n), dtype=np.float64)
    multi_scores = np.zeros((3, 10, n), dtype=np.float64)
    
    for seed_idx, seed in enumerate(seeds):
        for readout_idx in range(10):
            h1_pred = h1_per_seed[seed_idx, readout_idx].astype(np.int32)
            multi_pred = multi_per_seed[seed_idx, readout_idx].astype(np.int32)
            
            h1_scores[seed_idx, readout_idx] = compute_cross_community_coassignment(
                case_data["labels_true"], case_data["communities"], h1_pred
            )
            multi_scores[seed_idx, readout_idx] = compute_cross_community_coassignment(
                case_data["labels_true"], case_data["communities"], multi_pred
            )
    
    # Average over 10 readouts per seed
    h1_per_seed_avg = np.mean(h1_scores, axis=1)  # [3, n]
    multi_per_seed_avg = np.mean(multi_scores, axis=1)
    
    # Average over 3 seeds
    h1_final = np.mean(h1_per_seed_avg, axis=0)  # [n]
    multi_final = np.mean(multi_per_seed_avg, axis=0)
    delta = multi_final - h1_final
    
    return h1_final, multi_final, delta, h1_scores, multi_scores


def save_per_anchor_csv(out_path, dataset_name, case_data, h1_final, multi_final, delta):
    """Save per-anchor metrics to CSV."""
    n = case_data["n"]
    rows = []
    
    for i in range(n):
        # Denominator: same-class nodes outside community
        anchor_class = case_data["labels_true"][i]
        anchor_comm = case_data["communities"][i]
        
        same_class_mask = (case_data["labels_true"] == anchor_class)
        outside_comm_mask = (case_data["communities"] != anchor_comm)
        denom = np.sum(same_class_mask & outside_comm_mask)
        
        rows.append({
            "dataset": dataset_name,
            "anchor": i,
            "class_id": anchor_class,
            "community": anchor_comm,
            "H1": h1_final[i],
            "multihop": multi_final[i],
            "delta": delta[i],
            "denominator": denom
        })
    
    import csv
    csv_path = out_path / f"{dataset_name}_per_anchor.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "anchor", "class_id", "community", "H1", "multihop", "delta", "denominator"])
        writer.writeheader()
        writer.writerows(rows)


def plot_all_cases(cases_data, out_path):
    """Generate 3x4 figure with fixed layouts and cross-community scores."""
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    
    # Colormaps
    cmap_seq = LinearSegmentedColormap.from_list("seq", ["#F4F0EA", "#B5465D"])
    
    # Compute global delta range
    all_deltas = np.concatenate([cases_data[ds]["delta"] for ds in DATASETS])
    max_abs_delta = np.nanmax(np.abs(all_deltas[np.isfinite(all_deltas)]))
    cmap_div = LinearSegmentedColormap.from_list("div", ["#797A8E", "#F4F0EA", "#B5465D"])
    norm_div = TwoSlopeNorm(vmin=-max_abs_delta, vcenter=0, vmax=max_abs_delta)
    
    for row, dataset in enumerate(DATASETS):
        data = cases_data[dataset]
        tsne2 = data["tsne2"]
        labels = data["labels_true"]
        class_names = data["class_names"]
        h1 = data["h1_final"]
        multi = data["multi_final"]
        delta = data["delta"]
        
        # Col 0: Class-colored context
        ax = axes[row, 0]
        valid_mask = np.isfinite(h1)
        for cls_id in np.unique(labels):
            mask = (labels == cls_id)
            ax.scatter(tsne2[mask, 0], tsne2[mask, 1], color=plt.get_cmap("tab20")(int(cls_id) % 20), s=5, rasterized=True, alpha=0.7)
            centroid = tsne2[mask].mean(axis=0)
            ax.text(centroid[0], centroid[1], str(cls_id), fontsize=8, ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Col 1: H1
        ax = axes[row, 1]
        valid = np.isfinite(h1)
        ax.scatter(tsne2[~valid, 0], tsne2[~valid, 1], c="gray", s=5, rasterized=True, alpha=0.3)
        sc1 = ax.scatter(tsne2[valid, 0], tsne2[valid, 1], c=h1[valid], cmap=cmap_seq, vmin=0, vmax=1, s=5, rasterized=True)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Col 2: Multihop
        ax = axes[row, 2]
        valid = np.isfinite(multi)
        ax.scatter(tsne2[~valid, 0], tsne2[~valid, 1], c="gray", s=5, rasterized=True, alpha=0.3)
        sc2 = ax.scatter(tsne2[valid, 0], tsne2[valid, 1], c=multi[valid], cmap=cmap_seq, vmin=0, vmax=1, s=5, rasterized=True)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Col 3: Delta
        ax = axes[row, 3]
        valid = np.isfinite(delta)
        ax.scatter(tsne2[~valid, 0], tsne2[~valid, 1], c="gray", s=5, rasterized=True, alpha=0.3)
        sc3 = ax.scatter(tsne2[valid, 0], tsne2[valid, 1], c=delta[valid], cmap=cmap_div, norm=norm_div, s=5, rasterized=True)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Match row limits
        xlim = [tsne2[:, 0].min(), tsne2[:, 0].max()]
        ylim = [tsne2[:, 1].min(), tsne2[:, 1].max()]
        for col in range(4):
            axes[row, col].set_xlim(xlim)
            axes[row, col].set_ylim(ylim)
    
    # Column titles
    axes[0, 0].set_title("Class context", fontsize=10)
    axes[0, 1].set_title("H1 cross-comm", fontsize=10)
    axes[0, 2].set_title("Multihop cross-comm", fontsize=10)
    axes[0, 3].set_title("Delta (multi−H1)", fontsize=10)
    
    # Row labels
    for row, dataset in enumerate(DATASETS):
        axes[row, 0].set_ylabel({"Quake_Smart-seq2_Lung":"Lung", "student_grade":"student_grade"}.get(dataset,dataset), fontsize=10)
    
    # Colorbars
    cbar_ax1 = fig.add_axes([0.92, 0.35, 0.01, 0.3])
    fig.colorbar(sc1, cax=cbar_ax1, label="Co-assignment [0,1]")
    
    cbar_ax2 = fig.add_axes([0.92, 0.1, 0.01, 0.2])
    fig.colorbar(sc3, cax=cbar_ax2, label="Delta")
    
    fig.subplots_adjust(left=0.06, right=0.88, bottom=0.04, top=0.95, wspace=0.12, hspace=0.15)
    
    fig.savefig(out_path / "case_layouts.pdf", dpi=300)
    fig.savefig(out_path / "case_layouts.svg")
    fig.savefig(out_path / "case_layouts.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot case layouts with cross-community co-assignment")
    parser.add_argument("--out", required=True, type=Path, help="Output directory")
    parser.add_argument("--community-root", type=Path, default=None, help="Fallback root for community_ids.npy")
    args = parser.parse_args()
    
    args.out.mkdir(parents=True, exist_ok=True)
    
    cases_data = {}
    for dataset in DATASETS:
        print(f"Loading {dataset}...")
        case_data = load_case(dataset, args.community_root)
        h1_final, multi_final, delta, h1_scores, multi_scores = compute_metrics_for_case(case_data)
        
        cases_data[dataset] = {
            **case_data,
            "h1_final": h1_final,
            "multi_final": multi_final,
            "delta": delta,
            "h1_scores": h1_scores,
            "multi_scores": multi_scores
        }
        
        save_per_anchor_csv(args.out, dataset, case_data, h1_final, multi_final, delta)
        
        # Save per-seed npz
        np.savez(
            args.out / f"{dataset}_per_seed.npz",
            h1_scores=h1_scores,
            multi_scores=multi_scores,
            seeds=case_data["seeds"]
        )
    
    print("Generating figure...")
    plot_all_cases(cases_data, args.out)
    
    # Metadata
    metadata = {
        "input_sha256": {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for ds in DATASETS for f in (ASSETS/ds).glob("*") if f.name in ["fixed_input_layout.npz","frozen_readouts_10.npz","community_ids.npy"]},
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "labels_used_for_model_fit": False,
        "posthoc_full_x_labels_loaded": True,
        "date": "2026-09-27",
        "datasets": DATASETS,
        "class_names": {ds: cases_data[ds]["class_names"].tolist() for ds in DATASETS},
        "layout_interpretation": "Fixed input-space tSNE coordinates, not learned embedding movement; color is per-anchor cross-community same-class coassignment averaged within readout then seed.",
        "note": "Fixed input layout, full multi-seed average, gray=no eligible pairs, post-hoc only"
    }
    with open(args.out / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Done. Outputs in {args.out}")


if __name__ == "__main__":
    main()
