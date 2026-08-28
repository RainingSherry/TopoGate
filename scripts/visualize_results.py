"""Generate visualization plots for TopoGate vs CLUBench benchmark.

1. Critical-difference diagram (Demšar 2006)
2. Bucket-wise win rate bar chart
3. Per-dataset scatter plot (TopoGate vs best(CLUBench))
4. Heatmap of 29 algorithms × 13 datasets

Saves PNGs to /home/luolie/ToPoGate/papers/tab_figs/
"""
import csv
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = Path('/home/luolie/ToPoGate/papers/tab_figs')
CSV_PATH = OUTPUT_DIR / 'merged_comparison.csv'

DATASET_META = {
    'iris': ('tabular', 'small', 'low', 'multi'),
    'har': ('image', 'small', 'medium', 'multi'),
    'spambase': ('tabular', 'medium', 'low', 'binary'),
    'breast_cancer_wisconsin_original': ('tabular', 'small', 'low', 'binary'),
    'mammographic_mass': ('tabular', 'small', 'low', 'binary'),
    'sms_spam_collection': ('text', 'small', 'medium', 'binary'),
    'enron': ('text', 'large', 'high', 'binary'),
    'ISOLET': ('tabular', 'medium', 'medium', 'multi'),
    'reuters': ('text', 'medium', 'high', 'multi'),
    'cnae9': ('text', 'small', 'medium', 'multi'),
    'Campbell': ('bioinfo', 'large', 'high', 'multi'),
    'first-order-theorem-proving': ('tabular', 'medium', 'low', 'multi'),
    'Mouse_retina': ('bioinfo', 'medium', 'high', 'multi'),
}

DATASET_ORDER = list(DATASET_META.keys())


def main():
    data = defaultdict(lambda: defaultdict(dict))
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[row['algorithm']][row['dataset']] = {
                'acc': float(row['ACC']),
                'nmi': float(row['NMI']),
                'ari': float(row['ARI']),
            }

    algorithms = sorted(data.keys())
    topogate = 'TopoGate'

    # ---- 1. Bucket-wise win rate bar chart ----
    # Build a list of (algorithm, dataset) pairs
    pairs = []
    for ds in DATASET_ORDER:
        for algo in algorithms:
            if algo == topogate or ds not in data[algo]:
                continue
            pairs.append((algo, ds))

    # Compute bucket assignment for each pair
    bucket_types = ['modality', 'size', 'dim', 'K']
    bucket_values = {
        'modality': ['tabular', 'text', 'bioinfo', 'image'],
        'size': ['small', 'medium', 'large'],
        'dim': ['low', 'medium', 'high'],
        'K': ['binary', 'multi'],
    }

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, btype in zip(axes, bucket_types):
        wins = []
        labels = []
        for bval in bucket_values[btype]:
            cnt = 0
            total = 0
            for algo, ds in pairs:
                if DATASET_META[ds][bucket_types.index(btype)] != bval:
                    continue
                total += 1
                if data[topogate][ds]['nmi'] > data[algo][ds]['nmi']:
                    cnt += 1
            wins.append(100 * cnt / total if total > 0 else 0)
            labels.append(f'{bval}\n(N={total})')
        bars = ax.bar(labels, wins, color=['steelblue', 'seagreen', 'coral', 'gold'][:len(labels)])
        ax.set_ylabel('NMI Win Rate (%)')
        ax.set_title(f'By {btype}')
        ax.set_ylim(0, 100)
        ax.axhline(50, color='red', linestyle='--', alpha=0.5, label='50%')
        for bar, v in zip(bars, wins):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1, f'{v:.0f}%',
                    ha='center', va='bottom', fontsize=10)
        ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'bucket_win_rates.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Wrote {OUTPUT_DIR / "bucket_win_rates.png"}')

    # ---- 2. Per-dataset scatter: TopoGate vs best(CLUBench) ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric in zip(axes, ['acc', 'nmi', 'ari']):
        tg_vals = []
        best_vals = []
        ds_labels = []
        for ds in DATASET_ORDER:
            tg_vals.append(data[topogate][ds][metric])
            best = max((data[algo][ds][metric] for algo in algorithms
                        if algo != topogate and ds in data[algo]), default=0)
            best_vals.append(best)
            ds_labels.append(ds)
        ax.scatter(best_vals, tg_vals, s=80, alpha=0.7, edgecolors='k')
        for i, ds in enumerate(ds_labels):
            ax.annotate(ds[:4], (best_vals[i], tg_vals[i]),
                       fontsize=8, xytext=(3, 3), textcoords='offset points')
        lim = [0, 1]
        ax.plot(lim, lim, 'r--', alpha=0.5, label='y=x')
        ax.set_xlabel(f'Best of CLUBench ({metric.upper()})')
        ax.set_ylabel(f'TopoGate ({metric.upper()})')
        ax.set_title(f'{metric.upper()}: TopoGate vs Best(CLUBench)')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        # Count points above diagonal
        n_above = sum(1 for t, b in zip(tg_vals, best_vals) if t > b)
        ax.text(0.05, 0.95, f'TopoGate ≥ Best: {n_above}/{len(tg_vals)}',
                transform=ax.transAxes, fontsize=11, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'topogate_vs_best.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Wrote {OUTPUT_DIR / "topogate_vs_best.png"}')

    # ---- 3. Heatmap: 29 algorithms × 13 datasets for NMI ----
    # Sort algorithms by avg NMI rank
    avg_ranks = {}
    for algo in algorithms:
        ranks = []
        for ds in DATASET_ORDER:
            sorted_algos = sorted(
                algorithms,
                key=lambda a: data[a][ds]['nmi'] if a in data and ds in data[a] else -1,
                reverse=True,
            )
            try:
                ranks.append(sorted_algos.index(algo) + 1)
            except ValueError:
                pass
        avg_ranks[algo] = np.mean(ranks) if ranks else 99
    sorted_algos = sorted(algorithms, key=lambda a: avg_ranks[a])

    # Build matrix
    matrix = np.zeros((len(sorted_algos), len(DATASET_ORDER)))
    for i, algo in enumerate(sorted_algos):
        for j, ds in enumerate(DATASET_ORDER):
            matrix[i, j] = data[algo][ds]['nmi']

    fig, ax = plt.subplots(figsize=(13, 10))
    im = ax.imshow(matrix, cmap='viridis', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(DATASET_ORDER)))
    ax.set_xticklabels(DATASET_ORDER, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(sorted_algos)))
    ax.set_yticklabels(sorted_algos, fontsize=8)
    # Highlight TopoGate row
    topogate_idx = sorted_algos.index(topogate)
    ax.axhline(topogate_idx - 0.5, color='red', linewidth=2)
    ax.axhline(topogate_idx + 0.5, color='red', linewidth=2)
    # Annotate cells
    for i in range(len(sorted_algos)):
        for j in range(len(DATASET_ORDER)):
            color = 'white' if matrix[i, j] < 0.5 else 'black'
            ax.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center',
                    fontsize=7, color=color)
    plt.colorbar(im, ax=ax, label='NMI')
    ax.set_title(f'NMI Heatmap (29 algorithms × 13 datasets). TopoGate row highlighted.')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'nmi_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Wrote {OUTPUT_DIR / "nmi_heatmap.png"}')

    # ---- 4. Critical-difference-like bar chart ----
    # Compute average rank for each algorithm on NMI
    metrics = ['acc', 'nmi', 'ari']
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, metric in zip(axes, metrics):
        ranks = {}
        for algo in algorithms:
            all_ranks = []
            for ds in DATASET_ORDER:
                sorted_algos = sorted(
                    algorithms,
                    key=lambda a: data[a][ds][metric] if a in data and ds in data[a] else -1,
                    reverse=True,
                )
                try:
                    rank = sorted_algos.index(algo) + 1
                except ValueError:
                    rank = len(sorted_algos)
                all_ranks.append(rank)
            ranks[algo] = np.mean(all_ranks)
        sorted_algos = sorted(ranks.keys(), key=lambda a: ranks[a])
        avg_ranks = [ranks[a] for a in sorted_algos]
        # Color TopoGate in red
        colors = ['red' if a == topogate else 'steelblue' for a in sorted_algos]
        ax.barh(range(len(sorted_algos)), avg_ranks, color=colors)
        ax.set_yticks(range(len(sorted_algos)))
        ax.set_yticklabels(sorted_algos, fontsize=8)
        ax.set_xlabel(f'Avg Rank ({metric.upper()}, lower=better)')
        ax.set_title(f'Avg Rank — {metric.upper()}')
        ax.axvline(14.5, color='gray', linestyle='--', alpha=0.5, label='Median (14.5)')
        ax.invert_yaxis()
        ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'avg_rank_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Wrote {OUTPUT_DIR / "avg_rank_comparison.png"}')

    print('All plots generated.')


if __name__ == '__main__':
    main()
