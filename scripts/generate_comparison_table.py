"""Generate final comparison table (Table 1: general deep clustering SOTA).

Reads per-model CSVs from /home/luolie/ToPoGate/result/baseline_comparison/
and produces a wide table: rows = datasets, columns = (model, metric).
"""
import csv
from pathlib import Path
from collections import defaultdict

RESULT_DIR = Path('/home/luolie/ToPoGate/result/baseline_comparison')
OUTPUT_DIR = Path('/home/luolie/ToPoGate/papers/tab_figs')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ['GCEALS', 'IDC', 'TableDC', 'ZEUS', 'TopoGate']
METRICS = ['ACC', 'NMI', 'ARI']

# Order of datasets (paper: small → large, with PCA flag annotated)
DATASETS_ORDER = [
    'iris', 'har', 'spambase', 'breast_cancer_wisconsin_original',
    'mammographic_mass', 'sms_spam_collection', 'enron', 'ISOLET',
    'reuters', 'cnae9', 'Campbell', 'first-order-theorem-proving',
    'Mouse_retina', 'Quake_Smart-seq2_Lung', 'hrvatin_filtered',
]

# Read all data
all_rows = defaultdict(dict)  # (dataset, model) -> {metric -> value}
dataset_meta = {}  # dataset -> {n_clusters, n_samples, n_features}

for csv_path in sorted(RESULT_DIR.glob('*.csv')):
    if csv_path.name == 'summary.csv':
        continue
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ds = row['dataset']
            model = row['model']
            if model not in MODELS:
                continue
            all_rows[(ds, model)] = {
                'ACC': float(row['ACC']) if row['ACC'] else 0.0,
                'NMI': float(row['NMI']) if row['NMI'] else 0.0,
                'ARI': float(row['ARI']) if row['ARI'] else 0.0,
                'time_sec': float(row['time_sec']) if row['time_sec'] else 0.0,
            }
            dataset_meta[ds] = {
                'n_clusters': row['n_clusters'],
                'n_samples': row['n_samples'],
                'n_features': row['n_features'],
            }

# Build the table
print('\n# Table 1: General Deep Clustering SOTA vs TopoGate\n')
print('| Dataset | K | N | Feat | Metric | ', end='')
print(' | '.join(MODELS) + ' |')
print('|', '|'.join(['---'] * (5 + len(MODELS))), '|')

# Single metric table for ACC
print('\n## ACC (Accuracy)\n')
print('| Dataset | K | N | Feat | ', ' | '.join(MODELS) + ' |')
print('|', '|'.join(['---'] * (4 + len(MODELS))), '|')
for ds in DATASETS_ORDER:
    if ds not in dataset_meta:
        continue
    meta = dataset_meta[ds]
    row = [ds, meta['n_clusters'], meta['n_samples'], meta['n_features']]
    vals = []
    for m in MODELS:
        v = all_rows.get((ds, m), {}).get('ACC', None)
        vals.append(f'{v:.4f}' if v is not None else 'N/A')
    print('|', ' | '.join(str(x) for x in row + vals), '|')

# NMI
print('\n## NMI (Normalized Mutual Information)\n')
print('| Dataset | K | N | Feat | ', ' | '.join(MODELS) + ' |')
print('|', '|'.join(['---'] * (4 + len(MODELS))), '|')
for ds in DATASETS_ORDER:
    if ds not in dataset_meta:
        continue
    meta = dataset_meta[ds]
    row = [ds, meta['n_clusters'], meta['n_samples'], meta['n_features']]
    vals = []
    for m in MODELS:
        v = all_rows.get((ds, m), {}).get('NMI', None)
        vals.append(f'{v:.4f}' if v is not None else 'N/A')
    print('|', ' | '.join(str(x) for x in row + vals), '|')

# ARI
print('\n## ARI (Adjusted Rand Index)\n')
print('| Dataset | K | N | Feat | ', ' | '.join(MODELS) + ' |')
print('|', '|'.join(['---'] * (4 + len(MODELS))), '|')
for ds in DATASETS_ORDER:
    if ds not in dataset_meta:
        continue
    meta = dataset_meta[ds]
    row = [ds, meta['n_clusters'], meta['n_samples'], meta['n_features']]
    vals = []
    for m in MODELS:
        v = all_rows.get((ds, m), {}).get('ARI', None)
        vals.append(f'{v:.4f}' if v is not None else 'N/A')
    print('|', ' | '.join(str(x) for x in row + vals), '|')

# Compute average ranking for TopoGate
print('\n## Average Ranking (lower is better)\n')
print('| Metric | ', ' | '.join(MODELS) + ' |')
print('|', '|'.join(['---'] * (1 + len(MODELS))), '|')
for metric in METRICS:
    # For each dataset, rank models (1 = best)
    ranks = defaultdict(list)
    for ds in DATASETS_ORDER:
        if ds not in dataset_meta:
            continue
        sorted_models = sorted(
            MODELS,
            key=lambda m: all_rows.get((ds, m), {}).get(metric, -1),
            reverse=True,
        )
        for rank, m in enumerate(sorted_models, 1):
            ranks[m].append(rank)
    line = [metric]
    for m in MODELS:
        avg = sum(ranks[m]) / len(ranks[m]) if ranks[m] else 0
        line.append(f'{avg:.2f}')
    print('|', ' | '.join(line), '|')

# Time
print('\n## Time (seconds)\n')
print('| Dataset | K | N | Feat | ', ' | '.join(MODELS) + ' |')
print('|', '|'.join(['---'] * (4 + len(MODELS))), '|')
for ds in DATASETS_ORDER:
    if ds not in dataset_meta:
        continue
    meta = dataset_meta[ds]
    row = [ds, meta['n_clusters'], meta['n_samples'], meta['n_features']]
    vals = []
    for m in MODELS:
        v = all_rows.get((ds, m), {}).get('time_sec', None)
        vals.append(f'{v:.1f}' if v is not None else 'N/A')
    print('|', ' | '.join(str(x) for x in row + vals), '|')

# CSV output for downstream processing
csv_path = OUTPUT_DIR / 'comparison_table.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    header = ['dataset', 'n_clusters', 'n_samples', 'n_features']
    for m in MODELS:
        for k in METRICS + ['time_sec']:
            header.append(f'{m}_{k}')
    writer.writerow(header)
    for ds in DATASETS_ORDER:
        if ds not in dataset_meta:
            continue
        meta = dataset_meta[ds]
        row = [ds, meta['n_clusters'], meta['n_samples'], meta['n_features']]
        for m in MODELS:
            d = all_rows.get((ds, m), {})
            for k in METRICS + ['time_sec']:
                row.append(d.get(k, ''))
        writer.writerow(row)
print(f'\nWrote wide CSV: {csv_path}')
