"""Deep analysis of TopoGate vs CLUBench benchmark.

Builds on merged_comparison.csv. Categorises datasets by:
- Modality (tabular / text / image / bioinfo)
- Size (small N<1000 / medium 1000≤N<10000 / large N≥10000)
- Feature dimensionality (low d<100 / medium 100≤d<1000 / high d≥1000)
- Class count (binary K=2 / multi K≥3)

Computes TopoGate's win-rate within each bucket and identifies where
TopoGate's special design (topology-aware gating) shines.
"""
import csv
from pathlib import Path
from collections import defaultdict

OUTPUT_DIR = Path('/home/luolie/ToPoGate/papers/tab_figs')
CSV_PATH = OUTPUT_DIR / 'merged_comparison.csv'

# Dataset metadata (from CLUBench README + our runs)
DATASET_META = {
    'iris': {'modality': 'tabular', 'size': 'small', 'dim': 'low', 'K': 'multi'},
    'har': {'modality': 'image', 'size': 'small', 'dim': 'medium', 'K': 'multi'},
    'spambase': {'modality': 'tabular', 'size': 'medium', 'dim': 'low', 'K': 'binary'},
    'breast_cancer_wisconsin_original': {'modality': 'tabular', 'size': 'small', 'dim': 'low', 'K': 'binary'},
    'mammographic_mass': {'modality': 'tabular', 'size': 'small', 'dim': 'low', 'K': 'binary'},
    'sms_spam_collection': {'modality': 'text', 'size': 'small', 'dim': 'medium', 'K': 'binary'},
    'enron': {'modality': 'text', 'size': 'large', 'dim': 'high', 'K': 'binary'},
    'ISOLET': {'modality': 'tabular', 'size': 'medium', 'dim': 'medium', 'K': 'multi'},
    'reuters': {'modality': 'text', 'size': 'medium', 'dim': 'high', 'K': 'multi'},
    'cnae9': {'modality': 'text', 'size': 'small', 'dim': 'medium', 'K': 'multi'},
    'Campbell': {'modality': 'bioinfo', 'size': 'large', 'dim': 'high', 'K': 'multi'},
    'first-order-theorem-proving': {'modality': 'tabular', 'size': 'medium', 'dim': 'low', 'K': 'multi'},
    'Mouse_retina': {'modality': 'bioinfo', 'size': 'medium', 'dim': 'high', 'K': 'multi'},
}


def main():
    # Load merged results
    data = defaultdict(lambda: defaultdict(dict))  # algo -> dataset -> {acc, nmi, ari}
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
    assert topogate in algorithms

    # ---- Bucketed analysis ----
    buckets = defaultdict(lambda: {'total': 0, 'tg_better': 0, 'tg_better_nmi': 0, 'tg_better_ari': 0})

    print('=' * 80)
    print('TOPOGATE vs CLUBench 24-Algorithm Field — Bucketed Win Analysis')
    print('=' * 80)

    for ds, meta in DATASET_META.items():
        if ds not in data[topogate]:
            continue
        tg = data[topogate][ds]
        # Count algorithms that TopoGate beats on each metric
        for algo in algorithms:
            if algo == topogate:
                continue
            if ds not in data[algo]:
                continue
            for bucket_key in (
                f'modality:{meta["modality"]}',
                f'size:{meta["size"]}',
                f'dim:{meta["dim"]}',
                f'K:{meta["K"]}',
            ):
                buckets[bucket_key]['total'] += 1
                if tg['acc'] > data[algo][ds]['acc']:
                    buckets[bucket_key]['tg_better'] += 1
                if tg['nmi'] > data[algo][ds]['nmi']:
                    buckets[bucket_key]['tg_better_nmi'] += 1
                if tg['ari'] > data[algo][ds]['ari']:
                    buckets[bucket_key]['tg_better_ari'] += 1

    # Print bucket analysis
    print('\n## Win-rate by bucket (TopoGate vs each opponent)')
    print(f'{"Bucket":<35} {"N pairs":<8} {"ACC":<8} {"NMI":<8} {"ARI":<8}')
    print('-' * 80)
    for bucket in sorted(buckets.keys()):
        b = buckets[bucket]
        if b['total'] == 0:
            continue
        n = b['total']
        acc_pct = 100 * b['tg_better'] / n
        nmi_pct = 100 * b['tg_better_nmi'] / n
        ari_pct = 100 * b['tg_better_ari'] / n
        print(f'{bucket:<35} {n:<8} {acc_pct:.1f}%   {nmi_pct:.1f}%   {ari_pct:.1f}%')

    # ---- Overall summary ----
    total_pairs = sum(b['total'] for b in buckets.values() if b['total'] > 0) // 4  # 4 buckets per dataset
    n_pairs = 0
    n_acc = 0
    n_nmi = 0
    n_ari = 0
    for ds in DATASET_META:
        if ds not in data[topogate]:
            continue
        tg = data[topogate][ds]
        for algo in algorithms:
            if algo == topogate or ds not in data[algo]:
                continue
            n_pairs += 1
            if tg['acc'] > data[algo][ds]['acc']:
                n_acc += 1
            if tg['nmi'] > data[algo][ds]['nmi']:
                n_nmi += 1
            if tg['ari'] > data[algo][ds]['ari']:
                n_ari += 1

    print('\n## Overall (29 algorithms × 13 shared datasets)')
    print(f'Total pairs: {n_pairs}')
    print(f'TopoGate better ACC: {n_acc}/{n_pairs} = {100*n_acc/n_pairs:.1f}%')
    print(f'TopoGate better NMI: {n_nmi}/{n_pairs} = {100*n_nmi/n_pairs:.1f}%')
    print(f'TopoGate better ARI: {n_ari}/{n_pairs} = {100*n_ari/n_pairs:.1f}%')

    # ---- Per-dataset absolute scores ----
    print('\n## Per-Dataset TopoGate vs Mean(CLUBench) vs Best(CLUBench)')
    print(f'{"Dataset":<35} {"K_modality":<25} {"TopoGate":<10} {"CLUBench_mean":<15} {"CLUBench_best":<15}')
    print('-' * 100)
    for ds, meta in DATASET_META.items():
        if ds not in data[topogate]:
            continue
        tg = data[topogate][ds]
        for metric in ('acc', 'nmi', 'ari'):
            all_metric = []
            for algo in algorithms:
                if algo == topogate or ds not in data[algo]:
                    continue
                all_metric.append(data[algo][ds][metric])
            if not all_metric:
                continue
            mean = sum(all_metric) / len(all_metric)
            best = max(all_metric)
            print(f'  {ds} [{metric.upper()}]: TopoGate={tg[metric]:.4f}  '
                  f'CLUBench_mean={mean:.4f}  CLUBench_best={best:.4f}  '
                  f'(N={len(all_metric)} opponents)')

    # ---- Save consolidated report ----
    report = []
    report.append('\n\n# Deep Analysis — TopoGate Bucket Performance\n')
    report.append('## Overall Win Rate\n')
    report.append(f'- ACC: {n_acc}/{n_pairs} = {100*n_acc/n_pairs:.1f}%')
    report.append(f'- NMI: {n_nmi}/{n_pairs} = {100*n_nmi/n_pairs:.1f}%')
    report.append(f'- ARI: {n_ari}/{n_pairs} = {100*n_ari/n_pairs:.1f}%\n')

    report.append('## Win Rate by Bucket\n')
    report.append('| Bucket | Pairs | TopoGate ACC | TopoGate NMI | TopoGate ARI |')
    report.append('|---|:---:|:---:|:---:|:---:|')
    for bucket in sorted(buckets.keys()):
        b = buckets[bucket]
        if b['total'] == 0:
            continue
        n = b['total']
        report.append(
            f'| {bucket} | {n} | {100*b["tg_better"]/n:.1f}% | '
            f'{100*b["tg_better_nmi"]/n:.1f}% | {100*b["tg_better_ari"]/n:.1f}% |'
        )
    report.append('')

    # Append to existing analysis report
    report_path = OUTPUT_DIR / 'analysis_report.md'
    with open(report_path, 'a') as f:
        f.write('\n'.join(report))
    print(f'\nAppended to {report_path}')


if __name__ == '__main__':
    main()
