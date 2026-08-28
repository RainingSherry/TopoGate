"""Statistical significance testing for TopoGate vs each CLUBench algorithm.

Tests: Wilcoxon signed-rank test on NMI across 13 shared datasets.
"""
import csv
from pathlib import Path
from collections import defaultdict
from scipy import stats
import numpy as np

OUTPUT_DIR = Path('/home/luolie/ToPoGate/papers/tab_figs')
CSV_PATH = OUTPUT_DIR / 'merged_comparison.csv'

METRICS = ['acc', 'nmi', 'ari']
shared = ['iris', 'har', 'spambase', 'breast_cancer_wisconsin_original',
          'mammographic_mass', 'sms_spam_collection', 'enron', 'ISOLET',
          'reuters', 'cnae9', 'Campbell', 'first-order-theorem-proving', 'Mouse_retina']


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

    # Build vectors
    tg_vec = {m: [] for m in METRICS}
    for ds in shared:
        for m in METRICS:
            tg_vec[m].append(data[topogate][ds][m])

    # Wilcoxon signed-rank test: TopoGate vs each algorithm
    # H1: TopoGate > opponent (one-sided)
    results_table = []

    for metric in METRICS:
        for algo in algorithms:
            if algo == topogate:
                continue
            opp_vec = []
            for ds in shared:
                if ds in data[algo]:
                    opp_vec.append(data[algo][ds][metric])
                else:
                    opp_vec.append(np.nan)
            if any(np.isnan(opp_vec)):
                continue
            # Wilcoxon signed-rank (one-sided)
            diff = np.array(tg_vec[metric]) - np.array(opp_vec)
            diff = diff[diff != 0]  # remove zeros
            if len(diff) < 3:
                continue
            try:
                stat, p_two = stats.wilcoxon(diff, alternative='greater')
                # Bonferroni correction
                p_bonf = min(p_two * 28, 1.0)
                results_table.append({
                    'metric': metric,
                    'opponent': algo,
                    'mean_diff': float(np.mean(diff)),
                    'stat': float(stat),
                    'p_raw': float(p_two),
                    'p_bonf': float(p_bonf),
                    'wins': int(np.sum(diff > 0)),
                    'losses': int(np.sum(diff < 0)),
                })
            except ValueError:
                pass

    # Sort by (metric, p_bonf)
    results_table.sort(key=lambda x: (x['metric'], x['p_bonf']))

    # Write report
    report = []
    report.append('\n\n# Statistical Significance: TopoGate vs CLUBench (Wilcoxon signed-rank)\n')
    report.append('Null hypothesis H0: TopoGate ≡ opponent on the metric across 13 shared datasets.\n')
    report.append('Alternative hypothesis H1: TopoGate > opponent (one-sided).\n')
    report.append('Significance level: α=0.05 after Bonferroni correction (×28 comparisons per metric).\n')

    for metric in METRICS:
        report.append(f'\n## {metric.upper()}\n')
        report.append('| Opponent | Wins/Losses | Mean Δ | p (raw) | p (Bonferroni) | Significant |')
        report.append('|---|:---:|:---:|:---:|:---:|:---:|')
        for r in results_table:
            if r['metric'] != metric:
                continue
            sig = '✅ p<0.05' if r['p_bonf'] < 0.05 else 'ns'
            report.append(
                f'| {r["opponent"]} | {r["wins"]}/{r["losses"]} | '
                f'{r["mean_diff"]:+.4f} | {r["p_raw"]:.4f} | {r["p_bonf"]:.4f} | {sig} |'
            )

    report.append('\n## Summary\n')
    for metric in METRICS:
        sig_wins = sum(1 for r in results_table if r['metric'] == metric and r['p_bonf'] < 0.05)
        ns = sum(1 for r in results_table if r['metric'] == metric and r['p_bonf'] >= 0.05)
        worse = sum(1 for r in results_table if r['metric'] == metric and r['mean_diff'] < 0)
        report.append(f'- **{metric.upper()}**: TopoGate significantly better than {sig_wins}/28 opponents, '
                      f'not significant vs {ns}/28, worse than {worse}/28.')

    report_path = OUTPUT_DIR / 'analysis_report.md'
    with open(report_path, 'a') as f:
        f.write('\n'.join(report))
    print(f'Appended to {report_path}')

    # Also print to terminal
    for metric in METRICS:
        sig_wins = sum(1 for r in results_table if r['metric'] == metric and r['p_bonf'] < 0.05)
        ns = sum(1 for r in results_table if r['metric'] == metric and r['p_bonf'] >= 0.05)
        worse = sum(1 for r in results_table if r['metric'] == metric and r['mean_diff'] < 0)
        print(f'{metric.upper()}: significantly better than {sig_wins}/28, '
              f'not significant vs {ns}/28, mean diff < 0 against {worse}/28')


if __name__ == '__main__':
    main()
