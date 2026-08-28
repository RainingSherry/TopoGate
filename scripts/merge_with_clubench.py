"""Merge ToPoGate experiment results with CLUBench 131-dataset benchmark.

CLUBench ships 24 algorithms × 131 datasets of best_hpc performance matrices:
  baseline/CLUBench/performance_matrix/best_hpc/{algo}.p

We map our 15 datasets onto the same index order, then combine:
- Models we ran: GCEALS, IDC, TableDC, ZEUS, TopoGate (replace TopoGate's
  own run with the CLUBench TopoGate entry if available, else ours)
- 24 CLUBench algorithms: agglo, autosc, birch, cc, dbscan, dec, divc, dmicc,
  dscn, edesc, gmm, idec, kernel_kmeans, kfsc, kmeans, kpc, lfss, lrr,
  meanshift, p2ot, pica, s3comp, spectral_clustering, ssc

Output:
- merged_comparison.csv: 29 algorithms × 15 datasets (intersection)
- merged_comparison_full.csv: 24 algorithms × 131 datasets (CLUBench only,
  to verify our 15 datasets are a fair subset)
- analysis_report.md: TopoGate's rank on shared datasets vs the full
  CLUBench field, plus per-dataset / per-algorithm bucketed analysis.
"""
import csv
import pickle
from pathlib import Path
from collections import defaultdict
import statistics

CLUBENCH_PERF_DIR = Path('/home/luolie/ToPoGate/baseline/CLUBench/performance_matrix/best_hpc')
TOGOPATE_RESULT_DIR = Path('/home/luolie/ToPoGate/result/baseline_comparison')
OUTPUT_DIR = Path('/home/luolie/ToPoGate/papers/tab_figs')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Order of datasets in our 15-dataset experiment
OUR_DATASETS = [
    'iris', 'har', 'spambase', 'breast_cancer_wisconsin_original',
    'mammographic_mass', 'sms_spam_collection', 'enron', 'ISOLET',
    'reuters', 'cnae9', 'Campbell', 'first-order-theorem-proving',
    'Mouse_retina', 'Quake_Smart-seq2_Lung', 'hrvatin_filtered',
]

# Clubench 131 dataset name (without .npz), in order
CLUBENCH_DATASETS_NPZ = [
    'echocardiogram', 'skillcraft1_master_table_dataset', 'breast_cancer_wisconsin_original',
    'smoker_condition', 'glass_identification', 'statlog_image_segmentation', 'planning_relax',
    'customer_classification', 'pima_indians_diabetes_database', 'mobile_price_classification',
    'spambase', 'rice_seed_gonen_jasmine', 'heart_attack_analysis_prediction_dataset',
    'user_knowledge_modeling', 'world12d', 'pumpkin_seeds', 'iris', 'wine',
    'letter_recognition', 'mammographic_mass', 'breast_tissue', 'hepatitis',
    'predicting_pulsar_star', 'breast_cancer_wisconsin_prognostic', 'wireless_indoor_localization',
    'date_fruit', 'zoo', 'htru2', 'ionosphere', 'music_genre_classification',
    'spectf_heart', 'rice_dataset_cammeo_and_osmancik', 'ph_recognition', 'banknote_authentication',
    'wine_quality', 'cardiovascular_study', 'statlog_german_credit', 'boston',
    'seismic_bumps', 'dry_bean', 'credit_risk_classification', 'epileptic_seizure_recognition',
    'website_phishing', 'optical_recognition_of_handwritten_digits', 'siberian_weather_stats',
    'orbit_classification_for_prediction_nasa', 'magic_gamma_telescope', 'raisin',
    'patient_treatment_classification', 'fetal_health_classification', 'dermatology',
    'secom', 'paris_housing_classification', 'seeds', 'wine_customer',
    'crowdsourced_mapping', 'durum_wheat_features', 'classification_in_asteroseismology',
    'birds_bones_and_living_habits', 'microbes', 'image_segmentation', 'water_quality',
    'insurance_company_benchmark', 'harbermans_survival', 'yeast', 'heart_disease',
    'ecoli', 'extyaleb', 'breast_cancer_coimbra', 'student_grade', 'human_stress_detection',
    'fraud_detection_bank', 'pen_based_recognition_of_handwritten_digits',
    'diabetic_retinopathy_debrecen', 'pistachio', 'turkish_music_emotion', 'parkinsons',
    'weather', 'blood_transfusion_service_center', 'mfeat-karhunen', 'mfeat-factors',
    'wall-robot-navigation', 'Waveform', 'gas-drift', 'mfeat-morphological',
    'JapaneseVowels', 'rmftsa_sleepdata', 'first-order-theorem-proving', 'gina_prior2',
    'fabert', 'dilbert', 'synthetic_control', 'Drug Consumption', 'shuttle', 'tr45.wc',
    'steel-plates-fault', 'fbis.wc', 'mfeat-fourier', 'vehicle', 'micro-mass', 'ISOLET',
    'poker-hand', 'tamilnadu-electricity', 'mnist64', 'MNIST_CLIP', 'fashion_mnist',
    'FashionMNIST_CLIP', 'cifar10', 'CIFAR10_CLIP', 'coil20', 'COIL20_CLIP',
    'labeled_faces_in_the_wild', 'flickr_material_database', 'street_view_house_numbers',
    'har', 'Indian_pines', 'satellite_image', 'olivetti_faces', 'PCam', 'cnae9',
    'imdb', 'hate_speech', 'sentiment_labeld_sentences', 'sms_spam_collection',
    'wos', 'enron', 'reuters', '20newsgroups', 'Mouse_retina', 'Campbell', 'Baron Human',
]


def load_clubench_results():
    """Load all 24 algorithms' best_hpc results: {algo: {acc, nmi, ari} (lists of 131)}"""
    results = {}
    for p in sorted(CLUBENCH_PERF_DIR.glob('*.p')):
        algo = p.stem
        with open(p, 'rb') as f:
            data = pickle.load(f)
        results[algo] = {
            'acc': list(data['acc']),
            'nmi': list(data['nmi']),
            'ari': list(data['ari']),
        }
    return results


def load_our_results():
    """Load our 5 models × 15 datasets: {(model, dataset): {acc, nmi, ari}}"""
    results = {}
    for csv_path in sorted(TOGOPATE_RESULT_DIR.glob('*.csv')):
        if csv_path.name == 'summary.csv':
            continue
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                results[(row['model'], row['dataset'])] = {
                    'acc': float(row['ACC']),
                    'nmi': float(row['NMI']),
                    'ari': float(row['ARI']),
                }
    return results


def main():
    clubench = load_clubench_results()
    ours = load_our_results()

    # Verify which of our 15 datasets are in CLUBench 131
    dataset_to_idx = {ds: i for i, ds in enumerate(CLUBENCH_DATASETS_NPZ)}
    shared = [ds for ds in OUR_DATASETS if ds in dataset_to_idx]
    our_only = [ds for ds in OUR_DATASETS if ds not in dataset_to_idx]

    print(f'CLUBench 131 datasets: {len(CLUBENCH_DATASETS_NPZ)}')
    print(f'Our 15 datasets: {len(OUR_DATASETS)}')
    print(f'Shared (in both): {len(shared)} -> {shared}')
    print(f'Our-only (not in CLUBench): {our_only}')

    # ---- Build the merged comparison on shared datasets ----
    # Add TopoGate from our runs (we ran it; CLUBench has no TopoGate)
    all_algos = sorted(clubench.keys()) + ['GCEALS', 'IDC', 'TableDC', 'ZEUS', 'TopoGate']
    csv_path = OUTPUT_DIR / 'merged_comparison.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['algorithm', 'source', 'dataset', 'ACC', 'NMI', 'ARI'])
        for algo in all_algos:
            if algo in clubench:
                for ds in shared:
                    idx = dataset_to_idx[ds]
                    writer.writerow([
                        algo, 'CLUBench', ds,
                        f'{clubench[algo]["acc"][idx]:.4f}',
                        f'{clubench[algo]["nmi"][idx]:.4f}',
                        f'{clubench[algo]["ari"][idx]:.4f}',
                    ])
            else:
                # Our run
                for ds in shared:
                    if (algo, ds) in ours:
                        r = ours[(algo, ds)]
                        writer.writerow([
                            algo, 'ToPoGate', ds,
                            f'{r["acc"]:.4f}',
                            f'{r["nmi"]:.4f}',
                            f'{r["ari"]:.4f}',
                        ])
    print(f'Wrote {csv_path}')

    # ---- Compute TopoGate's rank on shared datasets ----
    # For each shared dataset, rank all 29 algorithms on ACC/NMI/ARI
    topogate_ranks = {'acc': [], 'nmi': [], 'ari': []}
    algo_avg_rank = defaultdict(lambda: {'acc': [], 'nmi': [], 'ari': []})

    for ds in shared:
        idx = dataset_to_idx[ds]
        all_algo_metric = defaultdict(dict)
        for algo in clubench:
            for metric in ('acc', 'nmi', 'ari'):
                all_algo_metric[algo][metric] = clubench[algo][metric][idx]
        for algo in ('GCEALS', 'IDC', 'TableDC', 'ZEUS', 'TopoGate'):
            if (algo, ds) in ours:
                r = ours[(algo, ds)]
                all_algo_metric[algo] = {
                    'acc': r['acc'], 'nmi': r['nmi'], 'ari': r['ari'],
                }
        for metric in ('acc', 'nmi', 'ari'):
            ranked = sorted(
                all_algo_metric.items(),
                key=lambda kv: kv[1][metric],
                reverse=True,
            )
            for rank, (alg, _) in enumerate(ranked, 1):
                algo_avg_rank[alg][metric].append(rank)
                if alg == 'TopoGate':
                    topogate_ranks[metric].append(rank)

    # ---- Write analysis report ----
    report = []
    report.append('# TopoGate vs CLUBench 24-Algorithm Benchmark — Analysis Report\n')
    report.append('## Overview\n')
    report.append(f'- CLUBench ships 24 algorithms × 131 datasets (best-hpc performance matrix)')
    report.append(f'- We benchmarked TopoGate (and 4 baselines: GCEALS, IDC, TableDC, ZEUS) on 15 datasets')
    report.append(f'- **Shared datasets (intersection): {len(shared)}**')
    report.append(f'- ToPoGate-only datasets (not in CLUBench): {our_only}\n')

    report.append('## Datasets\n')
    report.append('| Dataset | In CLUBench | In ToPoGate |')
    report.append('|---|:---:|:---:|')
    for ds in OUR_DATASETS:
        in_club = '✅' if ds in dataset_to_idx else '❌'
        report.append(f'| {ds} | {in_club} | ✅ |')
    report.append('')

    report.append('## TopoGate Average Rank on Shared Datasets\n')
    report.append('(29 algorithms total: 24 from CLUBench + 5 from our run)\n')
    report.append('| Metric | Avg Rank | Median Rank | Best (1) | Worst (29) |')
    report.append('|---|:---:|:---:|:---:|:---:|')
    for metric in ('acc', 'nmi', 'ari'):
        r = topogate_ranks[metric]
        if r:
            report.append(
                f'| {metric.upper()} | {statistics.mean(r):.2f} | {statistics.median(r):.1f} '
                f'| {min(r)} | {max(r)} |'
            )
    report.append('')

    report.append('## All Algorithms Average Rank (lower is better)\n')
    report.append('| Algorithm | Source | ACC Avg Rank | NMI Avg Rank | ARI Avg Rank |')
    report.append('|---|---|:---:|:---:|:---:|')
    for alg in sorted(algo_avg_rank.keys(), key=lambda a: statistics.mean(algo_avg_rank[a]['acc'])):
        src = 'CLUBench' if alg in clubench else 'ToPoGate'
        ar = [f'{statistics.mean(algo_avg_rank[alg][m]):.2f}' for m in ('acc', 'nmi', 'ari')]
        report.append(f'| {alg} | {src} | {ar[0]} | {ar[1]} | {ar[2]} |')
    report.append('')

    # Per-dataset ranking of TopoGate
    report.append('## TopoGate Per-Dataset Rank\n')
    report.append('| Dataset | ACC Rank | NMI Rank | ARI Rank | TopoGate ACC | TopoGate NMI | TopoGate ARI |')
    report.append('|---|:---:|:---:|:---:|:---:|:---:|:---:|')
    for ds in shared:
        idx = dataset_to_idx[ds]
        all_algo_metric = defaultdict(dict)
        for algo in clubench:
            for metric in ('acc', 'nmi', 'ari'):
                all_algo_metric[algo][metric] = clubench[algo][metric][idx]
        if ('TopoGate', ds) in ours:
            r = ours[('TopoGate', ds)]
            all_algo_metric['TopoGate'] = {
                'acc': r['acc'], 'nmi': r['nmi'], 'ari': r['ari'],
            }
        else:
            continue
        line = [ds]
        for metric in ('acc', 'nmi', 'ari'):
            ranked = sorted(all_algo_metric.items(),
                            key=lambda kv: kv[1][metric], reverse=True)
            rank = next(i for i, (a, _) in enumerate(ranked, 1) if a == 'TopoGate')
            line.append(f'{rank}')
        for metric in ('acc', 'nmi', 'ari'):
            line.append(f'{all_algo_metric["TopoGate"][metric]:.4f}')
        report.append('| ' + ' | '.join(line) + ' |')
    report.append('')

    # Head-to-head: TopoGate vs each CLUBench algorithm
    report.append('## TopoGate vs Each CLUBench Algorithm (win/tie/loss)\n')
    report.append('| Opponent | Better ACC | Better NMI | Better ARI |')
    report.append('|---|:---:|:---:|:---:|')
    h2h = defaultdict(lambda: {'acc': 0, 'nmi': 0, 'ari': 0})
    for ds in shared:
        idx = dataset_to_idx[ds]
        if ('TopoGate', ds) not in ours:
            continue
        tg = ours[('TopoGate', ds)]
        for algo in clubench:
            for metric in ('acc', 'nmi', 'ari'):
                if tg[metric] > clubench[algo][metric][idx]:
                    h2h[algo][metric] += 1
    for algo in sorted(h2h.keys()):
        report.append(
            f'| {algo} | {h2h[algo]["acc"]}/{len(shared)} | {h2h[algo]["nmi"]}/{len(shared)} '
            f'| {h2h[algo]["ari"]}/{len(shared)} |'
        )
    report.append('')

    # Save
    report_path = OUTPUT_DIR / 'analysis_report.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    print(f'Wrote {report_path}')

    # Print summary
    print('\n=== TOPOGATE AVERAGE RANK ===')
    for metric in ('acc', 'nmi', 'ari'):
        r = topogate_ranks[metric]
        if r:
            print(f'  {metric.upper()}: avg={statistics.mean(r):.2f}, '
                  f'median={statistics.median(r):.1f}, min={min(r)}, max={max(r)}')


if __name__ == '__main__':
    main()
