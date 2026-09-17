from __future__ import annotations
import csv, json, statistics
from pathlib import Path

FIXED = Path('/data/luolie/ToPoGate/result/v0_rg_biology_ablation_transductive_0911winners_20260916_fixed')
OLD   = Path('/data/luolie/ToPoGate/result/v0_rg_biology_ablation_transductive_old_winner_20260915')
ARCH  = Path('/data/luolie/ToPoGate/result/topogate_transductive_clubench131_scmae16_64x3_20260911/datasets')
DATASETS = ['Bach','Baron','Guo','Limb_Muscle','Macosko','Melanoma_5K','Quake_10x_Spleen','Quake_Smart-seq2_Lung','Tosches','Young','worm_neuron_cell']
VARIANTS = ['full','base_scmae','anchor_equal_mask','neighbor_equal','no_edge_modulation','fixed_mix_original_aux','mixed_observation_reconstruction']
SEEDS = [42,123,7]

def root_of(ds):
    return FIXED if (FIXED/'datasets'/ds).exists() else OLD

def load(ds, variant, estimator, seed):
    r = root_of(ds); p = r/'datasets'/ds/variant/estimator/f'seed_{seed}'/'final_summary.json'
    if not p.exists(): return None
    return json.loads(p.read_text())

rows = []
for ds in DATASETS:
    for v in VARIANTS:
        for s in SEEDS:
            rec = load(ds, v, 'current', s)
            if rec is None:
                rows.append({'dataset':ds,'variant':v,'estimator':'current','seed':s,'status':'MISSING'}); continue
            m = rec['metrics']
            row = {'dataset':ds,'variant':v,'estimator':'current','seed':s,'status':rec.get('status','completed'),
                   'ari':m['ari'],'nmi':m['nmi'],'acc':m['acc'],'fit_scope':rec.get('fit_scope')}
            if v=='full' and rec.get('old_full_metrics'):
                om = rec['old_full_metrics']; d = rec.get('full_reproduction_delta') or {}
                row.update(old_ari=om['ari'],old_nmi=om['nmi'],old_acc=om['acc'],
                           d_ari=d.get('ari'),d_nmi=d.get('nmi'),d_acc=d.get('acc'))
            rows.append(row)
for ds in ['Macosko','Quake_Smart-seq2_Lung']:
    for est in ['uniform_sample','full']:
        for s in SEEDS:
            rec = load(ds,'full',est,s)
            if rec: rows.append({'dataset':ds,'variant':'full','estimator':est,'seed':s,'status':rec.get('status','completed'),'ari':rec['metrics']['ari'],'nmi':rec['metrics']['nmi'],'acc':rec['metrics']['acc']})

out = FIXED/'csv_final'; out.mkdir(parents=True, exist_ok=True)
perms = [r for r in rows if r['status']!='MISSING']
with open(out/'ablation_per_seed.csv','w',newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
summ = []
for ds in DATASETS:
    for v in VARIANTS:
        vals = [r for r in rows if r['dataset']==ds and r['variant']==v and r['estimator']=='current' and r['status']!='MISSING']
        if len(vals)!=3: continue
        def ms(key):
            mean = statistics.mean(r[key] for r in vals)
            std  = statistics.stdev(r[key] for r in vals) if len(vals)>1 else 0.0
            return f'{mean:.4f}±{std:.4f}'
        summ.append({'dataset':ds,'variant':v,'n_seeds':len(vals),'ARI':ms('ari'),'NMI':ms('nmi'),'ACC':ms('acc')})
with open(out/'ablation_summary.csv','w',newline='') as f:
    w = csv.DictWriter(f, fieldnames=['dataset','variant','n_seeds','ARI','NMI','ACC']); w.writeheader(); w.writerows(summ)
audit = []
for ds in DATASETS:
    arch_fp = None
    cand = sorted((ARCH/ds).glob('candidate_*'))
    if cand:
        p = cand[0]/'seed_42'/'preprocessor_seed_42.json'
        if p.exists(): arch_fp = json.loads(p.read_text()).get('fingerprint')
    run_fp = None
    p = root_of(ds)/'datasets'/ds/'full'/'current'/'seed_42'/'preprocessor.json'
    if p.exists(): run_fp = json.loads(p.read_text()).get('fingerprint')
    audit.append({'dataset':ds,'archive_fingerprint':arch_fp,'run_fingerprint':run_fp,
                  'match':'YES' if arch_fp and arch_fp==run_fp else ('NO' if arch_fp and run_fp else 'MISSING')})
with open(out/'fingerprint_audit.csv','w',newline='') as f:
    w = csv.DictWriter(f, fieldnames=['dataset','archive_fingerprint','run_fingerprint','match']); w.writeheader(); w.writerows(audit)
fraud = [r for r in rows if r['variant']=='full' and r['estimator']=='current' and 'old_ari' in r and r['status']!='MISSING']
with open(out/'full_reproduction_audit.csv','w',newline='') as f:
    w = csv.DictWriter(f, fieldnames=['dataset','seed','ari','old_ari','d_ari','d_nmi','d_acc']); w.writeheader()
    w.writerows({k:r[k] for k in ['dataset','seed','ari','old_ari','d_ari','d_nmi','d_acc']} for r in fraud)
mac = []
for ds in DATASETS:
    for v in VARIANTS:
        vals = [r for r in rows if r['dataset']==ds and r['variant']==v and r['estimator']=='current' and r['status']!='MISSING']
        if len(vals)==3: mac.append({'dataset':ds,'variant':v,'ari_mean':round(statistics.mean(r['ari'] for r in vals),4)})
with open(out/'macro_summary.csv','w',newline='') as f:
    w = csv.DictWriter(f, fieldnames=['dataset','variant','ari_mean']); w.writeheader(); w.writerows(mac)
print('per_seed', len(perms), 'summary', len(summ), 'fingerprint_audit', len(audit), 'full_repro', len(fraud), 'macro', len(mac))
