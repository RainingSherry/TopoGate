from __future__ import annotations
import csv, json, math, statistics
from pathlib import Path

TAB = Path('/data/luolie/ToPoGate/result/v0_rg_ablation_0911winners_first26_20260917_tabular_fixed')
FIX = Path('/data/luolie/ToPoGate/result/v0_rg_biology_ablation_transductive_0911winners_20260916_fixed')
OLD = Path('/data/luolie/ToPoGate/result/v0_rg_biology_ablation_transductive_old_winner_20260915')
ARCH = Path('/data/luolie/ToPoGate/result/topogate_transductive_clubench131_scmae16_64x3_20260911/datasets')
VARIANTS = ['full','base_scmae','anchor_equal_mask','neighbor_equal','no_edge_modulation','fixed_mix_original_aux','mixed_observation_reconstruction']
EST='current'; SEEDS=[42,123,7]
ROOTS={'tab':TAB,'fix':FIX,'old':OLD}

def find_root(ds):
    for k in ('tab','fix','old'):
        if (ROOTS[k]/'datasets'/ds).exists(): return k
    return None

def load_final(ds, variant, seed):
    rk=find_root(ds)
    if rk is None: return None,None,None
    p=ROOTS[rk]/'datasets'/ds/variant/EST/f'seed_{seed}'/'final_summary.json'
    if not p.exists(): return None,None,None
    try: d=json.loads(p.read_text())
    except Exception: return None,None,None
    m=d.get('metrics') or {}
    return rk,d,m

def fmt(x):
    return f'{x:.4f}' if isinstance(x,(int,float)) and math.isfinite(x) else ''

# ---- collect ----
datasets=sorted({p.name for p in TAB.joinpath('datasets').iterdir() if p.is_dir()})
for rk in ('fix','old'):
    for p in (ROOTS[rk]/'datasets').iterdir():
        if p.is_dir() and p.name not in datasets: datasets.append(p.name)
datasets=sorted(datasets)
out=TAB/'csv_final'; out.mkdir(parents=True,exist_ok=True)
rows=[]
for ds in datasets:
    for v in VARIANTS:
        vals={m:[] for m in ('ari','nmi','acc')}
        for s in SEEDS:
            rk,d,m=load_final(ds,v,s)
            if m:
                for k in vals: vals[k].append(float(m[k]))
        if not vals['ari']: continue
        row={'dataset':ds,'variant':v,'n_seeds':len(vals['ari'])}
        for k in vals:
            row[f'{k}_mean']=statistics.mean(vals[k]); row[f'{k}_std']=statistics.stdev(vals[k]) if len(vals[k])>1 else 0.0
        rows.append(row)
with (out/'ablation_summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
# per-seed
prows=[]
for ds in datasets:
    for v in VARIANTS:
        for s in SEEDS:
            rk,d,m=load_final(ds,v,s)
            if m and d:
                prows.append({'dataset':ds,'variant':v,'estimator':EST,'seed':s,'source_root':rk,
                              'ari':float(m['ari']),'nmi':float(m['nmi']),'acc':float(m['acc']),
                              'config_hash':d.get('config_hash',''),'fingerprint':(d.get('preprocessor') or {}).get('fingerprint','')})
with (out/'ablation_per_seed.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['dataset','variant','estimator','seed','source_root','ari','nmi','acc','config_hash','fingerprint']); w.writeheader(); w.writerows(prows)
# macro over datasets per variant
macro=[]
for v in VARIANTS:
    sub=[r for r in rows if r['variant']==v]
    if not sub: continue
    mrow={'variant':v,'n_datasets':len(sub)}
    for k in ('ari','nmi','acc'):
        mrow[f'{k}_mean']=statistics.mean(r[f'{k}_mean'] for r in sub)
        mrow[f'{k}_std']=statistics.stdev(r[f'{k}_mean'] for r in sub) if len(sub)>1 else 0.0
    macro.append(mrow)
with (out/'macro_summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['variant','n_datasets','ari_mean','ari_std','nmi_mean','nmi_std','acc_mean','acc_std']); w.writeheader(); w.writerows(macro)
# paired deltas full vs each branch (per dataset-seed), plus signed-rank p
deltas=[]
try:
    from scipy import stats as st
    have_scipy=True
except Exception:
    have_scipy=False
for ds in datasets:
    base={}
    for s in SEEDS:
        rk,d,m=load_final(ds,'full',s)
        if m: base[s]=m
    if not base: continue
    for v in VARIANTS:
        if v=='full': continue
        ds_deltas=[]
        for s in SEEDS:
            rk,d,m=load_final(ds,v,s)
            if s in base and m:
                dl=float(m['ari'])-float(base[s]['ari']); ds_deltas.append(dl)
                deltas.append({'dataset':ds,'variant':v,'seed':s,'delta_ari_full_minus':round(dl,6)})
        if ds_deltas:
            pass
with (out/'paired_deltas.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['dataset','variant','seed','delta_ari_full_minus']); w.writeheader(); w.writerows(deltas)
# full reproduction audit
audit=[]
for ds in datasets:
    for s in SEEDS:
        rk,d,m=load_final(ds,'full',s)
        if not (d and m): continue
        oldm=d.get('old_full_metrics'); delta=d.get('full_reproduction_delta')
        fp=(json.loads((Path(d.get('preprocessor_path',''))).read_text()) if False else None)
        audit.append({'dataset':ds,'seed':s,'source_root':rk,'full_ari':round(float(m['ari']),6),
                      'archive_ari':round(float(oldm['ari']),6) if oldm else '',
                      'delta_ari':round(float(delta['ari']),6) if delta else '',
                      'config_hash':d.get('config_hash','')[:16],'archive_config_hash':str(d.get('archive_config_hash') or '')[:16]})
with (out/'full_reproduction_audit.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['dataset','seed','source_root','full_ari','archive_ari','delta_ari','config_hash','archive_config_hash']); w.writeheader(); w.writerows(audit)
# fingerprint comparison
fps=[]
for ds in datasets:
    prep_ref=ARCH/ds/'final'/'seed_42'/'preprocessor_seed_42.json'
    arch_fp=json.loads(prep_ref.read_text()).get('fingerprint') if prep_ref.exists() else None
    for s in SEEDS:
        pp=ROOTS[find_root(ds)]/'datasets'/ds/'full'/EST/f'seed_{s}'/'preprocessor.json'
        if pp.exists():
            new_fp=json.loads(pp.read_text()).get('fingerprint'); fps.append({'dataset':ds,'seed':s,'fingerprint':new_fp,'archive_fingerprint':arch_fp,'match':(new_fp==arch_fp) if arch_fp else ''})
with (out/'fingerprint_audit.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['dataset','seed','fingerprint','archive_fingerprint','match']); w.writeheader(); w.writerows(fps)
print('OK datasets=',len(datasets),'summary_rows=',len(rows),'per_seed=',len(prows),'deltas=',len(deltas),'audit=',len(audit),'fp=',len(fps))
print('VARIANTS done:', {v:sum(1 for r in rows if r['variant']==v) for v in VARIANTS})
