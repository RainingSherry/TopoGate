from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
ROOT=Path('/data/luolie/ToPoGate/result/v0_rg_biology_ablation_transductive_old_winner_20260915')
def main():
 rows=[]
 for p in ROOT.glob('datasets/*/*/*/seed_*/final_summary.json'):
  d=json.loads(p.read_text()); m=d['metrics']; old=d.get('old_full_metrics') or {}; delta=d.get('full_reproduction_delta') or {}
  rows.append({'dataset':d['dataset'],'variant':d['variant'],'estimator':d['estimator'],'seed':d['seed'],
               'ari':m['ari'],'nmi':m['nmi'],'acc':m['acc'],'mean_gate_train':d.get('mean_gate_train'),
               'old_ari':old.get('ari'),'old_nmi':old.get('nmi'),'old_acc':old.get('acc'),
               'ari_delta_vs_old':delta.get('ari'),'nmi_delta_vs_old':delta.get('nmi'),'acc_delta_vs_old':delta.get('acc'),
               'full_reproduction_max_abs_delta':max(map(abs,delta.values())) if delta else None})
 outdir=ROOT/'summaries'; outdir.mkdir(exist_ok=True); fields=list(rows[0]) if rows else []
 if rows:
  with (outdir/'ablation_per_seed.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(sorted(rows,key=lambda x:(x['dataset'],x['variant'],x['estimator'],x['seed'])))
 groups={}
 for r in rows: groups.setdefault((r['dataset'],r['variant'],r['estimator']),[]).append(r)
 summary=[]
 for (d,v,e),rs in sorted(groups.items()):
  x={'dataset':d,'variant':v,'estimator':e,'n_seeds':len(rs)}
  for k in ['ari','nmi','acc']:
   vals=[r[k] for r in rs]; x[k+'_mean']=float(np.mean(vals)); x[k+'_std']=float(np.std(vals,ddof=1)) if len(vals)>1 else 0.0
  summary.append(x)
 if summary:
  with (outdir/'ablation_summary.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
 neighbor=[r for r in rows if r['variant']=='full' and r['estimator'] in {'current','uniform_sample','full'} and r['dataset'] in {'Quake_Smart-seq2_Lung','Macosko'}]
 if neighbor:
  with (outdir/'neighbor_estimator_per_seed.csv').open('w',newline='') as f:
   fields=['dataset','estimator','seed','ari','nmi','acc']; w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows([{k:r[k] for k in fields} for r in sorted(neighbor,key=lambda x:(x['dataset'],x['estimator'],x['seed']))])
 references={(r['dataset'],r['seed']):r for r in rows if r['variant']=='full' and r['estimator']=='current'}
 paired=[]
 for r in rows:
  reference=references.get((r['dataset'],r['seed']))
  if reference is None or (r['variant']=='full' and r['estimator']=='current'): continue
  if r['variant']=='full':
   family='neighbor_estimator'; method=r['estimator']
  elif r['estimator']=='current':
   family='core_ablation'; method=r['variant']
  else:
   continue
  item={'comparison_family':family,'dataset':r['dataset'],'method':method,'reference':'full/current','seed':r['seed']}
  for metric in ['ari','nmi','acc']: item[metric+'_delta']=r[metric]-reference[metric]
  paired.append(item)
 if paired:
  with (outdir/'paired_deltas.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(paired[0])); w.writeheader(); w.writerows(sorted(paired,key=lambda x:(x['comparison_family'],x['dataset'],x['method'],x['seed'])))
 full=[r for r in rows if r['variant']=='full' and r['estimator']=='current']; maxdelta=max([r['full_reproduction_max_abs_delta'] for r in full],default=None)
 if full:
  audit_fields=['dataset','seed','ari','old_ari','ari_delta_vs_old','nmi','old_nmi','nmi_delta_vs_old','acc','old_acc','acc_delta_vs_old','full_reproduction_max_abs_delta']
  with (outdir/'full_reproduction_audit.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=audit_fields); w.writeheader(); w.writerows([{k:r[k] for k in audit_fields} for r in sorted(full,key=lambda x:(x['dataset'],x['seed']))])
 status={'expected_tasks':96,'completed_tasks':len(rows),'failed_or_pending':96-len(rows),'full_reproduction_records':len(full),'full_reproduction_max_abs_delta':maxdelta,'full_reproduction_pass':len(full)==12 and maxdelta is not None and maxdelta<=1e-10}
 (outdir/'completion.json').write_text(json.dumps(status,indent=2)); print(status)
if __name__=='__main__': main()
