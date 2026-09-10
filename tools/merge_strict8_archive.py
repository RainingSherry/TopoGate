import csv,sys
from pathlib import Path
import numpy as np
arc=Path(sys.argv[1])
def rd(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
rows=rd(arc/'test_metrics_per_seed.csv')+rd(arc/'strict8_topogate_test_metrics_per_seed.csv')
fields=list(rows[0])
for n in ('test_metrics_per_seed.csv','summaries/test_metrics_per_seed.csv'):
    with (arc/n).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
summary=[]
for ds in sorted({r['dataset'] for r in rows}):
    for model in sorted({r['model'] for r in rows if r['dataset']==ds}):
        g=[r for r in rows if r['dataset']==ds and r['model']==model]; x={'panel':'CLUBench','protocol':g[0]['protocol'],'dataset':ds,'model':model,'n_seeds':len(g),'direct_comparable':'true'}
        for m in ('ARI','NMI','ACC'):
            a=np.array([float(r[m]) for r in g]);x[m+'_mean']=float(a.mean());x[m+'_std']=float(a.std(ddof=1))
        summary.append(x)
sf=list(summary[0])
for n in ('matched_budget_summary.csv','per_dataset_comparison.csv','summaries/matched_budget_summary.csv','summaries/per_dataset_comparison.csv'):
    with (arc/n).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=sf);w.writeheader();w.writerows(summary)
pan=[]
for model in sorted({r['model'] for r in summary}):
    g=[r for r in summary if r['model']==model];pan.append({'panel':'CLUBench','protocol':g[0]['protocol'],'model':model,'dataset_count':len(g),'coverage':len(g)/131,'ARI_macro_mean':float(np.mean([float(r['ARI_mean']) for r in g])),'NMI_macro_mean':float(np.mean([float(r['NMI_mean']) for r in g])),'ACC_macro_mean':float(np.mean([float(r['ACC_mean']) for r in g])),'status':'complete'})
pf=list(pan[0])
for n in ('panel_summary.csv','summaries/panel_summary.csv'):
    with (arc/n).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=pf);w.writeheader();w.writerows(pan)
print(len(rows),len(summary))
