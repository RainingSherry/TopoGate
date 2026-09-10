import csv,sys
from pathlib import Path
import numpy as np
a=Path(sys.argv[1])
rows=list(csv.DictReader((a/'matched_budget_summary.csv').open(encoding='utf-8-sig')))
out=[]
for ds in sorted({r['dataset'] for r in rows}):
    g={r['model']:r for r in rows if r['dataset']==ds}
    if not {'TopoGate_strict8','KMeans','PCA+KMeans'} <= g.keys(): continue
    for model in ('KMeans','PCA+KMeans'):
        out.append({'panel':'CLUBench','dataset':ds,'model_a':'TopoGate_strict8','model_b':model,'ARI_delta':float(g['TopoGate_strict8']['ARI_mean'])-float(g[model]['ARI_mean']),'NMI_delta':float(g['TopoGate_strict8']['NMI_mean'])-float(g[model]['NMI_mean']),'ACC_delta':float(g['TopoGate_strict8']['ACC_mean'])-float(g[model]['ACC_mean'])})
for n in ('paired_comparisons.csv','summaries/paired_comparisons.csv'):
    with (a/n).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
print(len(out))
