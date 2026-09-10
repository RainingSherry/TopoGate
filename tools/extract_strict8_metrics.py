import csv, glob, json, os, sys
root=sys.argv[1]; out=sys.argv[2]
rows=[]
for d in sorted(glob.glob(root+'/*')):
    if not os.path.exists(os.path.join(d,'final_summary.json')): continue
    dataset=os.path.basename(d)
    for seed in (42,123,7):
        files=glob.glob(os.path.join(d,'final','*',f'seed_{seed}.json'))
        if len(files)!=1: raise RuntimeError(f'{dataset} seed {seed}: {len(files)} records')
        r=json.load(open(files[0])); m=r['test_metrics']
        rows.append({'panel':'CLUBench','protocol':'strict8_topogate_v1','dataset':dataset,'model':'TopoGate_strict8','seed':seed,'ARI':m['ari'],'NMI':m['nmi'],'ACC':m['acc'],'direct_comparable':'true','status':'complete'})
with open(out,'w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(len(rows))
