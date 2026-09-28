"""Parent aggregation of the audited Claude-authored cohort worker outputs."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;out=P/'outputs/E3_cohort';out.mkdir(parents=True,exist_ok=True)
rows=[];inputs={}
for folder in sorted((P/'outputs/cohort').iterdir()):
    records=list(folder.glob('attempt_*/record.json'))
    if len(records)!=1:raise ValueError((folder,len(records)))
    rec=json.loads(records[0].read_text());base=records[0].parent
    assert rec['status']=='completed' and rec['labels_used_for_model_fit'] is False
    f=base/'e3_per_seed.csv';digest=hashlib.sha256(f.read_bytes()).hexdigest()
    assert digest==rec['artifact_sha256']['e3_per_seed.csv'];inputs[folder.name]=digest
    d=pd.read_csv(f);h=d[d.arm=='H1'].set_index('seed');m=d[d.arm=='multihop'].set_index('seed')
    assert set(h.index)==set(m.index)=={7,42,123}
    for seed in [7,42,123]:
        rows.append(dict(dataset=folder.name,seed=seed,ARI_H1=h.loc[seed,'ARI_mean'],ARI_multi=m.loc[seed,'ARI_mean'],
            delta_ARI=m.loc[seed,'ARI_mean']-h.loc[seed,'ARI_mean'],
            split_reduction_pp=100*(h.loc[seed,'split_rate_mean']-m.loc[seed,'split_rate_mean']),
            merge_reduction_pp=100*(h.loc[seed,'merge_rate_mean']-m.loc[seed,'merge_rate_mean']),
            macro_split_reduction_pp=100*(h.loc[seed,'macro_split_rate_mean']-m.loc[seed,'macro_split_rate_mean']),
            macro_merge_reduction_pp=100*(h.loc[seed,'macro_merge_rate_mean']-m.loc[seed,'macro_merge_rate_mean'])))
d=pd.DataFrame(rows);assert d.dataset.nunique()==59
d.to_csv(out/'paired_per_seed.csv',index=False)
s=d.groupby('dataset').mean(numeric_only=True).drop(columns='seed')
for key in ['delta_ARI','split_reduction_pp','merge_reduction_pp']:
    s[key+'_seed_sd']=d.groupby('dataset')[key].std(ddof=1)
s['positive_ARI_seeds']=d.groupby('dataset').delta_ARI.apply(lambda x:int((x>1e-12).sum()))
s.to_csv(out/'summary.csv')
fig,ax=plt.subplots(figsize=(8.5,6.5))
colors=np.where(s.delta_ARI>1e-12,'#B5465D','#797A8E')
ax.scatter(s.split_reduction_pp,s.merge_reduction_pp,s=22+900*np.abs(s.delta_ARI),c=colors,alpha=.7,edgecolors='white',linewidth=.5)
for name,offset in [('Baron',(8,8)),('Quake_Smart-seq2_Lung',(8,-16)),('student_grade',(8,8))]:
    r=s.loc[name];ax.scatter([r.split_reduction_pp],[r.merge_reduction_pp],s=110,facecolors='none',edgecolors='#302D36',linewidth=1.2)
    ax.annotate(name.replace('Quake_Smart-seq2_Lung','Quake Lung'),(r.split_reduction_pp,r.merge_reduction_pp),xytext=offset,textcoords='offset points',fontsize=9)
ax.axhline(0,c='#ACBAC1',lw=.8);ax.axvline(0,c='#ACBAC1',lw=.8)
ax.set(xlabel='Reduction in same-class splitting (percentage points)',ylabel='Reduction in different-class merging (percentage points)',title='How multihop changes the partition: all 59 matched datasets')
ax.spines[['top','right']].set_visible(False)
fig.text(.5,.012,'Each point: mean of 3 training seeds × 10 readouts. Rose: ΔARI > 0; gray: otherwise. Area scales with |ΔARI|.',ha='center',fontsize=8)
fig.tight_layout(rect=[0,.04,1,1])
for ext in ['png','pdf','svg']:fig.savefig(out/f'partition_gain_plane.{ext}',dpi=250)
stats=dict(datasets=59,positive=int((s.delta_ARI>1e-12).sum()),negative=int((s.delta_ARI < -1e-12).sum()),
           tied=int((s.delta_ARI.abs()<=1e-12).sum()),all_three_positive=int((s.positive_ARI_seeds==3).sum()),
           mean_delta_ARI=float(s.delta_ARI.mean()),both_pair_errors_reduced=int(((s.split_reduction_pp>1e-10)&(s.merge_reduction_pp>1e-10)).sum()),
           cases=s.loc[['Baron','Quake_Smart-seq2_Lung','student_grade']].to_dict('index'))
(out/'summary.json').write_text(json.dumps(stats,indent=2));(out/'input_hashes.json').write_text(json.dumps(inputs,indent=2))
print(json.dumps(stats,indent=2))
