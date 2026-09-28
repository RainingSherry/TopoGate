"""Parent assembly of the reviewed, frozen response probe results."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent/'outputs/E5_response'
cases=['Baron','Quake_Smart-seq2_Lung','student_grade'];rows=[]
for ds in cases:
    for seed in [7,42,123]:
        d=pd.read_csv(P/f'{ds}_seed{seed}'/'probe_rows.csv');d['dataset']=ds;d['seed']=seed;rows.append(d)
f=pd.concat(rows,ignore_index=True)
s=f.groupby(['dataset','seed','model_role','family','kind','lambda'])[['encoder_response_normalized','decoder_anchor_mse_delta']].mean().reset_index()
s.to_csv(P/'response_per_seed.csv',index=False)
a=s.groupby(['dataset','model_role','family','kind','lambda']).agg(response_mean=('encoder_response_normalized','mean'),response_seed_sd=('encoder_response_normalized','std'),reconstruction_delta=('decoder_anchor_mse_delta','mean')).reset_index()
a.to_csv(P/'response_summary.csv',index=False)
fig,axes=plt.subplots(2,3,figsize=(12,7))
roles=['reference_h1','record_multihop'];colors=['#797A8E','#B5465D']
for col,ds in enumerate(cases):
    d=s[(s.dataset==ds)&(s.family=='plan_far')]
    for j,role in enumerate(roles):
        for k,kind in enumerate(['source','random_norm_matched']):
            vals=d[(d.model_role==role)&(d.kind==kind)&(d['lambda']==.5)].encoder_response_normalized
            assert len(vals)==3
            x=k+(j-.5)*.3
            axes[0,col].bar(x,vals.mean(),width=.28,color=colors[j],alpha=.85)
            axes[0,col].scatter(x+np.linspace(-.04,.04,3),vals,color='#302D36',s=12,zorder=5)
        for k,lam in enumerate([.25,.5]):
            vals=d[(d.model_role==role)&(d.kind=='source')&(d['lambda']==lam)].decoder_anchor_mse_delta
            x=k+(j-.5)*.3
            axes[1,col].bar(x,vals.mean(),width=.28,color=colors[j],alpha=.85)
            axes[1,col].scatter(x+np.linspace(-.04,.04,3),vals,color='#302D36',s=12,zorder=5)
    axes[0,col].set(title=ds.replace('Quake_Smart-seq2_Lung','Quake Lung'),xticks=[0,1],xticklabels=['Remote source','Norm-matched random'],ylim=(0,.82))
    axes[1,col].set(xticks=[0,1],xticklabels=['λ = 0.25','λ = 0.50'],ylim=(-.015,.29))
    for ax in axes[:,col]:ax.spines[['top','right']].set_visible(False);ax.axhline(0,color='#ACBAC1',lw=.7)
axes[0,0].set_ylabel('Normalized latent displacement')
axes[1,0].set_ylabel('Anchor reconstruction MSE increase')
fig.legend([plt.Rectangle((0,0),1,1,color=c) for c in colors],['H1','Locked multihop'],loc='upper center',ncol=2,frameon=False)
fig.text(.5,.015,'Same anchors and directions for both models; bars = 3-seed means; dots = seeds. Ungated, unmasked evaluation; remote = hop ≥ 2.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.06,1,.94])
for ext in ['png','pdf','svg']:fig.savefig(P/f'response_probe.{ext}',dpi=250)

