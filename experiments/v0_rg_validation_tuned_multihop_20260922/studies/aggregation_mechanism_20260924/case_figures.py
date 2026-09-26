"""Four-panel descriptive dossiers for fixed all-seed selected examples."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

def draw(root: Path,dataset: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result=root/'results/examples'/dataset
    selection=json.loads((result/'selection_provenance.json').read_text())
    chosen=selection['selected_top2']
    s=root/'outputs/source'/dataset/'seed_42'
    xy=np.load(s/'original_layout.npy');y=np.load(s/'labels_true.npy');fr=np.load(s/'fragment_ids.npy')
    links=pd.read_csv(s/'anchor_source_links.csv')
    for ix,item in enumerate(chosen):
        isclass=selection['selection_kind']=='class'
        ids=[int(item['class_id'])] if isclass else [int(item['class_a_id']),int(item['class_b_id'])]
        fig,axs=plt.subplots(2,2,figsize=(10,8));aa,bb,cc,dd=axs.ravel()
        background=np.sort(np.random.default_rng(1234).choice(len(y),min(2500,len(y)),replace=False))
        aa.scatter(xy[background,0],xy[background,1],s=2,color='#ACBAC1',alpha=.2)
        nodes=np.flatnonzero(np.isin(y,ids))
        aa.scatter(xy[nodes,0],xy[nodes,1],c=fr[nodes],cmap='tab20',s=4,alpha=.5)
        subset=links[(links.anchor_class_id.isin(ids))&(links.hop>=5)&(links.mc_actual_weight>0)]
        same=subset[(subset.same_class==1)&(subset.same_fragment==0)]
        edges=same.sort_values(['mc_actual_weight','anchor_index','source_index'],ascending=[False,True,True]).head(40)
        for e in edges.itertuples():
            a,b=int(e.anchor_index),int(e.source_index)
            aa.plot(xy[[a,b],0],xy[[a,b],1],color='#B5465D',lw=.4,alpha=.4)
        aa.set_title('A  Original PCA; colors: graph fragments\nseed 42, strongest 40 cross-fragment same-class links',fontsize=9)
        aa.set_xlabel('Original PCA 1');aa.set_ylabel('Original PCA 2')
        cover=[];geometry=[];repeats=[]
        for seed in (42,123,7):
            sp=root/'outputs/source'/dataset/f'seed_{seed}'
            p=pd.read_csv(sp/'source_per_anchor.csv');p=p[p.anchor_class_id.isin(ids)].copy();p['seed']=seed;cover.append(p)
            gp=root/'outputs/geometry'/dataset/f'seed_{seed}'
            g=pd.read_csv(gp/('per_class.csv' if isclass else 'per_class_pair.csv'))
            g=g[g.class_id.isin(ids)] if isclass else g[(g.class_a_id==ids[0])&(g.class_b_id==ids[1])]
            geometry.append(g)
            rp=root/'outputs/readout'/dataset/f'seed_{seed}'
            r=pd.read_csv(rp/('class_split_rates.csv' if isclass else 'pair_merge_rates.csv'))
            r=r[r.class_id.isin(ids)] if isclass else r[(r.class_a_id==ids[0])&(r.class_b_id==ids[1])]
            r['seed']=seed;repeats.append(r)
        cover=pd.concat(cover);geometry=pd.concat(geometry);repeats=pd.concat(repeats)
        # Each seed has equal weight; zero mass anchors remain in denominator.
        mass=cover.groupby(['seed','hop_group']).same_class_cross_fragment_mc_mass.mean().groupby('hop_group').mean().reindex(['H1','H2_4','H5_8'])
        bb.bar(range(3),mass,color=['#797A8E','#ACBAC1','#B5465D']);bb.set_xticks(range(3),['H1','H2–4','H5–8'])
        bb.set_title('B  Same-class cross-fragment source mass');bb.set_ylabel('Mean MC mixture mass')
        metric='compactness_over_global_spread' if isclass else 'centroid_distance_over_pooled_scatter'
        for seed,g in geometry.groupby('seed'):
            vals=g.set_index('arm')[metric].reindex(['h1','h8'])
            cc.plot([0,1],vals,marker='o',lw=1,label=str(seed))
        cc.set_xticks([0,1],['H1','H8']);cc.set_title('C  Class radius / global radius' if isclass else 'C  Class distance / pooled scatter');cc.legend(title='Train seed',fontsize=8)
        rate='split_rate' if isclass else 'merge_rate'
        for arm,x,c in [('h1',0,'#797A8E'),('h8',1,'#B5465D')]:
            vals=repeats[repeats.arm==arm][rate].to_numpy()
            dd.scatter(x+np.linspace(-.12,.12,len(vals)),vals,color=c,s=8,alpha=.5)
        dd.set_xticks([0,1],['H1','H8']);dd.set_title('D  Split rate across 3 seeds × 20 readouts' if isclass else 'D  Merge rate across 3 seeds × 20 readouts');dd.set_ylabel('Error rate; lower is better')
        fig.suptitle(f'{dataset}: exploratory class IDs {ids}; all-seed selection',fontsize=11);fig.tight_layout()
        stem=result/f'case_{ix+1}'
        fig.savefig(str(stem)+'.png',dpi=180);fig.savefig(str(stem)+'.pdf');plt.close(fig)
        cover.to_csv(str(stem)+'_coverage.csv',index=False);geometry.to_csv(str(stem)+'_geometry.csv',index=False);repeats.to_csv(str(stem)+'_readouts.csv',index=False);edges.to_csv(str(stem)+'_shown_edges.csv',index=False)
