"""Posthoc source coverage and aggregation analysis. No training or parameter selection.
Written by parent after Claude service interruption; paired training seeds are the units.
"""
from pathlib import Path
import argparse,csv,json
import numpy as np
import scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score
from sklearn.decomposition import PCA
from common import EXP,R,ARMS,read_json,atomic_json,atomic_npy,atomic_npz,sha256,json_safe
from v0rg_tunable_multihop.input_adapter import load_matrix,encode_labels,load_npz_matrix_only
import factorial_plan as fp

STATES=range(10,30)
COLORS=['#797A8E','#B5465D','#ACBAC1','#E29DAB','#66576D']

def csvwrite(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def labels_and_features(info):
    manifest=read_json(info['tuning_manifest_path'])
    assert sha256(manifest['feature_path'])==manifest['feature_sha256']
    X=load_npz_matrix_only(manifest['feature_path']).X
    errors=[]
    for source in info['posthoc_label_source_paths']:
        try:
            loaded=load_matrix(source)
            if loaded.X.shape!=X.shape:raise ValueError('source matrix shape differs')
            delta=sp.csr_matrix(loaded.X)-sp.csr_matrix(X)
            if delta.nnz and np.max(np.abs(delta.data))>1e-6:raise ValueError('source matrix rows/values differ')
            y,names=encode_labels(loaded.labels)
            y=np.asarray(y).reshape(-1)
            if len(y)!=X.shape[0]:raise ValueError('label length differs')
            return X,y,[str(s) for s in names],manifest,{'source':source,'source_sha256':sha256(source),'feature_sha256':manifest['feature_sha256'],'matrix_row_alignment_verified':True}
        except Exception as exc:errors.append(str(exc))
    raise ValueError('No feature-aligned label source: '+str(errors))

def pair_metrics(y,p):
    _,p=np.unique(p,return_inverse=True);k=int(y.max())+1
    tab=np.zeros((k,int(p.max())+1),dtype=np.int64);np.add.at(tab,(y,p),1)
    sizes=tab.sum(1);clusters=tab.sum(0)
    together=(tab*(tab-1)/2).sum(1);denom=sizes*(sizes-1)/2
    split=denom-together;merge=(clusters*(clusters-1)/2).sum()-together.sum()
    anchor_split=(sizes[y]-tab[y,p])/np.maximum(1,sizes[y]-1)
    anchor_merge=(clusters[p]-tab[y,p])/np.maximum(1,len(y)-sizes[y])
    return tab,split,denom,float(merge),anchor_split,anchor_merge

def readout(z,y,k,out,seed,arm,names):
    out.mkdir(parents=True,exist_ok=True)
    rows=[];classes=[];predictions=[];splits=[];merges=[]
    scale=max(float(np.mean(np.sum((z-z.mean(0))**2,axis=1))),1e-12)
    centers=np.array([z[y==c].mean(0) for c in range(k)])
    for state in STATES:
        km=KMeans(k,n_init=20,random_state=state).fit(z);p=km.labels_
        tab,spl,den,merged,a,b=pair_metrics(y,p)
        rows.append(dict(seed=seed,arm=arm,readout=state,ARI=adjusted_rand_score(y,p),NMI=normalized_mutual_info_score(y,p),inertia=km.inertia_,normalized_inertia=km.inertia_/(len(y)*scale),split_pairs=float(spl.sum()),merge_pairs=merged))
        for c in range(k):
            radius=float(np.mean(np.sum((z[y==c]-centers[c])**2,axis=1))/scale)
            classes.append(dict(seed=seed,arm=arm,readout=state,class_id=c,class_name=names[c],n=int((y==c).sum()),split_rate=float(spl[c]/den[c]) if den[c] else None,normalized_radius=radius))
        predictions.append(p);splits.append(a);merges.append(b)
    preds=np.asarray(predictions,dtype=np.int32)
    atomic_npz(out/'readouts.npz',predictions=preds,anchor_split=np.mean(splits,axis=0),anchor_merge=np.mean(merges,axis=0))
    csvwrite(out/'readouts.csv',rows);csvwrite(out/'classes.csv',classes)
    best=min(rows,key=lambda r:r['inertia'])
    summary=dict(seed=seed,arm=arm,ARI_mean=float(np.mean([r['ARI'] for r in rows])),ARI_sd_readout=float(np.std([r['ARI'] for r in rows],ddof=1)),min_inertia_ARI=best['ARI'],NMI_mean=float(np.mean([r['NMI'] for r in rows])),split_pairs_mean=float(np.mean([r['split_pairs'] for r in rows])),merge_pairs_mean=float(np.mean([r['merge_pairs'] for r in rows])))
    return summary,rows,classes,preds,np.mean(splits,0),np.mean(merges,0)

def communities(indices,weights):
    import networkx as nx
    n=len(indices);r=np.repeat(np.arange(n),indices.shape[1])
    mat=sp.coo_matrix((weights.ravel(),(r,indices.ravel())),shape=(n,n)).tocsr()
    mat=(mat+mat.T)*.5;mat.setdiag(0);mat.eliminate_zeros()
    graph=nx.from_scipy_sparse_array(mat)
    groups=nx.community.louvain_communities(graph,weight='weight',resolution=1.,seed=20260925)
    ids=np.empty(n,dtype=np.int32)
    for i,g in enumerate(sorted(groups,key=min)):ids[list(g)]=i
    return ids

def source_stats(record,graph,comm,y,out,data=None,feature_ids=None):
    """Exact accumulated training weights, or explicitly marked historical MC reconstruction."""
    from run_factorial import TRACE_DTYPE
    from v0rg_tunable_multihop.config import V0_RGConfig
    n=len(y);c=int(comm.max())+1
    near=np.zeros((n,c),dtype=bool)
    near[np.arange(n)[:,None],comm[graph['indices']]]=True
    masses=np.zeros((n,c));total=np.zeros(n);same=np.zeros(n);far=np.zeros(n);novel=np.zeros(n);same_novel=np.zeros(n)
    folder=Path(record['_folder']);path=folder/'source_trace.bin'
    if path.exists():
        trace=np.memmap(path,dtype=TRACE_DTYPE,mode='r');mode='actual_full_training_trace'
        chunks=(trace[i:i+2000000] for i in range(0,len(trace),2000000))
    else:
        import probe_directions as probe
        from v0rg_tunable_multihop.hop_sampling import build_exact_hop_plan
        config=V0_RGConfig(**record['config']);frozen=record['_frozen']
        plan=build_exact_hop_plan(frozen['graph'],frozen['edge_weights'],config.hop_variant,config.mix_neighbors,config.high_hop_probability_multiplier)
        rng=np.random.default_rng(20260925);parts=[]
        for anchor in range(n):
            nodes=frozen['graph'].indices[anchor] if plan is None else plan.indices[anchor]
            probs=frozen['edge_weights'][anchor] if plan is None else plan.probs[anchor]
            probs=np.asarray(probs,dtype=float);probs=probs/probs.sum()
            picks=rng.choice(len(nodes),size=128,p=probs)
            h=np.ones(len(picks),dtype=int) if plan is None else plan.hop_labels[anchor][picks]
            block=np.zeros(len(picks),dtype=TRACE_DTYPE);block['anchor']=anchor;block['source']=nodes[picks];block['hop']=h;block['weight']=1/128
            parts.append(block)
        chunks=[np.concatenate(parts)];mode='MC_candidate_probability_128_draws_per_anchor_not_actual_mixture_weights'
    chosen=np.sort(np.random.default_rng(20260925).choice(n,min(512,n),replace=False));feature_draws=[]
    for block in chunks:
        a=block['anchor'];s=block['source'];w=block['weight'].astype(float);h=block['hop'];sc=comm[s]
        good=y[a]==y[s];new=(h>=2)&~near[a,sc]
        take=np.isin(a,chosen)
        if mode=='actual_full_training_trace':take &= block['epoch']==1
        if take.any():feature_draws.append(block[take].copy())
        np.add.at(masses,(a,sc),w)
        for arr,values in ((total,w),(same,w*good),(far,w*(h>=2)),(novel,w*new),(same_novel,w*new*good)):
            np.add.at(arr,a,values)
    den=np.maximum(total,1e-20);p=masses/den[:,None]
    entropy=-(p*np.log(np.maximum(p,1e-20))).sum(1)
    out.mkdir(parents=True,exist_ok=True)
    atomic_npz(out/'source_coverage.npz',same_mass=same/den,far_mass=far/den,new_community_mass=novel/den,same_new_community_mass=same_novel/den,community_entropy=entropy,source_community_mass=p)
    if data is not None and feature_draws:
        sampled=np.concatenate(feature_draws);aa=sampled['anchor'];ss=sampled['source'];hh=sampled['hop'];ww=sampled['weight'].astype(float)
        atomic_npz(out/'sampled_source_pairs.npz',anchor=aa,source=ss,hop=hh,weight=ww,community_anchor=comm[aa],community_source=comm[ss])
        feature_rows=[]
        for klass in np.unique(y[aa]):
            for group in ['H1','far_same_class','far_other_class']:
                take=(y[aa]==klass)&((hh==1) if group=='H1' else ((hh>=2)&((y[aa]==y[ss]) if group=='far_same_class' else (y[aa]!=y[ss]))))
                if not take.any():continue
                delta=np.average(data[ss[take]]-data[aa[take]],axis=0,weights=ww[take])
                for f in np.argsort(-np.abs(delta))[:20]:feature_rows.append(dict(class_id=int(klass),source_group=group,selected_column=int(f),original_column=int(feature_ids[f]),standardized_source_minus_anchor=float(delta[f]),sampled_draws=int(take.sum()),sampling='fixed512anchors_first_epoch' if mode.startswith('actual') else 'fixed512anchors_MC'))
        csvwrite(out/'source_feature_contrasts.csv',feature_rows)
    return dict(mode=mode,same_mass=float(np.mean(same/den)),far_mass=float(np.mean(far/den)),new_community_mass=float(np.mean(novel/den)),same_new_community_mass=float(np.mean(same_novel/den))),same_novel/den

def figures(out,summaries,classes,layout,y):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4))
    arms=list(dict.fromkeys(r['arm'] for r in summaries))
    for i,arm in enumerate(arms):
        vals=[r['ARI_mean'] for r in summaries if r['arm']==arm]
        ax.scatter(np.full(len(vals),i),vals,c=COLORS[i%len(COLORS)],s=24)
        ax.plot([i-.2,i+.2],[np.mean(vals)]*2,c=COLORS[i%len(COLORS)],lw=2)
    ax.set_xticks(range(len(arms)),arms,rotation=20,ha='right');ax.set_ylabel('ARI averaged over 20 readouts');fig.tight_layout();fig.savefig(out/'paired_ARI.png',dpi=180);fig.savefig(out/'paired_ARI.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5));ax.scatter(layout[:,0],layout[:,1],c=y,cmap='tab20',s=2,rasterized=True);ax.set_title('Fixed input PCA layout; posthoc classes');ax.set_xlabel('PC1');ax.set_ylabel('PC2');fig.tight_layout();fig.savefig(out/'fixed_layout.png',dpi=180);plt.close(fig)

def analyze(args):
    lock=read_json(args.lock);info=lock['datasets'][args.dataset]
    X,y,names,manifest,alignment=labels_and_features(info)
    from prepare_factorial import resolve_preprocessing
    from v0rg_tunable_multihop.tuning import SplitPreprocessor
    from v0rg_tunable_multihop.config import V0_RGConfig
    prepinfo=resolve_preprocessing(info,True);prep=SplitPreprocessor(prepinfo['input_kind'],prepinfo['feature_limit']).fit(X);data=prep.transform(X)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if args.command=='historical':
        records=[]
        for arm,key in [('H1_anchor','selected_h1'),('flat_anchor','selected_multihop')]:
            for seed,item in info[key]['per_seed'].items():
                path=Path(item['record_path']);assert sha256(path)==item['record_sha256']
                r=read_json(path);r.update(arm=arm,_folder=str(path.parent),_embedding=item['embedding_path'],_embedding_sha256=item['embedding_sha256']);records.append(r)
    else:
        records=[]
        for path in sorted((EXP/'outputs/train'/args.dataset).glob('*/seed_*/record.json')):
            r=read_json(path)
            if r['status']=='completed':r.update(_folder=str(path.parent),_embedding=str(path.parent/'embedding.npy'),_embedding_sha256=r['artifact_sha256']['embedding.npy']);records.append(r)
        if args.command=='baseline':
            records=[]
            for path in sorted((EXP/'outputs/schntl'/args.dataset).glob('seed_*/record.json')):
                r=read_json(path);r.update(_folder=str(path.parent),_embedding=str(path.parent/'embedding.npy'),_embedding_sha256=r['artifact_sha256']['embedding.npy']);records.append(r)
    if not records:
        atomic_json(out/'summary.json',{'status':'partial','error':'no complete records'});return
    for r in records:
        assert r['test_labels_loaded'] is False and r['labels_used_during_fit'] is False
        assert r['feature_sha256']==manifest['feature_sha256']
    first=next((r for r in records if r['arm']=='flat_anchor'),records[0])
    graph=None;comm=None
    if args.command!='baseline':
        frozen=fp.build_frozen_graph(data,V0_RGConfig(**first['config']),int(first['seed']))
        graph={'indices':frozen['graph'].indices,'edge_weights':frozen['edge_weights']};comm=communities(graph['indices'],graph['edge_weights'])
        atomic_npy(out/'community_ids.npy',comm)
    layout=PCA(2,random_state=20260925).fit_transform(data);atomic_npy(out/'layout.npy',layout);atomic_npy(out/'labels_true.npy',y)
    sums=[];allrows=[];allclasses=[];cache={};coverage=[]
    for r in records:
        seed=int(r['seed']);arm=r['arm'];dest=out/f'{arm}_seed_{seed}'
        ep=Path(r['_embedding']);assert sha256(ep)==r['_embedding_sha256']
        if ep.suffix=='.npz':
            with np.load(ep,allow_pickle=False) as packed:z=packed['embedding']
        else:z=np.load(ep,allow_pickle=False)
        assert len(z)==len(y) and np.isfinite(z).all()
        summary,rows,cls,pred,a,b=readout(z,y,int(manifest['n_clusters']),dest,seed,arm,names)
        sums.append(summary);allrows.extend(rows);allclasses.extend(cls);cache[(seed,arm)]=(pred,a,b)
        if arm=='flat_anchor':
            r['_frozen']=fp.build_frozen_graph(data,V0_RGConfig(**r['config']),seed)
            # Communities remain fixed; near coverage uses this seed's own graph.
            own={'indices':r['_frozen']['graph'].indices,'edge_weights':r['_frozen']['edge_weights']}
            st,cov=source_stats(r,own,comm,y,dest,data,prep.features);coverage.append(dict(seed=seed,arm=arm,**st));cache[(seed,'coverage')]=cov
    contrasts=[];anchors=[];coassign=[]
    for seed in sorted({r['seed'] for r in sums}):
        vals={r['arm']:r['ARI_mean'] for r in sums if r['seed']==seed}
        for name,terms in [('multihop_minus_H1',{'flat_anchor':1,'H1_anchor':-1}),('prior',{'flat_anchor':1,'geometric_anchor':-1}),('target',{'flat_anchor':1,'flat_mixed':-1}),('interaction',{'flat_anchor':1,'geometric_anchor':-1,'flat_mixed':-1,'geometric_mixed':1})]:
            if all(t in vals for t in terms):contrasts.append(dict(seed=seed,contrast=name,delta_ARI=sum(vals[t]*w for t,w in terms.items())))
        if (seed,'H1_anchor') in cache and (seed,'flat_anchor') in cache:
            hp,hs,hm=cache[(seed,'H1_anchor')];mp,ms,mm=cache[(seed,'flat_anchor')];cov=cache.get((seed,'coverage'),np.zeros(len(y)))
            for i in range(len(y)):anchors.append(dict(seed=seed,anchor=i,class_id=int(y[i]),class_name=names[y[i]],community=int(comm[i]),same_new_community_mass=float(cov[i]),split_improvement=float(hs[i]-ms[i]),merge_improvement=float(hm[i]-mm[i]),h1_split=float(hs[i]),class_frequency=float(np.mean(y==y[i]))))
            rng=np.random.default_rng(20260925)
            for c in range(len(names)):
                ids=np.flatnonzero(y==c);aa=rng.choice(ids,20000);bb=rng.choice(ids,20000);keep=(aa!=bb)&(comm[aa]!=comm[bb]);aa=aa[keep];bb=bb[keep]
                if len(aa):coassign.append(dict(seed=seed,class_name=names[c],n_pairs=len(aa),H1_coassignment=float(np.mean(hp[:,aa]==hp[:,bb])),multihop_coassignment=float(np.mean(mp[:,aa]==mp[:,bb]))))
    csvwrite(out/'per_seed.csv',sums);csvwrite(out/'all_readouts.csv',allrows);csvwrite(out/'all_classes.csv',allclasses);csvwrite(out/'contrasts.csv',contrasts);csvwrite(out/'anchor_correspondence.csv',anchors);csvwrite(out/'source_coverage.csv',coverage);csvwrite(out/'cross_community_coassignment.csv',coassign)
    effects={}
    for name in sorted({r['contrast'] for r in contrasts}):
        vals=np.array([r['delta_ARI'] for r in contrasts if r['contrast']==name]);effects[name]={'mean':float(vals.mean()),'sd_training_seeds':float(vals.std(ddof=1)) if len(vals)>1 else None,'positive':int((vals>0).sum()),'n_training_seeds':len(vals),'per_seed':vals.tolist()}
    figures(out,sums,allclasses,layout,y)
    expected=6 if args.command=='historical' else 5 if args.command=='baseline' else info.get('expected_formal_records',25)
    atomic_json(out/'summary.json',json_safe({'status':'completed' if len(records)==expected else 'partial','dataset':args.dataset,'mode':args.command,'n_records':len(records),'expected':expected,'effects':effects,'alignment':alignment,'labels_used_for_model_fit':False,'posthoc_full_x_labels_loaded':True,'selection_performed':False,'source_coverage':coverage,'community_policy':'Louvain resolution1 seed20260925 fixed reference graph per dataset','readout_policy':'20 states10..29 n_init20; mean primary; minimum inertia secondary','limitations':['new seeds are optimization replicates, not new datasets','source coverage correlations are posthoc associations','historical source draws are MC candidate probabilities, not true training mixture mass','donor and gene-program semantics require separate aligned metadata analysis'] }))

def summarize(args):
    rows=[]
    for parent in ['factorial','p0_cases','baseline']:
        for p in (EXP/'outputs'/parent).glob('*/summary.json'):
            s=read_json(p);rows.append({'path':str(p),'status':s['status'],'dataset':s.get('dataset'),'effects':s.get('effects',{})})
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'summary.json',{'status':'completed','step_results':rows,'scope':'available step summaries; pipeline marker determines completeness'})
    lines=['# 实验结果自动汇总（待父代理阅读图表后形成最终分析）','', '各结果均按训练seed配对，全部保留；均值不替代逐类与来源分析。','']
    for r in rows:lines.extend([f"## {r['dataset']} / {Path(r['path']).parent.parent.name}",json.dumps(r['effects'],ensure_ascii=False,indent=2),''])
    (out/'RESULTS_SUMMARY.md').write_text('\n'.join(lines),encoding='utf-8')

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['historical','factorial','baseline','summarize']);p.add_argument('--dataset');p.add_argument('--lock',default=str(EXP/'outputs/p0/selection_lock_v2.json'));p.add_argument('--out',required=True);a=p.parse_args()
    if a.command=='summarize':summarize(a)
    else:analyze(a)
if __name__=='__main__':main()
