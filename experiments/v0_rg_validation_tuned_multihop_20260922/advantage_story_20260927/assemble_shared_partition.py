"""Parent assembly using the audited Claude-authored E4 computation."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from compute_e3_e4_analysis import compute_e4_analysis
P=Path(__file__).resolve().parent
assets=P.parent/'visual_argument_rebuild_20260926/case_assets/Baron'
partition=P/'inputs/Baron_trace_partition';out=P/'outputs/E5_shared_partition'
digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
y=np.load(assets/'fixed_input_layout.npz')['labels_true']
assert np.array_equal(y,np.load(partition/'labels_true.npy'))
inputs=[assets/'fixed_input_layout.npz',assets/'frozen_readouts_10.npz',partition/'community_ids.npy',
        partition/'labels_true.npy',P/'outputs/E5_trace/per_class_five_seed.csv']
hashes={str(p):digest(p) for p in inputs}
compute_e4_analysis('Baron',inputs[0],inputs[1],inputs[2],out)
pairs=pd.read_csv(out/'e4_community_pair_coassignment.csv')
src=pd.read_csv(inputs[-1]);rows=[]
for c,g in pairs.groupby('class_id'):
    g=g[g.comm_a!=g.comm_b]
    if g.empty:continue
    s=src[(src.class_id==c)&(src.category=='same_class_cross_community_new')].iloc[0]
    rows.append(dict(class_id=int(c),class_name=g.iloc[0].class_name,class_n=int(s.class_n),
        sameclass_new_source_mass=float(s.five_seed_mean_mass),
        cross_H1=np.average(g.H1_coassign_mean,weights=g.support),
        cross_multihop=np.average(g.multihop_coassign_mean,weights=g.support),
        cross_delta=np.average(g.change,weights=g.support)))
pd.DataFrame(rows).to_csv(out/'class_source_relation.csv',index=False)
assert hashes=={str(p):digest(p) for p in inputs}
record=dict(status='completed',dataset='Baron',posthoc_full_x_labels_loaded=True,
    labels_used_for_model_fit=False,new_encoder_training=False,label_alignment='exact_rowwise_equality',
    source_training_seeds=[1019,2027,3037,4051,5059],readout_training_seeds=[7,42,123],
    interpretation='descriptive_class_correspondence_across_independent_seed_groups_not_paired_causal_effect',
    input_sha256=hashes,code_sha256={p.name:digest(p) for p in [Path(__file__),P/'compute_e3_e4_analysis.py']},
    artifact_sha256={p.name:digest(p) for p in out.iterdir() if p.suffix in ['csv','.csv','.png','.pdf','.svg']})
tmp=out/'record.tmp';tmp.write_text(json.dumps(record,indent=2),encoding='utf-8');tmp.replace(out/'record.json')
print('Shared partition record and all hashes saved.')
