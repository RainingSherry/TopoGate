"""Parent assembly with the audited E4 community-pair statistic."""
from pathlib import Path
import numpy as np
import pandas as pd
from compute_e3_e4_analysis import compute_community_coassignment
P=Path(__file__).resolve().parent;assets=P.parent/'visual_argument_rebuild_20260926/case_assets';rows=[]
for ds,hop in [('Baron','H8'),('Quake_Smart-seq2_Lung','H8'),('student_grade','H4')]:
    y=np.load(assets/ds/'fixed_input_layout.npz')['labels_true'];g=np.load(assets/ds/'community_ids.npy')
    pred=np.load(P/'outputs/E2_E6'/ds/'readout_predictions.npz')
    n={(int(c),int(k)):int(((y==c)&(g==k)).sum()) for c in np.unique(y) for k in np.unique(g)}
    for seed in [7,42,123]:
        for arm in ['H1',hop,'Kwide']:
            vals=[]
            for arr in pred[f'{arm}_seed{seed}']:
                co=compute_community_coassignment(y,g,arr)
                pairs=[(v,n[(c,a)]*n[(c,b)]) for (c,a,b),v in co.items() if a!=b]
                vals.append(np.average([x[0] for x in pairs],weights=[x[1] for x in pairs]))
            assert len(vals)==10
            rows.append(dict(dataset=ds,strategy=arm,seed=seed,cross_coassignment=float(np.mean(vals))))
d=pd.DataFrame(rows);d.to_csv(P/'outputs/E2_E6/cross_coassignment_per_seed.csv',index=False)
d.groupby(['dataset','strategy']).cross_coassignment.agg(['mean','std']).reset_index().to_csv(P/'outputs/E2_E6/cross_coassignment_summary.csv',index=False)
