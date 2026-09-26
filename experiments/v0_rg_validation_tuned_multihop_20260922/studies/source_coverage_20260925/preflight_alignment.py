import json
from common import EXP,read_json,atomic_json
from analyze_cases import labels_and_features
rows=[]
for lp,ds in [(EXP/'outputs/p0/selection_lock_v2.json',['Baron','worm_neuron_cell','student_grade']), (EXP/'prereg/transfer_lock.json',['Pollen','Wang','hrvatin_geo'])]:
 lock=read_json(lp)
 for d in ds:
  X,y,n,m,a=labels_and_features(lock['datasets'][d]);rows.append({'dataset':d,'shape':list(X.shape),'classes':len(n),'matrix_row_alignment_verified':a['matrix_row_alignment_verified']});print(rows[-1],flush=True)
atomic_json(EXP/'review/label_alignment_preflight.json',{'status':'completed','posthoc_full_labels_loaded':True,'labels_used_for_model_fit':False,'rows':rows})
