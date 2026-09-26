from pathlib import Path
import numpy as np
from sklearn.metrics import adjusted_rand_score
from analyze_cases import pair_metrics,communities,labels_and_features
from common import read_json,EXP

def test_pair_metrics_exact_enumeration():
    y=np.array([0,0,0,1,1,2]);p=np.array([0,1,0,1,1,0]);tab,spl,den,merge,aa,bb=pair_metrics(y,p)
    assert spl.sum()==sum(y[i]==y[j] and p[i]!=p[j] for i in range(len(y)) for j in range(i))
    assert merge==sum(y[i]!=y[j] and p[i]==p[j] for i in range(len(y)) for j in range(i))
    assert aa[0]==.5 and bb[0]==1/3

def test_communities_seeded():
    ix=np.array([[1,2],[0,2],[0,1],[4,5],[3,5],[3,4]])
    a=communities(ix,np.ones_like(ix,dtype=float))
    assert np.all(a[:3]==a[0]) and np.all(a[3:]==a[3]) and a[0]!=a[3]
    assert np.array_equal(a,communities(ix,np.ones_like(ix,dtype=float)))

def test_new_trace_schema(tmp_path):
    from run_factorial import TRACE_DTYPE
    from probe_directions import trace_donors
    arr=np.zeros(4,dtype=TRACE_DTYPE);arr['anchor']=[0,0,1,1];arr['source']=[2,3,4,5];arr['hop']=[1,2,3,4]
    p=tmp_path/'trace.bin';arr.tofile(p)
    donor,hop,counts=trace_donors(p,np.array([0,1,9]),8,trace_dtype=TRACE_DTYPE.descr)
    assert counts.tolist()==[2,2,0] and donor[2]==-1
    assert donor[0] in [2,3] and donor[1] in [4,5]
