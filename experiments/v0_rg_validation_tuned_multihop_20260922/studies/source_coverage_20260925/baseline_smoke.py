import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import numpy as np
import run_schntl as s
from common import EXP,atomic_json,atomic_npz,sha256
root=EXP/'outputs/adapter_smoke';root.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(123);X=rng.poisson(3,(40,12)).astype(float)
atomic_npz(root/'X.npz',X=X);atomic_npz(root/'prep.npz',features=np.arange(12));atomic_json(root/'prep.json',{'fingerprint':'synthetic_smoke','input_kind':'raw_count','width':12})
task={'id':'baseline_adapter_smoke','dataset':'synthetic','seed':11,'algorithm':'scHNTL_official','feature_path':str(root/'X.npz'),'feature_sha256':sha256(root/'X.npz'),'n_clusters':3,'K_provenance':'benchmark_oracle_from_y','input_kind':'raw_count','locked_preprocessor_npz':str(root/'prep.npz'),'locked_preprocessor_json':str(root/'prep.json'),'locked_preprocessor_sha256':sha256(root/'prep.npz'),'expected_preprocessing_hash':'synthetic_smoke'}
atomic_json(root/'task.json',task)
s.PRE_EPOCHS=1;s.EPOCHS=1
result=s.run_task(root/'task.json',root/'run',4)
result.update(smoke=True,counts_as_formal=False,epochs_override=1);atomic_json(root/'run/record.json',result)
print('ADAPTER_SMOKE_COMPLETED')
