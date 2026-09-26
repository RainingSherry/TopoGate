"""Parent-run functional checks, including a tiny CPU fit for patched loss paths."""
from contextlib import ExitStack
from pathlib import Path
import importlib
import inspect
import json
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT.parent),'/home/luolie/ToPoGate']

class MonkeyPatch:
    def __init__(self):self.stack=ExitStack()
    def setattr(self,obj,key,value):self.stack.enter_context(patch.object(obj,key,value))

def main():
    passed=[]
    for name in ['tests_step1','tests_steps234','tests_interventions']:
        module=importlib.import_module(name)
        if hasattr(module,'CORE_PARENT'):module.CORE_PARENT=ROOT.parent
        for key,func in list(vars(module).items()):
            if not key.startswith('test_') or not callable(func):continue
            mp=MonkeyPatch()
            try:
                args={'monkeypatch':mp} if 'monkeypatch' in inspect.signature(func).parameters else {}
                func(**args)
            finally:mp.stack.close()
            passed.append(name+'.'+key);print('PASS',passed[-1],flush=True)
    from v0rg_tunable_multihop.config import V0_RGConfig
    from v0rg_tunable_multihop import trainer
    from step5_interventions import TraceCollector,_patched_training
    X=np.random.default_rng(12).normal(size=(36,12)).astype(np.float32)
    for arm in ['h1_matched_amplitude','mixed_target','replace_far_le4']:
        cfg=V0_RGConfig(hop_variant='H1' if arm.startswith('h1') else 'H8_B1',
                       epochs=2,hidden_size=8,batch_size=16,neighbor_k=2,
                       knn_pca_dim=4,mix_neighbors=2)
        with TemporaryDirectory(prefix='topogate_mechanism_unit_') as temp:
            c=TraceCollector(Path(temp)/'trace.bin',len(X));c.config=cfg
            c.norm_match_reference=arm=='replace_far_le4'
            with _patched_training(arm,42,np.count_nonzero(X,axis=1),c):
                p,z,diag=trainer.fit_predict(X,fit_X=X,n_clusters=3,config=cfg,seed=42,device='cpu')
            audit=c.finish()
            assert len(p)==len(X) and len(z)==len(X) and np.isfinite(z).all()
            assert audit['draws']>0
        passed.append('tiny_cpu_fit.'+arm);print('PASS',passed[-1],flush=True)
    (ROOT/'CHECK_RESULTS.json').write_text(json.dumps({'status':'passed','checks':passed},indent=2))

if __name__=='__main__':main()
