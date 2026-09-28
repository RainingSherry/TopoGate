"""Synthetic Linux controller smoke: no real training or dataset access."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import tempfile

import controller


DUMMY = r'''import argparse, hashlib, json, os, sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--mode',required=True)
p.add_argument('--device');a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
attempt=out.name
if a.mode=='fail':
    print('deterministic synthetic failure',flush=True);sys.exit(3)
if a.mode=='flaky' and attempt=='attempt_001':
    print('OOM synthetic transient failure',flush=True);sys.exit(3)
payload=out/'payload.txt';payload.write_text(os.environ.get('CUDA_VISIBLE_DEVICES','missing'))
sha=hashlib.sha256(payload.read_bytes()).hexdigest()
(out/'record.json').write_text(json.dumps({'status':'completed','artifact_sha256':{'payload.txt':sha}}))
'''


def test_dependency_validation_and_completion(tmp: Path):
    queue = {"tasks": [
        {"id": "a", "kind": "cpu", "command": ["dummy_worker.py", "--out", "outputs/a", "--mode", "ok"], "out": "outputs/a"},
        {"id": "b", "kind": "cpu", "command": ["dummy_worker.py", "--out", "outputs/b", "--mode", "ok"], "out": "outputs/b", "depends_on": ["a"]},
    ]}
    controller.validate_queue(queue)
    state={"tasks":{"a":{"status":"failed"}}}
    assert controller.dependencies_terminal(queue["tasks"][1],state)
    state["tasks"]["a"]["status"]="failed_transient"
    assert not controller.dependencies_terminal(queue["tasks"][1],state)
    bad=json.loads(json.dumps(queue));bad["tasks"][0]["depends_on"]=["b"]
    try: controller.validate_queue(bad)
    except ValueError: pass
    else: raise AssertionError("dependency cycle accepted")
    bad=json.loads(json.dumps(queue));bad["tasks"][0]["kind"]="gpu"
    bad["tasks"][0]["command"].extend(["--device","cuda:0"])
    try: controller.validate_queue(bad)
    except ValueError: pass
    else: raise AssertionError("non-fit worker accepted as GPU task")
    out=tmp/'empty_record';out.mkdir();(out/'record.json').write_text('{"status":"completed","artifact_sha256":{}}')
    assert not controller.task_completed(queue["tasks"][0],out)[0]
    payload=out/'payload.txt';payload.write_text('x')
    base={"status":"completed","artifact_sha256":{"payload.txt":controller.sha256(payload)}}
    (out/'record.json').write_text(json.dumps(base))
    fit={"command":["fit_worker.py"]}
    assert controller.task_completed(fit,out)[1]=="fit label/core provenance mismatch"
    cohort={"command":["cohort_worker.py"]}
    assert controller.task_completed(cohort,out)[1]=="cohort label/training provenance mismatch"
    print('PASS dependency terminal rule, cycle rejection, empty hash and label provenance rejection')


def test_gpu_pick_authorization():
    old=controller.nvidia_smi_memory
    controller.nvidia_smi_memory=lambda:{2:{"free_mb":18000},4:{"free_mb":13000}}
    try:
        resources={"gpus":[4],"max_workers_per_gpu":16,"initial_workers_per_gpu":4,
                   "start_free_mb":6000,"reserve_mb_per_task":4000}
        assert controller.pick_gpu({"tasks":{}},resources)==4
        assert controller.pick_gpu({"tasks":{}},resources,{4:1},{} ) is None
        controller.nvidia_smi_memory=lambda:{2:{"free_mb":100000}}
        resources["gpus"]=[2]
        state={"tasks":{}}
        baseline={}
        admissions={}
        for i in range(4):
            assert controller.pick_gpu(state,resources,admissions,baseline)==2
            state["tasks"][f"fit{i}"]={"status":"running","kind":"gpu","gpu":2}
            admissions[2]=i+1
        assert controller.pick_gpu(state,resources,admissions,baseline) is None
        assert controller.pick_gpu(state,resources,{}, {2:4})==2
        assert controller.pick_gpu(state,resources,{2:1}, {2:4}) is None
    finally:
        controller.nvidia_smi_memory=old
    print('PASS GPU allow-list, initial-four cap, one-extra cap, memory reservation')


def test_full_controller(tmp: Path):
    shutil.copyfile(controller.__file__,tmp/'controller.py')
    (tmp/'dummy_worker.py').write_text(DUMMY)
    tasks=[
        {"id":"base_fail","kind":"cpu","command":["dummy_worker.py","--out","outputs/base_fail","--mode","fail"],"out":"outputs/base_fail"},
        {"id":"after_failure","kind":"cpu","command":["dummy_worker.py","--out","outputs/after_failure","--mode","ok"],"out":"outputs/after_failure","depends_on":["base_fail"]},
        {"id":"flaky","kind":"cpu","command":["dummy_worker.py","--out","outputs/flaky","--mode","flaky"],"out":"outputs/flaky"},
        {"id":"after_flaky","kind":"cpu","command":["dummy_worker.py","--out","outputs/after_flaky","--mode","ok"],"out":"outputs/after_flaky","depends_on":["flaky"]},
    ]
    controller.atomic_json(tmp/'queue.json',{"tasks":tasks})
    controller.atomic_json(tmp/'resources.json',{"gpus":[2,4],"cpu_workers":2,"cpu_threads_per_task":2,
        "max_workers_per_gpu":16,"initial_workers_per_gpu":4,"start_free_mb":6000,
        "reserve_mb_per_task":4000,"min_host_available_mb":16000,"reserve_host_mb_per_task":1,
        "poll_seconds":.1,"python":sys.executable})
    (tmp/'REUSE_MANIFEST.json').write_text('{}')
    (tmp/'COHORT_LOCK_SHA256.txt').write_text('synthetic-lock')
    controller.atomic_json(tmp/'REVIEW_APPROVED.json',{"status":"approved",
        "queue_sha256":controller.sha256(tmp/'queue.json'),
        "code_sha256":{name:controller.sha256(tmp/name) for name in ('controller.py','dummy_worker.py')},
        "core_sha256":{str(tmp/'dummy_worker.py'):controller.sha256(tmp/'dummy_worker.py')},
        "input_sha256":{name:controller.sha256(tmp/name) for name in
                        ('REUSE_MANIFEST.json','COHORT_LOCK_SHA256.txt','resources.json')}})
    old=(controller.HERE,controller.STATE_DIR,controller.LOGS_DIR,
         controller.nvidia_smi_memory,controller.host_available_mb)
    controller.HERE=tmp;controller.STATE_DIR=tmp/'controller_state';controller.LOGS_DIR=tmp/'logs'
    controller.nvidia_smi_memory=lambda:{2:{"free_mb":16000},4:{"free_mb":0}}
    controller.host_available_mb=lambda:100000
    try:
        controller.run_controller()
        state=controller.read_json(tmp/'controller_state/controller_state.json')['tasks']
        assert state['base_fail']['status']=='failed'
        assert state['after_failure']['status']=='completed'
        assert state['flaky']['status']=='completed' and state['flaky']['attempts']==2
        assert state['after_flaky']['status']=='completed'
        assert (tmp/'outputs/flaky/attempt_001').is_dir() and (tmp/'outputs/flaky/attempt_002/record.json').exists()
        assert controller.read_json(tmp/'PIPELINE_STATUS.json')['status']=='PIPELINE_PARTIAL'
        events=[json.loads(line) for line in (tmp/'logs/events.jsonl').read_text().splitlines()]
        done={e['task_id']:i for i,e in enumerate(events) if e['event']=='task_done'}
        start={e['task_id']:i for i,e in enumerate(events) if e['event']=='task_start'}
        assert done['base_fail']<start['after_failure']
        assert done['flaky']<start['after_flaky']
        gpu_task={"id":"gpu_env","kind":"gpu","command":["dummy_worker.py","--out","outputs/gpu_env","--mode","ok","--device","cuda:0"],"out":"outputs/gpu_env"}
        direct=controller.run_task_worker('gpu_env',{'tasks':[gpu_task]},{'python':sys.executable,'cpu_threads_per_task':2},gpu_id=2)
        assert direct['status']=='completed'
        assert (tmp/'outputs/gpu_env/attempt_001/payload.txt').read_text()=='2'
        print('PASS controller singleton queue, terminal dependencies, retry once, attempt preservation, GPU2 worker visibility')
    finally:
        (controller.HERE,controller.STATE_DIR,controller.LOGS_DIR,
         controller.nvidia_smi_memory,controller.host_available_mb)=old


def test_fit_input_pin(tmp: Path):
    (tmp/'fit_worker.py').write_text(DUMMY)
    (tmp/'tasks').mkdir()
    (tmp/'tasks/a.json').write_text('{"seed": 7}')
    fit={"id":"a","kind":"gpu","command":["fit_worker.py","--task","tasks/a.json",
         "--out","outputs/train/a","--device","cuda:0"],"out":"outputs/train/a"}
    controller.atomic_json(tmp/'queue.json',{"tasks":[fit]})
    controller.atomic_json(tmp/'REVIEW_APPROVED.json',{"status":"approved",
        "queue_sha256":controller.sha256(tmp/'queue.json'),
        "code_sha256":{name:controller.sha256(tmp/name) for name in ('controller.py','fit_worker.py')},
        "core_sha256":{str(tmp/'fit_worker.py'):controller.sha256(tmp/'fit_worker.py')},
        "input_sha256":{name:controller.sha256(tmp/name) for name in
            ('REUSE_MANIFEST.json','COHORT_LOCK_SHA256.txt','resources.json','tasks/a.json')}})
    old=controller.HERE
    controller.HERE=tmp
    try:
        controller.verify_approval(controller.load_queue())
        (tmp/'tasks/a.json').write_text('{"seed": 42}')
        try: controller.verify_approval(controller.load_queue())
        except RuntimeError as e: assert 'input hash differs: tasks/a.json' in str(e)
        else: raise AssertionError('mutated fit task was accepted')
    finally:
        controller.HERE=old
    print('PASS frozen task JSON approval pin rejects mutation')


if __name__=='__main__':
    with tempfile.TemporaryDirectory(prefix='topogate-controller-review-') as name:
        tmp=Path(name)
        old=controller.HERE
        controller.HERE=tmp
        try: test_dependency_validation_and_completion(tmp)
        finally: controller.HERE=old
        test_gpu_pick_authorization()
        test_full_controller(tmp)
        test_fit_input_pin(tmp)
    print('ALL SYNTHETIC CONTROLLER CHECKS PASSED')
