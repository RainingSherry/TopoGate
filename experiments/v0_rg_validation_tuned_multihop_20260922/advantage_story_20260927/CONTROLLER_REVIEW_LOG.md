# Base queue controller review — 2026-09-27

The frozen base queue is prepared and copied to `yazhouwan` under
`/data/luolie/ToPoGate/experiments/v0_rg_validation_tuned_multihop_20260922/advantage_story_20260927`.
It has **21 GPU fits** and **59 CPU posthoc cohorts**. The fits rotate among the
three cases; all 59 cohorts satisfy the exact historical matched-config rule.
The excluded unmatched lock entry is not queued. No `REVIEW_APPROVED.json` was
created, and no real fit or cohort job was launched by this review.

The controller now enforces one controller and one live attempt per task via
`flock`; persists a task claim and GPU assignment before child launch; adopts
live `run-task` children after restart; recovers finished attempts from their
result files; and retains each attempt in its own output directory. Only
transient OOM/connection/timeout failures get one retry. A dependency becomes
ready only after every prerequisite reaches a terminal status; a failed fit can
therefore still permit a posthoc report explaining missing inputs. Every
completed fit/cohort record must have the expected artifacts and hashes plus
the label-scope provenance flags. Completed outputs are verified at first
observation/restart and again at pipeline closure, avoiding full trace rehash
on every polling pass.

GPU tasks must invoke `fit_worker.py --device cuda:0`. The controller maps this
visible device to only physical GPU 2 or 4, checks free GPU and host memory,
admits at most four initial workers per GPU, grows by at most one additional
worker per GPU per polling pass, and never exceeds sixteen per GPU. The CPU
budget is four workers at two threads each. The approved queue, controller
and worker scripts, 21 task JSON files, manifest, cohort lock hash file,
resource file, and 27 frozen core files are hash-gated before each launch.

Remote synthetic check (using
`/data/luolie/conda/envs/scssl_bench_py310/bin/python`, no datasets):

```
Frozen base queue: 21 fits, 59 cohorts; no approval or run
PASS dependency terminal rule, cycle rejection, empty hash and label provenance rejection
PASS GPU allow-list, initial-four cap, one-extra cap, memory reservation
PASS controller singleton queue, terminal dependencies, retry once, attempt preservation, GPU2 worker visibility
PASS frozen task JSON approval pin rejects mutation
ALL SYNTHETIC CONTROLLER CHECKS PASSED
```

`prepare_base_queue.py --check` also passed remotely: it compares the saved
queue/resources to the manifest, draft, all 21 task JSON files, the pinned
59-cohort selection lock, and the controller's queue/resource validators.
This does not replace a real training run; runtime data/model failures remain
possible and will be retained as explicit task failures.

Final remote SHA256:

| file | SHA256 |
|---|---|
| `controller.py` | `5bbab4d2aa5ae8253cf2ab74626fd466f2d76daf8626d6bed8db68f270d38ddd` |
| `queue.json` | `3a29a67f19ba58aa9fd2b5a298fd800d1ec9052226b5e5587656d699448cce7d` |
| `resources.json` | `9ec0f76721d804e3576ce355144971b6481e309bb4c142d7b78d01aa6f8ae0d0` |
| `launch_base_queue.sh` | `685bee2b6ff2b2240d408fdcc3be6236c81df73c03ead49235b39bcf9c6ee969` |

The parent may now review and create `REVIEW_APPROVED.json` with
`status="approved"`, `queue_sha256`, `code_sha256` for `controller.py`,
`fit_worker.py`, and `cohort_worker.py`, `input_sha256` for 21 `tasks/*.json`,
`REUSE_MANIFEST.json`, `COHORT_LOCK_SHA256.txt`, and `resources.json`, and
`core_sha256` for the 27 source paths from `REUSE_MANIFEST.json`. The manual
launch command after approval is:

```
cd /data/luolie/ToPoGate/experiments/v0_rg_validation_tuned_multihop_20260922/advantage_story_20260927
bash launch_base_queue.sh
```

The launch script verifies the approval and resources before starting the
controller; it does not create approval. Read-only status is
`python controller.py status`; the live log is `logs/controller.stdout.log`.
