# Shared-resource engineering execution

- Mode: `shared_resource_engineering_only`
- GPUs: physical `1` and `6` only; 45 cells each
- Matrix: `90/90` completed
- Wall time: 1977.2 seconds
- Failures: none
- Labels during fit: none
- Formal scientific use: **none**

All 90 summaries are explicitly marked `shared_resource_engineering_only` with `audit_ok=false`.
The run shared both GPUs with pre-existing external processes under explicit user authorization; no
foreign process was killed or preempted. GPU 0/7 were not used. The formal `MAIN` tree and
`MAIN_DISPATCH.json` were not modified by this run.

This artifact records execution/resource integrity only. It intentionally reports no clustering
metrics and must not be used for a performance or generalization claim.

Audit: `result/raw_sparse_mask_schedule_probe/SHARED_RESOURCE_AUDIT.json`
Manifest: `result/raw_sparse_mask_schedule_probe/SHARED_RESOURCE_MANIFEST.json`
