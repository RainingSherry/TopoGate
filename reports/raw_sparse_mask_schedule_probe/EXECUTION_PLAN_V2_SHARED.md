# Execution plan — shared v2

1. Preserve all v1 artifacts and launch no v1 guarded waiter.
2. Prepare the v2 manifest without rerunning v1 P0/P1.
3. Reuse the audited v1 SVD32 baseline through `SVD_REUSE_MANIFEST.json`.
4. Dispatch the unchanged 90-cell MAIN matrix on legal GPUs `[1,2,3,4,5,6]`
   regardless of foreign occupancy. GPU 0/7 remain forbidden.
5. Use the configured worker slots and retry policy until every cell is
   complete or the operator stops the run.
6. Aggregate v2 only from v2 summaries with `status=completed_valid`, valid
   source/adapter/mask/label audits, and complete paired coverage.

The v2 run is explicitly a shared-resource protocol; it must not be mixed with
v1 `MAIN` or `SHARED_RESOURCE_MAIN` directories.
