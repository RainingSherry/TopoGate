# E1/E3 integrity audit

## Completed checks

- E0 `audit_ok=true`; corrected constructive D1 gate is false; old D1 audit and
  decision hashes are recorded; `d2_authorized=false`, `gpu_runs_started=0`.
- E1 logical matrix: `54/54` completed-valid; 18 reused C2 controls and 36 new
  GPU runs. All new GPU summaries record physical GPU 6, legal single-device
  visibility, finite checkpoint metrics, exact corruption budget, and no raw
  arrays/checkpoints/embeddings persisted.
- E1b no-fit matrix: `54/54` completed-valid; labels are marked outer-readout
  only and the H0-derived feature matrix is built before label loading.
- E3 raw audit: `6/6`, `audit_ok=true`, `labels_not_loaded=true`,
  `gpu_runs_started=0`.
- Focused tests: `9 passed`; `compileall` passed.
- GPU resource firewall: preflight selected idle legal pool `[6]`; physical
  GPUs 0 and 7 were never selected, and busy external processes on GPUs 1–5
  were not interrupted.

## Claim firewall

E1/E2 use the frozen threshold-defined dense H0 proxy. E3's raw `.npz`
zero/nonzero support is descriptive only and cannot enter fitting, gates, or
the decision. No adaptive policy, GAN, learned generator, new Gate, matcher
optimization, architecture search, corruption-rate sweep, or holdout was
started.

## External review boundary

The Claude review route was invoked four times. The first review scored 4/10
and blocked launch; the subsequent same-thread route timed out, a fresh route
reported plan-mode filesystem unavailability, and an inline-evidence route
timed out. None was treated as an acquittal. The E1 launch was an explicitly
recorded manual executor gate based on the local checks above and the user's
instruction that the final decision be made by the executor.
