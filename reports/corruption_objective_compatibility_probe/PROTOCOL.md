# Protocol — corruption_objective_compatibility_probe

## Scientific question

The study has one bounded question: does the C2 P2-vs-P0 observation extend
outside the three scRNA development datasets, and does changing only the
reconstruction learning signal change the P2 incremental utility? It is not a
support-causality study and it is not a new TopoGate/backbone search.

## Frozen input and panel

All fit inputs are the existing, audited dense H0 snapshots and the existing
labels are loaded only after feature transformation/model fitting for
benchmark-known-K outer readout. The six sentinel datasets are:

- biological: Mouse_retina, Baron Human, Campbell;
- non-biological: cnae9, hate_speech, sms_spam_collection.

The first three are development/motivation data, not holdout evidence. The raw
`.npz` files are read only by E3 for a descriptive audit.

## E0 and E1/E1b

E0 reruns the corrected constructive matcher in this new project, records the
closed D1 audit/decision hashes, and never edits the closed support project.
The D2 support-attribution line remains unauthorized regardless of the E0
result.

E1 uses the exact small matched autoencoder contract: clean-H0 column
standardization, `d -> 64 -> 32 -> 64 -> d`, ReLU, Adam `1e-3`, 30 epochs,
batch 512, and checkpoints 1/5/10/20/30. Arms are exactly Clean, P0_Random,
and P2_SupportTarget. Biological P0/P2 controls are reused from audited C2
only after current H0, budget, label, and status hashes match; 36 new GPU runs
are otherwise authorized by this protocol (9 biological Clean + 27
non-biological all arms). E1b runs the no-fit H0 -> KMeans diagnostic for all
54 logical cells on CPU.

Primary E1 quantities are `ARI(P2)-ARI(P0)` and `ARI(P2)-ARI(Clean)`.
Training amplification is the model P2-vs-P0 delta minus the matched no-fit
P2-vs-P0 delta. It is diagnostic, not a paper claim by itself.

## Automatic Gate

E2 is launched only when E1 is complete and, among the three non-biological
datasets, at least two satisfy all of the following: both E1 deltas are at
least `0.03 ARI`, and both deltas are positive in at least two of three paired
seeds. In addition, at least two non-biological datasets must have training
amplification at least `0.03 ARI`. If either condition fails, E2 is not run.

## E2 objective matrix

E2 compares the same P0/P2 corruptions under O0 Global-MSE, O1 Changed-only
MSE, and O2 Balanced-MSE (`0.5 changed + 0.5 unchanged`). The matrix is
6 datasets × 2 corruptions × 3 objectives × 3 seeds = 108 logical cells.
The 36 O0 cells reuse the audited C2 controls; the 72 O1/O2 cells are new GPU
runs. The primary interaction is

```text
I_O(d) = [ARI(P2,O)-ARI(P0,O)] - [ARI(P2,O0)-ARI(P0,O0)].
```

An objective is a candidate only if its interaction is at least `+0.03` on at
least four of six datasets, includes at least one biological and one
non-biological dataset, and has at most one material negative dataset.

## E3 and interpretation firewall

E3 records raw shape, dtype, nnz/zero fraction, zero rows, and source hashes.
Raw-X zero/nonzero support is descriptive only and does not change the H0
estimand, model input, E1/E2 gates, or decision.

## Engineering contract

One process per physical GPU is allowed on `[1,2,3,4,5,6]`; `[0,7]` are
forbidden. Each run has a 30-minute timeout, at most one same-command retry
for a transient CUDA/I/O error, and an explicit `incomplete_compute` record for
timeouts, hard-wall expiry, or protocol failures. The total GPU wall is 11.5
hours. Existing valid runs are reused only after current source hashes and
exact stage/dataset/arm/objective/seed fields are checked. No batch-size,
epoch, architecture, or dataset rescue is allowed.
