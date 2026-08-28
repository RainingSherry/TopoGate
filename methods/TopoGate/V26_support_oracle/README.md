# V26 Support Oracle Study v1

V26 tests whether sparse support structure is a cluster-relevant learning signal on
the fixed eleven-dataset panel selected by the user.  It reports support-only and
value-only clustering, then compares `P0_RANDOM`, `P1_SUPPORT_PRESERVE`,
`P2_SUPPORT_TARGET`, and one simple class-conditional label oracle under the same
row-wise swap budget.

The support-only diagnostic binarizes matrix entries but retains coordinates.  Its
complementary value-only diagnostic uses fixed quantiles of each row's nonzero
values: it retains a value distribution but has neither feature locations nor
zero padding for active-coordinate count.  For multiclass data, the oracle
contrasts the own-class support profile with a class-size-weighted mean across
all non-own classes; it never uses model state.

Labels are unavailable to the reconstruction model.  They are used only for outer
benchmark evaluation, except that `O_LABEL_ORACLE` uses labels to precompute the
corruption-coordinate score.  That arm is explicitly diagnostic and not deployable.

Run the preparation and pack-first matrix with:

```bash
/data/luolie/conda/base/bin/python3 scripts/V26/run_matrix.py --all
```

The dispatcher forbids GPU 0/7 and fills a currently used legal GPU while memory
admission remains safe before it spills to another legal GPU.  It reserves every
active V26 job at its measured preflight peak, retains one 4 GiB safety margin per
GPU, and has no fixed worker-count cap.  It never kills or pre-empts foreign
processes.
