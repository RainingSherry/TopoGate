# V24-Q1 v2 Protocol

V24-Q1 tests one bounded question: after clean State, effective Support, and
sample x intervention x nine-dimensional Marginal controls, does the frozen
V23 cycle-response fingerprint add local same-label pair discrimination in an
outer evaluation? It does not test independence, causality, functional
redundancy, or clustering gain.

## Pre-fit Contracts

The generated worlds are audited before any V23 fit or profile stage. W0 is
an iid exact-sparsity global null; W1, W2 and W3 isolate mean, support and
marginal-dispersion controls; W4 holds support and featurewise nonzero
marginals fixed while changing within-block dependence; W5 mixes realistic
signals. Synthetic labels are available only for this generator audit and the
outer pair evaluation. Fit and profile receive only the matrix path.

For a single W0 seed, the probe contract is a one-sided detectability guard:
macro OVR AUC must not exceed `classifier_chance_ceiling=0.52`. Its conditional
bootstrap interval is retained as a diagnostic, but is not required to cover
0.5: independently generated null datasets can have an interval entirely on
one side of 0.5 by finite-sample variation. The fixed five-seed panel adds the
predeclared centering guard: both support and featurewise-marginal mean AUCs
must lie within `null_panel_mean_auc_margin=0.01` of 0.5. Any per-seed or
panel failure writes `invalid_design` and blocks P1.

The two standardized-amplitude nuisance channels use a feature-relative scale
floor of `0.01` and are clipped at `10`. This protects the residualizer from
rare almost-constant nonzero sparse features: they should be retained as a
marginal-control event, not allowed to create an arbitrarily large leverage
point. The control diagnostics record the minimum reference scale and the
effective-position clip fraction. This affects only V24's outer nuisance
controls, never the V23 fit/profile probe.

## Promotion Sequence

`prepare` validates all six worlds and the W0 panel. `calibrate` separately
checks the estimator on matched null and weak-alternative representations.
Independent calibration replicates may run in a deterministic process pool;
the worker count is execution metadata, not a selected estimator parameter.
`p0` is a read-only V23 postmortem. Only a successful prepare and calibration
allow the six-world, five-seed P1 fit/profile/analyze matrix. The Q1 decision
requires all 30 records, per-seed bootstrap CIs, calibration and P0. Passing
Q1 merely permits a preregistered Q2 design; DCBoost remains a frozen
downstream attribution target until then.

## V1 Boundary

The earlier v1 preflight rejected W0 seeds because it required every
conditional bootstrap interval to include 0.5. No P1 job ran. V2 preserves
the existing per-seed AUC ceiling, introduces the fixed-panel mean-null guard,
and writes to a new result root; v1 artifacts remain untouched as a failed
preflight record.
