# RS1 relation information / solvability audit

RS1 extracts geometry, local topology, and cross-view stability features from
the frozen candidate pool. It then runs grouped, diagnostic-only logistic
probes for class and pool-reference targets. The output separates
predictability (`AP`, `Delta AP`, `Lift@b`) from downstream clustering utility.

No selector is trained in RS1. A positive probe is not evidence that a feature
has clustering utility; a negative probe is an information bottleneck under
the frozen feature family and grouped split.
