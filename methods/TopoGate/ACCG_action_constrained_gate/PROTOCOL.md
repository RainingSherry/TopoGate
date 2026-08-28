# ACCG Frozen Protocol

## Research object

ACCG studies sample-graph-guided, feature-coordinate donor replacement for
unsupervised clustering. It keeps the audited V25-E1 V21 backbone, warmup
branchpoint, donor schedule, eligible set, effective budget, Gumbel tensor,
optimizer schedule, known-K protocol, and clean-embedding KMeans readout.

The only primary method change is the admissible hard action selected for the
`T_c` arm. ACCG does not add an encoder, decoder, cluster head, MLP, or readout.

## Feature-conditional model

The first paper uses a capped model interface with `d <= 2000`.

For each feature, ACCG applies a label-free robust z-score transform with a
frozen clip. Fold-specific feature graphs are fit without the held-out rows.
Each feature uses its top-k positive cosine neighbors, with non-negative weights
normalized to sum to one and no self-edge.

For transformed row `z_i`:

```text
m_ij(x)   = sum_{l in N_f(j)} w_jl z_i,l
rho_ij(x) = ((z_i,j - m_ij(x)) / scale_j)^2
F_i(M)    = M union {l: l in N_f(j), j in M}
R_i^M(x)  = mean_{j in F_i(M)} rho_ij(x)
```

The same hard footprint is used before and after the action. The declared
energy is conditional structural consistency under this feature model. It is
not task relevance, causality, semantic correctness, or an ARI predictor.

## Actual joint action

For the same V21 donor row `d_i` and eligible set `E_i`:

```text
x_i^M = x_i + M_i * (d_i - x_i)
kappa_i(M) = R_i^M(x_i) - R_i^M(x_i^M)
```

The primary admissibility constraint is:

```text
|M_i| = b_i
R_i^M(x_i^M) - R_i^M(x_i) <= epsilon_i
```

`epsilon_i` is calibrated from label-free random joint donor actions at the
same mask ratio. It is never selected from ARI, NMI, labels, or dataset
outcomes. A coordinate-wise singleton delta is a control and ranking aid only.

The selector evaluates the post-action residual jointly. Pair lookahead exists
because a coherent same-donor pair can be inadmissible as either singleton but
admissible when replaced together. Small instances are audited against a
brute-force exact solver.

## Budget and infeasibility

The primary `accg_joint` policy retains V21's exact requested effective budget.
If the greedy admissible pool cannot fill it, the selector adds the
least-violating remaining coordinates and records:

```text
constraint_infeasible = true
constraint_violated
fallback_count
safe_selected_count
budget_fill
```

This preserves dose matching while exposing violations. `accg_joint_abstain`
is a secondary dose sensitivity that may select fewer coordinates; it cannot
replace the exact-budget primary result.

## Matched arms

```text
N   no assignment intervention
R   matched random assignment intervention
T_s original V21 sample-only adversarial selection
T_c ACCG joint action-constrained selection
```

All arms branch after the same warmup and head initialization. They replay the
same post-branch batches, reconstruction corruption, assignment donor,
eligibility, budget rule, Gumbel noise, and optimizer budget. The branchpoint
stores seed, K, model-input shape/hash, schedule hashes, topology-statistics
hash, model/head/Gate/optimizer state, and RNG state.

Ablations run only `T_c` and must reuse a completed canonical `accg_joint`
branchpoint. The runner rejects a branchpoint with a different seed, K, model
input, warmup, V21 config, source hash, or incomplete `N/R/T_s/T_c` controls.

## Optimization

The encoder/head update uses the exact hard action. The constrained Gate update
uses the hard selector in the forward pass and V21's straight-through smooth
path in the backward pass:

```text
L_gate = -JS(q_clean, q_action)
         + lambda_cov * coverage
         + lambda_struct * relu(structural_delta - epsilon)
```

The structural footprint remains the declared hard footprint during the
gradient calculation. This differentiable barrier does not redefine the hard
admissibility audit.

## Controls

Frozen variants are:

```text
accg_joint          joint energy, real feature graph, exact-budget fallback
accg_coordinate     singleton/coordinate constraint control
accg_shuffled_graph shuffled feature-neighbor control
accg_marginal_only  no feature relation, coordinate marginal control
accg_joint_abstain  joint constraint with abstention sensitivity
```

`T_s` is the matched unconstrained V21 adversary. The main paper comparison is
`T_c - T_s`, not a comparison against an unrelated reconstruction baseline.

## Label and K boundary

The full label vector is loaded only by the outer runner. It is unavailable to
preprocessing, sample graph, feature graph, null calibration, selector, Gate,
loss, and model updates. Benchmark labels may determine K and are used after
fit to calculate ARI/NMI. The protocol is therefore a real-ground-truth,
known-K benchmark, not fully label-free fitting.

## Required artifacts

Every completed main panel must contain resolved config, runner/source hashes,
branchpoint, epsilon calibration, schedule manifest, per-arm predictions,
metrics, histories, structural audits, gradient probes, selection traces, and
checkpoints. The summarizer recomputes ARI/NMI from saved predictions and outer
labels and rejects source/config drift or failed schedule matching.

No ACCG training result existed when this protocol was written. Code validity
does not count as evidence that the structural constraint improves clustering.

