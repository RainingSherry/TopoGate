# Protocol

## Questions

1. Does the P2 gain survive a support-preserving matched support-role contrast
   with the same active source coordinates and closely matched row-wise
   perturbation L1?
2. Only after that question is answered may a later protocol ask whether the
   dense-H0 phenomenon corresponds to raw-X zero/nonzero support.

## Frozen M1 matrix

| item | frozen value |
|---|---|
| datasets | Mouse_retina, Baron Human, Campbell |
| seeds | 42, 123, 7 |
| new arm | `P2_MM_SupportPreserve` |
| reused arm | C2 `P2_SupportTarget` (read-only) |
| encoder | exact C2 small matched reconstruction probe |
| epochs | 30 |
| readout | clean embedding, benchmark-known-K KMeans after fit |
| legal physical GPUs | 1, 2, 3, 4, 5, 6 |
| forbidden physical GPUs | 0, 7 |

For every epoch and row, M1 deterministically replays C2's ordered
active-source/inactive-destination pairs. For each P2 active source, a distinct
active partner is selected from all active coordinates excluding the complete P2
source set. A square/rectangular Hungarian assignment minimizes

```text
abs(2*abs(H0[source] - H0[active_partner])
    - 2*abs(H0[source] - H0[P2_inactive_destination])).
```

The control swaps source/partner values. It therefore keeps the row value
multiset and threshold support while retaining the P2 source positions.

## Structural contract

A run is valid only when all of the following hold:

- changed-coordinate counts equal P2 exactly;
- threshold-support change rate is exactly zero;
- P2 source positions are identical;
- no partner is repeated or overlaps a source;
- every row's value multiset is unchanged;
- labels and clustering metrics are absent from fit and partner selection;
- the embedding is finite.

## Magnitude contract

The dataset-level relative mismatch is computed over all audited epochs:

```text
abs(sum L1_MM - sum L1_P2) / sum L1_P2 <= 0.05
```

The median row/epoch relative mismatch must be at most `0.10`. A failure is
`magnitude_match_not_estimable`/`protocol_mismatch`, never a negative result.
The full 30-epoch, nine-row preflight runs before model construction. If any
dataset×seed fails the frozen tolerance, the formal M1 GPU matrix is not
authorized because its two-of-three promotion gate would no longer be
estimable under this control.

## Primary estimand and gate

```text
Delta_cross(d) = ARI(P2_reused, d) - ARI(P2_MM_SupportPreserve, d)
```

This is a descriptive estimand, not a strict causal isolation. The
active-active control may be easier for reconstruction, so `Delta_cross` can be
conservative (downward-biased) for a support-crossing interpretation.

The M1 gate is passed only when at least two of three datasets have mean
`Delta_cross >= 0.03`, each passing dataset has at least two positive paired
seed differences, and no dataset has mean `Delta_cross <= -0.03`.

If the gate fails, the support-crossing claim is downgraded to a matched
row-wise permutation observation; M2 is not run.
