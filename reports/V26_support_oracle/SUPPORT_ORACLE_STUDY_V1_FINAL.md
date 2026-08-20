# V26 Support Oracle Study v1 — Final Report

**Status:** completed valid
**Run window:** 2026-08-20 04:41:08–05:38:41 UTC (57m 33s)
**Formal matrix:** 165 / 165 cells, 11 datasets × 5 arms × 3 paired seeds
**Primary metric:** ARI

## Scope and protocol

The study used exactly the following frozen datasets:

- scRNA: Mouse, Baron, Campbell, Macosko, Melanoma, Quake, Wang
- non-biological: news20, rcv1, arcene, sms spam

For each dataset, the study reports two diagnostics and trains the same
reconstruction probe with five arms:

1. `CLEAN`
2. `P0_RANDOM` — random coordinate-pair value swaps
3. `P1_SUPPORT_PRESERVE` — active-to-active swaps
4. `P2_SUPPORT_TARGET` — active-to-inactive swaps
5. `O_LABEL_ORACLE` — diagnostic-only class-conditional support oracle

The support-only diagnostic uses binary feature coordinates.  The value-only
diagnostic uses fixed quantiles of each row's nonzero values, without feature
locations or zero-padding.  Labels are unavailable to the model in every arm;
they are used after fitting for benchmark metrics, and only the oracle uses
labels to choose its corruption coordinates.

## Integrity and execution checks

- All 165 cells ended with exit code 0; no worker log contains a traceback,
  CUDA OOM, or runtime error.
- The frozen 11-source manifest has a unique, complete dataset-ID join to all
  formal cells.  Every completed cell's source path, size, and mtime agree
  with the frozen source record.
- The implementation digest is
  `8bb750e215d9b159aa0ee33319e4048880fabdc549adda9a720271cee5c9eb4b`.
- GPU policy held: GPU 0 and 7 were never selected.  GPU 6 ran 157 cells;
  GPU 1 ran 6 and GPU 5 ran 2, only after GPU 6's measured-reservation budget
  was exhausted.
- The preliminary diagnostic that allowed sparse support leakage in its
  value-only representation is retained as invalid prelaunch audit material;
  it does not enter any result below.

## Results

All entries below are ARI.  `support−value` is the diagnostic support advantage;
`oracle−P2` is the paired-seed mean difference.

| Dataset | Case | Support | Value | Support−value | Clean | P0 | P1 | P2 | Oracle | Oracle−P2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Mouse | A | .8131 | .2829 | +.5302 | .8285 | .8355 | .8106 | .8055 | .8054 | −.0001 |
| Baron | Inconclusive | .4103 | .0179 | +.3924 | .2275 | .1948 | .1974 | .2267 | .0295 | −.1972 |
| Campbell | Inconclusive | .1887 | −.0133 | +.2021 | .2056 | .1880 | .1828 | .1591 | .0577 | −.1014 |
| Macosko | Inconclusive | .2685 | .1115 | +.1570 | .5482 | .5756 | .5500 | .5558 | .3952 | −.1606 |
| Melanoma | Inconclusive | .1955 | .0738 | +.1217 | .3980 | .4124 | .4953 | .3679 | .0700 | −.2979 |
| Quake | Inconclusive | .3922 | .0229 | +.3693 | .1218 | .1106 | .1190 | .1354 | .0553 | −.0800 |
| Wang | A* | .3208 | −.0310 | +.3518 | .0584 | −.0455 | −.0501 | −.0510 | −.0504 | +.0006 |
| news20 | C | .0325 | .0055 | +.0270 | .0035 | .0001 | .0020 | .0000 | .0000 | +.0000 |
| rcv1 | C | .0569 | −.0001 | +.0570 | −.0002 | −.0002 | −.0000 | −.0000 | −.0003 | −.0002 |
| arcene | C | .0792 | .0050 | +.0742 | .0853 | .0656 | .0851 | .0727 | .0255 | −.0472 |
| sms spam | C | −.0574 | −.0332 | −.0242 | .6400 | .6571 | .6380 | .6580 | .5888 | −.0692 |

## Decision

The frozen decision rule classifies support as strong when support-only ARI is
at least .10 and exceeds value-only ARI by at least .03.  A material oracle gap
is .03 ARI.

- **Case A:** Mouse and Wang have strong support diagnostics with negligible
  oracle–P2 gaps.  Mouse is the clear positive result.  Wang is only a
  rule-based A: its formal Clean ARI is .0584 and all corruption arms are near
  or below zero, so it should not be treated as a robust positive confirmation.
- **Case C:** all four non-biological datasets (news20, rcv1, arcene, sms spam)
  support freezing the support route under the preregistered rule.
- **Case B:** none.  No dataset shows a positive material oracle advantage.
- **Inconclusive:** Baron, Campbell, Macosko, Melanoma, and Quake all have
  strong support diagnostics but a large *negative* oracle–P2 gap.

## Interpretation

The study establishes that support structure is a strong cluster-relevant signal
on much of the scRNA panel: the median scRNA support-minus-value advantage is
.3518 ARI.  It does not establish that the present simple label oracle is a
useful performance upper bound.  On five scRNA datasets the oracle is much
worse than P2, indicating that its class-conditional active-to-inactive swap is
too destructive or has the wrong causal direction for denoising training.
This is an oracle-design finding, not evidence that support is irrelevant.

The next support study should preserve the validated support diagnostic but
replace the destructive swap oracle with an oracle that selects useful support
regions without forcing class-conflicting values into inactive coordinates.

## Canonical machine-readable artifacts

- `result/V26_support_oracle/FREEZE/manifest.json`
- `result/V26_support_oracle/DISPATCH/final.json`
- `result/V26_support_oracle/ANALYSIS/support_oracle_decision.json`
- `AUTO_REVIEW.md` and `REVIEW_STATE.json`

Raw datasets and all 165 per-cell result files are intentionally excluded from a
publication bundle; the frozen manifest records their paths and SHA-256 hashes.
