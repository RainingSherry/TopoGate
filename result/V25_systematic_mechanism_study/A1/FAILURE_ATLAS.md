# V25 A1 Failure Atlas

This report is descriptive and provenance-aware. It does not treat rows, seeds, variants, coordinates, or pair counts as independent population samples.

The atlas imports `paired_delta_ari` and `ari_mean` from the audited historical table; it does not reload labels. This is an evidence-ingestion boundary, not a label-free evaluation claim: the original benchmark metrics may have used dataset labels, and no A1 row is re-evaluated here.

## Scope

- V1-V22 paired records: `1637`.
- V1-V22 dataset/protocol/readout units represented: `239`.
- Positive (`Delta ARI > 0.03`): `194`; negative (`Delta ARI < -0.03`): `680`; observed-small: `763`.
- V23 local/global rows are boundary evidence only and are not pooled into the intervention atlas.

## Version/family summary

| Version | Family | Rows | Datasets | Mean Delta ARI | Positive | Negative | Small |
|---|---|---:|---:|---:|---:|---:|---:|
| V09 | learned | 7 | 7 | 0.015356458195667081 | 2 | 0 | 5 |
| V09 | random | 37 | 22 | -0.013462429005758818 | 5 | 7 | 25 |
| V09 | static | 22 | 22 | 3.341738844288289e-05 | 2 | 2 | 18 |
| V10 | learned | 9 | 3 | -0.019822938397388412 | 0 | 2 | 7 |
| V11 | fixed | 30 | 5 | -0.00254918755422382 | 0 | 2 | 28 |
| V11 | learned | 15 | 5 | -0.002555200201501895 | 0 | 1 | 14 |
| V11 | random | 15 | 5 | -0.002536907452850068 | 0 | 1 | 14 |
| V12 | learned | 16 | 4 | 0.01016842373442294 | 4 | 0 | 12 |
| V13 | hard | 15 | 5 | -0.15559566122999674 | 3 | 5 | 7 |
| V14 | learned | 15 | 5 | 0.004373099308058459 | 1 | 2 | 12 |
| V16.1 | fixed | 8 | 8 | 0.045994957365592413 | 2 | 0 | 6 |
| V16.1 | learned | 8 | 8 | 0.0008035987344635089 | 0 | 0 | 8 |
| V16.1 | random | 8 | 8 | 0.00039873296132428293 | 0 | 0 | 8 |
| V18 | auxiliary | 745 | 149 | -0.09315512128792038 | 107 | 328 | 310 |
| V18 | fixed | 146 | 146 | -0.23622337689072673 | 7 | 100 | 39 |
| V18 | learned | 149 | 149 | -0.08245027462777364 | 25 | 62 | 62 |
| V18 | random | 149 | 149 | -0.14062412067327734 | 9 | 95 | 45 |
| V18 | static | 149 | 149 | -0.045876417461800816 | 23 | 58 | 68 |
| V19 | learned | 65 | 24 | -0.0022741239235652896 | 0 | 1 | 64 |
| V19 | static | 11 | 11 | 0.006559400447262201 | 1 | 0 | 10 |
| V21 | assignment-adversarial | 18 | 6 | -0.21088608960435784 | 3 | 14 | 1 |

## Baseline/headroom analysis

The baseline table is a fixed-bin descriptive sensitivity. It cannot establish that a strong baseline causes intervention harm; ceiling/headroom remains a competing explanation.

| Baseline bin | Rows | Datasets | Mean baseline | Mean Delta ARI | Harm count | Positive count |
|---|---:|---:|---:|---:|---:|---:|
| 0.2-0.4 | 274 | 41 | 0.3136113690139822 | -0.06237567078379133 | 123 | 43 |
| 0.4-0.6 | 198 | 27 | 0.5188008983738338 | -0.09012802654052687 | 108 | 29 |
| 0.6-0.8 | 225 | 29 | 0.7028346417689915 | -0.24258769433487598 | 192 | 13 |
| <0.2 | 726 | 93 | 0.05631618252428615 | 0.0012736425336224387 | 113 | 105 |
| >=0.8 | 198 | 27 | 0.8999666671819617 | -0.31238645071280724 | 144 | 0 |
| unavailable | 16 | 4 | NA | 0.01016842373442294 | 0 | 4 |

## Structural opportunity and intervention magnitude

The audited long table has no common fixed-graph/null opportunity endpoint. The opportunity table is therefore a stratified missingness record. Magnitude strings are post-treatment descriptors and cannot be interpreted as causes.

- Structural opportunity groups: `9`; uniform opportunity endpoint available: `False`.
- Magnitude descriptor groups: `10`; causal magnitude inference: `False`.

## Artifact-complete replay gate

- Rows admitted to offline E3 replay: `0`.
- Metadata-only rows are excluded. An empty replay set is an explicit boundary, not a reconstructed result.

## Local/global boundary evidence

These rows come from V23 M0 and are retained as a separate boundary branch.

| Condition | Local delta (kNN purity@10) | Global delta (ARI) | Disconnect |
|---|---:|---:|---|
| cycle_minus_precycle | 0.1306111111111111 | 0.01166109752358898 | False |
| cycle_minus_support | 0.09232222222222221 | -0.003670781195910501 | True |
| gain_minus_precycle | -0.0027555555555555575 | 3.604090477705507e-05 | False |
| cycle_minus_untrained | 0.038177777777777776 | 0.0060075292826640495 | False |
| conditional_null_cycle_minus_precycle | 0.04360000000000001 | 0.00742835515346323 | False |
| global_null_cycle_minus_precycle | 0.00018888888888888658 | -0.00016360852312741344 | True |

## Missing evidence boundary

The long registry does not contain a uniform graph-quality, embedding-drift, or artifact-complete replay field across V1-V22. Those quantities remain unavailable rather than being reconstructed from ARI or gate strings. Gate strings are retained only as post-treatment magnitude proxies. V23 kNN purity is explicitly post-hoc supervised geometry; no label-free local metric is claimed.
