# E2/E6 independent code and data review (2026-09-27)

Scope: `depth_worker.py` and its three completed posthoc case outputs. The worker reads frozen historical H1/H2/H4 records and completed new H2/H8/Kwide fits; it does not train an encoder or alter core model files.

## Checks completed

- `test_depth_worker_review.py` passed on the remote CPU environment. Its hand-computable cases cover the selected H4 Kwide reference, fixed non-depth `q` and amplitude, candidate novelty including hop 9, invalid-index hard failure, ten KMeans readouts, and float32 graph-row normalization.
- All three real cases have `status=completed`, 15/15 cells, 150/150 readouts, three training seeds per strategy, and no missing cells. Each of the 10 listed output artifacts per case matches the SHA-256 in `record.json`. Per-seed statistics and the mean/SD aggregation were independently recalculated in the review test.
- New-fit candidate plans were checked against the saved training graph. Kwide has exactly the selected multihop candidate count **for every anchor**, not only on average. The same graph is used across strategies within each seed; the posthoc coverage baseline uses each anchor's seed-specific H1 graph neighbors. Historical plans are reconstructed from frozen graph and configuration and labeled as such in `coverage_summary.csv`.
- `student_grade` H4 and H8 were checked for seeds 7, 42, and 123. Reconstructed H4 and saved H8 candidate offsets, node indices, probabilities, hop labels, and draw counts are elementwise identical. Both reach maximum hop 4. Historical H4 and new H8 embeddings are bitwise identical for all three seeds. This explains their identical ARI as path reach saturation in this graph, not an independent validation of a higher hop effect. Full machine-readable result: `outputs/E2_E6/student_h4_h8_saturation_audit.json`.

## Reviewable results

| Case | H1 ARI | H2 | H4 | H8 | Kwide | Selected depth − Kwide |
|---|---:|---:|---:|---:|---:|---:|
| Baron | .771040 | .837090 | .772907 | .837703 | .783289 | +.054414 (H8) |
| Quake Lung | .706870 | .713798 | .713484 | .738201 | .703739 | +.034462 (H8) |
| student_grade | .328167 | .327394 | .367869 | .367869 | .350063 | +.017805 (H4) |

Each table entry first averages 10 KMeans states per training seed, then averages three training seeds. The standard deviations in `summary.csv` are across those three seed means (`ddof=1`). The candidate probability mass assigned to communities absent from each anchor's H1 graph is 40.33% versus 2.40% (Baron selected H8 versus Kwide), 28.54% versus 9.22% (Quake), and 14.76% versus 5.91% (student H4 versus Kwide). This is **candidate-plan probability**, not realized training-source mass or final gated input mass. E5 trace provides realized-source measurements separately.

The depth response is not monotonic: Baron H4 returns near H1 while H2/H8 improve; student H4/H8 coincide exactly. Thus the three-case result supports a topology-path versus count-matched wide-one-hop comparison under frozen selected settings, not a general claim that greater numerical depth always improves clustering. The Kwide comparison locks per-anchor candidate count; it does not claim equal feature displacement magnitude. These three cases are prespecified examples and their three-seed SDs are descriptive rather than a population-level significance test.

## Related final-scope checks

- E3 fresh full-cohort output has 59 matched datasets; `RESULT_E3.md` accurately reports 44 positive, 10 negative, 5 tied mean-ARI differences and the three case values. The archived E1 46-positive readout is retained as a separate historical result; no cause is asserted for its small near-zero flips.
- The revised depth cohort has 236 complete dataset-depth groups: H1 59, H2 36, H3 27, H4 25, H5 23, H6 24, H7 15, H8 27. Ignoring `high_hop_probability_multiplier` only for candidate H1 is valid because H1 has no higher-hop source on which that parameter can act; all active non-depth parameters remain checked. Missing depth cells should remain visibly missing in the heatmap.
- Baron E5 trace `labels_true` and `community_ids` arrays equal the arrays in `inputs/Baron_trace_partition` elementwise for all 8,569 anchors. The five-seed realized trace and historical three-seed relation table are independent cohorts; `RESULT_E5_SOURCE.md` correctly limits the joint interpretation.
- Every one of the 11 rows in `class_source_relation.csv` was independently recomputed: the same-class-new source mass from `per_anchor_mass.npz` (mean over five seeds and class anchors), and H1/multihop cross-community coassignment from `e4_community_pair_coassignment.csv` weighted by pair support. All three recomputed values per row agree within `1e-12`.
- All nine E5 response records have completed status, `labels_used_for_model_fit=false`, `test_labels_loaded=false`, `core_source_unchanged=true`, and valid hashes for the two listed output artifacts. The only relaxed historical-plan comparison is the named scalar `path_mass_retained_after_topk` with absolute tolerance `1e-4`; the three Baron seeds record raw maximum differences 4.2844e-5, 1.0508e-5, and 3.3155e-5. This response is a posthoc ungated finite-difference feature response along reconstructed current Monte Carlo directions; it is not an exact replay of historical training draws or a causal training ablation.

The previously noted shared-partition provenance gap is closed. The rerun `assemble_shared_partition.py` verifies exact rowwise label equality before invoking the audited E4 analysis, compares input hashes again after computing, and atomically writes `outputs/E5_shared_partition/record.json` last. I independently verified all five recorded input hashes, both code hashes, and all seven artifact hashes against the current files. Its five source-trace seeds and three readout seeds match the respective saved NPZ files exactly; the record limits the result to descriptive class correspondence.
