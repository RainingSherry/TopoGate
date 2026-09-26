# Audited V0_RG noise results (2026-09-23)

- [Final report](FINAL_REPORT.md): all four cases, evidence boundaries, and
  interpretation.
- [`tables/summary.csv`](tables/summary.csv): equal-weight model-seed means
  after averaging five noise seeds within each model seed.
- [`tables/seed_summary.csv`](tables/seed_summary.csv): each model seed and
  within-model noise variation.
- [`tables/metrics.csv`](tables/metrics.csv): every accepted numeric metric from
  1,218 observed/direction probes. Source identifiers are run-relative rather
  than machine-specific paths.
- [`tables/paired_ablations.csv`](tables/paired_ablations.csv),
  [`tables/paired_summary.csv`](tables/paired_summary.csv): strictly paired
  full–control effects; 18 structural no-op states are excluded.
- [`figures/`](figures/): PDF, SVG, PNG, source CSV, and captions for all 12
  final figures.
- [`FILE_AUDIT.json`](FILE_AUDIT.json): 840/840 raw-input observations,
  378/378 direction probes, and 870/870 validated comparisons.
- [`inventories/checkpoints.json`](inventories/checkpoints.json): artifact hashes
  and effective/no-op intervention decisions, with machine paths replaced by
  symbolic roots.

The original full archive of embeddings, predictions, logs, and per-run arrays
is excluded from Git; its SHA-256 is in `PUBLISH_MANIFEST.json`. The 18-dataset
multi-baseline expansion is separate and was not a completed result of this
study. No noisy-Top1 or general robustness claim is made here.

The exact Python snapshot of V0_RG used by these checkpoints is in
[`experiments/V0_RG_noise_four_step_20260923/source_snapshot/`](../../experiments/V0_RG_noise_four_step_20260923/source_snapshot/).
