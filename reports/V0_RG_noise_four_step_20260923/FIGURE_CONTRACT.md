# Figure contract

## Claim and evidence boundary

These figures describe corruption and frozen-model responses in four deliberately selected advantage cases. They do not estimate robustness across datasets or establish a causal mediation mechanism. Discovery is exploratory; confirmation noise seeds are reused existing realizations, not a previously unseen holdout.

The immediately available discovery figure asks whether input neighbourhood damage and retained clustering quality coexist. It uses replacement p=0.2 only. Its neighbourhood diagnostics are seed-42 frozen-preprocessed, all-row Euclidean kNN, **not** the model's clean-fitted PCA / row-normalized cosine graph. Macro and micro purity must both be visible. Comparator identity is selected from existing discovery results and is not an independently selected challenger. HAR's small advantage must not be presented as decisive.

## Panel contract

- Discovery: four columns (Mouse_retina, dermatology, har, micro-mass), top row micro/macro purity clean-to-noisy slopes, bottom row ToPoGate/strongest reported comparator clean-to-noisy ARI slopes. Clean ARI is reconstructed as noisy ARI minus reported noisy-minus-clean delta. Each panel has actual source values; no significance stars or inferred confidence intervals.
- Mechanism, only when successful probe results exist: actual input damage, normalized latent displacement, and clean/noisy ARI across conditions. Use separate axes, never a connected causal chain through incomparable units. Retain model-seed variation and distinguish within-model noise variation.
- Ablation, only when strict pairing and inventory effectiveness audits pass: difference between full and ablated noisy-minus-clean ARI. Positive values indicate a smaller full-model ARI decline. No-op, unknown, mismatched, failed, or missing controls are recorded and excluded. Three arms cannot establish a two-factor interaction.
- Layout, only when actual paired embeddings and sample identities exist: fit PCA on clean embeddings once and apply the same transform to noisy embeddings. Show a deterministic label-free subset of arrows. Two-dimensional displacement is illustrative, not proof of preservation in full dimensions. Never fit separate noisy layouts.
- Input structure: actual normalized displacement versus neighbor retention and micro/macro purity changes, only from completed full-model probes under the same protocol. Different nominal Gaussian/Replacement strengths do not imply matched damage.
- Direction probes: model-input interventions only, with actual float32 norm checks and an identical common-valid sample set within each multiplier. Compare full and strictly paired effective controls; report valid counts and separate all-test quantities. No pooling with raw-input corruption.

## Rendering and statistics

Quantitative grid, Python/matplotlib only. White background; ToPoGate #B5465D, comparison #797A8E, distinguish series by markers/linestyles as well. Sans-serif 7–9 pt, editable SVG text and PDF TrueType text. Export PDF, SVG and 300-dpi PNG; nominal double-column width 183 mm (large exploratory grid may use a taller page). Source CSV and caption accompany each figure.

Aggregate over noise seeds within each model seed, then average model-seed means with equal weight. Report sample SD across model-seed means separately from mean within-model noise SD. Three model seeds crossed with five noise seeds are not 15 independent trainings. No p-values or confidence interval claims. Missing or unequal seed coverage is explicit in manifests and summary tables.

## QA and missing data

Only real observations may be plotted. Missing panels/figures are skipped and named with reasons in figure_manifest.json. No fabricated numbers, placeholder data, silent zero filling, or empty panels presented as results. The raw changed-entry fraction from the old snapshot is not used because its dtype mismatch invalidates interpretation. Final PNGs require visual inspection at intended size; machine-readable export/source checks complement but do not replace inspection.

Pipeline-validation figures must carry a visible smoke-validation label and matching caption. They verify code paths; one-case/one-seed examples are not mechanism findings.
