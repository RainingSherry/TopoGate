# V19 RG Full V2 Tuning Plan

## Scope

Only `rg_full` is searched. The existing `scmae_only` configuration remains a
fixed reference and is evaluated once with the same held-out-row protocol. A
separate matched scMAE control is required after the final RG configuration is
locked when a shared backbone profile was selected.

No tuning process reads `y`, derives `K`, runs KMeans, or writes ARI/NMI/SOTA
metrics. The final benchmark comparison is a separate post-freeze operation.

## Selection Protocol

- Split each dataset and seed into deterministic 80% fit rows and 20% evaluation
  rows. Model fitting, RG graph construction, pseudo mixing, and gate statistics
  use fit rows only. The current adapter still fits label-free input
  preprocessing (HVG/scaling) on the full feature matrix, so this is a
  transductive-X holdout rather than a strict inductive preprocessing split.
- Fit the RG graph, pseudo mixing, and autoencoder only on fit rows.
- Build an evaluation-only input graph on the evaluation rows using the fixed
  base reference graph profile. The graph is shared by RG candidates and the
  fixed scMAE reference and is not used for training.
- Compare each RG candidate with the fixed scMAE reference using the same
  evaluation mask ratio, masked recovery proxy, latent view cosine, and shared
  input-neighbor graph.
- The primary promotion score uses only layers marked
  `archived_sota_bridge_eligible`, aggregated once per underlying dataset.
  `internal_rg_native_only` layers are evaluated separately as native guardrails
  and cannot create a SOTA-comparable proxy win.
- A bottom-level dataset is an X-only proxy win when at least two of the three
  metrics pass their pre-registered improvement thresholds, with no severe
  regression or latent collapse, in at least two of three seeds when three
  seeds are available.
- Biological bridge layers contribute one comparable dataset unit; the paired
  native layer is reported as a separate guardrail and never creates a second
  SOTA-comparable win.

The selection artifact is marked `no_go` when fewer than 2 of the 8 comparable
underlying datasets pass the proxy-win rule. The downstream benchmark may still
run for diagnostic completeness, but a no-go artifact cannot be described as a
proxy-supported RG lock.

Thresholds are recorded in `selected_config.json`: recovery 1%, latent cosine
0.005, neighborhood overlap 0.01, severe-regression floors -5%, -0.05, -0.10,
and latent feature standard deviation collapse below `1e-4`.

## Funnel

The formal V19 v2 search is RG-mechanism-only. The scMAE backbone profile is
frozen to the base configuration; `mask_ratio`, learning rate, hidden size, and
epoch budget are not promoted as topology-gating parameters. A matched
scMAE-only control with that same frozen profile is run after the RG mechanism
configuration is locked.

| Stage | Candidates | Input layers | Seeds | Runs |
|---|---:|---:|---:|---:|
| Fixed scMAE reference | 1 | 8 comparable + 3 native = 11 | 3 | 33 |
| `mechanism_screen` | 48 | 8 SOTA-comparable layers | 42 | 384 |
| `mechanism_refine` | top 12 | all 11 layers | 42, 123, 7 | 396 |

The RG mechanism search total is 780 runs. `mechanism_screen` uses
`--comparable-only`; `mechanism_refine` uses all fixed manifest layers. The
backbone/joint stages from the earlier draft are retained only as non-formal
code paths and are disabled by the formal launcher contract.

## Launch Order

1. Confirm that the selected GPUs in `1..6` have no external compute process.
   GPU 0 and GPU 7 are forbidden. Each launcher worker receives an explicit
   `CUDA_VISIBLE_DEVICES` value.
2. Launch the fixed reference with `--kind reference` and all 11 layers.
3. Launch `mechanism_screen` with `--comparable-only`, summarize with
   `--top-k 12`, and pass the recorded `top_candidate_ids` to
   `mechanism_refine`.
4. Summarize `mechanism_refine` with `--top-k 1` and freeze the single RG
   mechanism configuration.
5. Run a fresh final evaluation and matched scMAE control. Only then reveal
   labels for benchmark K and ARI/NMI/SOTA reporting.

Every stage has a frozen `stage_spec.json`, resumable run records, and refuses
to resume under a different candidate, dataset, seed, or layer-selection
protocol. The old incomplete v2 draft roots are preserved and are not reused.
