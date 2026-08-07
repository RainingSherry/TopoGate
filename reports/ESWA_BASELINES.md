# ESWA candidate-baseline registry

Last reviewed: **2026-07-31**.  This is the authoritative local inventory for
the ESWA methods considered alongside TopoGate.  A directory being present is
not equivalent to its being a usable benchmark baseline: the status and
fairness column below are the deciding records.

## Status legend

| Status | Meaning |
| --- | --- |
| **Ready — clean-room** | Local, label-isolated implementation with a tested Python API. |
| **Conditional — adapter** | Downloaded upstream source plus an adapter; a runtime or paper-parity limitation remains. |
| **Archive only** | Source is preserved, but must not be invoked as a fair baseline. |
| **Blocked** | Neither sufficient paper detail nor usable source is locally available for a faithful implementation. |
| **Excluded** | The method's published input modality does not match the feature-matrix TopoGate protocol. |

## Usable or conditionally usable methods

| Method / ESWA paper | Local source and status | Interface / runner | Verification (2026-07-31) | Key fair-comparison constraint |
| --- | --- | --- | --- | --- |
| **AHDPC** — Wang et al., *AHDPC: Adaptively hyperbolic density peak clustering*, ESWA 299 (2026) 130065, [DOI](https://doi.org/10.1016/j.eswa.2025.130065) | [Clean-room reproduction](AHDPC/README.md), mapped to locally archived paper equations; [provenance](AHDPC/PROVENANCE.md). **Ready — clean-room.** | `AHDPC(...).fit_predict(X, n_clusters=K)`; `python -m methods.AHDPC.run --data_path ... --n_clusters K --save_dir ...` | `PYTHONPATH=source-repository pytest -q methods/AHDPC/tests` → **7 passed** | `K` is explicit; never pass `y` into fitting.  The article's per-dataset epsilon tuning is not allowed; use a preregistered fixed/label-free epsilon.  Eq. (1) is internally contradictory, so record `normalization_mode`.  O(n²) memory/time. |
| **DPC-GFNN** — Waqas et al., *Density peaks clustering based on Gaussian fuzzy neighborhood with noise parameter*, ESWA 255 (2024) 124782, [DOI](https://doi.org/10.1016/j.eswa.2024.124782) | [Clean-room reproduction](DPC_GFNN/README.md) from the archived full paper; [provenance](DPC_GFNN/PROVENANCE.md). **Ready — clean-room.** | `DPCGFNN(...).fit_predict(X, n_clusters=K)` or native `fit_predict(X)`; `python methods/DPC_GFNN/run.py ...` | `PYTHONPATH=source-repository pytest -q methods/DPC_GFNN/tests` → **8 passed** | Paper scans `k`/`lambda` and reports metric-specific best values; do not select either using test labels/ACC/NMI/ARI.  Explicit-K and native-unknown-K results are different protocols. O(n²). |
| **GCC** — Kuwil et al., *A novel data clustering algorithm based on gravity center methodology*, ESWA 156 (2020) 113435, [DOI](https://doi.org/10.1016/j.eswa.2020.113435) | [Independent reimplementation](GCC/README.md) from archived accepted manuscript; no author source found. **Ready — clean-room.** | `GravityCenterClustering(...).fit_predict(X)`; optional `n_clusters=K` is an adapter. `python methods/GCC/run.py ...` | `PYTHONPATH=source-repository pytest -q methods/GCC/tests` → **5 passed** | Native GCC is unknown-K.  The known-K multiplier search/merge/split logic is a local adapter, not the paper algorithm.  Do not use the CLI's `y`-to-K convenience unless that cardinality protocol is explicitly declared. Actual current implementation can incur repeated active-set sorts, so treat its worst-case time as higher than the paper's nominal O(n²) claim. |
| **HARR-V / HARR-M** — Zhang et al., *Learning unified distance metric for heterogeneous attribute data clustering*, ESWA 273 (2025) 126738, [DOI](https://doi.org/10.1016/j.eswa.2025.126738) | [Clean-room implementation](HARR/README.md) plus separately quarantined [third-party snapshot](HARR/PROVENANCE.md). **Ready — clean-room.** | `HARRV` / `HARRM` with explicit `n_clusters`, numerical/nominal/ordinal metadata and ordinal order; `python methods/HARR/run.py ...` | `PYTHONPATH=source-repository pytest -q methods/HARR/tests` → **5 passed** | Attribute types/orders must come from dataset documentation, never `y`.  Numerical discretisation is under-specified (default: five label-free quantile bins); the implementation materializes an n×K×d_hat distance tensor.  Do not run/import the third-party snapshot: its Heart-Failure loader includes `DEATH_EVENT` as both target and feature. |
| **LCG-DSC** — *Graph structure enhancement with local cluster guidance for discrete spectral clustering*, ESWA, [DOI](https://doi.org/10.1016/j.eswa.2025.129961) | [Author-associated MATLAB snapshot](LCG_DSC/PROVENANCE.md) at `1df5c7c…`, retained under `upstream/`; local [label-free adapter/status](LCG_DSC/README.md). **Conditional — adapter.** | MATLAB: `fit_predict(X, K, neighbor_k, lambda_weight, num_self, max_iter, seed)`; NPZ bridge: `python -m methods.LCG_DSC.run ...` | `PYTHONPATH=source-repository pytest -q methods/LCG_DSC/tests` → **4 passed** (Python-side adapter/validation only; MATLAB/Octave unavailable) | Never run `upstream/main.m`: it reads `Y` to infer K and uses dataset-specific choices.  Pre-register `neighbor_k`, `lambda_weight`, and `num_self`; end-to-end numerical validation still requires MATLAB/Octave. |
| **UEC** — *Unified embedding and clustering*, ESWA 238 (2024) 121923, [DOI](https://doi.org/10.1016/j.eswa.2023.121923) | [Author-associated Python snapshot](UEC/PROVENANCE.md) at `b6b103a…`, plus separate [adapter/status](UEC/README.md). **Conditional — adapter.** | `methods.UEC.fit_predict(X, n_clusters=K)`; `python -m methods.UEC.run_uec --input ... --n-clusters K --output-dir ...` | `PYTHONPATH=source-repository pytest -q methods/UEC/tests` → **7 passed, 2 upstream/deprecation warnings** | The adapter uses compatibility shims and disables upstream JIT on the current stack; this proves a smoke path, not historical-paper parity or timing.  `K` and all UMAP/training settings must be fixed without labels. |

## Archived, blocked, or excluded candidates

| Method / ESWA paper | Local source and status | Interface / test | Why it must not currently enter a TopoGate results table |
| --- | --- | --- | --- |
| **NNVDC** — *A new versatile density-based clustering method using k-Nearest Neighbors*, ESWA 227 (2023) 120250, [DOI](https://doi.org/10.1016/j.eswa.2023.120250) | [Author-provided Drive file](NNVDC/PROVENANCE.md), SHA-256 recorded; [status](NNVDC/STATUS.md). **Archive only.** | No `fit_predict` or CLI; no test. | `NNVDC.R` is a Windows-path global script that assumes a truth column, scans K, and reports labelled metrics.  R is unavailable.  A safe wrapper would be a new, separately documented reproduction. |
| **IEPC** — *Information entropy peaks clustering*, ESWA 288 (2025) 128197, [DOI](https://doi.org/10.1016/j.eswa.2025.128197) | [Repository snapshot](IEPC/PROVENANCE.md) at `0876785…`; [status](IEPC/STATUS.md). **Archive only.** | No safe API or test. | Upstream MATLAB is incomplete (`L2_dist.m` and an expected demo dataset are absent), hard-codes K, scans `k=5:50`, and retains maximum NMI/ARI against truth. |
| **ATSDPC** — *Adaptive two-stage density peaks clustering with hybrid distance based on dispersion coefficient*, ESWA 282 (2025) 127639, [DOI](https://doi.org/10.1016/j.eswa.2025.127639) | [PyPI 1.1.5 sdist archive](ATSDPC/PROVENANCE.md), SHA-256 recorded; [status](ATSDPC/STATUS.md). **Archive only.** | No safe API or test. | Confirmed direct leakage: source extracts `y_true`, requires selected cores from distinct true classes, and selects the best of ten rounds by ACC.  Never import `atsdpc.main.run_clustering` for a fair comparison. |
| **scSCDT** — *Self-contrastive neural network with deep topology mining for scRNA-seq data clustering*, ESWA 298 (2026) 129751, [DOI](https://doi.org/10.1016/j.eswa.2025.129751) | [Author-associated snapshot](scSCDT/PROVENANCE.md) at `bb27b3c…`; [status](scSCDT/STATUS.md). **Archive only.** | No method code, API, or test. | Upstream contains only “Coming soon...” and data; no model/loss/training/inference implementation exists to download or adapt. |
| **BNFW** — Li et al., *Boundary and noise detection clustering for data with fuzzy boundaries and weak connectivity*, ESWA 299 (2026) 129714, [DOI](https://doi.org/10.1016/j.eswa.2025.129714) | [Material-status record](BNFW/STATUS.md). **Blocked.** | No source/API/test. | Metadata and searches do not supply equations, pseudocode, or code.  Do not label a generic boundary/noise method as BNFW. |
| **GADPC** — Xu & Jiang, *A Graph Adaptive Density Peaks Clustering algorithm for automatic centroid selection and effective aggregation*, ESWA 195 (2022) 116539, [DOI](https://doi.org/10.1016/j.eswa.2022.116539) | [Material-status record](GADPC/STATUS.md). **Blocked.** | No source/API/test. | Centroid-selection, graph-adaptation, aggregation, and unknown-K rules cannot be faithfully recovered from metadata alone. |
| **PDCSN** — Xing et al., *A partition density clustering with self-adaptive neighborhoods*, ESWA 227 (2023) 120195, [DOI](https://doi.org/10.1016/j.eswa.2023.120195) | [Material-status record](PDCSN/STATUS.md). **Blocked.** | No source/API/test. | Partition-density/neighbourhood/cluster-extraction equations and a parameter protocol are absent; a guessed DPC/DBSCAN hybrid would misrepresent PDCSN. |
| **DICN** — Wang et al., *Deep Inference Clustering Network with Information Maximization*, ESWA 292 (2025) 128578, [DOI](https://doi.org/10.1016/j.eswa.2025.128578) | [Material-status record](DICN/STATUS.md). **Blocked.** | No source/API/test. | Architecture, information-maximisation objective, training schedule, input domain, and K protocol are unavailable. |
| **Neighborhood context-aware contrastive clustering** — Yin et al., ESWA 302 (2026) 130574, [DOI](https://doi.org/10.1016/j.eswa.2025.130574) | [Material-status record](Neighborhood_Context_Aware_Contrastive_Clustering/STATUS.md). **Blocked.** | No source/API/test. | No full text/code establishes its input modality, neighbourhood construction, objective, or K protocol; a generic contrastive implementation would be misleading. |
| **CGC** — Xie et al., *Contrastive graph clustering with adaptive filter*, ESWA 219 (2023) 119645, [DOI](https://doi.org/10.1016/j.eswa.2023.119645) | [Eligibility record](CGC/STATUS.md). **Excluded.** | Not applicable. | Published method requires an attributed graph plus features.  Constructing a kNN graph from TopoGate's `X` would add an unreported graph-building baseline and invalidate a like-for-like comparison. |
| **Robust CGC** — Li et al., *Robust contrastive graph clustering via reliable augmentation and two-stage self-supervision*, ESWA 302 (2026) 130538, [DOI](https://doi.org/10.1016/j.eswa.2025.130538) | [Eligibility record](Robust_CGC/STATUS.md). **Excluded.** | Not applicable. | Like CGC, it is an attributed-graph method.  Do not silently manufacture a graph and call the outcome a paper reproduction. |

## Recommended benchmark registration

1. **Primary tabular baselines:** AHDPC, DPC-GFNN, GCC (clearly label native
   versus known-K GCC), and HARR only for datasets with defensible heterogeneous
   feature metadata.
2. **Conditional supplementary baselines:** LCG-DSC after a MATLAB/Octave
   end-to-end validation, and UEC with its compatibility-mode limitation stated.
3. **Do not register as runnable:** NNVDC, IEPC, ATSDPC, scSCDT, BNFW, GADPC,
   PDCSN, DICN, Neighborhood context-aware contrastive clustering, CGC, and
   Robust CGC.  Their directory records are evidence/provenance, not an
   implementation claim.

For every registered method, retain its `summary.json` (where provided), state
whether K is explicit or native, apply the same quadratic-method sample cap,
and never use ACC/NMI/ARI/F1 to choose an algorithm parameter, seed, centre,
or stopping point.
