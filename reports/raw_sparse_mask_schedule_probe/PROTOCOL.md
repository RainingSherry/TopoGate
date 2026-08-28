# Protocol

`raw_sparse_mask_schedule_probe_v1` consumes exactly the raw numeric `x`
field resolved from the audited local E3 summaries. Each source is hashed;
missing or ambiguous sources terminate the project as `INCOMPLETE_COMPUTE`.

For feature `j`, fit once on the full unsupervised matrix
`s_j=sqrt(mean_i X_ij^2)` with a floor of `1e-6`, then use `X0=X/s`. No
centering, clipping, feature selection, PCA, or label-derived transform is
allowed. Exact zeros remain zeros. A row with no active coordinate has a zero
mask budget and is never changed.

The fixed mask budget is `ceil(0.25 * nnz_i)`, capped by the available
coordinates. Variable schedules draw a deterministic per-row ratio uniformly
from `[0.05, 0.45]` each epoch. `ALL` samples from all coordinates; `ACTIVE`
samples only from exact active coordinates. Masked arms use selected-coordinate
reconstruction loss, while `CLEAN_AE` uses all-coordinate MSE. Every epoch
records counts and scalar mask/loss audits without persisting masks.

The encoder is `d→64→32`, followed by `32→64→d`, ReLU activations, Adam
(`lr=1e-3`, `weight_decay=0`), and 30 epochs. Batch size is selected once per
dataset by the outcome-independent `[512,256,128,64]` forward/backward
preflight and then frozen. For each dataset/seed, model initialization and
batch-order hashes are paired across arms.

After fit, clean `X0` is encoded; only then are labels loaded for known-K
KMeans, ARI (primary), NMI and ACC (secondary). Effective rank and variance
diagnostics are computed before labels.

The SVD32 baseline is a separate label-free `TruncatedSVD` representation.
Fixed-ratio and representation-space probes are conditional and cannot alter
the primary matrix or gates.
