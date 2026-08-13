# NeighborMix-scMAE

NeighborMix-scMAE keeps the original scMAE `AutoEncoder` and mask-prediction
loss, then adds one conservative NeighborMix branch during training.

Default variant: `nm_scmae_mid`

- `alpha = 0.9`
- `neighbor_k = 5`
- `mix_neighbors = 4`
- `use_pseudo = true`
- `pseudo_weight = 0.3`
- `mask_ratio = 0.4`

The current pseudo-cell branch is an anchor-recovery training view: the mixed
neighbor expression is corrupted, but reconstruction is still supervised against
the original real cell. `mix_weight`, `consistency_weight`, and `target_mode`
are retained only for CLI compatibility with older runs and are not used by the
current training loop.
