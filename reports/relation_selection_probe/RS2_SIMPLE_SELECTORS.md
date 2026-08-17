# RS2 fixed simple selectors

RS2 evaluates five deterministic selectors on the same candidate pool, row
budget, edge weights, symmetrization, Spectral consumer, KMeans readout, and
paired seeds as the closed S1 reference. It changes only edge membership.

The selector scores are B0 cosine, B1 mutual-first, B2 Jaccard/SNN, B3
stability recurrence, and B4 fixed equal-rank fusion. No learned parameter or
post-result tuning is allowed.
