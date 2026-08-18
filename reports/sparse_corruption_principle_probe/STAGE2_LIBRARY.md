# C2 finite static corruption library

The six primary principles are frozen in `protocol.py` and implemented in
`corruption_library.py`. Every row uses the same exact changed-coordinate budget:

```text
m_i = min(ceil(0.25*a_i), floor(a_i/2), inactive_i)
q_i = 2*m_i
```

P0 may alter support; P1/P3/P4/P5 operate on active coordinates and preserve the clean-reference
support; P2 swaps active values with threshold-inactive coordinates, moving the support role while
preserving the row nonzero-value multiset even when dense H0 contains small proxy-inactive entries.
P4 requires an explicitly frozen residual score for formal use. P5 uses a label-free local
cosine-neighbour contrast. GeometrySafe is a low-score paired fixture only.

The primary C2 estimand for the completed matrix is
`Delta_P(d)=ARI(P,d)-ARI(P0_Random,d)`. The tested maximum is not an oracle. Reconstruction loss is
reported as a diagnostic and cannot promote a principle by itself.
