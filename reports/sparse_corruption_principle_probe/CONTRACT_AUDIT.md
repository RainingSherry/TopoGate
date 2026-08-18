# Local contract audit

The deterministic local audit completed with `audit_ok=true` after the C0/C1 artifacts were written.
This file is the pre-C2 contract snapshot; the later C2 decision and result are recorded in
`C2_RESULTS.md` and `C2_INTEGRITY_AUDIT.md`.

After boundary checks found that geometry scoring requested `n_neighbors=n` on small fixtures and
that P2 could overwrite raw nonzero values in dense H0 proxy-inactive positions, the library was
corrected to request at most `n-1` non-self neighbours and to swap P2 source/destination values. The
focused regression suite then had `16 passed`, including small-`n` and dense-proxy checks; the compact
C0/C1 audit remained unchanged before C2 authorization.

| check | result |
|---|---|
| holdout audit / minimum 8 / no development overlap | pass; 12 selected |
| outcome features used for holdout selection | empty |
| toy S/V/M and 18 toy rows | pass |
| C1 rows / finite values / zero fit / no labels loaded | pass; 54 rows |
| forbidden adaptive/GAN/performance artifacts under new result root | none |
| external auto-review | unavailable, no score; not used as evidence |

This audit was the pre-authorization record; it did not itself authorize or run the 54-job C2 GPU
matrix. The protocol owner subsequently authorized C2. See `C2_RESULTS.md` for the terminal result and
`C2_INTEGRITY_AUDIT.md` for the independent execution audit.
