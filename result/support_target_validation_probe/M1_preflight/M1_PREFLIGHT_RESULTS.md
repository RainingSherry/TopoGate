# M1 Magnitude Estimability Preflight

Status: `magnitude_match_not_estimable`; rows: `9/9`; GPU runs started: `0`.

This is a no-training structural preflight. A tolerance failure is `magnitude_match_not_estimable`, not a performance negative.

| Dataset | Seed | Total L1 mismatch | Median row mismatch | Estimable |
|---|---:|---:|---:|---:|
| Mouse_retina | 42 | 0.001582 | 0.028331 | True |
| Mouse_retina | 123 | 0.001647 | 0.028435 | True |
| Mouse_retina | 7 | 0.001686 | 0.028431 | True |
| Baron Human | 42 | 0.094640 | 0.084183 | False |
| Baron Human | 123 | 0.095877 | 0.083825 | False |
| Baron Human | 7 | 0.094646 | 0.083729 | False |
| Campbell | 42 | 0.005726 | 0.062658 | True |
| Campbell | 123 | 0.006001 | 0.062549 | True |
| Campbell | 7 | 0.005828 | 0.062785 | True |

- Frozen total-L1 tolerance: `0.05`.
- Frozen median-row tolerance: `0.1`.
- Formal M1 GPU authorization: `False`.
- No M1 model was constructed or trained; M2/M3/M4/adaptive/GAN remain locked.

> Support in C2/M1 denotes threshold-defined support of dense H0, not raw-X zero/nonzero support; raw sparse-support claims require a separate validation.
