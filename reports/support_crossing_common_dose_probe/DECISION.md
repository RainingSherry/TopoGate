# Decision

The D1 computation is complete and structurally audited, but the frozen
constructive common-dose gate fails:

- Mouse_retina passes;
- Baron Human fails both common-row coverage and magnitude tolerances;
- Campbell fails the dataset-total magnitude tolerance.

Therefore the terminal decision is:

```text
common_dose_not_estimable
```

This means the Cross/Preserve comparison is not estimable under this specific
constructive witness and tolerance contract. It is not an ARI negative result,
does not overturn C2, and does not prove that no alternative matching could
find a common dose. D2, raw-X bridge, holdout, adaptive policy and GAN remain
locked.
