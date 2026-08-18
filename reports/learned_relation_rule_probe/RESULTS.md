# A1 results — diagnostic supervised ceiling

The formal A1 matrix completed the three burned development datasets × two
diagnostic scorers × three frozen feature views with five GroupKFold-by-anchor
folds.  OOF coverage was 100% and anchor groups were disjoint in every fold.
The best frozen configuration per dataset was:

| dataset | best scorer/view | `Delta_sup` mean | `H_pool` mean |
|---|---|---:|---:|
| cnae9 | TinyMLP / No-rank | `-0.015740` | `+0.215720` |
| Campbell | Logistic / No-geometry | `+0.024503` | `+0.191444` |
| sms_spam_collection | Logistic / Full | `-0.300232` | `+0.367108` |

The frozen A1 gate therefore terminates as
`predictable_reference_not_actionable_for_selection`.  Diagnostic AP/AUPRC
does not override the clustering-action gate, and this supervised ceiling is
not label-free utility.  A2--A5 are not authorized.

The twelve-dataset holdout remains locked and unused.  Raw scores, graphs,
embeddings and predictions remain local and are excluded from publication.
