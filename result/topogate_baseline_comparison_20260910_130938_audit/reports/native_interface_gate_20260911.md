# Native interface gate (2026-09-11)

The CLUBench implementations were inspected directly on the remote server.

| model | exposed fitting API | test-safe inductive API | decision |
|---|---|---|---|
| IDEC | `fit_predict(X)` | no `predict`/`transform` | blocked for strict inductive panel |
| EDESC | `fit_predict(X)` | no `predict`/`transform` | blocked for strict inductive panel |
| TableDC | `fit_predict(X)` | no `predict`/`transform` | blocked for strict inductive panel |
| ZEUS | `fit_predict(X)`; internally fits PCA/scaler and KMeans | no embedding-only API | blocked until wrapper exposes representation-only inference |

The strict split protocol cannot call these methods on the complete matrix,
because that would fit a transformation or clustering readout using test
samples. The new `tools/run_strict_zeus_baseline.py` is a refusal gate that
records this condition rather than producing invalid formal metrics.

This is not a scientific failure of the methods. It is an interface/protocol
failure for the requested inductive main panel. Such runs may be placed in a
separate transductive panel only after the protocol is explicitly changed and
the corresponding ToPoGate transductive control is run too.
