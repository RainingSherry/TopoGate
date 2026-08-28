# Adaptive-corruption probe

`adaptive_corruption_probe` is an independent, non-V-series mechanism study.
It starts from the audited relation-selection release at frozen commit
`c80877cf904e41950315d37b95374825c33a7362`, but it shares no selector,
corruption, loss, model or holdout outcome with
`learned_relation_rule_probe`.

## Finite question

> In naturally sparse, high-dimensional data, does the corruption principle
> target clustering-relevant information, and is an adaptive or adversarial
> corruption policy necessary?

GAN is a late candidate, never the starting hypothesis.  The first question
is whether fixed corruption is a bottleneck at all.

## Stage order and status

```text
B0/S0 Problem and resource freeze
  -> B1 Corruption opportunity library
  -> B2 Adaptive-location necessity (conditional)
  -> B3 Generator/GAN necessity (conditional)
  -> B4 Minimal adaptive or generator model (conditional)
  -> B5 Independent holdout (conditional)
```

Only B1 is authorized at creation.  B2 requires a material heterogeneous
opportunity in B1; B3 requires adaptive location to beat frozen static rules;
B4/B5 remain locked until their preceding gates pass.

## Information decomposition

The protocol distinguishes support (`1[x_ij != 0]`), non-zero value magnitude,
and mixed/co-occurrence information.  A corruption that merely raises
reconstruction loss is not automatically useful for clustering.

## Non-negotiable boundaries

- B1 uses a fixed matched corruption library; no GAN, contrastive objective,
  graph module or architecture search is authorized.
- All C0--C4 arms share backbone, decoder, optimizer, epochs, readout, K
  protocol, seeds, requested/effective budget and audited change statistics.
- Support/value semantics are frozen before seeing clustering results.
- Encoder, decoder and corruption selector never receive `y`, ARI, NMI or ACC.
- `L_rec` and ARI are audited separately; harder reconstruction is not a
  success criterion.
- GPU use is explicit (`[1, 2, 3, 4, 5, 6]`); physical GPUs `0` and `7` are
  forbidden.  GPU availability does not authorize skipping B1.
- Raw data, checkpoints, embeddings, predictions and logs remain local and
  are excluded from GitHub.

The repository currently contains only the freeze contract.  No corruption
performance run is represented by this README.
