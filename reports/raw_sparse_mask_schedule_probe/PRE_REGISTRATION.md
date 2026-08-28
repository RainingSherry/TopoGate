# Raw Sparse Mask Schedule Probe — pre-registration

This is an independent, local-only mechanism study. It is not a V-series
release and it does not reopen TopoGate, ACCG, C2, or any earlier project.

## Frozen question

On six predeclared sentinel matrices, does a small masked autoencoder obtain a
cross-domain representation-learning advantage by masking exact raw active
(non-zero) coordinates rather than all coordinates, and by using a variable
5–45% schedule rather than fixed 25% masking?

The six datasets are Mouse_retina, Baron Human, Campbell (biological) and
cnae9, hate_speech, sms_spam_collection (non-biological). They are a mechanism
panel, not a holdout or generalization benchmark.

## Matrix and estimands

The primary matrix is 6 datasets × 5 arms × 3 paired seeds `[42, 123, 7]`:
`CLEAN_AE`, `ALL_FIXED`, `ACTIVE_FIXED`, `ALL_VARIABLE`, and
`ACTIVE_VARIABLE`. Every arm uses the same `d→64→32→64→d` ReLU MLP, Adam
(`1e-3`), 30 epochs, clean embedding readout, and known-K KMeans only after
fit. `ACTIVE` means exact `X0 != 0`, where `X0` is zero-preserving scaled raw
input; it is not a biological missingness claim.

Primary paired effects are `ACTIVE_FIXED − ALL_FIXED`,
`ACTIVE_VARIABLE − ACTIVE_FIXED`, `ALL_VARIABLE − ALL_FIXED`, and their
interaction. The materiality margin is 0.03 ARI with at least two of three
paired seed deltas in the same direction.

## Frozen gates

- G0 requires all 90 valid cells, complete source and adapter hashes, exact
  zero-pattern preservation, paired initialization and batch schedules, mask
  audits, labels-after-fit-only, and legal GPUs.
- G1 requires active-fixed positive material effects on at least 2/3 datasets
  in each domain.
- G2 requires active-variable positive material effects on at least 2/3 in each
  domain and at most one material-negative dataset.
- G3 compares the predeclared candidate (`ACTIVE_VARIABLE` if G2 passes,
  otherwise `ACTIVE_FIXED` if G1 passes) against both `CLEAN_AE` and SVD32 in
  both domains. A failure does not authorize a new model.

If both G1 and G2 fail, the input-level route closes. If G1/G2 pass but G3
fails, only the separately frozen representation-space localization probe may
run. No per-dataset best arm is a deployable result.

## Label and publication firewall

Training, scaling, masking, loss, and model selection receive no `y`, `K`,
ARI, NMI, or ACC. Labels are loaded only after fit for the outer benchmark
readout. Raw data, labels, scales, masks, arrays, embeddings, predictions,
weights, checkpoints, logs, and caches are excluded from compact publication
artifacts. The study has no holdout claim.

## Locked routes

No new Gate, topology selector, NeighborMix/ACCG rescue, support matcher
optimization, GAN, learned generator, Transformer/attention sweep,
corruption-rate tuning, feature-importance selector, residual/geometry
selector, dataset removal, or automatic `BUILD_NEW_MODEL` output.
