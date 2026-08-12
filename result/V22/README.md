# V22 Result Snapshot

This directory contains only public, auditable V22 metadata. Checkpoints,
topology memmaps, embeddings, predictions, caches, and launcher logs remain
outside the repository.

## Canonical cooperative Keep-Gate audit

`v22_full_cooperative_single_seed_20260812/` contains 16 jobs: 14 completed
records with passing artifact audits and 2 `incomplete_compute` records
(`real_sim` and `covtype`). The protocol is
`v22_topology_discriminator_cooperative_keep_gate_v1`. This is single-seed,
full-component engineering evidence, not a multi-seed efficacy claim or a
configuration-selection result.

Completed labeled ARI values in the canonical audit are:

| Dataset | ARI |
| --- | ---: |
| cnae9 | 0.204229 |
| Mouse_retina | 0.395226 |
| Baron Human | 0.316141 |
| Campbell | 0.135521 |
| sector | 0.023828 |
| news20 | 0.018747 |
| rcv1_train | 0.001252 |
| sms_spam_collection | -0.050868 |
| sentiment_labeld_sentences | 0.002705 |
| hate_speech | 0.051966 |
| imdb | -0.000161 |
| mnist | 0.314758 |

The two PBMC records are unlabelled and therefore have no ARI/NMI/ACC in this
snapshot. The aggregate report is the authoritative status table.

## Controls and diagnostics

- `v22_full_single_seed_20260812/` is the original hard-gate control; it has
  10 completed jobs and 2 `incomplete_compute` records.
- `v22_full_resource_recovery_20260812/` is a separate resource-recovery
  audit for the hard-gate protocol; all 12 jobs completed and passed artifact
  audits. It must not be read as a cooperative Keep-Gate result.
- `engineering_smoke_*` contains compact mechanism diagnostics and smoke
  metadata. These records are engineering evidence only.

## Provenance

The files in `dataset_manifests/` record source URLs, hashes, dataset profiles,
and label-isolation metadata. Local filesystem paths have been sanitized for
publication. Re-running requires obtaining the source data independently and
passing an explicit `n_clusters` for unlabelled inputs.

For a provenance-only dry run in this code snapshot, pass the selected manifest
with `--allow-missing-sources --dry-run`. Formal execution keeps strict source
file validation and requires the independently obtained data files.
