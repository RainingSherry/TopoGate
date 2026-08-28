# support_target_validation_probe publication manifest

Publish only:

- this protocol/pre-registration and compact result reports;
- `summary.json`, `audit.json`, resolved configs and compact CSVs after an
  independently audited M1 completion;
- focused source code and tests needed to reproduce the contract.

Never publish:

- raw input matrices or labels;
- H0/budget arrays, action masks, embeddings, predictions;
- model weights/checkpoints or training logs;
- dormant holdout source data.

The result directory is local evidence. Any GitHub release must stage only the
allowlisted files above and verify the remote commit explicitly.

