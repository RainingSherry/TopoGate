# A0/S0 freeze — learned-relation-rule probe

## Freeze result

At project creation this is a protocol-only freeze.  No A1 model, selector,
Spectral run or clustering result is implied.  The freeze audit must produce
`status=completed_valid` before an A1 launcher is allowed to run.

## Frozen contract

```yaml
project_id: learned_relation_rule_probe
protocol_id: learned_relation_rule_probe_a0_v1
base_commit: c80877cf904e41950315d37b95374825c33a7362
authorized_initial_stage: A1
locked_until_gate:
  - A2_transfer
  - A3_label_free
  - A4_learned_rule
  - A5_holdout
development_datasets: [cnae9, Campbell, sms_spam_collection]
sentinel_datasets: [Mouse_retina, Baron Human, hate_speech]
material_delta_ari: 0.03
capture_threshold: 0.25
primary_seeds: [42, 123, 7]
holdout_seeds: [42, 123, 7, 3032, 3033]
legal_gpu_pool: [1, 2, 3, 4, 5, 6]
forbidden_gpu_ids: [0, 7]
labels_used_during_fit: false
diagnostic_supervision_allowed_only_in: A1_target_builder
```

The twelve-dataset holdout is inherited by reference from the dormant
label-free-characteristics manifest and is independently hashed in the S0
artifact.  Its membership is not editable after this freeze.

## S0 audit checklist

- [ ] base commit and old-project terminal status recorded;
- [ ] old project and V-series paths are read-only inputs;
- [ ] development/sentinel/holdout roles are explicit;
- [ ] candidate-pool and budget reuse are explicit;
- [ ] A1 target is marked diagnostic-only;
- [ ] grouped-anchor split is frozen;
- [ ] GPU allow/deny list is explicit;
- [ ] publication exclusion list is explicit;
- [ ] no formal performance artifacts exist under this project's S0 result.

The machine-readable audit is the authority for launcher admission; this
checklist is explanatory documentation.
