# B0/S0 freeze — adaptive-corruption probe

## Freeze result

This is a protocol-only freeze.  No corruption arm, reconstruction model,
GAN, or clustering result has been run by creating this artifact.  B1 is
authorized only after the machine-readable audit reports
`status=completed_valid`.

## Frozen contract

```yaml
project_id: adaptive_corruption_probe
protocol_id: adaptive_corruption_probe_b0_v1
base_commit: c80877cf904e41950315d37b95374825c33a7362
authorized_initial_stage: B1
locked_until_gate:
  - B2_adaptive_location
  - B3_generator_necessity
  - B4_adaptive_or_generator_model
  - B5_holdout
development_panel: [sms_spam_collection, hate_speech, Mouse_retina, Baron Human, cnae9, Campbell]
arms: [C_clean_no_corruption, C0_MatchedRandom, C1_ValueOnly, C2_SupportOnly, C3_MixedMatched, C4_StaticHard]
backbone:
  input: audited_S0_H0
  encoder_decoder: "d_eff->64->32->64->d_eff ReLU; d_eff is frozen S0 H0 width"
  optimizer: Adam
  learning_rate: 0.001
  epochs: 30
  batch_size: 512
  corruption_rate: 0.25
support_definition: "abs(H0_ij)>=max(1e-6,0.05*row_max_abs)"
pair_budget_rule: "m_i=min(ceil(0.25*active_i),floor(active_i/2),inactive_i); all arms change 2*m_i coordinates"
material_delta_ari: 0.03
primary_seeds: [42, 123, 7]
holdout_seeds: [42, 123, 7, 3032, 3033]
legal_gpu_pool: [1, 2, 3, 4, 5, 6]
forbidden_gpu_ids: [0, 7]
labels_used_during_fit: false
positive_control_required_before_null: true
decision_hierarchy:
  level_1: "corruption effect vs C_clean via ARI(C0)-ARI(C_clean) and Delta_clean library"
  level_2: "structured-vs-random via Delta_random(C)=ARI(C)-ARI(C0), C1-C4"
  level_3: "B2 only if >=2 role classes have material, distinct structured winners"
  terminal_if_level_1_positive_but_level_2_negative: random_corruption_sufficient
cross_track_holdout_disjointness_required: true
```

## S0 audit checklist

- [ ] base commit and old projects are recorded as read-only inputs;
- [ ] structural dataset roles are frozen independently of ARI outcomes;
- [ ] support/value/mixed semantics and matched-budget fields are explicit;
- [ ] B1 is the only authorized performance stage;
- [ ] adaptive location and GAN are locked behind B1/B2 gates;
- [ ] hardness and clustering utility are separate endpoints;
- [ ] corruption impact and structured-vs-random utility are separate gates;
- [ ] GPU allow/deny list is explicit;
- [ ] label and publication firewalls are explicit;
- [ ] no formal performance artifacts exist under this project's S0 result.

The machine-readable audit is the authority for launcher admission.
