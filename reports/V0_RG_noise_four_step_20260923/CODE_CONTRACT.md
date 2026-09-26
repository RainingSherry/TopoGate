# Bundle interfaces

Python-only; no training. Modules in scripts/ are executable from that directory.

common.py (parent-owned) exposes:
- load_config(path) -> dict with _config_path, _config_sha256
- load_reference(config) -> existing audited raw-input evaluator module (V0_RG only)
- load_case(config, dataset, model_seed, variant='full') -> dict: ref, entry, checkpoint(Path), summary, cfg(V0_RGConfig), state(preprocessor), raw(float32 ndarray), raw_original_dtype(str), raw_cast_fraction(float), labels, x(frozen clean ndarray), test_indices, saved_embedding, saved_predictions, centers, provenance(dict). No model loaded until caller needs it.
- load_encoder(case, device) -> frozen model; cpu or permitted GPU only, audit physical mapping
- encode(case, model, matrix, device) -> embedding
- replay_clean(case, model, device) -> clean embedding and audit dict; must match saved embeddings and predictions
- corrupt(case, kind, severity, noise_seed) -> noisy_raw, noisy_x, metadata(raw dtype-normalized fraction, hash)
- atomic_json(path, obj), array_hash(array), sha256(path), code_hashes(), write_npz(path, **arrays), condition_slug(kind,severity,seed).

Probe script (agent-owned) CLI: --config, --dataset, --model-seed, --variant, --stage {observe,directions}, --device cpu(default) or cuda:4; optional --noise-seeds 3001 ... --condition replacement:0.2. Outputs each case/stage/variant/model_seed/condition/result.json plus samples.npz. JSON top level has protocol/dataset/variant/model_seed/corruption/severity/noise_seed/input/representation/prediction/provenance/clean_replay/status. Steps1-2 use observe; Step4 directions explicitly invoked. No rerun if matching task fingerprint successful; fail/record on mismatch.

Aggregation/plotting (agent-owned) receives --results ROOT, --out DIR (and optional --snapshot legacyCSV for discovery preview only). Aggregates seed means first; no fake independent n=15. Step3 paired comparisons preserve same dataset/model_seed/noise_seed/corruption/severity and corrupted input hash and matching test indices; full vs ablation effectiveness from audit inventory. Existing snapshot remains labeled exploratory legacy Euclidean/all-rows.

Inventory (parent-owned) writes inventories/checkpoints.json, checkpoints.csv and marks no-op controls; graph summary mean/min/max gates check. Current full configs: Mouse_retina edge_reliability_mode=none, gate_adaptivity=.25; micro-mass edge sim_mutual_snn, adaptivity=0; har and dermatology edge none and adaptivity=0. No-op controls cannot support component effects. Check actual artifacts and seed-specific config, do not hardcode expected statuses as facts.

Observation graph: PCA fitted on same clean fit scope as original, replay graph profile against graph_gate_summary where possible. Original build function fit_transform vs transform may differ at floating point precision; audit without silently accepting graph mismatch. Keep model's PCA/L2/cosine reproduction and self-excluded diagnostics distinct if original tie/self bug is detected. Original checkpoint k differs per dataset (Mouse_retina 10, micro-mass 20, others 30). Do not force k=30 everywhere.

Runtime constraints: allow GPU1/4/5, never0/7, no model refitting, no force-changing variants or hyperparameters to make ablations effective. Step4 uses common full-derived label-free directions/norms applied identically to all paired variants and is model-input-only (separate from raw-input observation noise).
