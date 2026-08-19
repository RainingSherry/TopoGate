# Pre-registration — corruption_objective_compatibility_probe

**Frozen before any new GPU run:** protocol ID
`corruption_objective_compatibility_probe_e0_e4_v1`, panel, arms, objectives,
seeds `[42, 123, 7]`, material margin `0.03`, objective interaction rule,
GPU allow/deny lists, timeout/retry policy, and the H0/raw-X interpretation
firewall are defined in `protocol.py` and `PROTOCOL.md`.

## Decision rules

1. E0 technical failure stops the unattended launch. A corrected D1 gate
   failure freezes support attribution and never authorizes D2.
2. E1 `G1` is a cross-domain opportunity gate: at least 2/3 non-biological
   datasets pass both model deltas and both two-of-three seed-positive checks.
3. E1 `G2` is a learning-amplification gate: at least 2/3 non-biological
   datasets have amplification ≥ 0.03.
4. E2 runs only after G1 and G2. A strong objective candidate follows the
   four-of-six, one-biological/one-non-biological, at-most-one-negative rule.
5. Missing or invalid cells never enter a mean, winner count, or interaction;
   the stage is `incomplete_compute` and no positive claim is emitted.

The allowed terminal labels are `STOP_GENERAL_CORRUPTION`,
`REPRESENTATION_NOT_OBJECTIVE`, `STATIC_CORRUPTION_REPLICATION`, and
`CORRUPTION_AWARE_OBJECTIVE_OPPORTUNITY`. The program cannot create a new
model or unlock any route.
