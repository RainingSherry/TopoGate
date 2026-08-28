# E1 results — cross-domain opportunity and no-fit diagnostic

Protocol: `corruption_objective_compatibility_probe_e0_e4_v1`.
The complete E1 matrix is `54/54` (six datasets × three arms × three paired
seeds). Biological P0/P2 entries are audited C2 controls reused only after
current H0, budget-manifest, and post-fit label hashes matched; 36 Clean/new
non-biological GPU cells were newly computed on physical GPU 6. No run used
GPU 0 or 7. The no-fit E1b control is complete `54/54` on CPU.

All E1 fit and corruption operations use standardized dense H0. Labels are
loaded only after the transformation/fit to obtain benchmark-known K and
outer metrics. The no-fit control materializes its H0-derived feature matrix
before loading labels. Raw-X support is not used.

## Dataset-level primary quantities

`Delta_random = ARI(P2)-ARI(P0)` and `Delta_clean = ARI(P2)-ARI(Clean)` use
the final epoch-30 clean embedding and are means over the three paired seeds.
`Training_amplification = Delta_random - nofit_Delta_random` is diagnostic.

| Dataset | Role | ARI Clean | ARI P0 | ARI P2 | Delta random | Delta clean | no-fit Delta random | Amplification |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Mouse_retina | biological scRNA development | 0.754456 | 0.439834 | 0.834732 | **+0.394898** | **+0.080276** | −0.474125 | +0.869024 |
| Baron Human | biological scRNA development | 0.144447 | 0.214700 | 0.340769 | **+0.126069** | **+0.196322** | −0.038151 | +0.164220 |
| Campbell | biological boundary control | 0.042775 | 0.023504 | 0.170387 | **+0.146883** | **+0.127612** | −0.126054 | +0.272937 |
| cnae9 | non-biological sentinel | 0.038605 | 0.035703 | 0.038144 | +0.002441 | −0.000461 | −0.034553 | +0.036994 |
| hate_speech | non-biological sentinel | 0.002692 | 0.010202 | 0.002908 | −0.007294 | +0.000216 | +0.031485 | −0.038780 |
| sms_spam_collection | non-biological sentinel | 0.161677 | 0.177139 | 0.156828 | −0.020311 | −0.004849 | −0.006942 | −0.013369 |

The first three rows reproduce the previously observed C2 P2-vs-P0 deltas;
they are development evidence, not independent generalization evidence. The
non-biological panel supplies the cross-domain test and does not pass it:

- G1 cross-domain opportunity: `0/3` datasets meet both material model deltas
  and both two-of-three seed-positive checks;
- G2 training amplification: `1/3` datasets meet `≥0.03` (cnae9 only);
- E2 objective matrix: **not launched** by the frozen G1+G2 gate.

## Decision boundary

The result supports a bounded statement only: P2 remains highly beneficial on
the three burned biological development datasets under the matched probe, but
the frozen six-dataset sentinel experiment did not establish a cross-domain
corruption opportunity and did not authorize objective-specific follow-up.
This is not evidence that P2 is universally useless, nor a raw-X support
causal test. The support-specific attribution line remains frozen and D2 was
not authorized.
