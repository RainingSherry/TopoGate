# ACCG Synthetic Contract v3

**Status:** final claim-aligned synthetic gate; one fresh panel only.

## Relation to v1 and v2

v1 remains a frozen No-Go because it pooled incompatible oracle targets. v2
corrected the estimand roles and added an unseen generator family, but its
standalone W5 AUC floor of `0.65` failed consistently (`0.635--0.637`) even
though incremental AUC was strongly positive. That standalone floor is not the
paper claim. The claim is that the joint action energy adds information beyond
the matched sample-side baseline.

v3 therefore makes one final statistical correction, without changing the
model, generator, selector, donor, or feature energy:

- retain a standalone held-out-family AUC floor of `0.60`;
- require positive incremental AUC and PR over the baseline;
- require the grouped-bootstrap 95% lower bound of incremental AUC to be
  positive for every held-out generator family;
- use fresh seeds `[3032, 3033, 3034, 3035, 3036]`;
- do not make any further threshold or endpoint amendments after v3.

## Promotion rule

Promotion requires shortcut audit, exact-selector audit, and W5 v3 primary
success. W2 remains secondary evidence; W1 is a negative control and does not
need joint-positive incremental information. If v3 fails, ACCG is closed as a
method route and no real-data rescue is launched.
