# Project status

Last project-level review: 2026-08-17.

## Archived / frozen method family

### TopoGate / ACCG intervention-selection line

Status: **FROZEN**

Evidence boundary:
- V1-V25 tested multiple node/edge, soft/hard, predictive, assignment and adversarial topology-gating variants.
- ACCG synthetic v3 demonstrated that non-additive joint action structure can be measured under its declared synthetic contract.
- The locked real ACCG panel did not support promotion as a clustering-improvement method: mean ARI(T_c)-ARI(T_s) was +0.007492, median +0.000363, dataset-bootstrap 95% CI [-0.000879,+0.018889], and joint selection underperformed the coordinate control on the development subset.

Decision:
- Do not continue outcome-driven tuning of epsilon/lambda, feature graphs, gate architectures, donor rules, attention blocks or larger benchmark panels to rescue the same claim.
- Preserve code, benchmark infrastructure, matched protocols and negative evidence as reusable assets.

Reopen condition:
- independent new evidence must show that topology/structural actionability reliably predicts or directly improves representation/clustering utility, **or** a new task formulation must materially change the scope of the previous negative evidence.

## Active project-level exploration

Goal: find a bounded computer-science problem in high-dimensional sparse representation/clustering that has:

1. a clear input/output/action space;
2. an important and documented gap;
3. measurable oracle/ceiling headroom;
4. a strong simple baseline;
5. remaining headroom that specifically motivates a new method;
6. a 1-3 day kill test before large-scale model development.

Candidate families are exploratory only:
- LLM/Agent-assisted clustering;
- algorithm selection / AutoML for clustering;
- self-supervised learning objective vs clustering geometry;
- sparse-native representation learning.

None is approved as the next method until it passes the Idea Gate.
