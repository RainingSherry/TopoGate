# Research OS

This directory is the **frozen evidence/protocol layer** for the TopoGate research program.

Live scientific state is maintained in Notion:
- Research OS: https://app.notion.com/p/3bf56f803877811799d4e5873398840e?pvs=204

## Division of responsibility

### Notion = live scientific state
Maintain:
- projects and paper-level goals;
- problems, observations, gaps and candidate ideas;
- hypotheses and predictions;
- experiment proposals and decisions;
- evidence interpretations and paper-safe claims.

Do **not** duplicate seed-level logs, full configs, result tables or source code there.

### GitHub = frozen and reproducible facts
Maintain:
- preregistered/frozen experiment contracts;
- exact code/config/commit used by an experiment;
- reproducible runners and environment requirements;
- curated metrics and reports;
- negative results and decision snapshots.

A Git commit or model version is **not** a unit of scientific progress. Scientific progress is measured by hypotheses resolved and evidence-backed claims gained.

## Scientific workflow

```text
Idea
  -> Idea Gate
  -> literature + local foundations
  -> bounded research question
  -> oracle / ceiling / simple baseline
  -> hypothesis + prediction
  -> cheapest discriminating experiment
  -> frozen experiment contract
  -> run / evidence
  -> decision (GO / NO-GO / FREEZE / REPLAN)
  -> claim or escalation
```

The experimental loop is allowed only after a project-level steering check confirms that the question still contributes to the intended computer-science paper.

## Stable IDs

Use the Notion-generated IDs and include them in frozen GitHub artifacts when available:

- `PRJ-*` project
- `RQ-*` problem / observation / gap / idea
- `HYP-*` hypothesis
- `EXP-*` scientific experiment
- `EVD-*` evidence item
- `DEC-*` research decision
- `CLM-*` paper claim

Compute runs are not scientific experiments. A single `EXP-*` may contain many datasets/seeds/runs.

## Required gates

Before creating a new model or launching a costly benchmark, answer:

1. What exact capability is being proposed?
2. What is the action space and evaluation target?
3. What oracle/ceiling estimates the available headroom?
4. What does a simple non-novel baseline already recover?
5. What remaining gap specifically requires the proposed method?
6. What is the cheapest result that would kill the formulation?
7. If the result is positive/negative, what project-level decision changes?

If the last question has no answer, the experiment should normally not run.

## Current TopoGate boundary

The topology-guided donor/mask/intervention-selection family is frozen as a default method-development route. V25 and ACCG remain valuable as negative-mechanism evidence and reusable experimental infrastructure, but new work should not revive the same core assumption by renaming the gate, donor policy, structural proxy, attention block, or backbone.

See `PROJECT_STATUS.md` and `templates/`.
