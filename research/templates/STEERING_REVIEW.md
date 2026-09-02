# Steering Review

Project: `PRJ-*`
Date:
Review interval: recommended every 2 weeks or after a major confirmatory result.

## 1. Paper-level relevance
- What computer-science problem are we solving now?
- If the current local mechanism were solved perfectly, would the target task materially improve?
- Is the intended contribution still a method/system/task/analysis contribution that the target community would care about?

## 2. Progress quality
- What hypotheses were actually resolved?
- What evidence-backed claims were added?
- Did we mostly add failure explanations without narrowing method design?
- Did any model-version count create a false sense of progress?

## 3. Loop audit
- Are we reviving a frozen assumption under a new module/name?
- Are we tuning after seeing target outcomes?
- Are we expanding datasets/seeds mainly to make the mean positive?
- Are we adding complexity before checking oracle headroom and simple baselines?
- Are benchmark/seed/hyperparameter lotteries plausible explanations?

## 4. Cost audit
For each planned experiment:
- What decision will it change?
- Is there a cheaper existing-results analysis, oracle, toy model, frozen embedding test or sentinel experiment?
- Expected information gain / cost: high / medium / low.

## 5. Resource decision
- Continue:
- Freeze:
- Park:
- Replan:
- Next cheapest decisive action:

## 6. Counterfactual check
If starting today with zero sunk cost, would we still spend the next month on this project in its current form? Why?
