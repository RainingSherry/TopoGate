# Notion Research OS links

Live dashboard:
- https://app.notion.com/p/3bf56f803877811799d4e5873398840e?pvs=204

Databases:
- Projects: https://app.notion.com/p/d5087e8d356941bbbc4c7582fb8c50ef
- Questions / Observations / Gaps / Ideas: https://app.notion.com/p/61887fc4626d492d83e37f5937f8f8ff
- Hypotheses / Predictions: https://app.notion.com/p/050ab220ee774469b244cf939e6ed62c
- Experiments: https://app.notion.com/p/efd599bd60914b64b8f475c12db560c1
- Evidence: https://app.notion.com/p/ab25b5c022fa45b09ca703070cad3cdb
- Decisions: https://app.notion.com/p/474041f8f8324fe3b9e5963e5138f188
- Claims: https://app.notion.com/p/54019e6958874609aa788383be6b7700

Existing literature database is reused rather than duplicated. Notion `Questions` can relate directly to existing paper records.

## Synchronization rule

There is no bidirectional automatic mirror by design.

- Notion is authoritative for the **current scientific state**.
- GitHub is authoritative for **frozen experiment facts**.

When an experiment changes from `candidate` to `frozen`:
1. create/finalize the Notion experiment record;
2. copy its stable `EXP-*` ID into a GitHub contract under `research/contracts/`;
3. commit the contract before the confirmatory run;
4. record the resulting commit SHA back in Notion;
5. after the run, write the curated GitHub report and create an `EVD-*` item in Notion;
6. record a `DEC-*` decision with explicit reopen condition.

This keeps the two systems linked without maintaining duplicate result copies.
