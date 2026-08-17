# RS0 freeze

RS0 freezes the project boundary, dataset roles, inherited references, feature
families, selector rules, seeds, margins, and output contract. It performs no
new model training. The inherited holdout is copied read-only from the closed
representation-consumer project before any selector result is inspected.

The old project is `CLOSED`; its S0/S1/S2 artifacts remain immutable inputs.
This project does not create `methods/TopoGate/V26` or `methods/RelationGate`.

The formal RS0 artifact records the source result-tree hashes, holdout
inheritance status, and resolved protocol configuration.
