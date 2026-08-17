# Representation-Consumer Probe implementation boundary

`protocol.py` contains the frozen label-free contract, row-specific budget builders, diagnostic
oracles, and the Spectral consumer apparatus. `s0_audit.py`, `s1_opportunity.py`, and
`s2_simple_cut.py` implement the completed S0, S1-v2, and conditional S2 stages. The current
adapter audit is terminal `adapter_not_estimable`, so no `T_adapter`, TopoCut, S3, S4, S5, or S6
implementation may be added here. A future sample-edge selector must use a new
`relation_selection_probe` project.
