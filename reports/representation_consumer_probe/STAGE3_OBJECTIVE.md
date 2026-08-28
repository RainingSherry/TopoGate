# S3 — Objective Isolation (Permanently Locked)

本阶段不属于当前 `representation_consumer_probe` 执行链。S0 的 `adapter_not_estimable` 是
T-related causal chain 的 terminal state，因此不会执行 Rec-vs-Cut factorial，也不会用本文件
解锁新的 representation backbone。

如果未来需要比较 reconstruction 与 cut objective，必须新建独立项目，重新冻结 sample-edge
adapter、candidate family、budget、consumer inputs、K/readout、seeds 和 holdout。当前项目只保留
S1 opportunity-only 与必要的 S2 SimpleCut confirmation；不能把历史 S3 草案当成授权或结果。
