# S4 — Strong Backbone Validation (Permanently Locked)

本阶段在当前项目中永久锁定。由于 S0 `adapter_not_estimable`，本项目不创建 TopoCut-v0、
不执行 endpoint shuffle、不启动 scMAE/backbone comparison，也不把 DCGC/scSGC 的 cut objective
作为创新实现。

未来若要研究 sample-edge selector 或 cut-informed backbone，必须新建独立项目，重新冻结
relation adapter、budget、weights、readout、topology-specific negative control 和 holdout。当前
`representation_consumer_probe` 的终点是 S1/S2 opportunity Decision。
