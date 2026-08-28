# TopoGate implementations

TopoGate 的各代实现按目录隔离。维护或复现实验前，请先阅读
[CORE_CODE_INDEX.md](CORE_CODE_INDEX.md)，再进入对应版本的 README、配置和 runner。

当前入口：

- V9 legacy：`learnable_gate/run_npz.py`
- V10：`v10_reliable_graph/run.py`
- V11：`V11/run.py`
- V12 latent topology：`V12_latent_topology/run_npz.py`
- V17 topology-native reference：`V17_topology_native/run.py`（机制验证，尚无性能结论）
- 历史 static/V6/V7：仅用于冻结对照和研究复盘

目录之间不共享可变训练状态。V9 的旧配置名不代表 V10/V11/V12 源码；每个新版本
都必须从自己的目录、配置和输出契约确认。正式结果位于 `result/`，短期 smoke
位于 `/tmp`，不得把 smoke 当作性能结论。
