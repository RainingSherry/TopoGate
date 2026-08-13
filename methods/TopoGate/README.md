# TopoGate implementations

TopoGate 的各代实现按目录隔离。维护或复现实验前，请先阅读
[CORE_CODE_INDEX.md](CORE_CODE_INDEX.md)，再进入对应版本的 README、配置和 runner。

当前入口：

- V9 legacy：`learnable_gate/run_npz.py`
- V10：`v10_reliable_graph/run.py`
- V11：`V11/run.py`
- V12 latent topology：`V12_latent_topology/run_npz.py`
- V13 hard gate：`V13_hard_gate/run_npz.py`
- V15 counterfactual gate：`V15_counterfactual_gate/run.py`
- V16 / V16.1 predictive graph gate：各目录的 `run.py`
- V17 topology-native reference：`V17_topology_native/run.py`（机制验证，尚无性能结论）
- V18 scMAE latent gate：`V18_scmae_latent_gate/run.py`
- V19 RG adapter：`V19_rg_adapter/run.py`
- V20 topology-conditioned adversarial mask：`V20_topology_conditioned_adv_mask/run.py`
- V21 assignment-adversarial gate：`V21_assignment_adversarial_gate/run.py`
- V22 discriminator-guided hard mask / cooperative Keep-Gate：
  `V22_topology_discriminator_hard_mask/run.py`
- 历史 static/V6/V7：仅用于冻结对照和研究复盘

目录之间不共享可变训练状态。V9 的旧配置名不代表 V10/V11/V12 源码；每个新版本
都必须从自己的目录、配置和输出契约确认。正式结果位于 `result/final_results/`，
短期 smoke 位于本地临时目录，发布快照不包含 smoke、checkpoint 或原始输出。
