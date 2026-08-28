# Representation-Consumer Probe — Decision Boundary

当前项目不是 V 系列，也不是 architecture-search。S0 已确认当前 V21/V25 代码没有可审计的
feature-mask → sample-edge relation adapter，因此 `adapter_not_estimable` 是本项目 T-related
causal chain 的 terminal state，而不是新增 selector 的许可。

## 当前执行链

```text
S0 protocol + adapter audit
        ↓
S1 opportunity-only: F / U / R / O_pool / O_full + Spectral
        ↓
S2 opportunity confirmation: SimpleCut（仅当 S1 需要排除 spectral 假阴性）
        ↓
Decision: opportunity_absent_under_frozen_relation_family
         或 opportunity_present_within_frozen_candidate_pool
         或 candidate_family_requires_review
```

在本项目内永久锁定：S3 objective isolation、S4 strong backbone、S5 holdout、S6
paper-scale expansion、TopoCut 和任何新的 T selector。若以后研究 sample-edge selector，必须
新建 `relation_selection_probe`，不能把本项目的 `adapter_not_estimable` 绕开。

## S1/S2 只估计 opportunity

当前没有可估计的 `T_adapter`，因此不生成 `S_graph = ARI(T)-ARI(R)`，也不把 V25 的
feature-assignment `S_d^{V21}` 改名为 sample-edge 结果。S1/S2 只报告：

```text
H_pool = ARI(O_pool) - ARI(R)
H_full = ARI(O_full) - ARI(R)
C      = H_full - H_pool
```

这些是 label-derived diagnostic upper bounds，不能写成可部署方法性能。

## 终局规则

S1 的终局同时报告两个正交字段，避免把“pool 内已有机会”和“扩大 relation family 仍有额外机会”
混成一个类别：

```text
within_pool_opportunity = absent / present
candidate_gap = absent / present
```

其中 `candidate_gap=present` 表示 matched-budget 下 `H_full` 相对 `H_pool` 达到预注册的
candidate-gap margin；它不是总 candidate-recall loss。`b_i=min(8, positive_count_i)` 由
candidate pool 预先决定，所以 `O_full` 仍严格匹配该 budget；`fraction(b_i<8)` 与
`fraction(b_i=0)` 另作 positive-support deficiency 诊断。

1. 若所有 tested consumers 上 `H_pool` 与 `H_full` 都低于预注册 materiality margin，结论为
   `within_pool_opportunity=absent` 且 `candidate_gap=absent`，终局标签为
   `opportunity_absent_under_frozen_relation_family`。它只适用于冻结的 H0/cosine/k/budget/
   consumer family，不外推为“topology 在所有表示或数据上没有价值”。
2. 若 `H_full` 明显高于 `H_pool`，记录 `candidate_family_requires_review`；不实现新 backbone。
3. 若 `H_pool` 明显为正且 candidate gap 不明显，记录
   `within_pool_opportunity=present`、`candidate_gap=absent`，终局标签为
   `opportunity_present_within_frozen_candidate_pool`。这只说明未来需要研究 relation
   selection，不授权在本项目中补 selector。
4. 若 S1 发现明显的 oracle opportunity，S2 只能确认 opportunity 是否依赖 spectral relaxation。
   它不能把当前项目升级为 selection 或 representation-consumer 主张。

## 不可主张

- 不把 oracle graph 当作无标签方法性能；标签只能进入 diagnostic graph 和外层指标。
- 不把六个 stress datasets 当作 generalization benchmark。
- 不把 seed/run 数当作独立 dataset 数。
- 不把 spectral-only negative 当作 topology No-Go；必须记录 SimpleCut 是否完成确认。
- 不把 DCGC/scSGC 的 cut objective 作为本项目创新点。
- 不用新 aggregation、threshold、trainable mapping 或 feature-mask→edge 规则补 T adapter。
- 不启动 S3/S4/S5/S6，不创建 TopoCut，不扩展 GPU 训练矩阵。

## 当前状态

`S0 → S1 opportunity-only → S2 opportunity confirmation → Decision` 已是唯一有效执行计划。
S5 holdout manifest 保留为历史冻结输入，但其状态是
`dormant_due_to_adapter_not_estimable`，不构成当前项目授权。

## S1 formal Spectral result (2026-08-17)

有效工件位于 `result/representation_consumer_probe/S1_oracle_v2/`，协议为
`representation_consumer_probe_s1_opportunity_spectral_v2`。6 个数据集、5 个 arms、3 个
paired seeds 共 `90/90` jobs 完成；所有 fit/graph/consumer 路径均未接收标签，标签只用于
`O_pool/O_full` diagnostic construction 与外层指标。

| Dataset | ARI(R) | H_pool | H_full | C=H_full-H_pool | within-pool | candidate gap | S2 |
|---|---:|---:|---:|---:|---|---|---|
| cnae9 | 0.474525 | +0.215720 | +0.106188 | -0.109533 | present | absent | optional |
| Mouse_retina | 0.942599 | +0.027426 | -0.006358 | -0.033784 | absent | absent | required before negative conclusion |
| sms_spam_collection | 0.229800 | +0.367108 | +0.542305 | +0.175197 | present | present | not required for opportunity existence |
| Baron Human | 0.672484 | +0.014306 | -0.169519 | -0.183825 | absent | absent | required before negative conclusion |
| Campbell | 0.238075 | +0.191444 | +0.038981 | -0.152463 | present | absent | optional |
| hate_speech | -0.007068 | +0.002176 | +0.636495 | +0.634319 | absent | present | not required for opportunity existence |

因此 Spectral 已在 `cnae9`、`Campbell`、`sms_spam_collection` 的 frozen candidate pool 内
观察到 material opportunity；`sms_spam_collection` 与 `hate_speech` 还显示 matched-budget
candidate gap。`Baron Human` 与 `Mouse_retina` 的 Spectral opportunity 未达到 `delta=0.03`，
只能进入 conditional S2，不能直接写成 frozen relation family 的 No-Go。

该结果不解锁 S3/S4/S5/S6、TopoCut 或任何 selector；也不把 `H_pool/H_full/C` 改写为
TopoGate 或 V25 feature-assignment effect。

## S2 conditional SimpleCut result (2026-08-17)

S2 按上述条件只运行 Baron Human 与 Mouse_retina 的 `R/O_pool/O_full`，使用同一个
128→64→32 SimpleCut、三 paired seeds，共 `18/18` completed-valid。S2 的 fit 不接收标签或
K；K 只用于 known-K 外层 KMeans/readout，O arms 继续是 label-derived diagnostic oracle。
结果位于 `result/representation_consumer_probe/S2_simple_cut/`，人类可读报告为
`reports/representation_consumer_probe/S2_RESULTS.md`。

| Dataset | H_pool | H_full | C | S2 interpretation |
|---|---:|---:|---:|---|
| Baron Human | +0.033242 | +0.033367 | +0.000125 | SimpleCut confirms a material opportunity; Spectral near-threshold negative may be a relaxation miss |
| Mouse_retina | +0.008880 | +0.009622 | +0.000742 | observed-small under both consumers; no material opportunity observed |

Baron Human 的 SimpleCut effect 由 seed 间较大的波动组成，不能写成稳定收益；它只排除了
“Spectral 阴性即可判定没有 frozen-family opportunity”。Mouse_retina 仍只支持
`H_pool/H_full` 在当前 H0/cosine/k/budget 与两个 consumer 下未达到 `delta=0.03` 的限定性
描述，不能外推为 topology 全局 No-Go。两数据集均没有 material candidate gap。

S2 的 terminal decision 为：

```text
opportunity_status = heterogeneous_with_spectral_relaxation_caveat
selector_status = not_estimable
representation_consumer_promotion = not_authorized
```

因此本项目在 S2 后收口。S3 objective isolation、S4 strong backbone、S5 holdout、S6
paper-scale expansion、TopoCut 和新 selector 继续永久锁定；若未来研究 sample-edge selector，
必须另建 `relation_selection_probe`。S2 的训练历史最后一行记录的是 optimizer step 前的
loss，而 `fit_metadata.final_loss` 是 step 后重算值；该时间点差异已在 S2 integrity audit 中
标为 WARN，不影响 ARI/NMI/ACC、graph reuse 或 no-collapse 检查，也不改变 terminal boundary。
