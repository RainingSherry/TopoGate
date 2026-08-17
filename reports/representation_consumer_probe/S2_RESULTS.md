# S2 SimpleCut opportunity-confirmation results

有效协议为 `representation_consumer_probe_s2_opportunity_simplecut_v1`。本阶段只运行
S1 已冻结的 `R/O_pool/O_full` graph，使用同一个 128→64→32 SimpleCut encoder、
normalized graph energy 加 orthogonality/variance penalties，以及 paired seeds
`[42,123,7]`。数据集仅为 S1 需要排除 Spectral relaxation 假阴性的 Baron Human 和
Mouse_retina；共 `2 × 3 × 3 = 18/18` completed-valid runs。

## Contract and audit boundary

- SimpleCut fit 只接收固定 `H0`、selected graph `W` 和 seed；`y/labels` 不进入 encoder、
  loss、optimizer 或 graph consumer fit。
- `K=int(unique(y).size)` 仅用于外层 known-K KMeans/readout；`K_source` 是
  `benchmark_oracle_from_y`。
- `O_pool/O_full` 的 label-derived graphs 是 diagnostic-only upper bounds，并非可部署方法。
- 每个 run 精确复用对应 S1 v2 selected graph；S2 不重建 candidate pool、不调 budget、不调
  consumer 或 seed。
- 物理 GPU 为 3；GPU 0 和 7 禁止使用。训练工件为 80 epochs，18/18 均完成，没有 OOM 或
  `incomplete_compute`。

## Dataset-level opportunity quantities

定义仍为 label-derived diagnostic quantities：

```text
H_pool = ARI(O_pool) - ARI(R)
H_full = ARI(O_full) - ARI(R)
C      = H_full - H_pool
```

`delta=0.03` 只是已冻结的 materiality margin，并非显著性检验；统计单位是 dataset，seed
只是 paired repeat。

| Dataset | mean ARI(R) | H_pool | H_full | C | within-pool | candidate gap |
|---|---:|---:|---:|---:|---|---|
| Baron Human | 0.261240 | +0.033242 | +0.033367 | +0.000125 | present (material) | absent |
| Mouse_retina | 0.916848 | +0.008880 | +0.009622 | +0.000742 | absent (observed-small) | absent |

Seed-level `H_pool` / `H_full` values分别为：

- Baron Human: `H_pool=[+0.101090,+0.000729,-0.002094]`，
  `H_full=[+0.099802,+0.001463,-0.001165]`；均值达到 materiality，但 seed 间波动较大，
  不应写成稳定收益。
- Mouse_retina: `H_pool=[+0.030344,+0.003889,-0.007592]`，
  `H_full=[+0.030360,+0.006098,-0.007592]`；均值低于 `0.03`，归为 observed-small，
  不能写成 opportunity-positive，也不能据此作 topology 全局 No-Go。

## Representation diagnostics

18 个 embedding 均 finite，`low_variance_dimension_ratio=0`，且没有出现 constant
embedding。Baron Human 的 effective rank 为 `16.7312–18.1273`，最小维度标准差为
`3.4113–3.5832`；Mouse_retina 的 effective rank 为 `20.5364–22.4333`，最小维度标准差为
`0.4675–0.5203`。因此 S2 的 materiality 差异不能归因于明显 representation collapse。

## Decision

1. **Baron Human：Spectral 的近阈值阴性可由另一种 consumer 改写。** SimpleCut 在同一
   frozen candidate family 上观察到 `H_pool≈H_full≈+0.033`，而 `C≈0`。这支持“Spectral
   relaxation 可能漏掉 opportunity”的有限解释；它不支持 selector、TopoGate gain 或
   candidate bottleneck 结论。
2. **Mouse_retina：在两个已审计 consumer 上都未达到 material opportunity。** 结论限定为
   当前 `H0/cosine/k/budget` relation family 与这两个 consumer；不能外推为 topology 普遍
   无价值。
3. S2 完成后本项目进入 terminal Decision。`S_graph` 仍不可估计，因为 S0 的
   `adapter_not_estimable` 状态未改变。S3 objective isolation、S4 strong backbone、S5
   holdout、S6 benchmark、TopoCut 和新 selector 均继续锁定。

## Claim boundary

S2 证明的是一个条件性的 opportunity diagnostic：在 Baron Human 上，Spectral 阴性不能单独
排除 frozen relation family 的 opportunity；在 Mouse_retina 上，当前两个 consumer 未观察到
material opportunity。S2 不是 system-level scMAE 对比，不是 TopoGate 选择效用，不是新 backbone
性能，也不是跨数据集 generalization 证据。

Raw artifacts: `result/representation_consumer_probe/S2_simple_cut/`。
