# TopoGate V11 sparse H0 TDA pilot

**日期**：2026-08-03
**状态**：代码、工程验证和预注册正式比较均已完成；固定协议内性能 no-go。
**结果位置**：本报告位于 `source-repository/result/analysis/`，实际写入
`external-result-storage/result/analysis/`。

## 1. 目的与边界

现有 V9--V11 使用 PCA、kNN、mutual/SNN、动态图和 edge reliability。这些是
依赖度量的有限图结构，不能直接称为 persistent homology。本 pilot 只回答一个
较窄的问题：固定 raw kNN 的稀疏 1-skeleton 上，0 维 component persistence 是否
能作为一个独立、无标签、可审计的 edge prior。

本实现不计算 H1、不构造 dense Vietoris--Rips complex、不改变 V11 的 encoder、
Student-t mixture head、loss、candidate graph 选择规则或标签边界。`tda_prior_mode=none`
是默认值，且 `tda_prior_weight=0.0`，因此默认 V11 的计算路径保持不变。

## 2. 数学对象

给定 raw PCA 表征 `z_i`，先对每一行做单位范数归一化，固定 raw kNN 的无向边集
`E_0`。对每条边使用单位球上的 Euclidean chord distance：

```text
d(i,j) = || z_i / ||z_i|| - z_j / ||z_j|| ||_2
```

在这个固定稀疏 1-skeleton 上定义边过滤：

```text
K_alpha = (V, { (i,j) in E_0 : d(i,j) <= alpha })
```

所有顶点在 `alpha=0` 出生。按 `d(i,j)` 从小到大扫描边，并用 union-find
记录两个不同 H0 component 首次合并的边。由于所有顶点出生时间相同为 0，有限
H0 bar 的 persistence 就是该 merge edge 的 death distance；每个最终连通分量
的 infinite bar 不进入 edge prior。这个算法是该固定稀疏 1-skeleton 上的精确
H0 计算，但不是完整 dense VR persistence。

距离尺度在固定 raw skeleton 上一次计算，默认使用正距离中位数 `s`。用于门控
prior 的有界分数为：

```text
q_ij = 1 - exp(-d(i,j) / s)       if (i,j) is an H0 merge edge
q_ij = 0                          otherwise
```

`tda_scale_mode` 支持 `median`、`quantile`、`max` 和 `none`；尺度、过滤度量和
raw kNN skeleton 都在训练开始前固定，不读取标签，也不随 EMA latent graph 改写。

## 3. 与 V11 的连接

代码位于：

- `methods/TopoGate/V11/tda.py`：skeleton、尺度、union-find H0 persistence 和候选边映射；
- `methods/TopoGate/V11/graph.py`：保留 raw kNN indices，并为候选图保存 `tda_prior`；
- `methods/TopoGate/V11/trainer.py`：把 prior 作为 detached 项加入现有 graph-prior score；
- `methods/TopoGate/V11/config.py`：校验模式、权重和尺度配置；
- `methods/TopoGate/V11/configs/topogate_v11_tda_h0_mst.yaml`：可回退 pilot 配置；
- `scripts/V11/run_v11_multiseed.py`：注册 H0、fixed-filtration 和 random controls。

当前 edge target 的新增部分只有：

```text
score_ij = raw_prior_weight * raw_similarity
         + latent_prior_weight * latent_similarity
         + tda_prior_weight * q_ij
```

随后仍由已有 temperature softmax 和 teacher assignment agreement 生成 target。
`q_ij` 来自 NumPy 固定计算，不带 Torch gradient；它不是伪标签，也不直接监督
真实簇标签。

## 4. 对照模式

| 模式 | 含义 | 是否是 persistence |
|---|---|---|
| `none` | 零 prior，原 V11 默认路径 | 否 |
| `h0_mst` | 只给 H0 component merge edges persistence score | 是，限 sparse H0 |
| `fixed_filtration` | 同一固定 raw skeleton 上给所有边 proximity score | 否，距离控制 |
| `random` | 按无向候选边和 seed 生成确定性随机 prior | 否，随机控制 |

正式比较已按预注册协议完成：保留 `V11_full`、`V11_nomix`、
`V11_tda_h0_mst`、`V11_tda_fixed_filtration` 和 `V11_tda_random`，使用同一
五数据集清单、同一 `[42, 123, 7]` seeds、同一输入预处理和同一 benchmark K
协议。没有逐数据集用标签选择 filtration scale、prior weight、variant 或最佳
seed。原始产物位于 `result/V11/tda_h0_pilot_2026-08-03/`，共 75/75 completed。

## 5. 已完成验证

- `python -m compileall -q methods/TopoGate/V11 scripts/V11`：通过；
- `PYTHONPATH=source-repository pytest -q methods/TopoGate/V11/tests/test_v11.py`：
  `19 passed`，仅有 3 条 FAISS/SWIG 第三方弃用警告；
- 回归测试覆盖 H0 只保留 MST merge edges、尺度归一化、默认关闭、随机 control
  确定性和候选边映射；
- `datasets/iris.npz` CPU、seed=42、3 epochs、缩小网络 smoke 成功加载
  `h0_mst`，`graph_history` 写入 H0 merge count、尺度、平均 prior 和非零比例，
  `history` 写入平均 prior；临时输出写入 `/tmp` 后已清理；
- smoke 的 `y` 只用于 benchmark K 和后验指标，未传入 `V11Trainer`、图构建或
  prior 计算。

这些 smoke 验证只能证明实现、梯度路径和输出契约；它们不构成性能证据。随后
完成了独立的正式五数据集批次：H0 相对 `V11_full` 的 head ARI 为 `+0.000010`、
KMeans ARI 为 `-0.000726`；fixed-filtration 为 `+0.000002/-0.000665`；
random 为 `+0.000018/-0.000274`。H0、fixed 和 random 的 head 结果均没有形成
稳定的独立收益，故在该固定协议内判定为性能 **no-go**。逐数据集和 source hash
审计见 `result/analysis/topogate_cross_version_landscape_2026-08-03_tda.csv`。

## 6. 与参考书结论的对应

拓扑学资料要求区分邻域图、单纯复形、链群/边界算子、同调和 persistence；因此
本实现只使用明确的 filtration 与 H0 component merge 语义，拒绝把 cycle-rank
当作 H1。数学分析资料提醒 kNN 是离散不连续算子、PCA/标准化会改变排序边界，
所以 raw skeleton、尺度和预处理必须固定并记录。机器学习资料中的 mixture
responsibility 与 reject/null expert 对应 V11 的原有概率门控；TDA prior 只提供
一个外生无标签先验，不被写成已校准的 cluster probability。

## 7. Go/no-go 边界

正式比较已经完成。结果满足 no-go 条件：H0、fixed-filtration 和 random 的
head/KMeans 差值都接近零，且没有跨数据集一致的 head 改善；因此当前实现保留
为可审计诊断，不扩展到 H1、persistence image 或 Mapper，也不写入论文主方法。
若未来继续，必须先修正“晚合并边可能是跨组件 bridge”的语义假设，并以默认关闭、
可回退的 detached prior 做 toy graph 单元测试；这只是下一候选假设，不是已验证收益。
