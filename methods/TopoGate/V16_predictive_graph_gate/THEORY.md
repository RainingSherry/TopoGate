# V16 理论边界与证明草图

V16 不宣称对任意高维数据有效。它只讨论以下稀疏计数模型：样本由稀疏
Poisson 或 multinomial mixture 生成；同簇样本共享可重复的 feature
support；在 latent cluster 条件下，Poisson thinning 的两个视图边际独立；
候选图对同簇边有非零召回；图污染率低于固定阈值；同簇 donor 的 held-out
predictive risk 比跨簇 donor 低至少 `delta > 0`。

实现使用固定观测计数上的 binomial split，并保持每个 feature 的计数守恒。
因此，给定完整观测计数时两个 split 视图并不独立；独立性只作为 latent
Poisson 模型下的边际建模假设，不能被解释为条件于观测矩阵的精确事实。

## 边支持分离

对 anchor `i` 和候选 donor `j`，V16 使用

`s_ij = R_base(i) - R_edge(i,j)`。

在条件独立视图下，`R_edge` 是 donor 的条件多项式/Poisson 画像对 anchor
计数的负对数似然。若模型假设给出同簇 donor 风险上界
`mu_same`、跨簇 donor 风险下界 `mu_cross`，且
`mu_cross - mu_same >= delta`，则

`E[s_ij | c_i = c_j] - E[s_ij | c_i != c_j] >= delta`。

这只是风险差的直接重排；它不等价于 ARI utility。

## 中位数聚合

不同重复使用独立随机 split；在固定观测矩阵条件下，它们可以作为随机化
support 估计的重复，但不应被当作独立的新样本。若每次 support 估计以概率
至少 `1-epsilon` 保持正确符号，且污染重复数小于中位数 breakdown point，
样本中位数仍具有稳健的符号保持性质。实现中使用重复 support 的逐边中位数，
不使用当前 gate 生成 target。

## 图质量

设候选图同簇/跨簇边概率为 `a_candidate` 和 `b_candidate`，正 support
门控的条件保留率为 `r_same` 和 `r_cross`。未归一化的保留概率是
`r_same*a_candidate` 与 `r_cross*b_candidate`；经过边类型归一化后，
同簇边 odds 的提升需要额外满足保留率比值和先验 odds 的条件，不能仅由
`r_same > r_cross` 推出 `a_gate-b_gate > a_candidate-b_candidate`。
因此 V16 只在这些附加条件明确成立时声称图优势改善；候选图没有召回时，
support 门控无法恢复缺失的同簇边。

## assignment 传播

Stage A 是 topology-disabled masked count MAE，随后用 KMeans 初始化并冻结
spherical prototype readout，得到 `z_self` 与 `q_self`。Stage B 输出是
`q_out = pi_null q_self + sum_j pi_ij q_j`，其中 `pi` 为
`sparsemax([0, s_i1/tau, ...])`。若所有 support 不为正，`pi_null=1`，输出
精确等于 `q_self`。若正边均来自同簇且传播质量不超过预算，则凸组合的
assignment 误差不超过 self 分支；当 donor assignment 的期望误差严格较低
时，误差严格下降。该结论依赖正边 purity，不能从距离或 support 单独推出，
也不等价于对 ARI 的保证。
