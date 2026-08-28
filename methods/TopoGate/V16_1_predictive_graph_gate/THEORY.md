# V16.1 理论边界

V16.1 的门控对象是 **predictive graph support**，不是 ARI utility，也不
声称对任意高维数据必然改善聚类。方法只在以下条件下提出结构性解释：

1. 输入可以解释为稀疏的非负 Poisson 或 multinomial 计数；计数语义必须有
   可追溯的来源声明，不能仅凭非负整数推断。
2. 同簇样本具有可重复的 feature support，且 view-A 的候选图对一部分真实
   同簇边具有非零召回。
3. 对 latent Poisson/multinomial 生成过程做 thinning 时，A、B 是给定潜在
   强度后的边际独立观测。对一个已经固定的观测计数做二项分割时，A 与 B
   在给定总计数条件下是互补的，因此不能把该条件分布写成严格独立。
4. 候选图污染率低于共识筛选和支持聚合所能承受的范围，并且 donor 的
   held-out predictive risk 在同簇与跨簇之间存在正间隔。

在这些条件下，V16.1 用 view-A 估计 donor/background profile，在独立的
view-B 上计算

\[
s_{ij}^{(r)} = \frac{1}{T_i^{B_r}}
\sum_f x_{if}^{B_r}
\left(\log \hat p_{j,f}^{A_r}-\log \hat p_{0,f}^{A_r}\right).
\]

若同簇 donor 的期望支持高于跨簇 donor 至少一个正间隔，median 聚合在其
污染 breakdown point 内可保持支持符号。此时正支持门控有条件地提高图的
同簇/跨簇 odds；该结论还需要 candidate recall、同簇/跨簇保留率和归一化
条件同时成立，不能从 support 分离单独推出 ARI 增益。

输出只在 assignment space 做 sparsemax readout。null/self 是同一个
abstention 分支；所有支持非正时精确回退 `q_self`。Stage-A 的 latent 和
prototype readout 完全 topology-disabled，故 topology 不会通过门控反向
改变 encoder 或 masked count reconstruction。

上述条件是适用域和可证伪边界，不是对 `Campbell`、`Mouse_retina` 或任意
候选数据集的预先性能承诺。若证书通过而固定协议仍无增益，数据集应记录为
`empirical_not_supported`，不通过重新调 gate 挽救。
