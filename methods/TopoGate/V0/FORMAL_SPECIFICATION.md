# TopoGate V0 / scVICAR 的形式化规格

本文档严格按照 `methods/TopoGate/V0` 当前源码描述模型。它是实现对齐的
方法规格，不是对未来版本的设计草案。V0 只有一个共享的 scMAE 网络，`F`
与 `T` 是同一 vicinal corruption 框架的两种参数化。

## 1. 一句话定义与总计算图

设预处理后的输入为

\[
X=\mathcal P_{\phi}(X^{\rm raw})\in\mathbb R^{N\times D},
\]

其中 `\(N\)` 是样本/细胞数，`\(D\)` 是进入网络的特征数，`\(\phi\)` 是
固定的预处理配置。给定参数化

\[
v\in\{F,T\},
\]

V0 先用无标签图和 corruption 构造 pseudo view，再用同一个 scMAE 学习
参数 `\(\theta\)`：

\[
\theta_v^*
=\underset{\theta}{\arg\min}\;
\left[\mathcal L_{\rm real}(\theta)
+\eta\,\mathcal L_{\rm pseudo}^{(v)}(\theta)\right],
\]

上式针对 pseudo 分支启用的情况；若 pseudo 分支关闭，目标只剩
`\(\mathcal L_{\rm real}\)`。其中 `\(\eta\)` 是 pseudo-loss 权重。训练结束后只把干净输入送入编码器：

\[
Z_v=E_{\theta_v^*}(X)\in\mathbb R^{N\times H},
\qquad
\hat y=\operatorname{KMeans}_{K}(Z_v).
\]

因此完整流程是

```text
X_raw --P_phi--> X --G--> graph --C_v--> pseudo view --R_rho--> masked view
                                  |                         |
                                  +-------------------------+--> shared scMAE
                                                               |
X ------------------------------> clean encoder -------------> Z --KMeans_K--> y_hat
```

`F` 与 `T` 共享 `\(E_\theta\)`、mask predictor、decoder、损失和优化器；二者
只在图边权、节点 gate、邻居混合和 pseudo 样本权重上不同。因此“统一模型”
表示模型身份统一，并不表示默认 F/T 的数值输出完全相同。

## 2. 输入与无标签预处理

原始矩阵记为

\[
X^{\rm raw}=[x^{\rm raw}_{ig}]
\in\mathbb R_{\ge 0}^{N\times D_0},
\]

`\(D_0\)` 是原始特征数，`\(x^{\rm raw}_{ig}\)` 是样本 `\(i\)` 的原始特征
`\(g\)`。NPZ 输入在 `input_mode=raw` 时执行

\[
\tilde x^{\rm raw}_{ig}
=x^{\rm raw}_{ig}\frac{T_{\rm sum}}{\sum_{g'}x^{\rm raw}_{ig'}},
\qquad
x^{\rm log}_{ig}=\log(1+\tilde x^{\rm raw}_{ig}),
\]

其中 `\(T_{\rm sum}\)` 是 `target_sum`。零总和样本的缩放因子按源码置为零。若输入
已经是连续/log1p 数据，使用 `input_mode=log1p`，不重复做上述变换。

若 `n_top_features>0` 且 `D_0` 大于该值，按训练数据中每个特征的方差排序，
保留前 `\(D\)` 个特征；排序相同时按原始特征索引稳定打破平局。若
`scale_input=true`，再对每个特征做训练矩阵上的 StandardScaler，得到

\[
X=\mathcal P_\phi(X^{\rm raw})\in\mathbb R^{N\times D}.
\]

所有这些步骤均不读取标签。标签 `\(y\)` 只由外层 runner 读取，用于 benchmark
指标，或在未显式提供 `n_clusters` 时确定 benchmark 的 `\(K\)`；它不进入
`fit_predict`、图、gate、corruption、loss 或 optimizer。

## 3. 无标签 PCA/cosine KNN 图 `\(\mathcal G\)`

### 3.1 图表示

设 `\(d_p\)` 为实际 PCA 维度：

\[
d_p=\min(\texttt{knn\_pca\_dim},D,N-1).
\]

当源码判断该维度无需降维时，直接使用 `\(X\)`；否则

\[
U=\operatorname{PCA}_{d_p}(X)
\in\mathbb R^{N\times d_p}.
\]

第 `\(i\)` 个样本的图向量为行归一化结果

\[
e_i=\frac{u_i}{\lVert u_i\rVert_2},
\]

零范数行保持为零。`\(e_i\)` 只用于建图，不是 scMAE 的最终 embedding。

### 3.2 邻居、距离、相似度和基础概率

令 `\(k_*=\min(k,N-1)\)`，其中 `\(k\)` 是配置项 `neighbor_k`。对每个样本
`\(i\)`，余弦 KNN 给出邻居集合

\[
\mathcal N_i=\{j_{i1},\ldots,j_{ik_*}\}.
\]

对 `\(j\in\mathcal N_i\)` 定义

\[
d_{ij}=1-e_i^\top e_j,
\qquad
s_{ij}=1-d_{ij}=e_i^\top e_j.
\]

`\(d_{ij}\)` 是余弦距离，`\(s_{ij}\)` 是余弦相似度。给定温度
`\(\tau=\texttt{tau}>0\)`，基础邻居概率为稳定 softmax：

\[
p_{ij}
=\frac{\exp(s_{ij}/\tau)}
{\sum_{\ell\in\mathcal N_i}\exp(s_{i\ell}/\tau)}.
\]

因此每个非空邻居行满足 `\(\sum_jp_{ij}=1\)`。

### 3.3 拓扑关系量

互为近邻指标为

\[
m_{ij}=\mathbf 1[i\in\mathcal N_j],
\]

其中 `\(\mathbf 1[\cdot]\)` 为示性函数，条件成立时取 1，否则取 0。

SNN 重叠为

\[
q_{ij}
=\frac{|\mathcal N_i\cap\mathcal N_j|}
{|\mathcal N_i\cup\mathcal N_j|},
\]

分母在源码中至少按 1 处理。`\(m_{ij}\)` 与 `\(q_{ij}\)` 都由无标签 KNN
图得到。

### 3.4 T 的解析 edge reliability

`F` 直接使用 `\(p_{ij}\)`。`T` 根据 `edge_reliability_mode` 组合以下因子：

\[
\begin{aligned}
r^{\rm sim}_{ij}&=\exp(\gamma_{\rm sim}s_{ij}),\\
r^{\rm mutual}_{ij}&=1+\gamma_{\rm mutual}m_{ij},\\
r^{\rm snn}_{ij}&=1+\gamma_{\rm snn}q_{ij},\\
r^{\rm dist}_{ij}&=\exp(-\gamma_{\rm distance}d_{ij}).
\end{aligned}
\]

若某个因子没有被 mode 选中，它的指数为 0；因此统一写为

\[
r_{ij}
=(r^{\rm sim}_{ij})^{I_{\rm sim}}
(r^{\rm mutual}_{ij})^{I_{\rm mutual}}
(r^{\rm snn}_{ij})^{I_{\rm snn}}
(r^{\rm dist}_{ij})^{I_{\rm dist}},
\]

`\(I_\bullet\in\{0,1\}\)` 是 mode 指示量：例如 `\(I_{\rm sim}=1\)` 时启用
similarity 因子，`\(I_{\rm mutual}=1\)` 时启用 mutual 因子，其余两项同理。
对应的五个 mode 为：

| edge_reliability_mode | I_sim | I_mutual | I_snn | I_dist |
| --- | ---: | ---: | ---: | ---: |
| `none` | 0 | 0 | 0 | 0 |
| `sim` | 1 | 0 | 0 | 0 |
| `sim_mutual` | 1 | 1 | 0 | 0 |
| `sim_mutual_snn` | 1 | 1 | 1 | 0 |
| `sim_mutual_snn_distance` | 1 | 1 | 1 | 1 |

源码随后把 `\(r_{ij}\)` 裁剪到 `\([10^{-6},10^6]\)`，并行归一化为

\[
a_{ij}
=\frac{p_{ij}r_{ij}}
{\sum_{\ell\in\mathcal N_i}p_{i\ell}r_{i\ell}}.
\]

`F` 的等价量为 `\(r_{ij}=1\)`、`\(a_{ij}=p_{ij}\)`。

## 4. F/T 节点 gate 与 pseudo 权重

### 4.1 F：固定参数化

`F` 对所有节点使用同一个邻居注入比例

\[
g_i^{F}=1-\alpha,
\qquad
\nu_i^{F}=1,
\]

其中 `\(\alpha=\texttt{alpha}\)` 是 anchor 保留比例，默认 `\(\alpha=0.9\)`。
因此默认每个样本保留 90% anchor、注入 10% 邻居均值。

### 4.2 T：拓扑参数化

首先计算节点级拓扑统计量

\[
\bar m_i=\frac1{k_*}\sum_{j\in\mathcal N_i}m_{ij},
\qquad
\bar q_i=\frac1{k_*}\sum_{j\in\mathcal N_i}q_{ij},
\]

以下两个平均量只在 `\(k_*>0\)` 时计算；无边时源码直接把 gate 设为 0。

以及基础概率加权的扰动代理

\[
\pi_i=1-\sum_{j\in\mathcal N_i}p_{ij}s_{ij}.
\]

当前 V0 主路径没有从 `fit_predict` 传入不确定性估计，因此
`\(u_i=0\)`。保留该符号是为了说明 `compute_node_gate` 的可选接口。节点
gate logit 为

\[
t_i
=\beta_{\rm mutual}\bar m_i
+\beta_{\rm snn}\bar q_i
-\beta_{\rm perturb}\pi_i
-\beta_{\rm uncertainty}u_i.
\]

令 sigmoid 函数为

\[
\sigma(t)=\frac1{1+\exp(-t)},
\]

则

\[
g_i^{T}
=g_{\min}+(g_{\max}-g_{\min})\sigma(t_i).
\]

源码先把 `\(t_i\)` 裁剪到 `\([-60,60]\)` 以防指数溢出。gate 的 pseudo-loss
样本权重为

\[
\nu_i^{T}
=\operatorname{clip}\left(
\frac{g_i^{T}}{\max_j g_j^{T}},0,1\right),
\]

其中 `\(\operatorname{clip}(z,0,1)=\min(1,\max(0,z))\)`。`\(g_{\min}\)`、
`\(g_{\max}\)`、`\(\beta_*\)` 都是固定配置，不是网络权重。若图没有边，
源码令 `\(g_i^T=0\)`，pseudo view 退化为 anchor。

## 5. Vicinal corruption `\(\mathcal C_v\)`

对一个 batch `\(B\subseteq\{1,\ldots,N\}\)` 中的样本 `\(i\)`，令

\[
M_{\rm mix}=\max(1,\min(\texttt{mix\_neighbors},k_*))
\]

为当前 batch 的邻居抽样数。令 `\(P_{ij}^{(v)}\)` 为抽样分布：

\[
P_{ij}^{(F)}=p_{ij},
\qquad
P_{ij}^{(T)}=a_{ij}.
\]

在默认 `neighbor_estimator=current` 下，有放回抽取

\[
c_{ir}\sim\operatorname{Categorical}(P_i^{(v)}),
\qquad r=1,\ldots,M_{\rm mix},
\]

并把抽中的概率重新归一化：

\[
\lambda_{ir}
=\frac{P_{i,c_{ir}}^{(v)}}
{\sum_{t=1}^{M_{\rm mix}}P_{i,c_{it}}^{(v)}}.
\]

邻居均值（更准确地说是概率加权邻居估计）为

\[
\bar x_i^{(v)}
=\sum_{r=1}^{M_{\rm mix}}\lambda_{ir}x_{c_{ir}}.
\]

于是 pseudo-cell 为

\[
x_i^{\prime(v)}
=(1-g_i^{(v)})x_i+g_i^{(v)}\bar x_i^{(v)}.
\]

`current` 是 V0 默认且对应历史 F/T runner 的路径；`uniform_sample` 把
`\(\lambda_{ir}\)` 固定为 `\(1/M_{\rm mix}\)`；`full` 不抽样，直接对全部 `\(k_*\)` 个
邻居使用 `\(P_i^{(v)}\)` 加权。pseudo-cell 以 detached tensor 返回，因而
图、gate 和邻居均值没有 autograd 梯度。

pseudo 分支的目标不是 `\(x_i^{\prime(v)}\)`，而是同一个真实 anchor `\(x_i\)`。
这使训练目标成为从 vicinal corruption 恢复真实样本。

## 6. scMAE elementwise row-swap noise `\(\mathcal R_\rho\)`

对真实 view 或 pseudo view 的 batch 矩阵 `\(V\in\mathbb R^{b\times D}\)`，其中
`\(b=|B|\)`，先独立生成

\[
S_{ig}\sim\operatorname{Bernoulli}(\rho),
\qquad \rho=\texttt{mask\_ratio}.
\]

再生成当前 batch 内的随机行置换 `\(\Pi_B\)`，得到

\[
\widetilde v_{ig}
=
\begin{cases}
V_{\Pi_B(i),g},&S_{ig}=1,\\
V_{ig},&S_{ig}=0.
\end{cases}
\]

训练实际使用的有效 mask 不是 `\(S\)`，而是

\[
M_{ig}=\mathbf 1[\widetilde v_{ig}\ne V_{ig}].
\]

因此，若随机置换后数值恰好相等，该位置最终记为未改变。真实 view 和
pseudo view 各使用一套独立的随机流。

## 7. 共享 scMAE 网络

### 7.1 编码器

设 `\(H=\texttt{hidden\_size}\)` 为 latent 维度，编码器中间宽度为

\[
W=\max(256,2H).
\]

对一个被噪声污染的样本 `\(\widetilde x_i\in\mathbb R^D\)`，网络按下式计算：

\[
\begin{aligned}
o_i&=\operatorname{Dropout}(\widetilde x_i),\\
a_i^{(1)}&=\operatorname{Mish}\left(\operatorname{LN}_{W}
(W_1o_i+b_1)\right),\\
a_i^{(2)}&=\operatorname{Mish}\left(\operatorname{LN}_{H}
(W_2a_i^{(1)}+b_2)\right),\\
z_i&=W_3a_i^{(2)}+b_3.
\end{aligned}
\]

`\(z_i\in\mathbb R^H\)` 是 latent embedding。参数形状为

\[
W_1\in\mathbb R^{W\times D},\quad
W_2\in\mathbb R^{H\times W},\quad
W_3\in\mathbb R^{H\times H},
\]

以及相应 bias `\(b_1,b_2,b_3\)`。`\(\operatorname{LN}\)` 是带 affine 参数的
LayerNorm；`Dropout` 在训练时随机置零，在 `eval()` 时关闭；Mish 定义为

\[
\operatorname{Mish}(t)=t\tanh(\log(1+e^t)).
\]

### 7.2 mask predictor 与 decoder

mask predictor 输出每个输入特征的原始 logits：

\[
\zeta_i=W_mz_i+b_m\in\mathbb R^D,
\]

其中 `\(W_m\in\mathbb R^{D\times H}\)`、`\(b_m\in\mathbb R^D\)`。`\(\zeta_i\)`
不是概率；概率只有在需要解释时才是 `\(\sigma(\zeta_i)\)`。

默认 V0 配置把原始 logits 与 latent 拼接：

\[
c_i=[z_i;\zeta_i]\in\mathbb R^{H+D},
\qquad
\hat x_i=W_dc_i+b_d\in\mathbb R^D.
\]

这里 `[;]` 表示向量拼接，`\(W_d\in\mathbb R^{D\times(H+D)}\)`，
`\(b_d\in\mathbb R^D\)`。V0 默认不对 decoder mask 特征做 sigmoid，也不 detach
该特征；这保持了历史 scMAE 行为。

### 7.3 可学习参数集合

所有可学习参数合并记为

\[
\theta=\{W_1,b_1,\gamma^{\rm LN}_1,\beta^{\rm LN}_1,
W_2,b_2,\gamma^{\rm LN}_2,\beta^{\rm LN}_2,W_3,b_3,
W_m,b_m,W_d,b_d\},
\]

其中 `\(\gamma^{\rm LN}_1,\beta^{\rm LN}_1,\gamma^{\rm LN}_2,\beta^{\rm LN}_2\)` 是两个 LayerNorm 的 affine
参数。F/T 不增加自己的神经网络权重；`\(\alpha,\gamma_*,\beta_*,g_{\min},
g_{\max}\)`、KNN 结果、edge reliability、node gate 和 sample weight 都是
固定配置或 NumPy 解析量。V0 主路径也不包含历史 T contrastive projector。

## 8. 单样本损失

以下定义对真实 view 或 pseudo view 都成立。目标记为 `\(y_i^{\rm target}\)`；
在 V0 中真实 view 和 pseudo view 的目标都为干净 anchor `\(x_i\)`。

### 8.1 加权重建损失

令 `\(\lambda_d=\texttt{masked\_data\_weight}\)`，特征级权重为

\[
w_{ig}=M_{ig}\lambda_d+(1-M_{ig})(1-\lambda_d).
\]

默认 `normalize_reconstruction_by_weight=false`，所以

\[
\mathcal L_i^{\rm rec}
=(1-\lambda_m)\frac1D
\sum_{g=1}^{D}w_{ig}
(\hat x_{ig}-y_{ig}^{\rm target})^2,
\]

其中 `\(\lambda_m=\texttt{mask\_loss\_weight}\)`。源码保留了一个兼容分支：
若启用按权重归一化，则把上式中的 `\(D^{-1}\sum_g\)` 换成
`\(\sum_g w_{ig}e_{ig}/\sum_gw_{ig}\)`，其中
`\(e_{ig}=(\hat x_{ig}-y_{ig}^{\rm target})^2\)`；V0 默认不启用该分支。

### 8.2 mask BCE 损失

\[
\mathcal L_i^{\rm mask}
=\lambda_m\frac1D\sum_{g=1}^{D}
\operatorname{BCEWithLogits}(\zeta_{ig},M_{ig}),
\]

即稳定计算

\[
\operatorname{BCEWithLogits}(a,m)
=-m\log\sigma(a)-(1-m)\log(1-\sigma(a)).
\]

单样本总损失为

\[
\ell_i=\mathcal L_i^{\rm rec}+\mathcal L_i^{\rm mask}.
\]

## 9. 两个 view 的训练目标

令 `\(B\)` 为一个 batch，`\(|B|=b\)`。真实分支为

\[
\mathcal L_{\rm real}
=\frac1b\sum_{i\in B}\ell_i^{\rm real}.
\]

F 的 pseudo 分支为

\[
\mathcal L_{\rm pseudo}^{F}
=\frac1b\sum_{i\in B}\ell_i^{F}.
\]

T 的 pseudo 分支使用 gate-derived sample weight：

\[
\mathcal L_{\rm pseudo}^{T}
=
\frac{\sum_{i\in B}\nu_i^{T}\ell_i^{T}}
{\max(\sum_{i\in B}\nu_i^{T},10^{-8})}.
\]

当 `graph_enabled=true` 时，最终 batch 目标为

\[
\mathcal L^{(v)}
=\mathcal L_{\rm real}
+\eta\mathcal L_{\rm pseudo}^{(v)},
\qquad v\in\{F,T\}.
\]

图被禁用、`use_pseudo=false`、`pseudo_weight=0` 或没有有效邻居时，pseudo
分支不参与，目标退化为 `\(\mathcal L_{\rm real}\)`。

## 10. 优化、表示提取与聚类

V0 使用 Adam：

\[
\theta\leftarrow\operatorname{AdamStep}
(\theta,\nabla_\theta\mathcal L^{(v)},\operatorname{lr}).
\]

每个 epoch 遍历 DataLoader；`seed` 同时控制 Python、NumPy、PyTorch、batch
shuffle、邻居抽样和两套 row-swap noise 随机流。当前 V0 唯一允许的随机协议是
`rng_protocol=isolated_v0`：DataLoader、邻居抽样、真实 view 和 pseudo view
使用相互独立的 V0-local 子流；不执行旧 PlantNet 随机状态回放或隐藏的辅助抽样。
优化器只接收 `model.parameters()`，不接收图或 gate。

训练完成后进入 `eval()`，对干净输入 `x_i` 提取

\[
z_i^{\rm clean}=E_{\theta_v^*}(x_i),
\qquad
Z=[z_1^{\rm clean};\ldots;z_N^{\rm clean}].
\]

若外层给出 `\(K\)`，才执行

\[
\hat y=\operatorname{KMeans}(Z;K,\texttt{kmeans\_n\_init},\texttt{seed}).
\]

若调用 `fit_predict(..., n_clusters=None)`，只返回 `\(Z\)`，不运行 KMeans。
ACC、ARI、NMI 等指标由 runner 在训练返回后计算；它们不参与表示学习。

## 11. 梯度、标签和等价性边界

### 11.1 梯度路径

训练中存在

\[
\frac{\partial\mathcal L}{\partial\theta}\ne0
\]

的正常神经网络路径。但图、edge reliability、node gate、抽样邻居、pseudo
cell 和 T sample weight 在 NumPy/随机采样后以 detached tensor 进入模型，因而
它们没有到 `\(\theta\)` 的反向传播路径。更准确地说，当前实现没有定义
`\(\partial\mathcal L/\partial p_{ij}\)`、`\(\partial\mathcal L/\partial r_{ij}\)`
或 `\(\partial\mathcal L/\partial g_i\)` 的 autograd 路径；这些量不是
`nn.Parameter`。

### 11.2 F/T 的“等价”含义

F/T 在以下层面等价：

- 同一 scMAE 网络拓扑和同一优化器；
- 同一 mask noise、重建目标和损失定义；
- 同一 clean embedding 与 KMeans readout 契约；
- 都不把标签传入核心拟合函数。

F/T 在以下层面不等价：

- F 使用固定 `\(g_i=1-\alpha\)`，T 使用拓扑解析 gate；
- F 使用基础 `\(p_{ij}\)`，T 使用 reliability 调整后的 `\(a_{ij}\)`；
- F pseudo loss 为单位样本权重，T 按 `\(\nu_i^T\)` 加权；
- 默认 F/T 的 `neighbor_k` 也不同（配置分别为 5 和 10）。

所以 V0 是“一个模型身份、两个 corruption parameterizations”，不是声称 F/T
在默认配置下逐点输出相同。

### 11.3 历史 T contrastive 边界

历史 retired T runner 的 `rg_neighbormix_scmae_contrast_safe` 可以额外创建
contrastive projector；V0 当前主路径没有该 projector，也没有 contrastive loss。
因此本规格只覆盖基础 F/T，不把该历史实验变体包装成 V0 的可学习 gate。

## 12. 数学函数与算子说明

| 函数/算子 | 在 V0 中的定义或作用 |
| --- | --- |
| `\(\mathcal P_\phi\)` | 按固定配置 `\(\phi\)` 完成 count normalization、`log1p`、方差特征选择和可选标准化 |
| `\(\operatorname{PCA}_{d_p}\)` | 在不读取标签的情况下把输入投影到最多 `\(d_p\)` 个主成分 |
| `\(\operatorname{normalize}\)` | 对每一行除以 `\(\ell_2\)` 范数，用于余弦 KNN；零行保持为零 |
| `\(\operatorname{KNN}\)` | 在归一化图表示上返回每个样本的 `\(k_*\)` 个最近邻 |
| `\(\exp(t)\)`、`\(\log(t)\)` | 自然指数和自然对数；用于 softmax、reliability、Mish 和 count `log1p` |
| `\(\lVert u\rVert_2\)` | 向量 `\(u\)` 的欧氏范数，`\(\sqrt{\sum_g u_g^2}\)` |
| `\(u^\top v\)` | 向量内积；归一化后等于余弦相似度 |
| `\(|A|\)`、`\(A\cap B\)`、`\(A\cup B\)` | 集合大小、交集和并集，用于 SNN overlap |
| `\(\mathbf 1[\cdot]\)` | 示性函数；条件真取 1，否则取 0 |
| `\(\operatorname{softmax}\)` | 把一行相似度变成和为 1 的概率；源码先减行最大值再 exponentiate 以保证数值稳定 |
| `\(\operatorname{clip}(t,a,b)\)` | 把 `\(t\)` 截断到 `[a,b]`；用于 gate 样本权重和指数溢出保护 |
| `\(\sigma(t)\)` | sigmoid，`\(1/(1+e^{-t})\)`；把 gate logit 映射到 `(0,1)` |
| `\(\operatorname{Bernoulli}(\rho)\)` | 以概率 `\(\rho\)` 产生 1 的二值随机变量，用于候选 mask |
| `\(\operatorname{Categorical}(P_i)\)` | 按邻居概率行 `\(P_i\)` 抽取邻居位置；V0 默认有放回抽样 |
| `\(\operatorname{Dropout}\)` | 训练时随机置零输入单元，测试时关闭；没有持久可学习权重 |
| `\(\operatorname{LN}_d\)` | 对 `\(d\)` 维向量做 LayerNorm：`\(\gamma_d^{\rm LN}\odot(u-\mu_d(u))/\sqrt{\varsigma_d^2(u)+\epsilon}+\beta_d^{\rm LN}\)`；其中 `\(\mu_d\)`、`\(\varsigma_d^2\)` 是该向量的均值/方差，`\(\epsilon\)` 是数值稳定常数，`\(\odot\)` 是逐元素乘法 |
| `\(\operatorname{Mish}(t)\)` | `\(t\tanh(\log(1+e^t))\)` 的平滑激活函数 |
| `\(\operatorname{MSE}\)` | 对每个特征计算平方误差；V0 再按 `\(w_{ig}\)` 加权 |
| `\(\operatorname{BCEWithLogits}\)` | 对 logits 和二值 mask 的稳定二元交叉熵，内部包含 sigmoid |
| `\(\arg\min\)` | 寻找使训练目标最小的网络参数；这里的变量是 `\(\theta\)` |
| `\(\nabla_\theta\)` | 对全部可学习网络参数求梯度 |
| `\(\operatorname{AdamStep}\)` | 用 Adam 优化器根据梯度更新 `\(\theta\)`，学习率为 `lr` |
| `\(\operatorname{KMeans}(Z;K)\)` | 仅在训练后把 `\(Z\)` 划为 `\(K\)` 个簇；不向训练回传梯度 |

## 13. 符号逐项说明

| 符号 | 含义 |
| --- | --- |
| `\(X^{\rm raw}\)` | 原始非负输入矩阵，形状 `\(N\times D_0\)` |
| `\(X\)` | 归一化/变换/特征选择后的模型输入，形状 `\(N\times D\)` |
| `\(N\)`、`\(D_0\)`、`\(D\)` | 样本数、原始特征数、实际输入特征数 |
| `\(x_i\)`、`\(x_{ig}\)` | 第 `\(i\)` 个样本及其第 `\(g\)` 个特征 |
| `\(T_{\rm sum}\)` | count 行归一化的目标总量 `target_sum` |
| `\(\phi\)` | 固定预处理配置，不是可学习参数 |
| `\(v\)` | 参数化标记，`\(F\)` 或 `\(T\)` |
| `\(d_p,U,e_i\)` | PCA 维度、PCA 表示、行归一化图向量 |
| `\(k,k_*\)` | 配置邻居数与实际有效邻居数 |
| `\(\mathcal N_i,j_{ir}\)` | 第 `\(i\)` 个节点的邻居集合及第 `\(r\)` 个邻居索引 |
| `\(d_{ij},s_{ij}\)` | 余弦距离与余弦相似度 |
| `\(\tau,p_{ij}\)` | softmax 温度与基础邻居概率 |
| `\(m_{ij},q_{ij}\)` | mutual-KNN 指标与 SNN 重叠 |
| `\(r_{ij},a_{ij}\)` | T 的 edge reliability 与最终行归一化边权 |
| `\(r_{ij}^{\rm sim},r_{ij}^{\rm mutual},r_{ij}^{\rm snn},r_{ij}^{\rm dist}\)` | reliability 的四个可选因子 |
| `\(I_{\rm sim},I_{\rm mutual},I_{\rm snn},I_{\rm dist}\)` | 当前 reliability mode 是否启用对应因子的 0/1 指示量 |
| `\(\gamma_*\)` | reliability 公式的固定系数 |
| `\(\alpha,g_{\min},g_{\max}\)` | F 的 anchor 保留比例与 T 的 gate 上下界 |
| `\(\bar m_i,\bar q_i,\pi_i,u_i\)` | 节点拓扑均值、SNN 均值、扰动代理、不确定性代理 |
| `\(\beta_{\rm mutual},\beta_{\rm snn},\beta_{\rm perturb},\beta_{\rm uncertainty}\)` | T gate logit 的四个固定系数 |
| `\(t_i,\sigma,g_i\)` | gate logit、sigmoid、节点混合强度 |
| `\(\nu_i\)` | pseudo-loss 样本权重 |
| `\(M_{\rm mix},c_{ir},\lambda_{ir}\)` | 抽样邻居数、抽中的邻居、插值权重 |
| `\(P_{ij}^{(v)},P_i^{(v)}\)` | F/T 当前参数化下的邻居抽样概率及其一行向量 |
| `\(\bar x_i,x_i'\)` | 邻居加权均值与 pseudo-cell |
| `\(\rho,S,\Pi_B,\widetilde v,M\)` | mask 比例、候选 mask、batch 行置换、污染输入、有效 mask |
| `\(H,W,o_i,a_i^{(1)},a_i^{(2)},z_i\)` | latent 维度、encoder 宽度、dropout 输出、两层隐藏激活、latent 向量 |
| `\(\gamma_d^{\rm LN},\beta_d^{\rm LN},\mu_d,\varsigma_d^2,\epsilon,\odot\)` | LayerNorm affine 参数、向量均值、方差、稳定常数、逐元素乘法 |
| `\(W_1,W_2,W_3,W_m,W_d,b_1,b_2,b_3,b_m,b_d\)` | encoder、mask predictor、decoder 的权重矩阵与 bias |
| `\(c_i,\zeta_i\)` | decoder 拼接向量与 mask predictor logits |
| `\(\ell_i\)` | 单样本总损失 |
| `\(\hat x_i\)` | decoder 的表达重建 |
| `\(\theta\)` | encoder、LayerNorm affine、mask predictor、decoder 的全部可学习参数 |
| `\(\theta_v^*\)` | 选择参数化 `\(v\)` 后训练得到的最优/最终网络参数 |
| `\(\mathcal L_{\rm real},\mathcal L_{\rm pseudo}^{(v)},\mathcal L^{(v)}\)` | 真实 view、pseudo view 和最终 batch 目标 |
| `\(\lambda_d,\lambda_m,w_{ig}\)` | mask 数据权重、mask loss 权重、特征重建权重 |
| `\(e_{ig}\)` | 单特征平方重建误差 |
| `\(\eta\)` | pseudo-loss 系数 `pseudo_weight` |
| `\(B,b\)` | 一个 batch 及其样本数 |
| `\(K,\hat y\)` | 最终 KMeans 簇数与预测簇编号 |

## 14. 函数与源码入口对照

| 函数/方法 | 数学角色和输入输出 |
| --- | --- |
| `normalize_parameterization` | 将 `F/-f/fixed` 映射为 `fixed`，将 `T/-t/topology` 映射为 `topology` |
| `V0Config.__post_init__` | 校验正值、边界、mode，并规范化参数化名称 |
| `V0Config.graph_enabled` | 判断 pseudo 图分支是否由 `use_pseudo`、权重、邻居配置共同启用 |
| `V0Config.mix_mode`/`gate_mode` | 把 canonical 参数化映射为记录用的 mix/gate 名称 |
| `V0Config.for_parameterization` | 在不改变其他字段的情况下生成 F 或 T 配置副本 |
| `V0Config.resolved_dict` | 输出可审计的固定配置和 effective F/T 字段 |
| `load_config` | 读取 YAML，应用显式 overrides，生成经过校验的 `V0Config` |
| `NeighborGraph` | 保存邻居 indices、概率、距离、相似度、mutual、SNN 和图 profile |
| `graph._empty_graph`/`graph._validate_data` | 生成空图哨兵并校验输入矩阵的维度、有限性 |
| `build_pca_knn_graph` | 实现 `\(\mathcal G(X)\)`，返回 indices、`\(p,d,s,m,q\)` 和 profile |
| `compute_edge_reliability` | 实现 `\(r_{ij}\)` 与 `\(a_{ij}\)`；不接收标签 |
| `summarize_edge_weights` | 计算 entropy、effective neighbor count 等无标签诊断 |
| `compute_node_gate` | 实现 F 固定 gate 或 T 解析 gate，并返回 `\(g_i,\nu_i\)` |
| `_row_and_probabilities` | 为单个节点选择 F 的 `p` 或 T 的 `a`，并重新归一化抽样行 |
| `make_pseudo_batch` | 执行邻居抽样、邻居均值、凸组合和 detached pseudo view |
| `apply_scmae_noise` | 执行 batch 内 row-swap，并返回污染输入与有效 mask |
| `AutoEncoder.__init__`/`WeightedAutoEncoder.__init__` | 创建共享 encoder、mask predictor、decoder 和其 LayerNorm 参数 |
| `WeightedAutoEncoder.forward_mask` | 计算 `\(z_i,\zeta_i,\hat x_i\)` |
| `WeightedAutoEncoder.forward` | `forward_mask` 的兼容别名，返回相同的三元组 |
| `WeightedAutoEncoder._check_expression_shape` | 检查输入是否为 `[batch, D]` 且特征维度匹配 |
| `WeightedAutoEncoder.loss_mask_weighted` | 计算逐样本 reconstruction/BCE/total loss，可选 T sample weight |
| `AutoEncoder._reconstruction_loss` | 历史 scMAE 的未逐样本拆分重建损失实现；V0 训练使用 weighted 版本 |
| `AutoEncoder.loss_mask` | 历史兼容 API，计算不带 T sample weight 的 mask 总损失 |
| `WeightedAutoEncoder.feature` | 在 `eval/no_grad` 下仅运行 encoder，提取 clean latent |
| `_build_operator_state` | 串联图、edge reliability、edge weights、node gates |
| `fit_predict` | 完整无标签训练、clean embedding 提取和可选 KMeans readout |
| `extract_embedding` | 分 batch 调用 `feature`，拼接为 `\(Z\)` |
| `resolve_device`/`resolve_runtime_device` | 解析 CPU/CUDA，并拒绝物理 GPU 0 和 7 |
| `seed_runtime`/`_torch_generator` | 初始化 Python、NumPy、PyTorch 和局部随机数流 |
| `_empty_graph` | 在无图配置下生成空邻居结构，保持形状契约 |
| `embedding_geometry` | 不看标签地统计 latent 范数、方差和有限性 |
| `neighbor_overlap` | 比较输入图与 latent kNN 的重叠，仅作诊断 |
| `evaluate_unsupervised_views` | 对两套确定性 mask view 评估 loss 和 latent 稳定性 |
| `_write_json` | 把可审计运行记录序列化到 JSON 文件 |
| `_file_sha256`/`_array_sha256` | 计算输入文件或数组内容 hash，供复现审计使用 |
| `_first_npz`/`_load_npz`/`_load_h5ad`/`load_input` | runner 外层读取 NPZ/h5ad 和可选标签，不是网络层 |
| `_encode_labels`/`_is_count_like`/`_prepare_array` | runner 外层标签编码、count-like 检测和无标签预处理 |
| `_mapped_predictions` | 仅评估阶段用匈牙利匹配把预测簇编号映射到标签编号 |
| `clustering_metrics` | runner 外层根据 `y` 计算 ACC/ARI/NMI 等，不回流到训练 |
| `run_one`/`_parse_args`/`main` | CLI 编排、输出契约和状态写入；不改变核心训练目标 |

## 15. 当前实现中不应误读的点

1. `mask_logits` 是 logits，不是概率；BCE 必须使用
   `BCEWithLogits`。
2. `M_{ig}` 是“替换后确实不同”的有效 mask，不一定等于 Bernoulli 候选
   `S_{ig}`。
3. pseudo target 是真实 anchor `\(x_i\)`，不是 pseudo-cell。
4. `F` 和 `T` 没有各自的可学习 gate 网络；T gate 是 NumPy 解析公式。
5. `K` 只属于最终 KMeans readout；无标签输入必须显式提供 `n_clusters`。
6. `precomputed_graph_embedding` 目前保留在 `fit_predict` 签名中，但当前
   `_build_operator_state` 没有消费它；不能把它描述成已经生效的图输入。
