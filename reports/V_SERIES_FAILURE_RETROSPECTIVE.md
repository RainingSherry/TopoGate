# TopoGate V1--V16.1 实验失败复盘与下一阶段主干选择

更新时间：2026-08-07

## 1. 文档目的与证据边界

本文档汇总 TopoGate 从早期 StaticGate 到 V16.1 的主要失败过程、直接原因、
深层机制原因和当前可支持的结论，并据此决定下一阶段是否继续使用 scMAE。

本文档不启动新实验，不重新计算数据或源码哈希，不把单 seed smoke、CUDA OOM、
下载失败、路径错误和未完成计算写成模型性能失败。数值来自当前
`CHANGELOG*.md`、`EXPERIMENT_PHASES.md`、`result/RESULTS_SUMMARY.md` 及其已核对
产物。旧记录若与当前事实表冲突，以当前源码和仍存在的产物为准。

失败统一分为五类：

1. `mechanism_no_go`：算法机制运行正常，但拓扑没有提高聚类；
2. `implementation_or_protocol_error`：实现、K、预处理、输出或 runner 改变了实验语义；
3. `theory_domain_not_supported`：输入语义或数据模型不满足方法假设；
4. `environment_or_data_error`：GPU、存储、下载、路径或外部数据错误；
5. `incomplete_compute`：计算没有形成完整产物，不能进入性能统计。

## 2. 总体结论

截至 V16.1，没有任何一个版本完成以下完整闭环：

\[
\text{候选图召回有效边}
\Rightarrow
\text{边级门控可识别}
\Rightarrow
\text{拓扑输出改善同一个聚类目标}
\Rightarrow
\text{最终 readout 获益}.
\]

目前证据支持：

- 近邻结构在部分数据集上有信息，近邻通常优于随机邻居或远邻；
- 邻居混合的收益高度依赖数据集、表示和图质量；
- 连续 node gate 不能精确拒绝具体坏边；
- softmax edge gate 容易退化为均匀平均；
- forced Top-k 可以产生稀疏选择，但会放大错误边；
- reconstruction、teacher agreement、recurrence 和 predictive support 都不自动等价于聚类收益；
- V16.1 的 abstention 语义正确，但在大量数据上因 support 全负而退化为 self-only；
- 当前不存在经过完整多数据集、多种子和因果消融确认的最终 TopoGate 版本。

## 3. 逐版本失败时间线

### 3.1 早期原型与 StaticGate

131 数据集的早期全量对照中，TopoGate 平均 ACC 约为 `0.6053`，CLUBench 最佳
方法平均约为 `0.7174`，胜/负/平为 `12/107/12`。典型弱项包括：

- `20newsgroups`：`0.197` 对最佳约 `0.913`；
- `wos`：`0.467` 对最佳约 `0.977`；
- `Baron Human`：`0.431` 对最佳约 `0.818`；
- `synthetic_control`：`0.568` 对最佳约 `0.938`。

StaticGate 消融中，core-5 的 `edge_only` 平均 ARI 约 `0.8054`，`nomix` 约
`0.8029`，`far_neighbors` 约 `0.7582`，`random_neighbors` 约 `0.7735`。
这证明局部近邻中存在信息，但不能证明 gate 能判断具体哪条边有益。

直接原因是数据域混杂：文本、scRNA、图像 embedding、表格和低维合成数据使用
相同 MAE、cosine 图和 mixing 公式。深层原因是“邻居近”没有被证明等价于
“传播其表示或 assignment 会改善聚类”。

### 3.2 V2 LearnableGate

初版四个 beta 是 argparse 浮点常量，不在计算图中；修复为 `nn.Parameter` 后，
15 数据集、3 seeds 的 LearnableGate 相对 StaticGate 总体 Delta ARI 约 `+0.003`，
按既定阈值为 4 胜、4 负、7 平。局部正例包括 Campbell `+0.036`、enron
`+0.044`、har `+0.028`、Mouse_retina `+0.011`；负例包括 sms_spam
`-0.017`、hrvatin `-0.040`。

beta 绝对值与 Delta ARI 的相关性仅 `0.205`，`p=0.464`。原因包括：

- beta 梯度只经过 `pseudo_weight 0.3 * gate_max 0.15`，有效通道上限约 4.5%；
- sigmoid 与固定 `gate_max=0.15` 把实际 gate 压在约 `0.06--0.11`；
- 连续 gate 不能精确回到 NoMix；
- pseudo reconstruction 不是聚类收益目标。

早期 Mouse_retina 还曾硬编码错误的 `K=7`，真实 `K=5`，造成约 `-0.22` 的假退化。
修正 K 后该结论消失，说明部分早期性能变化属于协议错误而不是模型失败。

### 3.3 Direction B BinaryRouter

BinaryRouter 用 Gumbel-Softmax 实现二元开关，希望在好图上开启、坏图上精确关闭。
single-seed 结果为：enron `0.052` 对 nomix `0.878`，Delta `-0.826`；har
Delta `-0.090`；Mouse_retina Delta `-0.245`。三个数据集全部崩溃，代码随后删除。

该方向解决了连续 gate 无法精确关断的问题，但 hard decision 产生不稳定近似梯度，
直接扰动 MAE encoder；更关键的是仍没有可信 edge target。因此“精确开关”只会
把错误决定变得更尖锐。

### 3.4 V3

V3 增加 learnable gate maximum、10 倍 gate 学习率、四个 gamma、degree 和
clustering coefficient、adaptive mask ratio。

- `v3_lr` 相对 baseline 约 `+0.0021`；
- `v3_full` 约 `-0.0010`；
- learnable gate maximum 与高学习率联用时，Mouse_retina 的 gate maximum
  飙到 `0.985`，接近全 mixing；
- lr 5x/3x 加 learnable maximum 分别约 `-0.0031/-0.0014`；
- v3_best 总体仅约 `+0.0013`。

四个 gamma 最终都收敛到约 `0.060`，说明同一乘法 reliability 链上的参数具有
对称梯度和结构冗余。`torch.bernoulli` 使 mask ratio 不可微，最终停在 `0.300`。
clustering coefficient 的 O(n^2) 实现对大数据被跳过并置零。新增组件增加了参数，
却没有增加独立且与聚类收益相关的监督。

### 3.5 V4

V4 将 V3 的 lr10 扩展到 8 个数据集、3 seeds。总体 Delta ARI 为 `-0.0013`，
Wilcoxon `p=0.5016`，5/8 数据集持平或退化。V3 smoke 的微小正差主要被单个
sms_spam 结果拉高，扩大数据后没有复现；部分 epoch/warmup 变化还削弱了可比性。

### 3.6 V5

V5 用单 gamma 修复对称冗余，用 Gumbel-STE 修复 mask ratio 不可微。实现阶段曾
出现 `_device_or_cpu` import、AutoEncoder 参数名、重复索引、node gate 维度、
embedding 提取、StandardScaler 和 LabelEncoder 等错误。缺少 StandardScaler
曾使 ARI 从约 `0.27` 变化到 `0.41`，足以改变性能结论。

修复接口后，STE mask 好于 fixed mask，但相对 V4 static 仍平均约 `-0.038`；
3-seed 总体约 `-0.068`，spambase 标准差约 `0.248`。V5 证明“让采样可微”并不
等价于“获得聚类相关监督”。

### 3.7 HVF 与 Adaptive PCA

HVF 在 enron、Quake 上分别出现约 `+0.083/+0.085` 的多种子局部正差，但在
Mouse_retina 约 `-0.006-- -0.031`，在 hrvatin 约 `-0.10-- -0.23`。

全局方差在文本中可能代表词项信息，在 scRNA 中却可能主要代表测序深度和技术
噪声；HVF 与 adaptive PCA 叠加还可能过度压缩。因此该预处理不能作为统一默认，
也不能用数据集类型标签自动替代正式消融。

### 3.8 V6 latent mixing

首轮 5 数据集 single-seed 平均 Delta 为 `-0.0083`，Campbell 曾出现 `+0.1835`
但无法排除单 seed 偶然性。审计发现 runner 漏传 warmup/ramp、learned static gate、
learnable maximum、freeze schedule，并有 latent consistency 重复权重错误。

修复后 har 3-seed 的 V6 为 `0.4813+-0.052`，LearnableGate 为
`0.5268+-0.037`，Delta `-0.0455`。这说明 har 上的主要问题是 latent mixing
位置本身有害，而不只是 runner bug。

### 3.9 V7 cross-attention

V7 用 cross-attention 替代简单 mean mixing。6 个数据集 single-seed 相对 V3 Full
平均约 `-0.0249`，enron `-0.128`，cnae9 `-0.063`，只有 sms_spam 超过旧消融
约 `+0.004`；运行时间也明显增加。

更强的融合模块没有更可靠的边监督，因此只是更有效地注入了错误邻居。该结果说明
表示融合能力不是主要瓶颈。

### 3.10 V9

V9 是局部结果最接近成功的一代。7 个多种子数据集的 Full-NoMix 平均约
`+0.015356`，但 Wilcoxon `p=0.3905`；balance_scale 有约 `+0.080941` 的局部
正差，spect_heart、vehicle、vertebral_column 更偏向 NoMix。额外的 Internet
Advertisements、webdata_wXa 和 SMS full TF-IDF 三集平均 Delta 为 `-0.00135`，
Wilcoxon `p=0.75`。

主要结构问题：

- warmup 把 gate 压为零时 beta 也没有梯度；
- static gate、learned gate 和外层 schedule 形成近似二次缩放；
-默认 dropout 为零时 MC-dropout uncertainty 恒为零；
- node gate 无法关闭具体坏边；
- similarity 与 `1-similarity` 同时输入，数学冗余；
- 固定 PCA 图不能随表示更新；
- detached/NumPy sample weight 切断梯度；
- reconstruction proxy 与最终 KMeans readout 不一致。

V9 只能支持“部分数据集存在条件性拓扑收益”，不能支持普遍有效的主张。

### 3.11 V10

V10 引入 edge gate、raw/EMA latent 动态图、可拒绝预算、独立 temporal target、
prototype KMeans 初始化、低秩 decoder、不均衡 prior，并修复冗余 distance、
duplicate-row self-loop、fixed-graph 消融和 output contract。

但 V10 只有短 iris engineering smoke，没有完成 5 个核心数据集、3 seeds 的
V10/V9/feature-only/fixed-graph 正式比较。因此 V10 不是已证伪的性能版本，而是
`insufficient_evidence`。它仍暴露过随机 prototype 主导首轮图、stability 自我确认、
uniform balance 压制非均衡簇、exact kNN 和高维 decoder 不可扩展等风险。

### 3.12 V11 与 TDA

iris 80 epoch、3 seeds 中，V11 Full 为 `0.6738+-0.0165`，NoMix 为
`0.6840+-0.0244`，Full 低约 `0.0102`。teacher assignment、动态图和 self/null
可以运行，但 teacher confidence、边 recurrence 和 reconstruction help 仍不等价于
edge clustering gain。

V11 H0/TDA 完成 5 数据集、3 seeds、75/75 runs。H0 相对 Full 的 head ARI 约
`+0.000010`，KMeans ARI 约 `-0.000726`。实现只是在固定 raw-kNN 1-skeleton
上做 H0 union-find，不是完整 persistent homology；detached prior 只改变 graph prior，
没有形成聚类监督。

`use_dynamic_graph=false` 与 `temporal_agreement` 曾静默产生全零 target，使所谓
fixed-graph 消融等价于强制 NoMix；该组合后来被配置校验禁止。

### 3.13 V12

V12 早期把 legacy `[latent, mask_logits] -> Linear` decoder 改为 `latent -> MLP`，
同时降低 mask loss。NoMix 从约 `0.4998/0.4764` 降到 `0.1843`，恢复 decoder 后
约 `0.4534`。因此早期 V12 下降首先是 decoder contract regression，不是 topology。

恢复后，K=5 的 softmax edge entropy 为 `1.6088`，几乎等于 `log(5)=1.6094`，
最大边权约 `0.2088`。NoMix ARI `0.6616`，edge-only `0.2015`；self/null lambda
从 0.01 增至 0.1 时结果约从 `0.6195` 降到 `0.1872`。

log-space rank hinge 虽能产生梯度，但 144/144 网格没有 entropy `<1.0` 的 cell。
V12 证明 softmax 竞争和 rank loss 能改变统计量，却不能提供边级有效性。

### 3.14 V13

V13 用 Gumbel-Top-k 强制选择两条边，effective neighbors 始终为 2，说明离散门控
机制本身成功。但 enron 约 `-0.73 ARI`，flame 约 `-0.084` 且 seed 翻转，只有
balance_scale 局部约 `+0.023`。

forced Top-k 没有 null/self fallback；kNN 一旦召回跨簇边，MSE topology alignment
就会强制 anchor 向错误簇移动。V13 证明稀疏选择形式不是瓶颈，候选正确性和拒绝
能力才是瓶颈。

### 3.15 V14

V14 用反事实 reconstruction help 与 assignment help 的逐边最小值构造 target。
5 数据集、Full/NoMix、3 seeds 中，Full ARI `0.133629`，NoMix `0.129256`，Delta
`+0.004373`，Wilcoxon `p=0.8139`，平均 target gate 仅 `0.006276`。

target 信号过弱，重构改善和 assignment 改善没有稳定映射到最终聚类；提高 loss
权重没有创造新的可识别信息。

### 3.16 V15 Counterfactual Gate

V15 用 topology-disabled EMA teacher 构造 detached `q_teacher/q_self/q_edge`，
训练逐边 utility scorer，并用 sparsemax null/self readout。Stage-1 panel 7/7 可运行，
但真实集 utility AUROC 仅 2/6 达标，candidate recall 中位数约 `0.70`，边界点、
离群点和低密度点的 null-AUROC 均为 `0.5`。

三项独立证书均缺失：teacher correctness `0/7`、held-out utility `0/7`、independent
cluster gain `0/7`；只有同一 run 的 in-sample utility `7/7` 可计算。

实现中还发现 `z_out/q_out` 语义不一致、local-consensus 未 dispatch、reconstruction
penalty 重复缩放、target/readout operator 不一致和 YAML 参数漂移。compound graph
pollution 下错误 donor 可以相互形成局部自洽，null mass 仍为零；raw/latent 图取交集
又将 recall 降到约 `0.15`。

V15 的根本 no-go 是 utility 属于人为 proxy，EMA teacher 没有独立正确性证书，
scorer 又在同一 run 中训练和评估，无法证明它对应真实聚类收益。

### 3.17 V16 与 V16.1

V16 放弃 ARI utility，改为稀疏 count thinning、raw sparse cosine candidate graph、
held-out predictive support 和 assignment-only sparsemax abstention。

fbis 三 seed 中，self-only `0.3314`，V16 `0.3295`，fixed graph `0.3985`。
Campbell 的 self/fixed/V16 为 `0.158261/0.217547/0.157655`；Mouse_retina 为
`0.404180/0.429160/0.404147`。两集 support 正边率仅约 `0.001531/0.000629`。

V16.1 扩展 count 输入后，完整筛选约 35 个候选，`candidate_positive=0`，达到预注册
停止条件。多数 clean Delta 精确为零，因为 support 全负，gate 正确回退 self。

`hrvatin_geo_maintype_counts` 是关键反例：graph purity `0.9968`、candidate recall
`0.9971`，support 正边率仅 `0.000524`，null mass `0.999118`；fixed graph ARI
`0.850403`，V16.1 ARI `0.617565`，clean Delta 相对 self 为 `-0.000309`。

这证明 candidate recall、edge purity、held-out single-donor predictive support 和
assignment gain 是不同条件。V16.1 的 gate/null 机制没有坏；失败的是 support 与
聚类传播收益之间的假设。

## 4. 不计为模型性能失败的事件

### 4.1 资源与未完成计算

- 早期 134 数据集 sweep 因系统盘满中断；
- 多 worker 共享 GPU 或 GPU 被外部进程占用导致 CUDA OOM；
- 默认隔离环境 CUDA/NVML 不可用，任务意外回退 CPU；
- `SRP224648` 的高维 decoder/Adam 状态超过单卡显存；
- `NormanWeissman2019_perturbation` Stage-0 约 4 小时 45 分钟仍未完成，按搜索上限停止；
- 这些事件均属于 `environment_error` 或 `incomplete_compute`，不进入 ARI 失败率。

### 4.2 Runner、路径与输出

- NPZ 键名 `X/x` 不一致；
- `--no_cuda` 被错误序列化；
- `scale_input=false` 曾被忽略；
- 部分数据集路径别名或 `.npz` 扩展名错误；
- `Path.stem` 把 `tr45.wc` 截为 `tr45`，丢失 word-count 语义；
- Stage-1 目录缺 condition，clean/compound 或 variant 发生覆盖；
- duplicated watcher 竞争同一输出目录；
- 初版汇总器按文件顺序而不是共同 seed 键配对；
- V11 显式数据集曾被旧默认列表静默过滤；
- V15 audit 初次广播索引错误；
- pytest `sys.path`、`jq`、`ruff`、Git metadata 和 CodeGraph 子命令缺失属于工具边界。

### 4.3 数据与外部 baseline

- 多个 scRNA H5AD 只有 normalized float，不能证明 raw count；
- dense NPZ 不满足 V16/V16.1 的 sparse-memory 协议；
- 缺标签、单一类别或含 `-1` 未过滤的数据不能直接进入统一 benchmark；
- OpenML SSL/网络失败、G2 和医学影像数据来源不明均保持 unresolved；
- 外部 baseline 的 known-K adapter、clean-room 复现和作者原版必须分开报告；
- OOM、下载失败或公式来源冲突不能归因于 TopoGate。

## 5. 跨版本的六个结构性根因

1. **图质量与干预收益混淆**：近、纯、稳定、recurrence 高均不等价于传播有益。
2. **代理目标错位**：距离、重构、teacher agreement、utility 和 support 没有稳定映射到最终聚类。
3. **门控形式不是核心瓶颈**：连续 gate 不能精确拒绝，hard gate 又放大错误边。
4. **缺少独立证书**：表示、图、target 和 gate 常由同一模型生成，形成自证循环。
5. **主干与拓扑目标不一致**：scMAE 优化 anchor reconstruction，拓扑只是额外扰动，最终又常由 KMeans 读取。
6. **版本间协议混杂**：decoder、PCA、mask、K、schedule、cluster head 和输出同时变化，削弱因果归因。

## 6. 是否应替换 scMAE 主干

### 6.1 判断

**是。下一阶段优先更换与拓扑门控目标一致的主干，比继续修补 scMAE 更合理。**

但“更换主干”不能理解为替换一个 encoder 名称。新主干必须满足：

\[
\boxed{
\text{主干优化对象}
=
\text{拓扑边对象}
=
\text{门控对象}
=
\text{最终聚类 readout 依据}
}
\]

scMAE 的主要不一致是：

- 主目标是恢复 anchor 特征，而不是形成可分簇的 pairwise/assignment geometry；
- 同簇细胞或文档也不要求相互精确重建，故 donor reconstruction risk 可能错误惩罚好边；
- 邻居 mixing 是 reconstruction 的扰动源，MAE 最容易学到的策略往往是忽略拓扑；
- 多个版本训练 reconstruction proxy，最终却用独立 KMeans 评价，训练目标和 readout 分裂；
- gate 的梯度长期只是重建损失中的弱支路。

因此 scMAE 可以保留为可选的稀疏去噪初始化器或对照，但不再适合作为下一代
TopoGate 的论文主干。

## 7. 主干候选比较

| 候选主干 | 目标一致性 | 优点 | 主要风险 | 当前建议 |
|---|---|---|---|---|
| scMAE/普通 MAE | 低 | 稀疏重建成熟 | 重建与边收益、最终聚类错位 | 不作为主干 |
| 图对比聚类 | 中高 | edge 权重可直接控制正样本对 | 错误图产生 false positives，自证循环仍存在 | 作为第二候选 |
| 鲁棒稀疏自表达/子空间聚类 | 高 | 系数矩阵同时是重建关系、图和 gate；有明确子空间假设 | O(n^2)、候选限制和 count 似然需设计 | **近期首选** |
| 污染图概率混合模型 | 很高 | gate 可解释为 clean-edge posterior，assignment 是最终输出 | 推断复杂、初始化与可识别假设要求高 | 理论备选 |
| 纯密度传播模型 | 中 | 对边界/噪声点有自然拒绝语义 | 高维距离集中，难处理极稀疏数据 | 仅作对照 |

仓库中的 `Neighborhood_Context_Aware_Contrastive_Clustering` 目前缺少全文和官方代码，
尚未确认其输入模态、图构造、cluster head 和 K 协议，不能直接重建为下一代主干。
GCC 是 gravity center clustering，不是 graph contrastive clustering，也不满足该用途。

## 8. 推荐主线：鲁棒门控自表达聚类

### 8.1 逻辑闭环

令 `C_ij` 表示用候选样本 `j` 解释样本 `i` 的稀疏自表达系数：

\[
\min_{Z,C}\;
L_{signal}(X,Z)
+ \lambda_{se}\,\rho(Z-ZC)
+ \lambda_1\lVert C\rVert_1,
\quad C_{ii}=0.
\]

其中：

- `Z` 是稀疏安全、低容量的信号表示；
- `rho` 使用 Huber、L1、Poisson deviance 或与输入计数语义一致的稳健损失；
- `C` 仅在候选边上优化，避免完整 dense `n x n` 参数；
- `C_ij=0` 就是精确拒绝；
- `|C_ij|` 是边权，也是 gate；
- `A=|C|+|C|^T` 是最终聚类 affinity；
- 最终 spectral/prototype assignment 直接读取 `A`，不再由独立 KMeans 读取另一个目标。

这样不再需要额外定义 utility scorer：拓扑、门控和聚类都围绕同一个 `C`。

### 8.2 保留“TopoGate”创新的位置

TopoGate 不再是叠在主干上方的 MLP，而是对自表达 topology 的显式门控：

1. candidate graph 只限定可优化支持集；
2. proximal/hard-concrete support 决定精确零边；
3. robust residual 决定边是否保留；
4. cluster co-assignment 只校准同一个 affinity，不生成独立 utility 标签；
5. affinity 直接进入最终聚类 readout。

### 8.3 为什么它比继续修 V16 更有希望

- V16 用单个 donor profile 预测 anchor，要求过强；自表达允许少量同簇邻居联合解释 anchor；
- V16 support 与最终 assignment readout 分离；自表达系数直接定义最终 affinity；
- V13 强制固定 Top-k；L1/proximal 解允许每个 anchor 使用 0 到多个邻居；
- V12 softmax 必须分配总质量；稀疏系数允许全零回退；
- V15 需要人为 utility；自表达目标在子空间/主题混合假设下有清晰生成语义。

### 8.4 主要风险

- 如果数据不满足 union-of-subspaces、局部主题混合或可重复 feature support，自表达也会失败；
- 候选图没有召回同结构样本时，限制支持集会产生不可恢复错误；
- unrestricted `C` 为 O(n^2)，必须使用 candidate-restricted sparse coefficients 或 anchor dictionary；
- count 数据不能直接套用高斯 L2 自表达，应使用 Poisson/multinomial deviance 或适当变换；
- representation `Z` 与 `C` 联合训练仍可能坍塌，需固定尺度、禁止自边并使用非平凡表示约束；
- spectral readout 的 K 仍需遵守 benchmark oracle/显式 K 协议。

## 9. 理论更强的备选：污染图概率混合模型

另一条路线是显式引入 latent cluster `c_i` 和 clean-edge indicator `e_ij`：

\[
p(X,E,C)=\prod_i p(x_i\mid c_i)
\prod_{(i,j)}p(e_{ij}\mid c_i,c_j,\epsilon).
\]

gate 是后验：

\[
g_{ij}=P(e_{ij}=1\mid x_i,x_j,q_i,q_j),
\]

而 `q_i=P(c_i|x_i,E)` 同时是训练对象和最终输出。此时 gate、graph likelihood 和
cluster assignment 属于同一个概率模型，不需要把 reconstruction help 命名为 ARI utility。

该路线理论闭环最强，但需要证明或明确假设混合分布、图污染模型和后验可识别性，
实现与推断风险高于稀疏自表达，不宜作为第一轮落地。

## 10. 为什么不直接选普通图对比聚类

图对比聚类确实比 scMAE 更接近目标：边权可以直接决定正样本对，cluster head 也可
作为最终输出。但它仍可能重演旧问题：

- candidate graph 错误时，跨簇边被当成 false positive；
- gate 由当前 embedding/assignment 生成时仍会自我确认；
- instance contrastive 和 cluster contrastive 可能相互竞争；
- 仅改变 attention 或 contrastive temperature 仍不定义边是否正确。

所以图对比聚类只能在具备独立边拒绝机制、同一 assignment readout 和明确污染模型时
成为候选；不能因为名称含 neighborhood/graph 就认为目标天然一致。

## 11. 下一阶段实施顺序

### Phase 0：冻结历史

- 冻结 V1--V16.1 代码和结果，不覆盖旧产物；
- 本文档作为跨版本失败和协议边界的统一入口；
- V16.1 的 35 候选零正例触发停止条件，不再扩充或调整 V16.2。

### Phase 1：先定义命题，不训练模型

固定一个数据模型，而不是同时覆盖所有稀疏数据。建议先选择：

- 稀疏 count/topic mixture；或
- noisy union-of-subspaces。

预先写清：

1. 什么是正确 topology edge；
2. `C_ij` 在何种条件下同簇非零、跨簇为零；
3. 图污染率和 candidate recall 的适用边界；
4. 为什么 affinity 的改善会降低同一个 clustering objective；
5. 哪些数据属于理论适用域，哪些只做外部对照。

### Phase 2：主干选择门槛

候选主干必须同时通过以下静态检查：

- topology 是主损失的一部分，不是额外 pseudo branch；
- gate 直接作用于主干使用的同一边；
- gate 允许精确零边且不强制固定邻居数；
- final prediction 读取同一 affinity/assignment；
- 不需要人为定义不可观测 ARI utility；
- feature-only、ungated topology、shuffled topology、output-disabled 消融语义清晰。

任一项失败，不进入真实数据实验。

### Phase 3：最小可证伪原型

只实现三个部件：

1. sparse-safe signal encoder 或固定 signal projection；
2. candidate-restricted robust self-expression `C`；
3. 由 `A=|C|+|C|^T` 直接得到的聚类 readout。

第一轮只回答：

- `C` 是否比 raw kNN 降低跨簇边比例；
- gate sparsity 是否随着人工图污染增加；
- ungated candidate graph 与 gated `C` 的差异是否进入最终 readout；
- output-disabled 是否消除拓扑收益。

不增加 teacher、EMA、动态图、utility scorer、多距离可靠性或复杂 attention。

### Phase 4：固定数据域确认

只在理论适用域内选择历史正例和新数据；负例若不满足生成假设，标记为域外，不改模型；
若满足假设仍失败，标记为 `empirical_not_supported`。先获得至少两个独立数据集的
同方向结果和完整消融，再扩大数据集。

### Phase 5：文献与实现门槛

在正式创建下一 V 系列前，按项目规则完成问题导向文献检索并归档全文，重点核对：

- robust sparse/subspace clustering；
- anchor graph 或 candidate-restricted self-expression；
- noisy/contaminated graph clustering；
- count-aware Poisson/multinomial factorisation；
- graph contrastive clustering 的 false-positive edge 处理。

没有全文、官方代码或可核对公式的命名模型，不作为主干来源。

## 12. 最终研究决策

下一步不应回到 V2/V9，也不应继续 V16 utility/support 修补。更合理的路线是：

> 将 scMAE 从“论文主干”降为可选初始化/对照，建立一个 topology-native clustering
> backbone。近期优先验证 candidate-restricted robust sparse self-expression，使门控
> 系数、图 affinity 和最终聚类读取同一个对象；若其理论假设不适合目标数据，再考虑
> 污染图概率混合模型，而不是继续增加独立 gate proxy。

这不是因为 scMAE 本身是错误模型，而是因为它解决的是 masked anchor reconstruction，
而 TopoGate 需要解决的是 noisy topology 下的可靠 co-assignment。两者目标不同，长期
通过弱 pseudo branch 强行耦合，正是 V1--V16 多次失败的共同来源。
