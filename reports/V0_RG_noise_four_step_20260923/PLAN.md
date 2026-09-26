# ToPoGate V0_RG 抗噪机制：四步执行计划

## 目标与执行边界

研究四个已筛选优势案例：Mouse_retina、dermatology、HAR（代码ID `har`）、micro-mass。回答“输入的联合关系与局部邻域改变后，为什么聚类仍可保持稳定”，并检验这种现象与实际启用的训练结构是否有关。研究对象是**已有样本在人工污染下的稳定性**。样本曾用于无标签 transductive fitting；固定 test_indices 是评价子集，不应称为完全未见样本测试。

交付包括计划、可运行 Python 代码、已有数据可视化、有限 smoke tests。2026-09-23已按本计划完成正式GPU4机制网格及逐文件审计；完成状态见 `RUN_STATUS.md`，结果解释见 `FINAL_REPORT.md`。原扩充实验仍独立完成。所有 ToPoGate 权重仅来自 `methods/TopoGate/V0_RG`，不改优胜参数，不重训模型。源 checkpoint、预处理、聚类中心保持只读。允许物理 GPU1/4/5，禁止0/7；小规模代码验证使用CPU。

四步链条：**输入改变 → 表示和决策响应 → 实际训练组件贡献 → 必要时做方向性检验。** 每一步都有可否定解释的结果，不能预先把预期写为结论。

## 必须先分配案例角色

读取真实 checkpoint 后发现，下列机制不是每个案例都启用。下表为已检查 seed42；inventory 将核对全部三个种子。

| 案例 | k | 额外 edge reliability | node mixing amplitude | 首要机制问题 |
|---|---:|---|---|---|
| Mouse_retina | 10 | none（仍有 similarity softmax） | 按样本变化，gate 约0.02152–0.02859 | 幅度异质性与恢复 anchor；每类结果避免少数类被总体ARI掩盖 |
| dermatology | 30 | none | 固定约0.03147 | 邻域辅助恢复与自身 anchor；输入纯度下降后的表示稳定 |
| har | 30 | none | 固定约0.02652 | 同上；sensor 案例，强对照差距小，不预设显著优越 |
| micro-mass | 20 | sim_mutual_snn | 固定约0.02246 | 来源可靠性调制与自身 anchor；检验扰动后改善是否重复 |

`no_edge_modulation` 在 full 已无额外 modulation 时属于 no-op；`fixed_mix_original_aux` 在 full 幅度已恒定时也属于 no-op（容许极小浮点差异）。这些对照可验证实现一致性，不能支持组件贡献。`base_scmae` 改变整个邻域辅助分支，不是只移除一个门。没有完整2×2，不估计来源×幅度协同交互。来源仍受 PCA/cosine 邻域和 similarity softmax 控制，none 不表示完全随机取邻居。

## 统一数据与统计约定

- 复用 model seeds 42/123/7。noise seeds 3001–3005 复用现有 confirmation 实现，已经部分被观察；不称为本计划未见的盲测。无需重跑旧基线训练。
- 主条件保持 Gaussian 标准差系数0.2/0.4与 Replacement 概率0.1/0.2，每个模型接收同一原始污染数组。程序记录源数据、污染矩阵、模型、预处理、readout、测试索引和代码的 hash。
- 预处理与 readout 冻结。原始矩阵进入污染器前使用float32；实际改变比例与该表示比较，另报原dtype转float32的量化比例，修正此前混淆。
- 每个模型先 clean replay，embedding 与保存结果在注册容差内一致，预测必须一致。失败单独记录并继续独立任务；不放宽容差掩盖差异。
- 图、PCA、noise、特征筛选、probe方向均不读标签；labels 仅用于 ARI、purity 和每类事后分析。K 已来自历史 benchmark，声明其原有设定。
- 每个 model seed 内先对 noise seeds 汇总，再跨 model seeds 报均值与分散程度。不得把3×5当作15次独立训练。案例选择来自既有优势筛选，新noise seed也不消除这种选择性。
- 不比较不同模型未经归一化的 latent 位移。不把纯度下降等同分布距离，不把低翻转率等同正确预测，不把在3个seed上的均值优势称为显著。

## Step 0：资源审计与实验清单（四步的前置）

**思路：**确认可比性和组件是否真的启用后，再运行机制分析。

**代码：**`scripts/audit_checkpoints.py`。逐个核对4 datasets ×5 variants ×3 model seeds =60个checkpoint状态。要求权重、预处理、centroids、embedding、predictions、test_indices、labels_true_test、config齐全；检查输入/输出维度、预测一致性线索、配置差异、hash、gate实际范围和no-op状态。

**输出：**`inventories/checkpoints.json`、`checkpoints.csv`，每个干预标明有效、no-op、不匹配或缺失。缺失不会触发重训。最大60个状态是审计范围，不等于必须做60个有效组件比较。

## Step 1：噪声是否改变了分布和聚类相关结构？

**假设H1：**Replacement 的跨样本替换可能保留单维边际，却改变多特征对应与局部类别一致性；Gaussian 更直接改变特征方差。需要量化，而非只展示加噪图。

**代码：**`scripts/probe.py --stage observe` 的 input 部分；数值实现在 `scripts/metrics.py`，原噪声函数由 `scripts/common.py` 从既有已审计 evaluator 导入。

**操作：**在每个原checkpoint固定测试行上，计算 raw/model输入变化比例、normalized displacement；用固定clean带宽的RBF MMD²比较点云（最多512个按固定seed抽取的测试样本）；用重复 clean-versus-clean 子集差异提供描述性参考量级。高维相关性仅在clean方差选出的最多128个特征上计算，保存索引，避免事后挑特征。

重新按原fit scope重放clean图：PCA → L2 normalization → cosine kNN，k来自原cfg；冻结clean PCA后投影noisy数据。对原图profile做重放核对；原样本数/fit scope无法确认时停止该组合。图诊断为离线分析，推理不使用noisy graph。若原实现的self排除在距离并列时不严格，保留原图复现并另记严格self排除诊断，不静默改变模型训练图。

**指标：**MMD²、相关矩阵变化、kNN retention、micro/macro purity及每个测试样本局部结构变化；报告zero norm和distance ties。MMD是描述性的effect size，成对数据不直接套独立两样本置换p值。

**可视化：**输入结构图：横轴为实际normalized displacement或图损伤量，纵轴为purity/retention；辅以固定clean投影下的同一样本位移。Gaussian和Replacement用不同点形，不把名义0.2视为相同难度。

**如何继续：**若总体分布距离小但paired结构变化大，讨论局部对应关系；若只同类邻居换位，收窄为几何变化；若预处理几乎消除了扰动，不能把稳定性归因于encoder。只有输入变化确实到达encoder，才检验H2。

## Step 2：输入改变如何传递到表示和聚类？

**假设H2：**ToPoGate可能抑制扰动造成的任务相关表示变化，或形成较大的聚类决策余量，两种解释需要区分。

**代码：**同一 `probe.py --stage observe` 完成冻结推理与representation/prediction输出，无需为Step1和Step2重复推理。

**操作：**固定encoder和clean centroids，保存clean/noisy的test embedding、预测、逐样本位移、相对原clean分配的signed margin、标签仅用于事后评价。二维图只拟合clean embedding的PCA，将noisy点投到同一坐标。源数据、test row和污染hash必须与Step1对齐。

**指标定义：**

- `D_input`：对应样本输入差的Frobenius范数，除以clean输入中心化后的Frobenius范数。分母零则报告未定义。
- `D_latent`：对同一个encoder的clean/noisy embedding使用同样定义；各模型按自身clean尺度归一化。
- `ΔARI = ARI_noisy − ARI_clean`，正值表示改善；不能自动叫去噪成功。
- prediction flip：固定centroids下，noisy预测是否不同于该样本的clean预测。
- signed margin：到原clean归属中心的距离，与最近其他中心距离之差构造正负余量；负值表示跨越原分配边界。规范化方案在metrics代码与字段中记录。

**可视化：**四案例的联动曲线：Input damage → normalized latent drift → ARI/flip。各模型先保留clean起点和noisy终点；实际强对照的历史结果只来自同一raw-input/test协议。经典方法没有encoder，不虚构其D_latent。

**如何继续：**若latent位移小且ARI保持，支持表示稳定；若位移大但margin保持，支持决策余量解释；若预测稳定但ARI低，是稳定错误。若只保持Top1但降幅大，归为精度优势。

## Step 3：稳定性与哪项训练设计有关？

**假设H3：**在实际启用相关组件的案例中，full相对精确定义消融应有可辨别的抗退化贡献。

**代码：**用原权重运行 `probe.py --stage observe --variant ...`，然后 `scripts/summarize.py` 进行严格配对；`scripts/plot_results.py` 出图。

| 比较 | 有效案例优先级 | 保留的其他结构 |
|---|---|---|
| full vs no_edge_modulation | micro-mass | similarity softmax与node gate仍保留 |
| full vs fixed_mix_original_aux | Mouse_retina | gate仍参与辅助损失加权，仅混合幅度固定 |
| full vs mixed_observation_reconstruction | 四案例，需配置审计通过 | 来源、幅度等配置相同，恢复目标变化 |
| full vs base_scmae | 四案例，作为整体分支对照 | 与独立历史scMAE结果分清，不混用输入协议 |

配对键为dataset/model_seed/noise_seed/corruption/severity，要求原始污染和test索引hash一致。对消融A定义 `I_A = ΔARI_full − ΔARI_A`；同时报告双方clean/noisy ARI。只有I_A与表示/决策指标方向一致且跨模型seed重复时，才给出相应的机制支持。full和单消融不构成统计协同证明。

**可视化：**配对消融图展示I_A的模型seed均值与SD，配套表同时列出双方clean/noisy ARI及逐seed结果；完整模型的clean/noisy起点见联动图。no-op用明确标签保留在一致性检查中，不混入“组件有贡献”的效应汇总。该图无需额外训练。

**如何继续：**只有clean起点差异则归因于基础表示质量；存在额外稳定性收益才支持抗噪贡献；没有差异就不保留对应机制解释。四案例无需得到同一个机制答案。

## Step 4：必要时检验对邻域方向的选择性稳定

**启动理由：**已有discovery在这四个预选案例中观察到邻域受损而聚类质量仍较高，但这些结果不能区分拓扑方向响应与一般平滑。用户要求启动完整机制实验后，本次队列预先安排四案例的方向探针，避免依据正式observe结果事后挑选方向实验对象。方向结果仍须在前三步正式证据的基础上解释；若正式结果不支持稳定性，方向图不能反向证明抗噪机制。单独运行observe命令不会自行触发方向探针，完整队列脚本会在observe与汇总后执行。

**代码：**`scripts/probe.py --stage directions`。按照full checkpoint的官方 `estimate_neighbors(..., estimator='current')` 采样和加权，构造原生邻域混合方向；用同一样本、同一范数构造random/shuffled-direction对照。所有变体看到共同full参考输入和共同方向，若预处理不同则不能配对。

**强度：**按full的原生混合位移范数乘0.5/1/2三个预注册倍率；它们是机制探针，不是重新选择最优gate。在实际float32加法之后验证三族位移均非零且匹配目标范数（rtol=1e-4、atol=1e-8），每个倍率取三族共同有效样本。ARI、flip和latent drift在这个共同子集比较，另保留all-test值及有效样本数，避免零位移或舍入误差产生虚假稳定。跨倍率有效集合可能不同，解释曲线时同时检查覆盖率。

**协议边界：**方向探针在model-input空间构造，独立标明，不能与raw-input Gaussian/Replacement平均混排。探针无标签构造；只比较冻结权重反应，不重启训练，不在测试时更新gate。

**可视化：**方向×倍率的latent drift与prediction flip曲线，full与有效消融明确区分；signed margin保存在逐样本结果中用于边界分析。优先查看full是否对真实邻域方向更稳定，并保持clean聚类分离；若所有方向同样平滑，不能写“拓扑方向选择性”；若只full派生方向有效，也须说明探针依赖full参考，不等价于普遍鲁棒。

## 工作量、顺序与决策边界

完整审计上限60个状态。若各seed配置一致，去掉结构性no-op后预计：full12、base_scmae12、mixed-target12、有效source3、有效amplitude3，共42个有效/参考状态；4条件×5noise seeds最多840次noisy forward，但12个full状态的240份既有ARI预测可复用验证，新增主要是embedding与结构统计。实际数字以inventory为准。

执行顺序：inventory → one-case smoke → four-case full observe → 有效消融observe → 中间汇总/figures → 四案例方向探针与有效对照 → 最终汇总。每个失败组合记录完整异常，其余独立组合继续；不自动调参修补。成本受Mouse_retina图与输入矩阵大小影响，先测量试跑wall time再估算，不预报未经测量的总耗时。

实际执行命令见 [README.md](README.md)，本轮完成范围和验证记录见 [VALIDATION.md](VALIDATION.md)。现有discovery预览使用旧Euclidean/all-row诊断，只作为选择案例的已有证据；新的PCA/cosine/test-query机制图由probe真实输出生成，两套结果不混为同一协议。

分布比较与聚类研究的文献依据：[MMD](https://www.jmlr.org/papers/v13/gretton12a.html)、[dataset-shift检测与影响](https://papers.neurips.cc/paper_files/paper/2019/file/846c260d715e5b854ffad5f70a516c88-Paper.pdf)、[SCARF empirical-marginal corruption](https://openreview.net/pdf?id=CuV_qYkmKb3)。图最终可服务于论文，但本轮可视化先准确标注探索性结果、样本范围及已完成部分。
