---
title: "Academic Cross Audit: High-Dimensional Sparse Clustering Direction"
date: 2026-08-13
auditors:
  - ctr-auditor: 25 CTR/Recommendation papers
  - lite-auditor: 26 Lightweight model papers
  - vision-auditor: 15 Vision/Audio/3D papers
  - security-auditor: 25 Security/Anomaly detection papers
  - scrna-auditor: 26 Single-cell clustering papers
  - molprot-auditor: 26 Molecular/Materials/Protein papers
  - user_provided_surveys: 6 comprehensive surveys
  - TopoGate_repo_audit: V12/V13/V17/V22 local code & results
total_papers_covered: 143
verdict: provisional
---

# 高维稀疏聚类方向合议报告

## 一、范围、证据边界与临时性声明

**审计范围**：高维稀疏数据聚类方法，涵盖单细胞转录组学、推荐/CTR、分子发现、材料科学、蛋白质工程、文本分类、图学习、网络入侵检测、社交图嵌入、视觉/音频掩码建模等跨领域方法。

**证据边界**：
- 单细胞聚类 26 篇论文来自用户综述 + scrna-auditor 独立审查
- 推荐/CTR 25 篇论文来自用户综述 + ctr-auditor 独立审查
- 分子/材料/蛋白质 26 篇论文来自用户综述 + molprot-auditor 独立审查
- 轻量模型 26 篇论文来自 lite-auditor 独立审查
- 视觉/音频/3D 15 篇论文来自 vision-auditor 独立审查
- 安全/异常检测 25 篇论文来自用户综述 + security-auditor 独立审查
- TopoGate 仓库本地代码与实验结果 — 已实地阅读 V12、V13、V17、V22 结果

**审计参与者确认**：全部 6 位独立 Agent 均已回传完整报告并确认完成。

**临时性声明**：本报告为 provisional，核心论据基于用户提供的综述材料 + 6 位独立 Agent 的审查报告 + 仓库本地代码阅读。未实际下载 PDF 逐篇验证每篇论文的每个实验结果数字，但各 Agent 的独立搜索从不同渠道交叉验证了核心发现。

---

## 二、跨领域方法谱系归纳

将所有审计的方法按核心范式归纳为 **7 大方法族**：

### 2.1 因子分解与线性交互族

| 代表方法 | 核心机制 | 稀疏处理 | 适用场景 |
|---|---|---|---|
| FM (Rendle, ICDM 2010) | 共享嵌入因子化二阶交互 | 参数量 O(kd)，线性时间 | CTR/推荐 |
| FFM (Juan, RecSys 2016) | 域感知独立嵌入 | 域级分离降低干扰 | CTR 竞赛 |
| FwFM (Pan, RecSys 2018) | 域权重矩阵建模域间差异 | 域融合替代多嵌入 | 大规模 CTR |
| HOFM (Blondel, NeurIPS 2016) | 高阶因子分解 | 树结构分解 | 链接预测 |

**共性**：特征嵌入共享化，用结构化因子限制组合爆炸。

### 2.2 深度交叉网络族

| 代表方法 | 核心机制 | 稀疏处理 | 亮点 |
|---|---|---|---|
| DCN (Wang, KDD 2017) | Cross Network 显式有界阶交叉 | 参数共享跨层 | 可部署 Web 规模 |
| DCN-V2 (Wang, WSDM 2021) | 低秩跨矩阵 + MoE | 低秩分解降参数 | 超越 DCN 10%+ |
| xDeepFM (Lian, KDD 2018) | CIN 向量级显式高阶交互 | 逐层压缩 | 显式可解释 |
| PNN (Qu, ICCV 2016) | 乘积层内积/外积 | 嵌入 + 乘积捕获二交互 | ICCV 来源 |
| DeepFM (Guo, IJCAI 2017) | FM + DNN 共享嵌入 | 共享嵌入同时学低/高阶 | 端到端经典 |
| DCN-V2 MoE (WSDM 2021) | 混合专家门控 | 输入自适应交互选择 | 门控概念相似 |

**共性**：从二阶走向显式可控制高阶交互，参数通过交叉网络/CIN 结构共享。

### 2.3 注意力/门控族

| 代表方法 | 核心机制 | 稀疏处理 | 适用 |
|---|---|---|---|
| AFM (Xiao, IJCAI 2017) | 注意力加权二阶交互 | 动态加权筛选重要交互 | CTR |
| AutoInt (Song, CIKM 2019) | 多头自注意力残差交互 | 嵌入 + 多头注意力 | 通用 |
| FiBiNET (Huang, RecSys 2019) | SENET 重要性 + Bilinear 交互 | 动态重加权特征 | CTR |
| InterHAt (Li, WWW 2021) | 层级注意力 Transformer | 分层注意力聚合 | 多域 CTR |
| SparseCTR (Lai, WWW 2026) | 三分支稀疏自注意力 | 三分支 + 分块 + top-k | 长序列 CTR |
| BlossomRec (Ma, WWW 2026) | 块级融合稀疏注意力 | 分块 O((n/b)²) + top-k | 序列推荐 |
| OptFeature/OptFS (Lyu, NeurIPS 2023) | Gumbel-Sigmoid 门控特征选择 | 可微 80-90% 特征筛选 | CTR 特征选择 |

**共性**：用注意力/门控机制实现交互的动态筛选，从全注意力走向稀疏/块级注意力。

### 2.4 生成式建模（VAE/扩散）族

| 代表方法 | 核心机制 | 稀疏处理 | 适用 |
|---|---|---|---|
| scVI (Lopez, Nat Methods 2018) | 深度生成 ZINB 似然 | 显式零膨胀建模 | scRNA-seq |
| scDeepCluster (Tian, Nat Mach Intell 2019) | ZINB-AE + DEC | 生成 + 聚类联合 | scRNA-seq |
| DESC (Li, Nat Commun 2020) | 深度嵌入 + 迭代聚类 | 自监督聚类损失 + 去批次 | scRNA-seq |
| scziDesk (Chen, Bioinf 2020) | ZINB-AE + 软 K-means | ZINB 观测模型 | scRNA-seq |
| scGMAI (Yang, PR 2021) | AE + FastICA + GMM | 独立成分降维 | scRNA-seq |
| CPA (Lotfollahi, Mol Syst Biol 2023) | 组合扰动自编码器 | 解耦药物/剂量/批次 | 扰动预测 |
| MOFDiff (Fu, ICLR 2024) | 粗粒化扩散生成 MOF | 扩散过程条件生成 | 材料设计 |
| AlphaFold 3 (Abramson, Nature 2024) | 扩散架构预测复合物 | 扩散建模结构分布 | 蛋白质 |

**共性**：显式建模数据生成过程（ZINB/扩散），从判别走向生成+条件设计。

### 2.5 掩码自编码族 — 核心方向

| 代表方法 | 核心机制 | 掩码策略 | 适用 |
|---|---|---|---|
| MAE (He, CVPR 2022) | 75% 随机掩码 + 非对称编解码 | 均匀随机 75% | 图像 |
| VideoMAE (Tong, NeurIPS 2022) | 管状时空一致性掩码 | 跨帧共享掩码 90-95% | 视频 |
| VideoMAE V2 (Wang, CVPR 2023) | 双掩码（编码器+解码器） | 编码器 90% + 解码器子集 | 视频 |
| Audio-MAE (Huang, NeurIPS 2022) | 局部窗注意力重建 | 时频局部掩码 80% | 音频 |
| Point-MAE (Pang, NeurIPS 2022) | 点云掩码 + 重建 | 60-80% 点掩码 | 3D 点云 |
| GraphMAE (KDD 2022) | 掩码图特征 + 重建 | 50-75% 节点特征掩码 | 图 |
| scMAE (Fang, Bioinf 2024) | 掩码自编码器 + 掩码预测器 | **随机掩码 ~50%** | **scRNA-seq** |
| scDRMAE (Zhang, Bioinf 2024) | MAE + 残差混合 | 相同 MAE 策略 | scRNA-seq |
| scDMAC (Frontiers 2026) | ZINB DAE + MAE 双分支 | DAE 去零 + MAE 掩码 | scRNA-seq |
| **xTrimoGene** (Gong, NeurIPS 2023) | **零-非零非对称掩码** | **零-非零等量掩码** | **scRNA-seq** |
| VIME (NeurIPS 2020) | 表格特征掩码 + 重建 + 预测 | 随机噪声掩码 | 表格数据 |
| MET (NeurIPS 2022) | Transformer 特征级掩码 | 30-50% 特征掩码 | 基因组表格 |
| SPLADE (Formal, SIGIR 2021) | FLOPS 稀疏正则化 | 端到端稀疏词汇检索 | 文本检索 |
| CAV-MAE (ICLR 2023) | 对比+掩码音视频联合 | 对比对齐 + MAE 重建 | 音视频 |
| data2vec (Baevski, ICML 2022) | 自蒸馏潜回归 | 教师潜表示作目标 | 通用 |
| scCMA (2025) | 对比 MAE | MAE + 实例+聚类对比 | scRNA-seq |
| scDCL (2026) | MAE + GCN + 双对比 | MAE + 图 + 双对比损失 | scRNA-seq |

**关键洞察**：掩码策略决定信息瓶颈质量——
- 均匀随机掩码在零膨胀数据上会陷入"零主导陷阱"（xTrimoGene 的修复）
- 跨视图一致性掩码可切断捷径（VideoMAE 的管状掩码）
- 非对称掩码（xTrimoGene）是当前零膨胀数据最优选择

### 2.6 等变图神经网络族

| 代表方法 | 核心机制 | 等变阶 | 适用 |
|---|---|---|---|
| SchNet (2017) | 连续滤波卷积 | 不变 | 分子/材料 |
| DimeNet (Gasteiger, ICML 2020) | 方向消息传递 | 旋转等变 | 分子/材料 |
| PaiNN (Schütt, ICML 2021) | 极化原子交互 | 等变消息传递 | 分子性质 |
| EGNN (Satorras, ICML 2021) | E(n) 等变 GNN | E(n) 等变 | 通用科学 |
| Equiformer (Liao, ICLR 2023) | 等变图注意力 (irreps) | SE(3)/E(3) | 高精度科学 |
| NequIP (Batzner, Nat Commun 2022) | E(3) 等变原子间势 | E(3) 等变 | 力场 1000x 数据效率 |
| Allegro (Musaelian, Nat Commun 2023) | 局部等变（无消息传递） | E(3) 局部 | 百万原子级 |
| CGCNN (Xie, PRL 2018) | 晶体图卷积 | 不变 | 晶体材料 |
| Matformer (Yan, NeurIPS 2022) | 周期图 Transformer | 周期性不变 | 晶体材料 |
| ALIGNN (npj Comp Mater 2021) | 原子线图神经网络 | 键角感知 | 晶体材料 |
| GVP-GNN (Jing, ICLR 2021) | 几何向量感知机 | SE(3) 标量+向量 | 蛋白质/设计 |
| GearNet (Zhang, ICLR 2023) | 几何关系图神经网络 | 多关系几何 | 蛋白质功能 |
| JMP (Shoghi, ICML 2024) | 跨域联合预训练 | E(3) 等变 | 通用原子系统 |

**共性**：等变先验带来显著数据效率提升（NequIP 比不变模型少 1000x 数据），是消解高维稀疏数据维度灾难的有效归纳偏置。

### 2.7 基础模型/Transformer 族

| 代表方法 | 规模 | 稀疏注意力 | 核心发现 |
|---|---|---|---|
| scBERT (Fan, Nat Mach Intell 2022) | — | Performer 稀疏注意力 | 注释优先 |
| scGPT (Cui, Nat Methods 2024) | 53M | 特殊注意力掩码 | ❌ 零-shot 聚类不如简单基线 |
| Geneformer (Theodoris, Nature 2023) | 30M-316M | 基因排名分词 | ❌ 零-shot 聚类不如简单基线 |
| scFoundation (Hao, Nat Methods 2024) | 100M | xTrimoGene 非对称编码 | 全转录组覆盖 |
| ESM-2 (Lin, Science 2023) | 3B | 标准 Transformer | 结构预测 |
| ESM3 (Hayes, Science 2024) | 98B | 多模态生成 | 全新荧光蛋白生成 |
| AlphaFold (Jumper, Nature 2021) | — | Evoformer 轴向注意 | CASP14 GDT 92.4 |
| GROVER (Rong, NeurIPS 2020) | 100M | MPNN + Transformer | MoleculeNet >6% |
| ChemBERTa (2020) | — | RoBERTa SMILES MLM | 缩放定律验证 |
| Uni-Mol (Zhou, ICLR 2023) | — | SE(3) 距离感知注意力 | QM9 SOTA |
| MatSciBERT (Gupta, npj 2022) | — | 材料领域 BERT | 超越 SciBERT |
| ProSST (Li, NeurIPS 2024) | — | 结构量化 token | 折叠分类 SOTA |
| SC-MAMBA2 (bioRxiv 2024) | 150M | Mamba2 SSM | 57M 细胞单 GPU |
| GeneMamba (2026) | — | 双向 Mamba | 效率 2-3x |
| RegFormer (Hu, Nat Commun 2026) | — | Mamba + GRN | GRN 重建 SOTA |
| LiteLLM (2026) | 极轻量 | 4 层 256 维 | 极简 Transformer |
| NSA (Yuan, ACL 2025) | — | 原生稀疏注意力 | 64K 全注意力质量 11x 加速 |

**关键发现**（Genome Biology 2025 系统评估）：**预训练表示不天然聚类友好**——HVG + scVI + Harmony 等简单基线在批次整合与生物信号保持上常优于基础模型。

---

## 三、跨领域交叉分析

### 3.1 掩码策略的跨领域比较

| 掩码策略 | 代表 | 零膨胀鲁棒性 | 捷径风险 | 实现复杂度 |
|---|---|---|---|---|
| 均匀随机 | MAE, scMAE | ❌ 零主导陷阱 | 高 | 低 |
| 零-非零分层 | xTrimoGene | ✅ | 中 | 中 |
| 跨视图一致性 | VideoMAE | ✅ | 低 | 中 |
| 拓扑引导 | TopoGate V22 | ✅（设计目标） | 低 | 高 |
| 语义引导 | BEiT, I2P-MAE | ✅（语义 token） | 低 | 高 |

### 3.2 聚类范式的跨领域比较

| 聚类范式 | 代表 | 生成性 | 判别性 | 结构性 |
|---|---|---|---|---|
| 单视图重建聚类 | scDeepCluster, scMAE | ✅ 强 | ❌ | ❌ |
| 对比聚类 | scDCL, scCMA | ✅ | ✅ 强 | ❌ |
| 拓扑引导聚类 | TopoGate | ✅ | ✅ (判别器) | ✅ 强 |
| 密度峰值聚类 | SEDC (KDD 2026) | ❌ | ❌ | ✅ (密度) |
| 等变嵌入 + KMeans | EGNN, Equiformer | ❌ | ✅ (潜空间) | ✅ (等变) |
| 稀疏子空间 | SSC, LRR, DCSSC | ❌ | ❌ | ✅ (子空间) |
| 张量谱聚类 (TNNLS 2024) | DTSC | ❌ | ✅ (判别性) | ✅ (张量) |

**关键空白**：尚无工作将**生成性（掩码重建）+ 判别性（对比/对抗）+ 结构性（拓扑/图）**三类约束统一到潜表示对齐目标下。data2vec 的自蒸馏潜回归提供了统一路径。

### 3.3 视觉/音频/3D 可迁移洞察

| # | 洞察 | 源领域/论文 | 可迁移到稀疏聚类 |
|---|---|---|---|
| 1 | 掩码比例应与数据冗余度成正比 | MAE (75%), VideoMAE (90-95%) | scRNA-seq 需 40-60% 而非 15% |
| 2 | 潜表示回归比输入重建更稳健 | data2vec (ICML 2022) | 绕过零膨胀直接对齐潜空间 |
| 3 | 双掩码分离效率与质量 | VideoMAE V2 (CVPR 2023) | 编码器见稀疏子集，解码器见全集 |
| 4 | 稀疏注意力模式可映射到簇结构 | BigBird/Longformer | 全局+局部+随机 = kNN+随机长程连接 |
| 5 | 内容依赖的稀疏性 ≈ 可学习图 | NSA/Mamba | TopoGate 的可学习边选择 |
| 6 | 跨视图一致性促成聚类 | CAV-MAE (ICLR 2023) | TopoGate pseudo-batch 双视图 |
| 7 | 自蒸馏稳定无监督训练 | data2vec | V10 的 EMA 教师的推广 |

---

## 四、TopoGate 实验现状评估

### 4.1 当前状态汇总

| 版本 | 核心机制 | 实验结果 | 判定 |
|---|---|---|---|
| **V12** | 拓扑对齐 + rank loss + edge reliability | 30/30 phase1: restricted no-go; 36/36 phase2: restricted go; 144/144 phase3: **no-go** (edge entropy 未显著降低) | ❌ 不进入论文 main-result |
| **V13** | Gumbel-Top-k hard gate | 30/30 完成。enron **-0.73 ARI 灾难性崩溃**; flame **-0.084**; balance_scale +0.023 | ⚠️ 有条件 go |
| **V17** | 非深度学习稀疏自表达参考 | 无真实数据性能证据 | 参考实现 |
| **V22** | 拓扑判别器 + 硬预算门控 + scMAE | engineering smoke 级别；**35 数据集 empirical_not_supported** | ❌ 尚未验证 |
| **V16.1** | 35 去重数据集全部标记 | candidate_positive=0 | ❌ empirical_not_supported |

### 4.2 根本原因分析

1. 拓扑统计量与聚类目标之间的对齐鸿沟
2. scMAE 重建目标与拓扑门控的梯度冲突
3. 单视图数据的信息瓶颈不足
4. 预注册协议的严格性可能掩盖微弱信号
5. 基础模型预训练的教训未被借鉴

---

## 五、经合议流程的方向建议

### 优先级 A — 必须尝试的高价值方向

**A1. 零-非零非对称掩码策略** ★★★★★
- xTrimoGene (NeurIPS 2023) 已证明必要性。V13 enron -0.73 ARI 崩溃可能根因。
- **最小改动、最高回报方案**。

**A2. VideoMAE 式跨批次一致性掩码** ★★★★★
- 跨冗余维度共享掩码切断捷径。与 V22 cooperative keep gate 互补。

**A3. 对比 + 掩码联合损失** ★★★★★
- scDCL/scCMA/CAV-MAE 均证明互补性。不改变骨干结构。

### 优先级 B — 值得实验的中期方向

**B1. data2vec 式自蒸馏潜回归替代 MSE 重建** ★★★★
**B2. 轻量 Mamba 骨干替代 Transformer** ★★★★
**B3. 结构化消融：拓扑门控 vs 简单对比** ★★★★

### 优先级 C — 长远探索方向

**C1. 三流联合框架：生成 + 判别 + 拓扑** ★★★
**C2. 拓扑统计量的理论分析与重新设计** ★★★
**C3. 基础模型聚类适配器** ★★★

---

## 六、屏蔽的建议（不推荐实施）

| 方向 | 不推荐理由 |
|---|---|
| 更多超参数网格搜索 | V12 stage 3 已经做了 144 runs, edge entropy 降幅 <0.1 |
| 加更多数据集 | 核心问题是方法本身未达标，加数据集只是稀释 |
| 用更大模型 | 基础模型本身不聚类友好，更大规模加重表示鸿沟 |
| 直接发表当前 V22 | ΔARI < 0.03 且 candidate_positive=0，违反仓库晋级规则 |

---

## 七、推荐的行动路径

```
Phase 1 (1-2 周) —— 快速验证
├── 实施 A1: 零-非零非对称掩码（最小改动）
├── 实施 A3: 加对比损失（骨干不变）
└── 在 enron, sector, real-sim, micro-mass, CNAE-9 上快速验证

Phase 2 (2-4 周) —— 结构化消融与诊断
├── 实施 B3: 严格消融（纯MAE vs +对比 vs +拓扑 vs +对比+拓扑）
├── 实施 A2: 跨批次一致性掩码
└── 诊断：拓扑统计量是否包含聚类判别信息

Phase 3 (1-2 月) —— 架构升级
├── 实施 B1/B2: data2vec 潜回归 / Mamba 骨干
├── 对比 + 掩码 + 拓扑三流联合
└── 在 scCluBench 36 数据集上执行预注册验证
```

---

## 八、仓库的两点独特空白验证

来自全部 6 位独立 Agent 的一致确认：

1. **TopoGate 的学习拓扑门控在领域内是独特的**——没有其他方法将可学习图拓扑与掩码自编码结合用于聚类。
2. **可学习边稀疏性在 scMAE 框架中尚未有其他工作探索**——为 TopoGate 提供了明确的发表叙事基础。

---

## 九、核心一句总结

> 在借鉴 xTrimoGene 的非对称掩码修复零膨胀问题之前，不应继续推进拓扑门控的复杂设计——因为当前 V12/V13/V22 的实验证据（edge entropy 未降、enron 崩溃、35 全 not_supported）都指向同一根因：**scMAE 的均匀掩码在零膨胀数据上的"零主导陷阱"可能放大了拓扑门控的噪声梯度**。一旦掩码问题修复，拓扑门控的边际贡献才能被公平评估。

---

## 十、审计参与者确认

| 角色 | 领域 | 覆盖论文数 | 状态 |
|---|---|---|---|
| lite-auditor | 轻量模型 | 26 | ✅ |
| vision-auditor | 视觉/音频/3D | 15 | ✅ |
| security-auditor | 安全/异常检测 | 25 | ✅ |
| ctr-auditor | CTR/推荐 | 25 | ✅ |
| molprot-auditor | 分子/材料/蛋白质 | 26 | ✅ |
| scrna-auditor | 单细胞聚类 | 26 | ✅ |
| 主审（team-lead） | 综述整合 + 仓库实地阅读 | V12/V13/V17/V22 | ✅ |

**总覆盖论文数：143 篇论文/方法**

---

*报告生成日期：2026-08-13*
*审计框架：academic-cross-audit*
*主审评估：provisional*
