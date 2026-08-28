#!/usr/bin/env python3
"""
生成 V9 (TopoGate LearnableGate) 详细学术流程图
保存为 PNG 和 SVG 格式
"""
import subprocess
import os

# V9 详细 Mermaid 流程图
mermaid_code = '''graph TD
    %% =============================================
    %% V9 TopoGate LearnableGate 完整流水线
    %% Variant: learnable_gate_v9_adaptive
    %% Key Features: Adaptive PCA + LearnableGate + Enhanced Stats + LearnableEdgeReliability
    %% =============================================

    %% ========== 样式声明 ==========
    classDef crit      fill:#ffcccc,color:#8b0000,stroke:#ff0000,stroke-width:2px,font-weight:bold
    classDef serious   fill:#ffe0b2,color:#7f3000,stroke:#ff6600,stroke-width:2px
    classDef warn      fill:#fff9c4,color:#6d4c00,stroke:#f9a825,stroke-width:2px
    classDef norm      fill:#d5e8d4,color:#1b5e20,stroke:#82b366,stroke-width:1px
    classDef start_end fill:#333,stroke:#000,stroke-width:2px,color:#fff,font-weight:bold
    classDef process   fill:#f9f9f9,stroke:#333,stroke-width:1px,color:#333
    classDef decision  fill:#fff2cc,stroke:#d6b656,stroke-width:1px,color:#333
    classDef data_node fill:#dae8fc,stroke:#6c8ebf,stroke-width:1px,color:#333
    classDef highlight fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#01579b
    classDef learnable fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#4a148c
    classDef optional  fill:#fafafa,stroke:#999,stroke-width:1px,color:#666,stroke-dasharray:5 5

    %% ========== 输入 / 输出节点 ==========
    Start(["🔷 输入原始数据矩阵<br>X ∈ ℝⁿˣᵈ (n样本, d特征)"]):::start_end
    End(["📊 输出聚类评价指标<br>ACC/NMI/ARI/F1"]):::start_end

    %% ========== 阶段一：数据加载与预处理 ==========
    subgraph 阶段一_数据加载与预处理["🔵 阶段一：数据加载与预处理"]
        LoadData["加载 .npz 或压缩格式<br>支持无标签/有标签数据集"]:::norm
        Log1p["log1p 变换 (可选)<br>input_mode=log1p 时启用"]:::optional
        FillNa["异常值处理<br>nan/posinf/neginf → 0"]:::warn
        HVF["高变异特征选择 (可选)<br>n_top_features=0 (V9禁用)"]:::optional
        Standardize["StandardScaler 标准化<br>with_mean=True, with_std=True"]:::norm
    end

    %% ========== 阶段二：自适应 PCA 降维 (V9 核心改进) ==========
    subgraph 阶段二_自适应PCA降维["🟢 阶段二：自适应 PCA 降维 (V9 核心改进)"]
        AdaptivePCA["自适应 PCA 维度选择<br>🟠待验证: 95%方差阈值<br>保留 ≥95% 方差, 上限2000维<br>公式: cumvar ≥ 0.95"]:::serious
        L2Norm["L2 归一化 (逐行)<br>‖x‖₂ = 1"]:::norm
    end

    %% ========== 阶段三：拓扑图构建 ==========
    subgraph 阶段三_拓扑图构建["🔵 阶段三：拓扑图构建"]
        KNNBuild["K+1 近邻搜索 (余弦距离)<br>k_eff = min(k, n_cells-1)"]:::norm
        RemoveSelf["剔除自身, 保留K个真邻居"]:::norm
        Distance2Sim["距离转相似度<br>sim = 1 - distance"]:::norm
        TempScale["温度尺度化<br>τ=0.2, softmax归一化"]:::norm
        CalcProb["计算采样概率 P(j|i)"]:::norm
        CalcMutual["互为邻居判定<br>mutual[i,j] = (i ∈ N(j))"]:::norm
        CalcSNN["共享邻居比例 SNN<br>SNN = |N(i)∩N(j)| / |N(i)∪N(j)|"]:::norm
    end

    %% ========== 阶段四：边可靠性计算 (可选可学习) ==========
    subgraph 阶段四_边可靠性计算["🔵 阶段四：边可靠性计算 (可选可学习)"]
        EdgeMode{"edge_reliability_mode<br>=?"}:::decision
        EdgeNone["rel = 1.0 (无可靠性加权)<br>weights = probs"]:::norm
        EdgeSim["rel = exp(γ_sim × sim)"]:::norm
        EdgeSimMutual["rel = exp(γ_sim×sim) × (1+γ_mutual×mutual)"]:::norm
        EdgeFull["rel = exp(γ_sim×sim) × (1+γ_mutual×mutual) × (1+γ_snn×SNN) × exp(-γ_dist×dist)"]:::norm
        LearnGamma["🟣 可学习 γ 参数 (V3)<br>γ_sim, γ_mutual, γ_snn, γ_dist<br>nn.Parameter, L2正则化 1e-4"]:::learnable
        RelL2Reg["γ_L2正则化损失项"]:::norm
        RowNorm["行归一化权重<br>weights[i] = probs[i] × rel[i]<br>weights[i] = weights[i] / Σⱼweights[i,j]"]:::norm
    end

    %% ========== 阶段五：随机邻居构建 (离线预计算) ==========
    subgraph 阶段五_随机邻居预计算["🔵 阶段五：随机/远邻预计算"]
        RandNeighbor["随机邻居 (无放回采样)<br>random_neighbors[i] ≠ graph.indices[i]"]:::norm
        FarNeighbor["远邻搜索 (相似度最低的k个)<br>far_neighbors[i] = argmin_sim(embedding[i])"]:::norm
    end

    %% ========== 阶段六：MC Dropout 不确定性估计 (V9 增强) ==========
    subgraph 阶段六_MC_Dropout不确定性估计["🟣 阶段六：MC Dropout 不确定性估计 (V9 增强)"]
        MCInit["初始化未训练 Encoder<br>model.eval() 激活 Dropout"]:::norm
        MCPasses["多次前向传播 (n_passes=5)<br>latent_p = encoder(X)"]:::norm
        MCVariance["计算潜变量方差<br>σ = std(latent_p, dim=0)"]:::norm
        MinMaxNorm["Min-Max 归一化到 [0,1]<br>uncertainty = (u - u_min) / (u_max - u_min)"]:::norm
    end

    %% ========== 阶段七：拓扑统计量构建 (V9: 4→6 维增强) ==========
    subgraph 阶段七_拓扑统计量构建["🟣 阶段七：拓扑统计量构建 (V9: 4→6 维)"]
        StatsMutual["mutual_ratio = mean(mutual[i,:])"]:::norm
        StatsSNN["snn_avg = mean(SNN[i,:])"]:::norm
        StatsPerturb["perturb = 1 - ΣⱼP(j|i)×sim(i,j)"]:::norm
        StatsUncert["uncertainty = MC Dropout 方差<br>(V9 新增, 替代恒为0)"]:::norm
        StatsDegree["degree_norm = k / n_cells<br>(V9 新增, 6维统计量)"]:::warn
        StatsCluster["clustering_coeff (V9 新增)<br>n≤5000: 精确计算 O(n²)<br>n>5000: 采样2000节点→全局均值"]:::crit
        ConcatStats["拼接为统计向量 (4维或6维)<br>[mutual, snn, perturb, uncertainty<br>, degree_norm, clustering_coeff]"]:::norm
    end

    %% ========== 阶段八：门控决策模块 (V9 核心) ==========
    subgraph 阶段八_门控决策["🟣 阶段八：门控决策 (V9 核心)"]
        GateMode{"gate_mode =?"}:::decision
        GateNone["gate = 0 (禁用混合)"]:::norm
        GateConst["gate = gate_max (固定混合)"]:::norm
        GateTopo["gate = f(stats) (静态拓扑门控)"]:::norm
        GateLearned["🟣 LearnableGate (V9 主要模式)<br>gate = gate_min + (gate_max - gate_min) × sigmoid(β·stats)"]:::learnable
        GateBinary["🟣 BinaryRouter (可选二值路由)<br>r = GumbelSoftmax(β·stats, temp)"]:::learnable

        %% LearnableGate 内部
        subgraph LearnableGate内部["LearnableGate 详细结构"]
            LG_Beta["β 参数组 (nn.Parameter)<br>β_mutual, β_snn, β_perturb<br>β_uncertainty, β_degree<br>β_cluster (enhanced_stats=6)"]:::learnable
            LG_Logits["logits = β·stats = βₘ×mutual + βₛ×snn - βₚ×perturb - βᵤ×uncertainty<br>+ β_d×degree_norm - β_c×cluster"]:::norm
            LG_Sigmoid["σ = sigmoid(logits) ∈ (0,1)"]:::norm
            LG_GateMax["🟣 可学习 gate_max (V3增强)<br>gate_max = gate_max_min + span × sigmoid(gate_max_raw)<br>gate_max_min=0.05, gate_max_max=1.0"]:::learnable
            LG_Gate["gate = gate_min + (gate_max - gate_min) × σ"]:::highlight
            LG_Schedule["β_scale 调度 (可选)<br>V9: β_scale=1.0 (禁用调度)<br>V3+: warmup_epochs=20, ramp_epochs=10"]:::optional
        end

        %% BinaryRouter 内部
        subgraph BinaryRouter内部["BinaryRouter 详细结构"]
            BR_Temp["温度调度<br>epoch≤20: temp=5.0 (软路由)<br>epoch>30: temp→0.01 (硬路由)"]:::norm
            BR_Gumbel["Gumbel-Softmax 采样<br>g = -log(-log(U)), U~Uniform(0,1)<br>r = sigmoid((logits+g)/temp)"]:::norm
            BR_Hard["推理时: r = argmax(logits)<br>r ∈ {0,1} 硬判决"]:::norm
            BR_Output["r=0: x'=anchor (自重构)<br>r=1: x'=mixed (邻居混合)"]:::highlight
        end

        GateMode -->|none| GateNone
        GateMode -->|constant| GateConst
        GateMode -->|topology| GateTopo
        GateMode -->|learned| GateLearned
        GateMode -->|binary| GateBinary

        GateNone --> GateResult
        GateConst --> GateResult
        GateTopo --> GateResult
    end

    %% ========== 阶段九：邻居特征混合 (Mixup) ==========
    subgraph 阶段九_邻居特征混合["🔵 阶段九：邻居特征混合"]
        MixMode{"mix_mode =?"}:::decision
        MixReliability["按 edge_weights 采样 m 个邻居"]:::norm
        MixMutual["仅采样互为邻居的节点"]:::norm
        MixRandom["采样 random_neighbors"]:::norm
        MixFar["采样 far_neighbors"]:::norm
        MixNone["不做混合"]:::norm

        CalcNeighborMean["计算邻居特征均值<br>neighbor_mean = Σⱼ wⱼ × Xⱼ"]:::norm
        GetAnchor["提取锚点特征<br>anchor = X[i]"]:::norm

        ContMix["x' = (1-gate)×anchor + gate×neighbor_mean"]:::norm
        BinMix["x' = anchor + r×(neighbor_mean - anchor)"]:::norm
        NoMix["x' = anchor"]:::norm

        PseudoData["伪数据 x'"]:::data_node
        RealData["真实数据 x"]:::data_node
    end

    %% ========== 阶段十：掩码噪声施加 ==========
    subgraph 阶段十_掩码噪声["🔵 阶段十：掩码噪声"]
        MaskMode{"mask_ratio 来源?"}:::decision
        MaskFixed["固定掩码比例<br>mask_ratio=0.4"]:::norm
        MaskLearn["🟣 可学习掩码比例 (V3)<br>mask_ratio = mask_min + span × sigmoid(raw)<br>mask_min=0.1, mask_max=0.6"]:::learnable

        ApplyMask["随机交换掩码 (行打乱)<br>should_swap ~ Bernoulli(mask_ratio)<br>corrupted = where(should_swap, X_shuffled, X)"]:::norm
        MaskTarget["mask = (corrupted ≠ X)"]:::norm
    end

    %% ========== 阶段十一：自编码器前向传播 ==========
    subgraph 阶段十一_自编码器前向传播["🔵 阶段十一：自编码器前向传播"]
        Encode["Encoder: Linear(d→hidden) → GELU → Linear(hidden→hidden)<br>输出 latent ∈ ℝⁿˣʰⁱᵈᵈᵉₙ"]:::norm
        MaskLogits["掩码预测头<br>Linear(hidden→d) → sigmoid → mask_logits"]:::norm
        Decode["Decoder: 拼接 [latent; mask] → Linear(d→hidden) → Linear(hidden→d)"]:::norm
        Reconstruct["重建输出<br>reconstruction ∈ ℝⁿˣᵈ"]:::norm
    end

    %% ========== 阶段十二：损失计算 ==========
    subgraph 阶段十二_损失计算["🔵 阶段十二：损失计算"]

        subgraph 真实分支["真实数据分支"]
            RealMSE["MSE_loss = mean((recon - X)² × weights)"]:::norm
            RealMaskLoss["mask_loss = BCE(sigmoid(mask_logits), mask)"]:::norm
            RealPer["rec_per = (1-mask_loss_weight)×MSE + mask_loss_weight×mask_loss"]:::norm
        end

        subgraph 伪数据分支["伪数据分支"]
            PseudoLoss["伪损失 = loss_mask_weighted(x'_corrupt, X, pseudo_mask, sample_weight)"]:::norm
            SampleWeight["sample_weight = gate / max(gate) (连续)<br>sample_weight = r (二值路由)"]:::norm
        end

        TotalLoss["总损失 = real_loss + pseudo_weight × pseudo_loss"]:::highlight
        EdgeReg["边可靠性 L2 正则化 (可选)<br>reg = γ_reg × (γ_sim² + γ_mutual² + γ_snn² + γ_dist²)"]:::optional
        FullLoss["full_loss = total_loss + edge_reg"]:::norm
    end

    %% ========== 阶段十三：反向传播与优化 ==========
    subgraph 阶段十三_反向传播与优化["🔵 阶段十三：反向传播与优化"]

        subgraph 参数分组["优化器参数分组"]
            MAEGroup["组0: MAE 参数<br>lr = 1e-3"]:::norm
            GateGroup["组1: Gate/Router/Gamma 参数<br>lr = 1e-2 (10×放大)"]:::learnable
            MaskGroup["组2: mask_ratio 参数<br>lr = 1e-3"]:::learnable
        end

        FreezeCheck{"epoch > freeze_mae_after?"}:::decision
        FullUpdate["全参数更新<br>optimizer.step()"]:::norm
        FreezeMAE["冻结 MAE 梯度<br>仅更新 Gate/Router 参数<br>zero grad for MAE params"]:::norm
    end

    %% ========== 阶段十四：推理与评估 ==========
    subgraph 阶段十四_推理与评估["🔵 阶段十四：推理与评估"]
        ExtractEmb["提取潜变量 embedding<br>z = encoder(X)"]:::norm
        KMeans["KMeans 聚类<br>n_clusters=K, n_init=10, seed固定"]:::norm
        Hungarian["匈牙利算法对齐<br>min Σ cost(y_true, y_pred)"]:::norm
        ComputeMetrics["计算指标<br>ACC/NMI/ARI/F1/..."]:::norm
    end

    %% ========== 边连接 ==========

    %% 阶段一
    Start --> LoadData
    LoadData --> Log1p
    Log1p --> FillNa
    FillNa --> HVF
    HVF --> Standardize

    %% 阶段二
    Standardize --> AdaptivePCA
    AdaptivePCA --> L2Norm

    %% 阶段三
    L2Norm --> KNNBuild
    KNNBuild --> RemoveSelf
    RemoveSelf --> Distance2Sim
    Distance2Sim --> TempScale
    TempScale --> CalcProb
    CalcProb --> CalcMutual
    CalcMutual --> CalcSNN

    %% 阶段四
    CalcSNN --> EdgeMode
    EdgeMode -->|none| EdgeNone
    EdgeMode -->|sim| EdgeSim
    EdgeMode -->|sim_mutual| EdgeSimMutual
    EdgeMode -->|sim_mutual_snn_distance| EdgeFull

    EdgeSimMutual --> RowNorm
    EdgeSim --> RowNorm
    EdgeFull --> RowNorm
    EdgeNone --> RowNorm

    EdgeFull --> LearnGamma
    LearnGamma --> EdgeFull
    LearnGamma --> RelL2Reg

    %% 阶段五
    RowNorm --> RandNeighbor
    RandNeighbor --> FarNeighbor

    %% 阶段六
    AdaptivePCA -->|PCA嵌入| MCInit
    Standardize -->|标准化数据| MCInit
    MCInit --> MCPasses
    MCPasses --> MCVariance
    MCVariance --> MinMaxNorm

    %% 阶段七
    RowNorm -->|probs, similarity| StatsMutual
    RowNorm -->|probs, similarity| StatsSNN
    RowNorm -->|probs, similarity| StatsPerturb
    MinMaxNorm -->|uncertainty| StatsUncert
    CalcProb -->|k| StatsDegree
    CalcSNN -->|indices, k| StatsCluster
    StatsMutual --> ConcatStats
    StatsSNN --> ConcatStats
    StatsPerturb --> ConcatStats
    StatsUncert --> ConcatStats
    StatsDegree --> ConcatStats
    StatsCluster --> ConcatStats

    %% 阶段八
    ConcatStats --> GateMode

    %% LearnableGate 流程
    GateLearned --> LG_Beta
    LG_Beta --> LG_Logits
    LG_Logits --> LG_Sigmoid
    LG_Sigmoid --> LG_GateMax
    LG_GateMax --> LG_Gate
    LG_Beta --> LG_Schedule
    LG_Schedule --> LG_Gate
    LG_Gate --> GateResult

    %% BinaryRouter 流程
    GateBinary --> BR_Temp
    BR_Temp --> BR_Gumbel
    ConcatStats -->|stats| BR_Gumbel
    BR_Gumbel --> BR_Hard
    BR_Hard --> BR_Output
    BR_Output --> GateResult

    GateConst --> GateResult
    GateTopo --> GateResult
    GateNone --> GateResult

    %% 阶段九
    GateResult --> MixMode
    CalcSNN -->|indices| MixReliability
    CalcSNN -->|indices| MixMutual
    RandNeighbor --> MixRandom
    FarNeighbor --> MixFar

    RowNorm -->|edge_weights| MixReliability
    MixMode -->|reliability| MixReliability
    MixMode -->|mutual| MixMutual
    MixMode -->|random| MixRandom
    MixMode -->|far| MixFar
    MixMode -->|none| MixNone

    MixReliability --> CalcNeighborMean
    MixMutual --> CalcNeighborMean
    MixRandom --> CalcNeighborMean
    MixFar --> CalcNeighborMean

    CalcNeighborMean --> GetAnchor
    GetAnchor --> ContMix
    GetAnchor --> BinMix
    GetAnchor --> NoMix

    LG_Gate -->|gate| ContMix
    BR_Output -->|r| BinMix

    ContMix --> PseudoData
    BinMix --> PseudoData
    NoMix --> PseudoData

    Standardize --> RealData

    %% 阶段十
    RealData --> ApplyMask
    PseudoData --> ApplyMask
    MixMode -->|none| ApplyMask

    ApplyMask --> MaskTarget

    MaskMode -->|固定| MaskFixed
    MaskMode -->|可学习| MaskLearn
    MaskFixed --> ApplyMask
    MaskLearn --> ApplyMask

    %% 阶段十一
    MaskTarget --> Encode
    Encode --> MaskLogits
    Encode --> Decode
    MaskLogits --> Decode
    Decode --> Reconstruct

    %% 阶段十二
    Reconstruct --> RealMSE
    MaskLogits --> RealMaskLoss
    RealMSE --> RealPer
    RealMaskLoss --> RealPer
    RealPer --> TotalLoss

    Reconstruct --> PseudoLoss
    LG_Gate -->|gate| SampleWeight
    BR_Output -->|r| SampleWeight
    SampleWeight --> PseudoLoss
    PseudoLoss --> TotalLoss

    LearnGamma -->|reg| EdgeReg
    EdgeReg --> FullLoss
    TotalLoss --> FullLoss

    %% 阶段十三
    FullLoss --> FreezeCheck
    FreezeCheck -->|否| FullUpdate
    FreezeCheck -->|是| FreezeMAE
    FullUpdate --> InferenceStart
    FreezeMAE --> InferenceStart

    %% 阶段十四
    InferenceStart["推理模式: model.eval()"]:::norm
    Reconstruct --> InferenceStart
    InferenceStart --> ExtractEmb
    ExtractEmb --> KMeans
    KMeans --> Hungarian
    Hungarian --> ComputeMetrics
    ComputeMetrics --> End

    %% 特殊边
    subgraph 数据流["双分支数据流"]
        direction LR
        RealBranch["真实分支: x → mask → MAE → loss_real"]:::data_node
        PseudoBranch["伪分支: x' → mask → MAE → loss_pseudo"]:::data_node
    end

    RealData --> RealBranch
    PseudoData --> PseudoBranch
'''

# 写入 Mermaid 文件
output_dir = "/home/luolie/ToPoGate/papers/figures"
os.makedirs(output_dir, exist_ok=True)

mermaid_path = os.path.join(output_dir, "v9_pipeline.mmd")
with open(mermaid_path, "w", encoding="utf-8") as f:
    f.write(mermaid_code)

print(f"Mermaid 文件已保存到: {mermaid_path}")

# 尝试使用 mmdc 转换为 PNG
png_path = os.path.join(output_dir, "v9_pipeline.png")
svg_path = os.path.join(output_dir, "v9_pipeline.svg")

# 检查 mmdc 是否可用
try:
    result = subprocess.run(
        ["which", "mmdc"],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print("找到 mmdc, 开始转换...")
        # 转换为 PNG (高分辨率)
        subprocess.run([
            "mmdc",
            "-i", mermaid_path,
            "-o", png_path,
            "-b", "white",
            "-s", "2",
            "-w", "4800"
        ], check=True)
        print(f"PNG 已保存到: {png_path}")

        # 转换为 SVG
        subprocess.run([
            "mmdc",
            "-i", mermaid_path,
            "-o", svg_path,
            "-b", "white",
            "-s", "2"
        ], check=True)
        print(f"SVG 已保存到: {svg_path}")
    else:
        print("mmdc 不可用, 请手动转换:")
        print(f"  mmdc -i {mermaid_path} -o {png_path} -b white -s 2 -w 4800")
except Exception as e:
    print(f"转换失败: {e}")
    print("Mermaid 文件已保存, 请手动使用 Mermaid Live Editor 或 mmdc 转换")
