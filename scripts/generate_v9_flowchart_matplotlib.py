#!/usr/bin/env python3
"""
使用 Matplotlib 生成 V9 TopoGate 详细流程图
生成 PNG 和 PDF 格式
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.path import Path
import matplotlib.patheffects as pe
import numpy as np

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False

def create_flowchart():
    """创建 V9 详细流程图"""
    
    fig, ax = plt.subplots(1, 1, figsize=(32, 48))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 160)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # 颜色定义
    colors = {
        'start_end': '#333333',
        'norm': '#d5e8d4',
        'norm_border': '#82b366',
        'learnable': '#f3e5f5',
        'learnable_border': '#7b1fa2',
        'warn': '#fff9c4',
        'warn_border': '#f9a825',
        'serious': '#ffe0b2',
        'serious_border': '#ff6600',
        'crit': '#ffcccc',
        'crit_border': '#ff0000',
        'decision': '#fff2cc',
        'decision_border': '#d6b656',
        'data': '#dae8fc',
        'data_border': '#6c8ebf',
        'highlight': '#e1f5fe',
        'highlight_border': '#0288d1',
        'optional': '#fafafa',
        'optional_border': '#999999',
    }
    
    def draw_box(ax, x, y, w, h, text, box_type='norm', fontsize=6, bold=False):
        """绘制带文本的矩形框"""
        color = colors.get(box_type, colors['norm'])
        border_color = colors.get(box_type + '_border', colors['norm_border'])
        
        # 绘制圆角矩形
        rect = FancyBboxPatch((x, y), w, h, 
                              boxstyle="round,pad=0.02,rounding_size=0.3",
                              facecolor=color, edgecolor=border_color,
                              linewidth=1.5, zorder=3)
        ax.add_patch(rect)
        
        # 添加文本
        weight = 'bold' if bold else 'normal'
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
               fontsize=fontsize, wrap=True, zorder=4,
               fontweight=weight)
    
    def draw_arrow(ax, x1, y1, x2, y2, label='', color='#333333'):
        """绘制连接箭头"""
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color=color, lw=1),
                   zorder=2)
        if label:
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            ax.text(mid_x, mid_y, label, fontsize=5, ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8))
    
    def draw_decision(ax, x, y, w, h, text, fontsize=6):
        """绘制菱形决策框"""
        # 菱形的顶点
        cx, cy = x + w/2, y + h/2
        half_w, half_h = w/2, h/2
        
        diamond = plt.Polygon([
            [cx, cy + half_h],
            [cx + half_w, cy],
            [cx, cy - half_h],
            [cx - half_w, cy]
        ], facecolor=colors['decision'], edgecolor=colors['decision_border'],
           linewidth=1.5, zorder=3)
        ax.add_patch(diamond)
        ax.text(cx, cy, text, ha='center', va='center', fontsize=fontsize, zorder=4)
    
    # ========== 阶段标签 ==========
    stage_labels = [
        (5, 155, "Stage 1: Data Loading & Preprocessing"),
        (5, 135, "Stage 2: Adaptive PCA (V9 Key)"),
        (5, 120, "Stage 3: Topology Graph Construction"),
        (5, 105, "Stage 4: Edge Reliability (Optional Learnable)"),
        (5, 93, "Stage 5: Random/Far Neighbors Precomputation"),
        (5, 83, "Stage 6: MC Dropout Uncertainty (V9 Enhanced)"),
        (5, 71, "Stage 7: Topology Stats (4→6 dim)"),
        (5, 57, "Stage 8: Gate Decision (V9 Core)"),
        (5, 41, "Stage 9: Neighbor Feature Mixing"),
        (5, 28, "Stage 10: Mask Noise"),
        (5, 18, "Stage 11: Autoencoder Forward"),
        (5, 10, "Stage 12: Loss Calculation"),
        (5, 4, "Stage 13: Backprop & Optimization"),
        (5, 0, "Stage 14: Inference & Evaluation"),
    ]
    
    for x, y, label in stage_labels:
        ax.text(x, y, label, fontsize=10, fontweight='bold', 
               color='#1976D2', ha='left', va='center')
    
    # ========== 输入/输出 ==========
    # 输入节点
    start_box = FancyBboxPatch((70, 153), 25, 4, 
                              boxstyle="round,pad=0.02,rounding_size=0.5",
                              facecolor='#333333', edgecolor='#000000',
                              linewidth=2, zorder=3)
    ax.add_patch(start_box)
    ax.text(82.5, 155, "Input: X ∈ ℝⁿˣᵈ (n samples, d features)", 
           ha='center', va='center', fontsize=8, color='white', fontweight='bold')
    
    # 输出节点
    end_box = FancyBboxPatch((70, -2), 25, 4, 
                             boxstyle="round,pad=0.02,rounding_size=0.5",
                             facecolor='#333333', edgecolor='#000000',
                             linewidth=2, zorder=3)
    ax.add_patch(end_box)
    ax.text(82.5, 0, "Output: ACC/NMI/ARI/F1 Metrics", 
           ha='center', va='center', fontsize=8, color='white', fontweight='bold')
    
    # ========== 阶段一：数据加载与预处理 ==========
    y_start = 149
    x_box = 30
    
    # Load Data
    draw_box(ax, x_box, y_start-1, 18, 3, "Load .npz / compressed format\nSupport labeled/unlabeled", 'norm', 6)
    draw_box(ax, x_box+20, y_start-1, 15, 3, "log1p transform (optional)\ninput_mode=log1p", 'optional', 6)
    draw_box(ax, x_box, y_start-5, 18, 3, "NaN/Inf → 0\nfill_na", 'warn', 6)
    draw_box(ax, x_box+20, y_start-5, 15, 3, "HVF feature selection\n(V9 disabled: n_top=0)", 'optional', 6)
    draw_box(ax, x_box+10, y_start-9, 18, 3, "StandardScaler\nmean=True, std=True", 'norm', 6, bold=True)
    
    # ========== 阶段二：自适应 PCA ==========
    y_pca = 132
    draw_box(ax, x_box, y_pca, 20, 4, "Adaptive PCA\nRetain ≥95% variance\nCap: 2000 dims\n🟠 Grid search on threshold", 'serious', 6, bold=True)
    draw_box(ax, x_box+22, y_pca, 15, 4, "L2 Normalize\n‖x‖₂ = 1", 'norm', 6)
    
    # ========== 阶段三：拓扑图构建 ==========
    y_graph = 118
    draw_box(ax, x_box, y_graph, 15, 3, "K+1 Nearest Neighbors\n(Cosine Distance)", 'norm', 6)
    draw_box(ax, x_box+17, y_graph, 12, 3, "Remove self\nKeep K neighbors", 'norm', 6)
    draw_box(ax, x_box+31, y_graph, 15, 3, "Distance → Similarity\nsim = 1 - dist", 'norm', 6)
    draw_box(ax, x_box, y_graph-4, 18, 3, "Temperature scaling τ=0.2\nSoftmax normalization", 'norm', 6)
    draw_box(ax, x_box+20, y_graph-4, 15, 3, "Sampling probability\nP(j|i)", 'norm', 6)
    draw_box(ax, x_box+37, y_graph-4, 15, 3, "Mutual neighbors\nmutual[i,j] = (i ∈ N(j))", 'norm', 6)
    draw_box(ax, x_box+20, y_graph-8, 15, 3, "SNN ratio\n|N(i)∩N(j)|/|N(i)∪N(j)|", 'norm', 6)
    
    # ========== 阶段四：边可靠性 ==========
    y_edge = 103
    draw_decision(ax, x_box+15, y_edge, 15, 4, "edge_reliability\n_mode=?")
    
    # 边可靠性选项
    draw_box(ax, x_box-20, y_edge-5, 12, 3, "none\nrel=1.0", 'norm', 6)
    draw_box(ax, x_box-6, y_edge-5, 12, 3, "sim\nrel=exp(γ·sim)", 'norm', 6)
    draw_box(ax, x_box+8, y_edge-5, 15, 3, "sim_mutual\nrel=exp(γ·sim)·(1+γ·mut)", 'norm', 6)
    draw_box(ax, x_box+25, y_edge-5, 20, 3, "sim_mutual_snn_distance\nrel=exp(γ·sim)·(1+γ·mut)·(1+γ·snn)·exp(-γ·dist)", 'norm', 6, bold=True)
    
    # 可学习 gamma
    draw_box(ax, x_box+47, y_edge-5, 20, 3, "🟣 Learnable γ (V3)\nγ_sim, γ_mut, γ_snn, γ_dist\nnn.Parameter, L2 reg 1e-4", 'learnable', 6)
    
    # 归一化
    draw_box(ax, x_box+25, y_edge-10, 20, 3, "Row normalization\nweights[i] = probs[i]·rel[i]", 'highlight', 6)
    
    # ========== 阶段五：随机邻居 ==========
    y_rand = 90
    draw_box(ax, x_box, y_rand, 18, 3, "Random neighbors\n(w/o replacement)", 'norm', 6)
    draw_box(ax, x_box+20, y_rand, 18, 3, "Far neighbors\n(min similarity)", 'norm', 6)
    
    # ========== 阶段六：MC Dropout ==========
    y_mc = 79
    draw_box(ax, x_box, y_mc, 15, 3, "Init untrained encoder\nmodel.eval()", 'norm', 6)
    draw_box(ax, x_box+17, y_mc, 15, 3, "Multiple passes (n=5)\nlatent = encoder(X)", 'norm', 6)
    draw_box(ax, x_box+34, y_mc, 18, 3, "Latent variance\nσ = std(latent, dim=0)", 'norm', 6)
    draw_box(ax, x_box+54, y_mc, 18, 3, "Min-Max Norm [0,1]\nuncertainty", 'highlight', 6, bold=True)
    
    # ========== 阶段七：拓扑统计量 ==========
    y_stats = 65
    draw_box(ax, x_box, y_stats, 10, 3, "mutual_ratio", 'norm', 6)
    draw_box(ax, x_box+12, y_stats, 10, 3, "snn_avg", 'norm', 6)
    draw_box(ax, x_box+24, y_stats, 12, 3, "perturb\n1-ΣP·sim", 'norm', 6)
    draw_box(ax, x_box+38, y_stats, 12, 3, "uncertainty\n(MC Dropout)", 'highlight', 6, bold=True)
    draw_box(ax, x_box+52, y_stats, 12, 3, "degree_norm\nk/n_cells", 'warn', 6)
    draw_box(ax, x_box+66, y_stats, 14, 4, "clustering_coeff\n🟡 n≤5000: exact\nn>5000: sample→global_mean", 'crit', 6)
    
    draw_box(ax, x_box+15, y_stats-5, 25, 3, "Concat: [mutual, snn, perturb, uncertainty,\ndegree_norm, clustering_coeff] → 6-dim stats", 'norm', 6, bold=True)
    
    # ========== 阶段八：门控决策 ==========
    y_gate = 52
    draw_decision(ax, x_box+15, y_gate, 15, 4, "gate_mode=?")
    
    # 门控选项
    draw_box(ax, x_box-20, y_gate-6, 10, 3, "none\ngate=0", 'norm', 6)
    draw_box(ax, x_box-8, y_gate-6, 10, 3, "const\ngate=max", 'norm', 6)
    draw_box(ax, x_box+4, y_gate-6, 12, 3, "topology\nstatic f(stats)", 'norm', 6)
    
    # LearnableGate
    draw_box(ax, x_box+18, y_gate-6, 15, 3, "🟣 LearnableGate\n(V9 main mode)", 'learnable', 6, bold=True)
    
    # BinaryRouter
    draw_box(ax, x_box+35, y_gate-6, 15, 3, "🟣 BinaryRouter\n(optional)", 'learnable', 6)
    
    # LearnableGate 内部结构
    lg_y = y_gate - 14
    draw_box(ax, x_box+18, lg_y, 15, 3, "β parameters\n(nn.Parameter)", 'learnable', 5)
    draw_box(ax, x_box+35, lg_y, 18, 3, "logits = β·stats\nβₘ·mutual + βₛ·snn - βₚ·perturb - βᵤ·uncertainty", 'norm', 5)
    draw_box(ax, x_box+55, lg_y, 15, 3, "σ = sigmoid(logits)\n∈ (0,1)", 'norm', 5)
    
    draw_box(ax, x_box+18, lg_y-5, 15, 3, "🟣 Learnable gate_max\n(V3 enhancement)\ngate_max_min + span·σ(gate_max_raw)", 'learnable', 5)
    draw_box(ax, x_box+35, lg_y-5, 18, 3, "gate = gate_min + (gate_max - gate_min)·σ", 'highlight', 5, bold=True)
    draw_box(ax, x_box+55, lg_y-5, 15, 3, "β_scale schedule\n(V9: disabled=1.0)", 'optional', 5)
    
    # BinaryRouter 内部
    br_y = y_gate - 14
    draw_box(ax, x_box+72, br_y, 15, 3, "Temperature schedule\nT≤20: T=5.0 (soft)\nT>30: T→0.01 (hard)", 'norm', 5)
    draw_box(ax, x_box+89, br_y, 15, 3, "Gumbel-Softmax\nr = σ((logits+g)/T)", 'norm', 5)
    draw_box(ax, x_box+72, br_y-5, 15, 3, "Inference: r=argmax\nr ∈ {0,1}", 'norm', 5)
    draw_box(ax, x_box+89, br_y-5, 15, 3, "r=0: x'=anchor\nr=1: x'=mixed", 'highlight', 5, bold=True)
    
    # ========== 阶段九：特征混合 ==========
    y_mix = 30
    draw_decision(ax, x_box+15, y_mix, 15, 4, "mix_mode=?")
    
    draw_box(ax, x_box-15, y_mix-5, 12, 3, "reliability", 'norm', 6)
    draw_box(ax, x_box+0, y_mix-5, 12, 3, "mutual", 'norm', 6)
    draw_box(ax, x_box+15, y_mix-5, 12, 3, "random", 'norm', 6)
    draw_box(ax, x_box+30, y_mix-5, 12, 3, "far", 'norm', 6)
    draw_box(ax, x_box+45, y_mix-5, 12, 3, "none", 'norm', 6)
    
    draw_box(ax, x_box+5, y_mix-10, 15, 3, "Sample m neighbors\nweighted by probs", 'norm', 6)
    draw_box(ax, x_box+22, y_mix-10, 15, 3, "Compute neighbor\nfeature mean", 'norm', 6)
    draw_box(ax, x_box+39, y_mix-10, 15, 3, "Get anchor\nanchor = X[i]", 'norm', 6)
    
    draw_box(ax, x_box+10, y_mix-15, 18, 3, "Continuous Mix:\nx'=(1-g)·anchor + g·neighbor", 'norm', 6)
    draw_box(ax, x_box+30, y_mix-15, 18, 3, "Binary Mix:\nx'=anchor + r·(neighbor-anchor)", 'norm', 6)
    
    # 伪数据
    pseudo_box = FancyBboxPatch((x_box+50, y_mix-18), 18, 5, 
                               boxstyle="round,pad=0.02,rounding_size=0.3",
                               facecolor=colors['data'], edgecolor=colors['data_border'],
                               linewidth=1.5, zorder=3)
    ax.add_patch(pseudo_box)
    ax.text(x_box+59, y_mix-15.5, "Pseudo Data x'", ha='center', va='center',
           fontsize=7, fontweight='bold')
    
    # 真实数据
    real_box = FancyBboxPatch((x_box, y_mix-18), 15, 5, 
                             boxstyle="round,pad=0.02,rounding_size=0.3",
                             facecolor=colors['data'], edgecolor=colors['data_border'],
                             linewidth=1.5, zorder=3)
    ax.add_patch(real_box)
    ax.text(x_box+7.5, y_mix-15.5, "Real Data x", ha='center', va='center',
           fontsize=7, fontweight='bold')
    
    # ========== 阶段十：掩码噪声 ==========
    y_mask = 11
    draw_decision(ax, x_box+15, y_mask, 15, 4, "mask_ratio=?")
    
    draw_box(ax, x_box, y_mask-5, 12, 3, "Fixed\nmask_ratio=0.4", 'norm', 6)
    draw_box(ax, x_box+14, y_mask-5, 15, 3, "🟣 Learnable (V3)\nmask_min + span·σ(raw)\n[0.1, 0.6]", 'learnable', 6)
    
    draw_box(ax, x_box+32, y_mask-5, 18, 3, "Random swap mask\n(row shuffle)\nBernoulli(mask_ratio)", 'norm', 6, bold=True)
    draw_box(ax, x_box+52, y_mask-5, 15, 3, "mask = (corrupted ≠ X)", 'norm', 6)
    
    # ========== 阶段十一：自编码器 ==========
    y_ae = 3
    draw_box(ax, x_box, y_ae, 15, 3, "Encoder\nLinear(d→hidden) → GELU → Linear", 'norm', 6, bold=True)
    draw_box(ax, x_box+17, y_ae, 12, 3, "Mask predictor\nLinear(hidden→d)", 'norm', 6)
    draw_box(ax, x_box+31, y_ae, 18, 3, "Decoder\nconcat[latent,mask] → Linear → output", 'norm', 6)
    draw_box(ax, x_box+51, y_ae, 15, 3, "Reconstruction\n∈ ℝⁿˣᵈ", 'norm', 6)
    
    # ========== 绘制主要连接线 ==========
    # 简化的连接线
    main_flow_x = 82
    
    # 从输入到阶段一
    draw_arrow(ax, 70, 155, 30, 151)
    
    # 阶段一到阶段二
    draw_arrow(ax, 48, 144, 30, 136)
    
    # 阶段二到阶段三
    draw_arrow(ax, 67, 132, 30, 124)
    
    # 阶段三到阶段四
    draw_arrow(ax, 30, 114, 50, 105)
    
    # 阶段四到阶段五
    draw_arrow(ax, 50, 96, 30, 93)
    
    # 阶段五到阶段六
    draw_arrow(ax, 30, 87, 30, 83)
    
    # 阶段六到阶段七
    draw_arrow(ax, 30, 76, 30, 73)
    
    # 阶段七到阶段八
    draw_arrow(ax, 48, 64, 50, 58)
    
    # 阶段八到阶段九
    draw_arrow(ax, 50, 47, 30, 37)
    
    # 阶段九到阶段十
    draw_arrow(ax, 30, 27, 30, 16)
    
    # 阶段十到阶段十一
    draw_arrow(ax, 30, 8, 30, 6)
    
    # 阶段十一到输出
    draw_arrow(ax, 48, 3, 70, -1)
    
    # ========== 添加图例 ==========
    legend_x = 5
    legend_y = -8
    
    ax.text(legend_x, legend_y, "Legend:", fontsize=10, fontweight='bold')
    
    legend_items = [
        ('norm', 'Standard Module'),
        ('learnable', '🟣 Learnable Parameter (nn.Parameter)'),
        ('serious', '🟠 Hyperparameter Needs Validation'),
        ('warn', '🟡 Potential Numerical Issue'),
        ('crit', '🔴 Methodological Issue'),
        ('decision', '◇ Decision Branch'),
        ('data', '📦 Data Flow'),
        ('highlight', '💎 Key Output/Formula'),
        ('optional', '...... Optional/Disabled'),
    ]
    
    for i, (style, label) in enumerate(legend_items):
        bx = legend_x + (i % 3) * 32
        by = legend_y - 3 - (i // 3) * 4
        rect = FancyBboxPatch((bx, by), 4, 2, 
                              boxstyle="round,pad=0.02,rounding_size=0.2",
                              facecolor=colors.get(style, '#fff'),
                              edgecolor=colors.get(style+'_border', '#999'),
                              linewidth=1, zorder=3)
        ax.add_patch(rect)
        ax.text(bx + 5, by + 1, label, fontsize=7, va='center')
    
    # ========== 添加标题 ==========
    ax.text(50, 159, "V9 TopoGate LearnableGate Pipeline (learnable_gate_v9_adaptive)",
           ha='center', va='center', fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='#E3F2FD', edgecolor='#1976D2', linewidth=2))
    
    # 保存
    plt.tight_layout()
    plt.savefig('/home/luolie/ToPoGate/papers/figures/v9_flowchart.png', 
                dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig('/home/luolie/ToPoGate/papers/figures/v9_flowchart.pdf', 
                bbox_inches='tight', facecolor='white', edgecolor='none')
    print("流程图已保存到:")
    print("  PNG: /home/luolie/ToPoGate/papers/figures/v9_flowchart.png")
    print("  PDF: /home/luolie/ToPoGate/papers/figures/v9_flowchart.pdf")
    plt.close()

if __name__ == '__main__':
    create_flowchart()
