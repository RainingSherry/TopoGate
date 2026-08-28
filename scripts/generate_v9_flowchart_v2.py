#!/usr/bin/env python3
"""
V9 TopoGate LearnableGate 详细学术流程图
使用纯文本符号，无 emoji 依赖
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import numpy as np

# 设置字体
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def create_flowchart():
    """创建 V9 详细流程图"""
    
    fig, ax = plt.subplots(1, 1, figsize=(36, 52))
    ax.set_xlim(0, 120)
    ax.set_ylim(-15, 175)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor('#fafafa')
    fig.patch.set_facecolor('#fafafa')
    
    # 颜色方案
    colors = {
        'start_end': ('#2c3e50', '#2c3e50'),
        'norm': ('#e8f5e9', '#4caf50'),
        'learnable': ('#f3e5f5', '#9c27b0'),
        'warn': ('#fff9c4', '#fbc02d'),
        'serious': ('#ffe0b2', '#ff9800'),
        'crit': ('#ffcdd2', '#f44336'),
        'decision': ('#fffde7', '#ffc107'),
        'data': ('#e3f2fd', '#2196f3'),
        'highlight': ('#e1f5fe', '#03a9f4'),
        'optional': ('#fafafa', '#bdbdbd'),
    }
    
    def draw_box(ax, x, y, w, h, text, box_type='norm', fontsize=7, bold=False):
        """绘制圆角矩形框"""
        fc, ec = colors.get(box_type, colors['norm'])
        rect = FancyBboxPatch((x, y), w, h, 
                              boxstyle="round,pad=0.01,rounding_size=0.15",
                              facecolor=fc, edgecolor=ec,
                              linewidth=1.5, zorder=3)
        ax.add_patch(rect)
        
        weight = 'bold' if bold else 'normal'
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
               fontsize=fontsize, wrap=True, zorder=4, fontweight=weight)
    
    def draw_decision(ax, x, y, w, h, text, fontsize=6):
        """绘制菱形决策框"""
        fc, ec = colors['decision']
        cx, cy = x + w/2, y + h/2
        diamond = plt.Polygon([
            [cx, cy + h/2], [cx + w/2, cy],
            [cx, cy - h/2], [cx - w/2, cy]
        ], facecolor=fc, edgecolor=ec, linewidth=1.5, zorder=3)
        ax.add_patch(diamond)
        ax.text(cx, cy, text, ha='center', va='center', fontsize=fontsize, zorder=4)
    
    def draw_arrow(ax, x1, y1, x2, y2, color='#666666'):
        """绘制箭头"""
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color=color, lw=1.2),
                   zorder=2)
    
    # ==================== 标题 ====================
    title_box = FancyBboxPatch((25, 170), 70, 5, 
                              boxstyle="round,pad=0.01,rounding_size=0.3",
                              facecolor='#1976d2', edgecolor='#0d47a1',
                              linewidth=2, zorder=3)
    ax.add_patch(title_box)
    ax.text(60, 172.5, "V9 TopoGate LearnableGate Pipeline  (learnable_gate_v9_adaptive)",
           ha='center', va='center', fontsize=12, color='white', fontweight='bold')
    
    # ==================== 输入/输出节点 ====================
    # 输入
    start_box = FancyBboxPatch((85, 165), 30, 4, 
                              boxstyle="round,pad=0.01,rounding_size=0.3",
                              facecolor='#2c3e50', edgecolor='#1a1a1a',
                              linewidth=2, zorder=3)
    ax.add_patch(start_box)
    ax.text(100, 167, "Input: X in R^(n x d)", ha='center', va='center',
           fontsize=8, color='white', fontweight='bold')
    
    # 输出
    end_box = FancyBboxPatch((85, -10), 30, 4, 
                             boxstyle="round,pad=0.01,rounding_size=0.3",
                             facecolor='#2c3e50', edgecolor='#1a1a1a',
                             linewidth=2, zorder=3)
    ax.add_patch(end_box)
    ax.text(100, -8, "Output: ACC/NMI/ARI/F1", ha='center', va='center',
           fontsize=8, color='white', fontweight='bold')
    
    # ==================== 阶段一：数据加载与预处理 ====================
    ax.text(5, 162, "[Stage 1] Data Loading & Preprocessing", fontsize=9, fontweight='bold', color='#1565c0')
    
    y1 = 158
    draw_box(ax, 35, y1-1, 18, 3, "Load .npz / compressed\nSupport: labeled/unlabeled", 'norm', 7)
    draw_box(ax, 55, y1-1, 15, 3, "log1p transform\n(optional)", 'optional', 7)
    draw_box(ax, 35, y1-5, 18, 3, "NaN/Inf handling\n-> 0", 'warn', 7)
    draw_box(ax, 55, y1-5, 15, 3, "HVF selection\n(V9: disabled)", 'optional', 7)
    draw_box(ax, 42, y1-9, 18, 3, "StandardScaler\nmean=T, std=T", 'highlight', 7, bold=True)
    
    # ==================== 阶段二：自适应 PCA ====================
    ax.text(5, 145, "[Stage 2] Adaptive PCA (V9 KEY)", fontsize=9, fontweight='bold', color='#2e7d32')
    
    y2 = 141
    draw_box(ax, 35, y2, 22, 4, "Adaptive PCA Dim Selection\nRetain >=95% variance\nCap: 2000 dims", 'serious', 7, bold=True)
    draw_box(ax, 60, y2, 15, 4, "L2 Normalize\n||x||_2 = 1", 'norm', 7)
    
    # ==================== 阶段三：拓扑图构建 ====================
    ax.text(5, 132, "[Stage 3] Topology Graph Construction", fontsize=9, fontweight='bold', color='#1565c0')
    
    y3 = 128
    draw_box(ax, 35, y3, 16, 3, "K+1 Nearest Neighbors\n(Cosine Distance)", 'norm', 6)
    draw_box(ax, 53, y3, 12, 3, "Remove self\nKeep K", 'norm', 6)
    draw_box(ax, 67, y3, 15, 3, "Distance -> Sim\nsim = 1 - dist", 'norm', 6)
    draw_box(ax, 84, y3, 15, 3, "Temperature scale\ntau=0.2, softmax", 'norm', 6)
    
    y3b = y3 - 4
    draw_box(ax, 35, y3b, 16, 3, "Sampling prob\nP(j|i)", 'norm', 6)
    draw_box(ax, 53, y3b, 12, 3, "Mutual check\ni in N(j)", 'norm', 6)
    draw_box(ax, 67, y3b, 15, 3, "SNN ratio\n|Ni∩Nj|/|Ni∪Nj|", 'norm', 6)
    
    # ==================== 阶段四：边可靠性 ====================
    ax.text(5, 118, "[Stage 4] Edge Reliability (Optional Learnable)", fontsize=9, fontweight='bold', color='#1565c0')
    
    y4 = 114
    draw_decision(ax, 40, y4, 14, 4, "edge_rel\n_mode=?")
    
    draw_box(ax, 20, y4-5, 14, 3, "none\nrel=1.0", 'norm', 6)
    draw_box(ax, 36, y4-5, 14, 3, "sim\nrel=exp(gam*sim)", 'norm', 6)
    draw_box(ax, 52, y4-5, 16, 3, "sim_mutual\nrel=exp*(1+gam*)", 'norm', 6)
    draw_box(ax, 70, y4-5, 20, 3, "sim_mutual_snn_distance\nrel=exp*(1+gam*)*(1+gam*snn)*exp(-gam*dist)", 'norm', 6, bold=True)
    
    # Learnable gamma
    draw_box(ax, 92, y4-5, 20, 3, "LEARNABLE gamma (V3)\ngam_sim, gam_mut, gam_snn\ngam_dist (nn.Parameter)", 'learnable', 6)
    
    draw_box(ax, 52, y4-10, 22, 3, "Row normalization\nweights = probs * rel", 'highlight', 7)
    
    # ==================== 阶段五：随机邻居 ====================
    ax.text(5, 99, "[Stage 5] Random/Far Neighbors Precomputation", fontsize=9, fontweight='bold', color='#1565c0')
    
    y5 = 95
    draw_box(ax, 35, y5, 18, 3, "Random neighbors\n(w/o replacement)", 'norm', 7)
    draw_box(ax, 55, y5, 18, 3, "Far neighbors\n(min similarity)", 'norm', 7)
    
    # ==================== 阶段六：MC Dropout 不确定性 ====================
    ax.text(5, 88, "[Stage 6] MC Dropout Uncertainty (V9 Enhanced)", fontsize=9, fontweight='bold', color='#7b1fa2')
    
    y6 = 84
    draw_box(ax, 35, y6, 15, 3, "Init encoder\nmodel.train()", 'norm', 6)
    draw_box(ax, 52, y6, 15, 3, "Multiple passes (n=5)\nlatent = encoder(X)", 'norm', 6)
    draw_box(ax, 69, y6, 16, 3, "Latent variance\nsigma = std(lat, dim=0)", 'norm', 6)
    draw_box(ax, 87, y6, 16, 3, "MinMax Norm [0,1]\nuncertainty", 'highlight', 6, bold=True)
    
    # ==================== 阶段七：拓扑统计量 ====================
    ax.text(5, 76, "[Stage 7] Topology Statistics (4 -> 6 dim)", fontsize=9, fontweight='bold', color='#7b1fa2')
    
    y7 = 72
    draw_box(ax, 35, y7, 9, 3, "mutual\n_ratio", 'norm', 6)
    draw_box(ax, 46, y7, 9, 3, "snn\n_avg", 'norm', 6)
    draw_box(ax, 57, y7, 11, 3, "perturb\n1-sum(P*sim)", 'norm', 6)
    draw_box(ax, 70, y7, 12, 3, "uncertainty\n(MC Dropout)", 'highlight', 6, bold=True)
    draw_box(ax, 84, y7, 11, 3, "degree\n_norm", 'warn', 6)
    draw_box(ax, 97, y7, 15, 4, "clustering\n_coeff\n[W] n>5000: approx", 'crit', 6)
    
    draw_box(ax, 35, y7-5, 40, 3, "Concat: [mutual, snn, perturb, uncertainty, degree_norm, clustering] -> 6-dim stats tensor", 'norm', 7, bold=True)
    
    # ==================== 阶段八：门控决策 ====================
    ax.text(5, 60, "[Stage 8] Gate Decision (V9 CORE)", fontsize=9, fontweight='bold', color='#c62828')
    
    y8 = 56
    draw_decision(ax, 40, y8, 14, 4, "gate\n_mode=?")
    
    # 门控选项
    draw_box(ax, 22, y8-5, 10, 3, "none\ngate=0", 'norm', 6)
    draw_box(ax, 34, y8-5, 10, 3, "const\ngate=max", 'norm', 6)
    draw_box(ax, 46, y8-5, 12, 3, "topology\nstatic f()", 'norm', 6)
    draw_box(ax, 60, y8-5, 14, 3, "LEARNABLE\nGATE (V9)", 'learnable', 6, bold=True)
    draw_box(ax, 76, y8-5, 14, 3, "BINARY\nROUTER", 'learnable', 6)
    
    # LearnableGate 详细
    lg_y = y8 - 13
    ax.text(5, lg_y+4, "  LearnableGate Internal:", fontsize=8, fontweight='bold', color='#7b1fa2')
    draw_box(ax, 22, lg_y, 14, 3, "beta params\n(nn.Parameter)", 'learnable', 6)
    draw_box(ax, 38, lg_y, 16, 3, "logits = beta . stats\nbm*mutual + ...", 'norm', 6)
    draw_box(ax, 56, lg_y, 12, 3, "sigmoid()\nin (0,1)", 'norm', 6)
    
    draw_box(ax, 22, lg_y-4, 14, 3, "LEARNABLE\ngate_max (V3)", 'learnable', 6)
    draw_box(ax, 38, lg_y-4, 16, 3, "gate = gate_min\n+ (gate_max - min)*sigmoid", 'highlight', 6, bold=True)
    draw_box(ax, 56, lg_y-4, 12, 3, "beta_scale\n(V9: disabled)", 'optional', 6)
    
    # BinaryRouter 详细
    br_y = lg_y
    ax.text(70, br_y+4, "  BinaryRouter Internal:", fontsize=8, fontweight='bold', color='#7b1fa2')
    draw_box(ax, 70, br_y, 14, 3, "Temp schedule\nT<=20: 5.0 (soft)", 'norm', 6)
    draw_box(ax, 86, br_y, 16, 3, "Gumbel-Softmax\nr = sigma((log+g)/T)", 'norm', 6)
    draw_box(ax, 70, br_y-4, 14, 3, "Inference: r=argmax\nr in {0,1}", 'norm', 6)
    draw_box(ax, 86, br_y-4, 16, 3, "r=0: x'=anchor\nr=1: x'=mixed", 'highlight', 6, bold=True)
    
    # ==================== 阶段九：特征混合 ====================
    ax.text(5, 30, "[Stage 9] Neighbor Feature Mixing", fontsize=9, fontweight='bold', color='#1565c0')
    
    y9 = 26
    draw_decision(ax, 40, y9, 14, 4, "mix\n_mode=?")
    
    draw_box(ax, 18, y9-5, 12, 3, "reliability", 'norm', 6)
    draw_box(ax, 32, y9-5, 12, 3, "mutual", 'norm', 6)
    draw_box(ax, 46, y9-5, 12, 3, "random", 'norm', 6)
    draw_box(ax, 60, y9-5, 12, 3, "far", 'norm', 6)
    draw_box(ax, 74, y9-5, 12, 3, "none", 'norm', 6)
    
    y9b = y9 - 10
    draw_box(ax, 30, y9b, 15, 3, "Sample m neighbors\nweighted by probs", 'norm', 6)
    draw_box(ax, 47, y9b, 14, 3, "Neighbor feature\nmean", 'norm', 6)
    draw_box(ax, 63, y9b, 14, 3, "Get anchor\nanchor = X[i]", 'norm', 6)
    
    y9c = y9 - 15
    draw_box(ax, 35, y9c, 18, 3, "Continuous:\nx'=(1-g)*anchor + g*neighbor", 'norm', 7)
    draw_box(ax, 55, y9c, 18, 3, "Binary:\nx'=anchor + r*(neighbor-anchor)", 'norm', 7)
    
    # 数据框
    pseudo_box = FancyBboxPatch((76, y9c-1.5), 15, 4, 
                              boxstyle="round,pad=0.01,rounding_size=0.15",
                              facecolor='#e3f2fd', edgecolor='#2196f3',
                              linewidth=1.5, zorder=3)
    ax.add_patch(pseudo_box)
    ax.text(83.5, y9c+0.5, "Pseudo x'", ha='center', va='center',
           fontsize=7, fontweight='bold')
    
    real_box = FancyBboxPatch((18, y9c-1.5), 14, 4, 
                             boxstyle="round,pad=0.01,rounding_size=0.15",
                             facecolor='#e3f2fd', edgecolor='#2196f3',
                             linewidth=1.5, zorder=3)
    ax.add_patch(real_box)
    ax.text(25, y9c+0.5, "Real x", ha='center', va='center',
           fontsize=7, fontweight='bold')
    
    # ==================== 阶段十：掩码噪声 ====================
    ax.text(5, 8, "[Stage 10] Mask Noise", fontsize=9, fontweight='bold', color='#1565c0')
    
    y10 = 4
    draw_decision(ax, 40, y10, 14, 4, "mask\n_ratio?")
    
    draw_box(ax, 22, y10-5, 12, 3, "Fixed\n=0.4", 'norm', 6)
    draw_box(ax, 36, y10-5, 16, 3, "LEARNABLE (V3)\n[0.1, 0.6]", 'learnable', 6)
    
    draw_box(ax, 54, y10-5, 18, 3, "Random swap mask\n(row shuffle)\nBernoulli(ratio)", 'norm', 7, bold=True)
    draw_box(ax, 74, y10-5, 16, 3, "mask = (corrupted != X)", 'norm', 6)
    
    # ==================== 阶段十一：自编码器 ====================
    ax.text(5, -4, "[Stage 11] Autoencoder Forward", fontsize=9, fontweight='bold', color='#1565c0')
    
    y11 = -8
    draw_box(ax, 25, y11, 15, 3, "Encoder\nLinear(d->h) -> GELU -> Linear", 'norm', 7, bold=True)
    draw_box(ax, 42, y11, 12, 3, "Mask head\nLinear(h->d)", 'norm', 6)
    draw_box(ax, 56, y11, 18, 3, "Decoder\nconcat[latent,mask] -> Linear -> out", 'norm', 6)
    draw_box(ax, 76, y11, 14, 3, "Reconstruction\nin R^(n x d)", 'norm', 6)
    
    # ==================== 阶段十二：损失计算 ====================
    ax.text(5, -15, "[Stage 12] Loss Calculation", fontsize=9, fontweight='bold', color='#1565c0')
    
    # 损失框
    loss_y = -19
    draw_box(ax, 22, loss_y, 14, 3, "Real MSE loss\n+ mask BCE", 'norm', 6)
    draw_box(ax, 38, loss_y, 14, 3, "Pseudo loss\n+ sample_weight", 'norm', 6)
    
    draw_box(ax, 54, loss_y, 16, 3, "Total = real_loss\n+ pseudo_w * pseudo_loss", 'highlight', 7, bold=True)
    draw_box(ax, 72, loss_y, 14, 3, "+ gamma L2 reg\n(optional)", 'optional', 6)
    
    # ==================== 主要流程箭头 ====================
    # 简化版本 - 主要连接
    draw_arrow(ax, 85, 167, 35, 160)  # 输入 -> 阶段一
    draw_arrow(ax, 51, 152, 35, 145)  # 阶段一 -> 阶段二
    draw_arrow(ax, 67, 141, 35, 135)  # 阶段二 -> 阶段三
    draw_arrow(ax, 35, 124, 40, 118)  # 阶段三 -> 阶段四
    draw_arrow(ax, 60, 108, 35, 102)  # 阶段四 -> 阶段五
    draw_arrow(ax, 35, 92, 35, 87)     # 阶段五 -> 阶段六
    draw_arrow(ax, 35, 81, 35, 77)     # 阶段六 -> 阶段七
    draw_arrow(ax, 55, 69, 40, 62)     # 阶段七 -> 阶段八
    draw_arrow(ax, 40, 50, 35, 42)     # 阶段八 -> 阶段九
    draw_arrow(ax, 35, 32, 35, 28)     # 阶段九 -> 阶段十
    draw_arrow(ax, 35, 15, 35, 11)     # 阶段十 -> 阶段十一
    draw_arrow(ax, 35, 0, 35, -4)      # 阶段十一 -> 阶段十二
    draw_arrow(ax, 35, -12, 85, -9)    # 阶段十二 -> 输出
    
    # ==================== 图例 ====================
    legend_y = -22
    ax.text(5, legend_y, "LEGEND:", fontsize=9, fontweight='bold')
    
    legend_items = [
        ('norm', 'Standard Module'),
        ('learnable', 'LEARNABLE (nn.Parameter)'),
        ('serious', '[S] Hyperparameter - Grid Search Needed'),
        ('warn', '[W] Numerical Risk'),
        ('crit', '[C] Methodological Issue'),
        ('decision', 'Decision Branch'),
        ('data', 'Data Flow'),
        ('highlight', 'KEY Output/Formula'),
        ('optional', '-- Optional / Disabled'),
    ]
    
    for i, (style, label) in enumerate(legend_items):
        col = i % 3
        row = i // 3
        lx = 5 + col * 38
        ly = legend_y - 3 - row * 4
        
        fc, ec = colors.get(style, colors['norm'])
        rect = FancyBboxPatch((lx, ly), 5, 2.5, 
                              boxstyle="round,pad=0.01,rounding_size=0.1",
                              facecolor=fc, edgecolor=ec,
                              linewidth=1, zorder=3)
        ax.add_patch(rect)
        ax.text(lx + 6, ly + 1.25, label, fontsize=7, va='center')
    
    # 保存
    plt.tight_layout()
    plt.savefig('/home/luolie/ToPoGate/papers/figures/v9_flowchart_detailed.png', 
                dpi=200, bbox_inches='tight', facecolor='#fafafa', edgecolor='none')
    plt.savefig('/home/luolie/ToPoGate/papers/figures/v9_flowchart_detailed.pdf', 
                bbox_inches='tight', facecolor='#fafafa', edgecolor='none')
    
    print("V9 详细流程图已保存:")
    print("  PNG: /home/luolie/ToPoGate/papers/figures/v9_flowchart_detailed.png")
    print("  PDF: /home/luolie/ToPoGate/papers/figures/v9_flowchart_detailed.pdf")
    plt.close()

if __name__ == '__main__':
    create_flowchart()
