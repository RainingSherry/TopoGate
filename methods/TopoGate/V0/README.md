# TopoGate V0: unified scVICAR

V0 是把 plantnet 中历史的 `NeighborMix_scMAE`（scVICAR-F）与
`RG_NeighborMix_scMAE`（scVICAR-T）合并后的模型身份重构。两者共享同一个
scMAE 编码器、掩码预测器、解码器、训练循环和 KMeans readout；只有 vicinal
corruption 的参数化不同。旧实现没有被移动、删除或修改，仍可作为数值和实验
协议对照。

## 历史位置与 V0 映射

| 论文/历史名称 | plantnet 中的 canonical 源码 | V0 设置 |
| --- | --- | --- |
| scVICAR-F / NeighborMix | `/home/luolie/biopipeline/dimension-reduction/plantnet/experimental_retired_models/NeighborMix_scMAE/`（本仓库同步入口为 `methods/NeighborMix_scMAE/`） | `parameterization: fixed` 或 CLI `F` |
| scVICAR-T / RG-NeighborMix | `/home/luolie/biopipeline/dimension-reduction/plantnet/experimental_retired_models/RG_NeighborMix_scMAE/` | `parameterization: topology` 或 CLI `T` |

统一后的计算路径为：

```text
X -> PCA/cosine kNN graph -> vicinal corruption -> shared scMAE -> clean embedding
                                      |
                         fixed (F)    |    topology (T)
```

两种设置都把最终 embedding 从干净真实细胞 `X` 提取出来。pseudo view 只用于
训练 anchor-recovery 分支，重建目标仍是对应的真实 anchor。

### F: fixed parameterization

图的 `probs` 只用于抽取邻居，抽到的邻居表达按抽样概率重新归一化。每个样本
使用相同的

```text
x_prime = alpha * x + (1 - alpha) * neighbor_mean
```

默认值沿用旧 F：`alpha=0.9`、`neighbor_k=5`、`mix_neighbors=4`、
`pseudo_weight=0.3`。F 的 pseudo loss 不使用 per-sample gate weighting。

### T: topology parameterization

图仍然是 PCA + cosine kNN，但先用 similarity、mutual-kNN、SNN 和 distance
计算 analytic edge reliability，再得到行归一化的 `edge_weights`。节点 gate 为

```text
gate_i = gate_min + (gate_max - gate_min) * sigmoid(
    beta_mutual * mutual_ratio_i
  + beta_snn * snn_i
  - beta_perturb * perturb_i
  - beta_uncertainty * uncertainty_i
)
x_prime_i = (1 - gate_i) * x_i + gate_i * neighbor_mean_i
```

T 的 pseudo loss 按 `clip(gate_i / max(gate), 0, 1)` 加权。默认值沿用旧 T：
`neighbor_k=10`、`mix_neighbors=4`、`gate_max=0.15`、
`edge_reliability_mode=sim_mutual_snn_distance`、`beta_perturb=2.0`。
`uncertainty` 在 V0 中没有启用，保持旧 T 的无标签 analytic 路径。

## 运行

先从仓库根目录准备一个 NPZ（键名可为 `X`/`x`/`features`/`data`，可选
`y`/`labels`），然后运行：

```bash
python -m methods.TopoGate.V0.run \
  --data-path datasets/example.npz \
  --save-dir /tmp/topogate_v0_fixed_smoke \
  --config methods/TopoGate/V0/configs/topogate_v0_fixed.yaml \
  --parameterization F \
  --device cpu \
  --epochs 2 \
  --n-clusters 3
```

T 只需替换配置或别名：

```bash
python -m methods.TopoGate.V0.run \
  --data-path datasets/example.npz \
  --save-dir /tmp/topogate_v0_topology_smoke \
  --config methods/TopoGate/V0/configs/topogate_v0_topology.yaml \
  --parameterization T \
  --device cpu \
  --epochs 2 \
  --n-clusters 3
```

`--data_path`、`--save_dir`、`--n_clusters` 等下划线形式也保留，便于替换旧
NeighborMix 命令。输入为 h5ad 时，V0 使用 scMAE family 的 raw/count、HVG 和
scale 约定；如没有标签列，必须显式传入 `--n-clusters`。GPU 运行使用
`--gpu 1..6`，物理 GPU 0 和 7 会被拒绝。

## Python API

核心 API 没有 `y` 参数，因而 graph、gate、loss 和 optimizer 不可能从调用签名
接收真值标签：

```python
import numpy as np
from methods.TopoGate.V0 import V0Config, fit_predict

X = np.asarray(..., dtype=np.float32)
config = V0Config(parameterization="topology", epochs=2, batch_size=32)
predictions, embedding, diagnostics = fit_predict(
    X,
    n_clusters=3,       # only used by the final KMeans readout
    config=config,
    seed=42,
    device="cpu",
)
```

无标签训练或 representation-only 诊断可使用 `n_clusters=None`；这时返回的
`predictions` 为 `None`，不会运行 KMeans。

## 输出契约

runner 在 `--save-dir` 写出一组可审计文件：

- `resolved_config.json`：解析后的参数、effective F/T 设置、seed、输入 hash 和 K 来源；
- `embedding_final.npy`：干净真实细胞的 latent embedding；
- `predictions.npy`：KMeans 预测簇编号；有标签时另写 `predictions_mapped.npy`、
  `labels_true.npy` 和 `label_mapping.json`（编码簇编号到原始标签值）；
- `neighbor_indices.npy`、`neighbor_base_probs.npy`、`edge_reliability.npy`、`edge_weights.npy`、`node_gate.npy`：图和 gate 诊断；
- `training_history.json`、`metrics.json`、`summary.json`、`status.json`、`run_record.json`：训练、benchmark 和状态记录；
- `embedding_geometry.json`、`unsupervised_diagnostics.json`：不读取标签的工程诊断；
- `model.pt`：共享 backbone 权重及 resolved config。

`metrics.json` 只在 runner 外层根据 `labels_true.npy` 计算，不能反向影响训练。
短期 smoke 只能证明代码和产物契约可运行，不构成性能结论；正式结果仍应写入
仓库规定的 `result/` 并登记相应 manifest。

## 兼容边界

- `methods/NeighborMix_scMAE/` 与 plantnet retired T 目录保持冻结，不由 V0 导入
  其 runner，也不改变其历史输出。
- V0 的统一训练循环只保留 F/T 两种论文参数化；历史 T 的 random/far/mutual
  压力消融仍应在 retired runner 中复现，不会被误称为 V0 主模型。
- F/T 的数据预处理、K 协议和 benchmark 标签来源由外层 runner 记录；核心
  `fit_predict` 只接收 `X`，不会猜测 K 或生成标签。
