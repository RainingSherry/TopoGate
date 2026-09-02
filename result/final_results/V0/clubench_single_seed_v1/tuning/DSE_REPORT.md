# CLUBench V0 无标签调参报告

- Seed: 42
- 预筛 epoch: 12（最终评估固定为 80，不把本次 screening 当性能结论）
- 校准数据集: iris.npz, wine.npz, letter_recognition.npz, mnist64.npz, cnae9.npz
- 标签/K/KMeans/ACC/NMI/ARI：均未读取或执行。
- 目标：按数据集宏平均的 held-out masked loss（低）、view cosine（高）与 input-neighbour overlap（高）排名；非有限或塌缩候选淘汰。

## 锁定配置
### fixed
- Winner: `F_anchor`
- Mean rank: 1.466667
- Resolved config:
```json
{
  "protocol_id": "topogate_v0_clubench_label_free_dse_v1",
  "parameterization": "fixed",
  "hidden_size": 128,
  "dropout": 0.0,
  "masked_data_weight": 0.75,
  "mask_loss_weight": 0.7,
  "mask_ratio": 0.4,
  "epochs": 12,
  "batch_size": 256,
  "lr": 0.001,
  "num_workers": 0,
  "drop_last": false,
  "use_pseudo": true,
  "pseudo_weight": 0.3,
  "alpha": 0.9,
  "neighbor_k": 5,
  "mix_neighbors": 4,
  "neighbor_estimator": "current",
  "knn_pca_dim": 50,
  "tau": 0.2,
  "edge_reliability_mode": "sim_mutual_snn_distance",
  "gamma_sim": 1.0,
  "gamma_mutual": 1.0,
  "gamma_snn": 1.0,
  "gamma_distance": 1.0,
  "gate_min": 0.0,
  "gate_max": 0.15,
  "beta_mutual": 1.0,
  "beta_snn": 1.0,
  "beta_perturb": 2.0,
  "beta_uncertainty": 1.0,
  "kmeans_n_init": 20,
  "n_top_features": 1000,
  "target_sum": 10000.0,
  "input_mode": "auto",
  "scale_input": true,
  "evaluate_unsupervised": false,
  "rng_protocol": "isolated_v0"
}
```

### topology
- Winner: `T_anchor`
- Mean rank: 1.600000
- Resolved config:
```json
{
  "protocol_id": "topogate_v0_clubench_label_free_dse_v1",
  "parameterization": "topology",
  "hidden_size": 128,
  "dropout": 0.0,
  "masked_data_weight": 0.75,
  "mask_loss_weight": 0.7,
  "mask_ratio": 0.4,
  "epochs": 12,
  "batch_size": 256,
  "lr": 0.001,
  "num_workers": 0,
  "drop_last": false,
  "use_pseudo": true,
  "pseudo_weight": 0.3,
  "alpha": 0.9,
  "neighbor_k": 10,
  "mix_neighbors": 4,
  "neighbor_estimator": "current",
  "knn_pca_dim": 50,
  "tau": 0.2,
  "edge_reliability_mode": "sim_mutual_snn_distance",
  "gamma_sim": 1.0,
  "gamma_mutual": 1.0,
  "gamma_snn": 1.0,
  "gamma_distance": 1.0,
  "gate_min": 0.0,
  "gate_max": 0.15,
  "beta_mutual": 1.0,
  "beta_snn": 1.0,
  "beta_perturb": 2.0,
  "beta_uncertainty": 1.0,
  "kmeans_n_init": 20,
  "n_top_features": 1000,
  "target_sum": 10000.0,
  "input_mode": "auto",
  "scale_input": true,
  "evaluate_unsupervised": false,
  "rng_protocol": "isolated_v0"
}
```

## 运行产物

逐点原始指标位于 `outputs/*/metrics.json`；程序化表位于 `dse_log.csv`；最终 runner 只能读取 `selection_manifest.json` 中的锁定配置。
