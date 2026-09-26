# ToPoGate V0_RG 四步抗噪机制实验：完成报告

完成时间：2026-09-23 04:55 UTC。原始服务器实验根目录不包含在公开仓库中；完整逐次归档的 SHA-256 见 `PUBLISH_MANIFEST.json`。本报告只讨论论文版本 `methods/TopoGate/V0_RG`。

## 覆盖与可复核性

四案例为Mouse_retina、dermatology、har和micro-mass；沿用已筛选的优势案例，不把结果推广到全部数据集。复用42个有效/参考checkpoint状态；18个结构性no-op只在inventory中记录，不计入机制对照。模型种子42/123/7；raw-input Gaussian 0.2/0.4、Replacement 0.1/0.2各使用noise seeds 3001–3005。方向探针另用full派生native/random/shuffled方向、倍率0.5/1/2，在model-input空间测试。没有新训练权重，也没有重调超参数。

最终文件审计：observe **840/840**、directions **378/378**；full–消融同污染观测配对**600/600**，同方向和共同有效样本配对**270/270**。汇总接收1218个结果、870个配对，拒收0。四个full布局、输入结构、表示/ARI、总消融、四张逐案例消融和方向图均由真实结果生成。证据见 `FILE_AUDIT.json`、`tables/summary_manifest.json` 和各图的source CSV；原始服务器归档保存每次 test embedding、指标、预测、hash 及日志；公开仓库只保留数值表、图和审计摘要。

统计上先在每个model seed内平均五个noise seeds，再对三个model seeds等权平均。图中误差条是**model-seed均值的样本SD**，不是置信区间。方向探针每个model seed只有一次预设方向采样，没有五次独立噪声重复。

## 四步链条：以Replacement 0.2为共同展示条件

下表是四案例各自的full checkpoint，所有数字均来自新PCA/row-normalized cosine诊断和固定test rows；与旧discovery的all-row Euclidean邻域图不能混称同一指标。`D_input`和`D_latent`为相对于各自clean尺度的RMS位移；`R_kNN`为clean/noisy邻居保留率；`Δpurity`为noisy−clean micro purity；flip为固定clean centroids后的预测变化率。

| Dataset | D_input | R_kNN | Δpurity | Clean→noisy ARI | D_latent | Flip |
|---|---:|---:|---:|---:|---:|---:|
| Mouse_retina | 0.636 | 0.215 | −0.001 | 0.970→0.969 | 0.331 | 0.0024 |
| dermatology | 0.634 | 0.567 | −0.105 | 0.948→0.906 | 0.391 | 0.0463 |
| har | 0.633 | 0.754 | −0.025 | 0.609→0.595 | 0.175 | 0.0313 |
| micro-mass | 0.660 | 0.904 | −0.018 | 0.694→0.696 | 0.359 | 0.0269 |

**Step 1 → Step 2。**输入变化确实到达冻结预处理后的特征空间。Mouse_retina在Replacement 0.2下只保留约21.5%的原邻居，ARI却几乎不变；但局部标签纯度也几乎不变，说明换掉的大部分邻居仍处于相近类别结构中，不能把这一结果描述为“类别拓扑被严重破坏仍全部恢复”。dermatology的邻居和类别纯度都明显下降，ARI也下降约0.042；它更适合展示性能保留的程度，而非无退化。micro-mass的输入位移虽大，约90.4%的原邻居仍保留，说明其原图结构没有同样程度受损。

上述数据在固定clean带宽的描述性MMD²、相关性变化及样本对应位移上也有正的变化；例如Replacement 0.2的MMD²在四案例分别为0.00273、0.00394、0.00443、0.00319。这些数值不跨数据集直接比较，也没有成对样本下未经校准的p值。更完整的MMD参考量级、宏观/逐类纯度与所有噪声条件保存在summary.csv及逐条件result.json。

**Step 3。**定义抗退化对照量 `I_A=(ARI_noisy−ARI_clean)_full−(ARI_noisy−ARI_clean)_A`，正数表示full比对照少退化。Replacement 0.2下：

| Dataset | 有效对照 | I_A，三model seeds均值 ± SD |
|---|---|---:|
| Mouse_retina | no neighbourhood branch；fixed mixing amplitude；mixed-target reconstruction | +0.0007±0.0019；−0.0027±0.0027；0.0000±0.0055 |
| dermatology | no neighbourhood branch；mixed-target reconstruction | +0.0008±0.0081；0.0000±0.0000 |
| har | no neighbourhood branch；mixed-target reconstruction | −0.0002±0.0178；−0.0001±0.0001 |
| micro-mass | no neighbourhood branch；mixed-target reconstruction；no edge modulation | +0.0099±0.0162；+0.0102±0.0214；+0.0009±0.0208 |

Mouse_retina的full模型稳定性明确，但把自适应幅度固定后的平均抗退化量并未变差；当前证据**不能**把这种稳定性归因于自适应幅度。micro-mass的部分对照均值为正，但各model seed符号不一致，且额外边调制的差异很小；不能据此主张边权模块带来稳定的抗噪增益。恢复目标对照在三个案例上基本为零，micro-mass的正均值也受seed差异影响。此处是配置对照，不构成两个组件协同或训练机制因果链的证明。

**Step 4。**在model-input空间用实际float32位移核验三族方向等幅，并在共同有效行上比较。倍率1时，full模型的native/random/shuffled normalized latent drift分别为：Mouse_retina **0.0158/0.0093/0.0111**，dermatology **0.0109/0.0090/0.0097**，har **0.0090/0.0038/0.0085**，micro-mass **0.0072/0.0084/0.0199**。只有micro-mass在此倍率呈现native方向位移较小；其余三个案例的native方向并未比random更稳定。方向图的ARI、flip、三倍率与有效对照全部保存在结果中。由于方向由full模型的clean图生成且每seed只有一次采样，这些值不能推出普遍的“拓扑方向选择性”。

## 当前能支持的论文表述

最扎实的现象是：**在Mouse_retina，明显的输入和邻居身份变化与近乎不变的聚类划分并存。** 本实验进一步限定了这一现象：类别邻域纯度仍被保留，现有自适应幅度消融与方向探针没有证明独有的拓扑选择性抗扰。四案例体现不同的响应机制和输入结构损伤程度，不应被写成统一、已验证的组件因果链。

本研究属于原样本经过无标签transductive fitting后的冻结评估；固定test rows是评价子集，不是未见样本泛化。当前四步只比较V0_RG及其既有有效消融；“扰动后仍为Top1”的竞争结论应查原扩充实验中的同协议多模型结果，不能由这份机制面板单独推出。论文若叙述“恢复到扰动前的真实无噪数据”也不准确：训练目标是原始预处理观测值，未观测到真实无噪真值。

## 文件索引

- `tables/summary.csv`：分model seed等权后的全部指标；`seed_summary.csv`保留逐model seed值与noise seed变异。
- `tables/paired_summary.csv`、`paired_seed_summary.csv`：有效消融的配对效应和seed符号；`pairing_rejections.csv`为空。
- `figures/input_structure.pdf`：输入损伤与邻域结构；`mechanism_observation.pdf`：表示与ARI。
- `figures/effective_ablations_<Dataset>.pdf`：每案例消融；`matched_directions.pdf`：等幅方向与有效对照。
- `FILE_AUDIT.json`：1218个结果的逐文件覆盖、来源和870个配对校验。
