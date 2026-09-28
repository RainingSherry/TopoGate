# E5：同一来源方向下模型如何响应

九个case×training-seed均完成。复用18个历史H1/multihop权重；相同anchor、相同方向、λ=.25/.5，最多512个anchor。比较远跳来源方向与逐anchor等范数随机方向；latent displacement按各模型全体embedding的RMS半径归一化。这里的远跳为hop≥2，不单指H5–H8。

| 案例（λ=.5） | H1远源latent响应 | multi远源latent响应 | H1随机响应 | multi随机响应 |
|---|---:|---:|---:|---:|
| Baron | .39425 | .39536 | .12287 | .12722 |
| Quake Lung | .21947 | .21661 | .09601 | .09633 |
| student_grade | .73805 | .73801 | .50144 | .50142 |

两种模型对数据来源方向的响应均明显不同于等范数随机方向；但H1和multi的归一化latent响应接近。因此不能把这张图写成multihop独有的“更强响应能力”。它更适合作为辅助观察：新增来源并非简单的各向同性扰动，而多跳优势也不是所有方向的latent变化都变大或都变小。

原anchor重构MSE增量（λ=.5）从H1到multi：Baron .088943→.076442，Lung .030201→.029535，student .263854→.264781。Baron和Lung在三个seed、两个λ均减小；student在λ=.25小幅减小而λ=.5小幅增加。这支持在前两个案例中，对这些固定来源扰动的恢复约束更稳定；不把它推成真实噪声恢复或所有数据的统一优势。

程序细节：这是无mask、无gate缩放的评估有限差分，不是训练输入的回放。历史缺失的preprocessor sidecar在新输出目录重建，逐一匹配历史fingerprint并重现保存embedding。Baron历史路径质量汇总存在小浮点差异，独立审查允许仅path_mass_retained_after_topk的逐阶绝对差≤1e-4，其余图/配置/权重核查保持原标准；差值明细逐probe保存。方向标记为当前重建plan的Monte Carlo，不能称历史真实draw。历史源码文件未改动。

建议把来源质量和跨社区整合作为主文主证据，response图放补充或简短辅助段落，不用这组近似相同的latent幅度承担核心新颖性。

产物：outputs/E5_response/各case-seed的probe_rows.csv/probe_summary.json/record.json，response_per_seed.csv、response_summary.csv、response_probe.pdf/svg/png；代码response_worker.py、summarize_response.py。无新增encoder训练、无标签参与这些响应计算。
