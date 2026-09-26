# 交付与验证记录

日期：2026-09-23。交付状态：**四步研究计划、执行代码、小规模验证及正式四案例机制网格均完成，逐文件结果审计通过。** 本状态不代表原扩充实验或论文机制因果结论已完成。

## 已检查的真实材料

- 远端60个 checkpoint 状态：42个 `ready`，18个 `no_op_control`；无缺失或审计失败。详见 `inventories/checkpoints.json`。
- 有效状态构成：full 12、base_scmae 12、mixed_observation_reconstruction 12、no_edge_modulation 3（micro-mass）、fixed_mix_original_aux 3（Mouse_retina）。每个案例均含model seeds 42/123/7。
- 保留已有原始数据、权重、预处理、centroids和测试行；新增训练次数为0。
- 原始discovery的四案例结构/ARI预览已从真实汇总表生成，不把它当作新协议的机制结果。

## 代码验证

本地及SSH环境中9项数值性质测试通过。覆盖Frobenius归一化及尺度不变性、退化分母、分布不变但样本对应改变、重复点严格self排除、决策边界翻转、模型seed等权、验证与正式扫描隔离，以及float32实际方向有效性。完整远端输出见 `validation/numerical_tests_remote.txt`。

当前有效smoke版本为 `validation/smoke_v3`：

- 数据集dermatology、model seed42、CPU；full与mixed_observation_reconstruction。
- observe：Replacement 0.2、noise seed3001，共2个结果。
- directions：native/random/shuffled、倍率1、direction seed9102027，共6个结果。
- 共8个结果完成，严格配对4组，无结果拒收或配对拒收。
- clean embedding最大绝对误差为1.1920928955078125e-6；预测与原checkpoint逐行完全一致。
- 方向共同有效测试样本72/72，最大实际范数匹配误差约1.029e-7；完整量、容差和hash保存在逐条件result.json。
- 保存14份V0_RG源码文件hash及三个运行测量脚本的hash。汇总和绘图更新不触发重复推理；运行代码改变则拒绝覆盖旧签名结果。

smoke仅验证接口、数值和图表，不能用于判断某组件是否有贡献。旧 `smoke_v2` 保留为历史验证，不进入正式扫描，也不作为最新交付依据。`config_smoke.json` 指向v3；正式 `config.json` 的结果根目录与验证子目录分开。

## 可视化

- `figures/discovery/discovery_preview`：真实已有结果，四案例，清楚注明旧Euclidean/all-row诊断与test ARI的范围差异。
- `validation/smoke_v3/figures/`：输入结构、表示/ARI联动、有效消融、等幅方向、同一clean PCA布局的五种图。
- 每幅均导出PDF、SVG、PNG及source CSV、图注；smoke图加可见验证标识。关键PNG已人工查看，未以缺失数据构造曲线或统计显著性。

## 审查与尚待研究部分

Astra medium交叉审查已完成，修正了协议混图、汇总来源校验、实际方向共同有效集合等问题，详情见 `AUTO_REVIEW.md`。Claude MCP审查尝试失败：当前服务没有可用的general-purpose agent；原始工具回复保存于 `reviews/claude_mcp_attempt.json`。不能把Astra审查记为Claude通过。

正式队列按README顺序完成四案例full observe、inventory判为有效的消融、方向性探针与最终汇总。Step4用于区分拓扑方向响应与一般平滑解释；四案例均保留，没有根据前几步结果事后挑选。完成数字和证据边界见 `FINAL_REPORT.md` 与 `results/FILE_AUDIT.json`。
