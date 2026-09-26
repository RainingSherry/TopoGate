# 四步机制计划：交叉审查记录

范围仅为本目录的研究计划、分析代码与有限smoke，不是全论文投稿评审，不给出投稿分数。

## 资源和方法映射审查

核对了实际V0_RG配置、训练/推理路径与既有消融权重。修正的核心解释是：Mouse_retina的自适应幅度与micro-mass的额外边调制分别启用，HAR与dermatology没有这两个组件同时启用的证据。恢复目标是原始预处理观测值，不是假定可获得的无噪真实数据；训练图不参与冻结encoder推理。

## Astra medium代码审查：原始最终回复

只读审查发现3项：

1. `plot_results.py:181`：消融图混入observe和directions。最小修复：筛选 `stage=="observe"`，方向继续独立绘图。
2. `summarize.py:124`：单模型汇总先接收completed结果；配置、模型哈希、clean replay仅在配对阶段检查。局部续跑可能混合旧配置/代码结果。最小修复：每条结果入库前统一验证来源，代码版本不一致拒绝或分组。
3. `probe.py:197–215`：共同有效mask依据加法前方向；实际float32扰动变零或范数失配仍被纳入。最小修复：按每个multiplier验证三族实际扰动范数，共同剔除无效行或拒绝超差条件。尚无证据表明现有smoke触发。

RMS归一化、同dtype变化比例、GPU允许池、clean replay、MMD描述性解释和有效消融配对未发现其他具体错误。本次未修改文件、未新增实验。

## 已实施修复与验证

1. raw-input observe与model-input directions分别出图、汇总；机制消融图只接受observe。
2. 所有完成结果进入统计前核验inventory、协议、config/model hash和clean replay。同一批次代码来源混杂则整组拒收；方向配对额外要求共同有效mask相同。
3. 实际float32加法后进行三族等范数核验，重新取共同有效行；ARI、flip、drift提供共同子集版本及all-test版本，不静默混合分母。
4. 各模型seed先汇总noise seeds，再等权汇总；分开记录模型和噪声变异。MMD仅为描述性差异，提供同大小、共享clean reference的对照量级，不套独立样本p值。
5. 修复数值指标的RMS/Frobenius定义、无效分母和self邻居并列；9项数值测试与最终8个CPU smoke探针通过。
6. 保存source CSV、图注、PDF/SVG/PNG；缩短消融显示名并给smoke图加可见标识，完整文件ID仍留在数据表中。

## Claude MCP尝试

2026-09-23调用 `mcp__claude_code__Agent`，返回：`Agent type 'general-purpose' not found. Available agents:`，`isError=true`。这是工具不可用，不是科研否定意见，也不是审查通过。原始回复见 `reviews/claude_mcp_attempt.json`。

## 交付边界

计划和代码已进入正式案例研究；只有后续真实结果支持时，才能把相应机制解释写入论文。完整四案例机制网格已完成且通过逐文件审计（observe 840/840、directions 378/378、配对 870/870、问题 0）。结果支持案例特异的抗扰动现象，但没有确证各门控组件的抗噪因果贡献；本记录没有伪造Claude结论、显著性或模型的广泛鲁棒性。
