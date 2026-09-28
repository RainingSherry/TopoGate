# 实验代码与结果交叉审核

## Round 1

用户指定 Claude Opus 5.5 经 MCP 编写代码；主编辑应用并审核；Sol Max 独立审计；审核后 GPU2/4 运行；Luna Max 轮询；完成后 Sol Max + Claude 复核、主编辑裁定。

当前从 E3/E4 的确定性统计模块开始请求代码，以便先核实模型服务及代码质量。原始返回保存在 claude_author_round1/。远端执行器与配置复用由 execution_audit 核查。外部模型返回之前不登记其代码或审核为已完成。

冻结范围见 EXECUTION_SPEC.md。各步骤的思路、判读与产物均在运行前列明。

### MCP 作者调用结果

claude_code 请求模型 claude-opus-5-5，300465 ms 后超时；backends=[]、outputProduced=false、reviewProduced=false、costUsd=0。原始结果见 claude_author_round1/raw.json。没有收到代码，不能登记为已编写或审核通过。已询问用户是否允许 Codex 编写、Sol Max 独立审核的替代流程；等待决定期间继续完成资产复用核查。尚未启动新训练，尚未创建无任务可监督的 Luna 轮询。

## Round 2：用户要求保留 Claude 作者，继续重试

MCP claude_code 指定 claude-opus-5-5，343891 ms 后返回代码。接口 backends 同时报 deepseek-v4.1-flash 与 claude-opus-5-5；无法独立验证纯 Opus 归属。计费3.24407125美元。原文见 claude_author_round2/response.md、raw.json。补丁已应用；不是未经审查直接运行。

主编辑修正：输出末尾无效 diff 标记；按训练seed均值计算SD；百分点图轴乘100；缺失读出不得静默跳过；singleton splitting记为未定义；加入真正等类macro列；社区用计数向量内积代替逐样本二重循环；增加社区内部对角线、真实样本对支持数和逐seed表；明确按群体大小选择展示；保存PDF/SVG/PNG；失败退出非零。三个手算测试的原期望值有误，按直接枚举纠正。

12项测试通过（直接执行test_e3_e4_analysis.py，环境无pytest），包括30组小样本穷举对照、cluster编号不变性、singleton处理和端到端文件产物。三例实际后处理已启动；全59的E3仍需独立遍历，不以三例代替。

远端18个历史端点权重和预处理重现检查全部通过，见ASSET_PREFLIGHT.json。训练包装代码已交Claude下一次串行请求；未启动新增GPU训练。

## 最终结果审阅（2026-09-27）
第一次claude_review超时600秒，无文字，未计通过。重试返回PASS_WITH_SCOPE，原文及元数据保存claude_final_retry目录。请求claude-opus-5-5；实际backends同时列DeepSeek与Claude，不能认证纯Opus。累计ledger18.76616825/20美元。

父编辑处理：21.4495%分母补充明确；ductal负向对应已在正文报告保留；student补3seed均值±SD，不制造显著性声明；H4/H8饱和在有效深度主结果解释。审阅者将三案例Kwide结果概括到59队列、将path/Kwide误写成H1/Kwide，均不采纳。所谓显著差异只作数值差异。

Sol Max独立检查与父编辑裁定：E1–E6完成，代码、记录、数值及图表通过；支持限定案例的来源组织与跨社区整合叙事，可以开始写稿，不代表投稿接受。
