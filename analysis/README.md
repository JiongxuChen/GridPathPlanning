# 验证器后续研究归档索引

本目录是根目录公开主实验之后的**研究归档**，不是
`publication/` 的原始发布内容；所有已公开 ID/OOD
面板均不得重新称作盲测。以下结果不改写 v1/v2/
`improvement_verify_v1` 的历史记录。

| 阶段 | 协议、实现、结果与结论 | 决策 |
| --- | --- | --- |
| [聚焦消融与失败分析](focused_ablation/report.md) | 去合法前缀损失的训练入口、开发配对审计与逐例失败分析；ID/OOD 为历史复核 | 前缀损失保留；消融仅开发评价 |
| [自然碰撞补齐](natural_collision_v1/report.md) | 冻结 G 池、V1000 等量替换清单、开发逐例预测与比较 | 未改善自然 G60，未采用；未跑新模型 ID/OOD |
| [局部占据 V](local_occupancy_v1/report.md) | 表示与训练、开发、冻结新布局、[鲁棒性／历史配对](local_occupancy_v1/robustness_and_historical_report.md)、[项目要求和边界](local_occupancy_v1/project_alignment_and_boundaries.md) | 原完整 3×3 V 为当前主候选；历史 ID/OOD 已公开 |
| [晚碰撞尝试](late_collision_v1/report.md) | 等量替换、聚焦损失、阈值校准及预留留出逐例结果均见 `results/late_collision_v1/` | 两个重训版未采用；0.7 门控只作独立探索变体 |
| [表示归因](representation_attribution_v1/report.md) | 预先冻结开发／留出、完整／仅中心／去中心训练对照；后验符号、冻结头、输入翻转探针见 `results/representation_attribution_v1/` | 仅中心在该分面打平完整，但未有同等广度的总体评价，不替换主 V |
| [原 V 新增封存评价](sealed_verifier_v1/report.md) | 事先写定协议、160 条新布局冻结挑战、逐例概率、CPU 复核和 GPU 账本见 `results/sealed_verifier_v1/` | 原完整 V 保留；晚碰撞 3/20 定位早一步 |
| [晚碰撞顺序实验](late_collision_v2/report.md) | 仅中心格优先配对、全新布局的 0.7 门控对照、平衡成对训练、边界损失和位置对齐；冻结样本与逐例结果见 `results/late_collision_v2/` | 平衡成对版淘汰；边界损失与对齐版为候选，原完整 V 暂保留 |
| [几何路线门控封存](sealed_gate_v1/report.md) | 原完整九格 V 加事先固定的 0.7 门控；200 个新布局、自然 G20、超长路径和双错；逐例概率、审计与账本见 `results/sealed_gate_v1/` | 当前几何预处理路线封存模型；191/200 完整标签、8 条源输入拒绝；不覆盖原始权重 |
| [原结构化 V 非局部输入探索](structured_pairs_v1/report.md) | 首错分层的 40 对同环境自然碰撞／合法路，原结构化偏移输入、V1000 和 750 更新不变；新布局 G60 与受控碰撞、未配对消融均归档 | 新候选：自然 G 碰撞联合 2/20→20/20，受控 1/20→18/20；尚不替换几何路线封存模型 |
| [固定行位正例消融](structured_pairs_v1/position_ablation_report.md) | 960 行逐行不变，仅在原位替换 40 条配对正例；750 更新，公开开发与已打开留出事后诊断 | 碰撞定位受控 13/20、自然 11/20；二分类均全对，显示配对内容与训练行位都值得进一步多 seed 检验 |
| [结构化 V 成对多 seed](structured_pairs_v1/multiseed_report.md) | seed 42/43/44 配对与原位未配对，新排重 80 布局，G60 冻结后七模型同候选 V700 | seed 44 反转，配对收益不跨 seed／分面稳定；不替换封存模型 |
| [可学习偏移集合 V](relational_set_v1/report.md) | 原偏移文本不变，增加原始数值 `(dr,dc)` 集合注意力头；seed 42/43/44、两批新排重 6×6 布局及通道消融 | 三 seed 在第二批受控碰撞 20/20、自然碰撞 21/21，合法路径全保留；后续同候选评价见下一行 |
| [集合 V 同候选边界评价](relational_boundary_v1/report.md) | 预冻结 120 个新 6×6 布局、G40 和八个受控分层；原结构化、九格门控及三 seed 集合 V 同候选推理，独立逐例审计 | 集合 V 各 111/120，九格门控 115/120；35 个碰撞均全对，差异是四条额外的长路径输入拒绝；集合 V 列为备选，不替换封存模型 |

各阶段的相关代码、数据来源、逐例预测和解释报告均有本地
归档；按阶段区分探索、开发、事后诊断和新增封存评价。
当前[本地检查点清单](relational_set_v1/archive_status.md)包括集合 V
三 seed、原完整局部 V、原结构化 V 与 v2 G，均有权重、分词器、
完成标记及训练状态，独立逐文件 SHA256 校验。其他历史权重
并未全量备份：晚碰撞重训版、表示归因消融／冻结头探针、
结构化配对／未配对模型、修复模型等仍需另行归档。
各阶段 CPU 审计能重算来源和保存预测的指标，但没有相应
权重时不能重新生成所有历史模型预测。研究用 `checkpoints/`
被 `.gitignore` 排除，不随公开仓库上传。

尚未补齐：原局部 V 专属的**事前**十步 pilot 记录
（已有事后资源审计不能冒充）；多 seed 方差、
更大规模自然生成和晚碰撞定位评价，
以及超长输入的表示／源上限改进。本轮只封存验证器，
不对修复器新增结论。
