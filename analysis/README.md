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

各阶段的相关代码、数据来源、逐例预测和解释报告均有本地
归档；按阶段区分探索、开发、事后诊断和新增封存评价。
本次公开分支纳入主要冻结面板、逐例预测及历史 ID/OOD
汇总；`local_occupancy_v1/historical_pipeline/` 中的大量
逐任务记录和推理缓存仅留在本地／远端隔离归档，未上传。
缓存不是训练权重，历史汇总也不能替代重新推理。
微调检查点及优化器状态**未本地归档**：原完整局部 V、
晚碰撞两个重训版与表示归因的两个消融／冻结头探针
保存在 A6000 隔离副本，不能只用本地 JSON 重新生成
模型预测。`natural_collision_v1` 和前缀消融分别在其
报告标明的其他远端副本。各阶段的 CPU 审计能重算
来源和保存预测的指标，但不代替权重再推理。

尚未补齐：原局部 V 专属的**事前**十步 pilot 记录
（已有事后资源审计不能冒充）；多 seed 方差、
自然生成上的新封存评价、较大规模晚碰撞定位评价，
以及超长输入的表示／源上限改进。本轮只封存验证器，
不对修复器新增结论。
