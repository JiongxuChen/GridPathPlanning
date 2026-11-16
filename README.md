# GridPathPlanning：生成、验证与反馈修复

## 1. 项目目标

在6×6单目标网格中，研究小型语言模型的路径生成、有效性验证、首次错误诊断与反馈修复。基线比较T5-base、T5-small、Flan-T5-base、Flan-T5-small；Part 2主线使用开发集选出的T5-base，其他模型的改进作为补充实验。

## 2. 三套实验的关系

| 实验 | 训练 | 验证 | 修复 | 已公开成绩 |
| --- | --- | --- | --- | --- |
| v1 | 4000生成记录 | 执行器 | 同一G提示修复／前缀续写 | 四模型开发G100/R100；未宣称已完成v1正式测试 |
| v2 | G3000＋V1000，独立G与朴素文本V | 模型V／执行器对照 | 两种反馈×提示／前缀四组合 | 四模型dev、单错、双错，各200实例 |
| improvement_verify_v1 | 在v2基础上复用其G，替换为结构化逐步V | 神经模型逐步预测，无执行器代答 | 与v2相同四组合 | T5-base主线及四模型补充，三个分区完整 |

**改进的直接对照是v2，而非v1。** 改进与v2使用相同数据、G权重、候选和评分；只改变V方案。v1与v2的数据预算分配和面板不同，不是单因素消融。其后的聚焦消融、自然碰撞、局部占据及归因研究另见[研究归档](analysis/README.md)，不改写上述原始对照身份。

## 3. 主要结果

T5-base单错测试：

| 指标 | v2 | 验证改进 |
| --- | ---: | ---: |
| 固定V80 Macro-F1 | 0.5827 | 0.9499 |
| 固定V80首错定位 | 12.50% | 80.00% |
| R60模型提示修复率 | 18.33% | 58.33% |
| G60模型前缀最终成功率 | 78.33% | 80.00% |

验证和受控修复明显改善，自然生成碰撞的诊断及反馈利用仍较弱。结果文件包含分母、失败类型、逐例轨迹及空后缀／神经后缀贡献；不能只凭单项分数评价整体规划能力。

后续在相同 V1000、T5-base、750 更新预算上，仅更换验证器输入为完整 3×3 局部占据表示；原结构化 V 的开发固定 V80 碰撞联合为 4/10，局部 V 为 10/10。局部 V 在已公开 ID/OOD 的固定 V80 无效首错／类型联合均为 40/40；这属于历史配对，不是新盲测。随后新增、先冻结后推理的 6×6 封存评价为完整标签 157/160、分类 160/160、无效首错／类型联合 97/100；三个错误均为晚碰撞早一步。该面板在既有研究结果可见后设计，详见[封存报告](analysis/sealed_verifier_v1/report.md)。

## 4. 文档与结果入口

- [phase1：环境、模型与上游版本准备](docs/phase1.md)、[phase2：执行器、BFS与评分](docs/phase2.md)。

- [v1数据](docs/phase3.md)、[v1执行器基线](docs/phase4.md)。
- [v2数据](docs/phase3_v2.md)、[v2模型验证基线](docs/phase4_v2.md)。
- [Part 2：T5-base主线](docs/improvement_verify_t5base_v1.md)、[四模型补充](docs/improvement_verify_all_v1.md)。
- [后续验证器研究归档](analysis/README.md)：聚焦消融、自然碰撞、局部占据、晚碰撞、表示归因及新增封存评价；这些并非原公开实验的增补盲测。
- `results/v1/`、`results/v2/`、`results/improvement_verify_v1/`：CSV、JSON、逐例预测与审查证据。
- `results/comparisons/`：v2与改进的配对结果；`results/training_summary.json`：轻量训练及成本摘要。

原始 `publication/` 发布副本不包含作者的本机路径、环境转储、归档目录、连接配置、权重、检查点、优化器、日志或模型推理缓存；后续公开的 `analysis/` 是独立研究归档，不改变原发布清单。保存的原始预测和指标保持原值；原发布副本的通用运行入口与原实验的源码指纹可能不同，变化见`manifests/public_release.json`，不将公开副本伪装为原训练运行。

## 5. 环境与权重

Python3.10、PyTorch2.5.1+CUDA12.1、Transformers4.40.1；实验参考GPU为RTX A6000。Conda安装位置由使用者决定，通过PATH中的`conda`或环境变量`GRIDPATH_CONDA`选择。

```bash
conda create -n gridpath python=3.10 pip -y
conda activate gridpath
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt safetensors==0.8.0 \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install --no-deps --no-build-isolation -e .
```

四个原始检查点的仓库、固定revision和文件SHA256在`manifests/models.json`、`flan_models.json`。在独立复现目录运行`bash scripts/run_phase4_v1.sh prepare all`下载并校验。使用下载设备自己的网络与HF镜像，不需要GitHub代理。微调权重未在公开副本发布、没有公共下载地址；需要重新训练，不能把预训练权重当作已微调权重。后续局部占据等研究的检查点目前仅在远端隔离副本，本地归档保存代码、冻结数据、逐例预测和来源审计，不等同于权重备份。

## 6. 查阅与离线复核

安装环境后可直接检查CSV／JSON，或只用已发布预测在CPU重算三个实验的指标：

```bash
python scripts/review_public_results.py
python -m pytest -q
```

公开测试集是面向公开副本的CPU单元测试；不包含依赖作者环境／旧检查点的维护测试。指标重算不加载权重、不调用模型、不改写发布结果；它验证保存预测的评分一致性，不等同于重新生成预测或验证旧权重身份。

## 7. 完整复现

为避免覆盖已发布结果，先创建独立运行目录：

```bash
python scripts/create_public_reproduction.py --output ../GridPathPlanning-run
cd ../GridPathPlanning-run
```

后续数据检查、原始资源、训练、开发、测试及对照顺序见[复现说明](docs/reproduce.md)。本仓库未随附训练状态或旧选模锁；新运行产生自己的数据源码清单、模型指纹、日志和成本。公开数据保留原文件及样本ID，训练数据与开发／测试不得混用。

## 8. 范围与局限

seed42，各适用子集最多三遍、20 GPU-hours预算；单seed、小面板、受控／自然错误分布差异均需披露。v2单错与双错共用G问题，不算两批独立生成证据。测试已经公开，后续调参与新增探索应使用独立开发设计，不把已有测试重新当作未查看的测试。

保留论文与上游来源标识，不转发课程说明、论文PDF或第三方代码副本。项目未指定开源许可证，本发布过程不自行添加授权声明。
