# 职业成长分析 MVP

本仓库是 Deepmanifold Take-home Challenge 的 MVP 版本，面向一个 AI 职业规划与岗位推荐产品，完整跑通本地数据科学流程：模拟数据生成、数据校验、漏斗分析、留存分析、A/B 实验分析、流失标签构造、流失预测建模、模型评估、可解释性分析和 Next Best Action 推荐。

MVP 版本不依赖云服务、外部 API、后端服务或前端页面。评审可以在本地通过命令行和 Notebook 复现实验流程。

## 项目范围

- 生成学生与早期职业人群的模拟用户数据和行为事件。
- 校验数据 schema、主键、时间戳和业务约束。
- 计算注册、激活、简历上传、岗位推荐、岗位收藏、成长任务和职业报告生成漏斗。
- 按 cohort 计算 D1、D7、D14 留存。
- 分析 onboarding A/B 实验，并检测 sample ratio mismatch。
- 仅使用注册后前 7 天行为构造训练特征，避免标签泄漏。
- 训练 Logistic Regression baseline 和 HistGradientBoostingClassifier。
- 通过 validation PR-AUC 选择模型，并仅在 test set 上做一次最终评估。
- 输出模型指标、特征解释、分群指标和规则型 Next Best Action。
- 使用 Jupyter Notebook 展示端到端流程。
- 使用 pytest 覆盖核心数据、分析、建模与推荐逻辑。

Enterprise 全栈系统不包含在本 MVP 导出仓库中。Enterprise 版本位于单独仓库，包含 FastAPI 后端、SQLite 数据层和 React 前端。

## 业务背景

模拟产品服务大学生和早期职业用户，帮助他们探索职业方向、上传简历、查看岗位推荐、完成成长任务并生成职业报告。增长分析系统关注用户从注册到激活、留存、流失风险识别和干预推荐的完整生命周期。

核心用户路径：

```text
signup
-> onboarding_complete
-> profile_complete
-> resume_upload
-> job_recommendation_view
-> job_save
-> growth_task_complete
-> career_report_generate
-> retained / churned
```

## 目录结构

```text
career-growth-analytics-mvp/
|-- artifacts/                  # 模型文件、指标和图表
|-- data/
|   |-- sample/                 # 1,000 用户样例数据
|   `-- processed/              # labels.csv 等派生结果
|-- docs/
|   |-- data_schema.md          # 数据表结构说明
|   |-- methodology.md          # 数据生成与标签方法
|   `-- model_card.md           # 流失模型卡
|-- notebooks/
|   |-- lifecycle_analysis.ipynb
|   `-- churn_modeling.ipynb
|-- scripts/                    # 数据生成、分析、建模脚本
|-- src/career_growth/          # 核心 Python 包
|-- tests/                      # pytest 测试
|-- PHASE2_MODELING_REPORT.md
|-- pyproject.toml
|-- README.md
`-- LICENSE
```

## 本地运行

```powershell
cd C:\Users\Administrator\Desktop\career-growth-analytics-mvp-export
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
$env:PYTHONPATH = "src"
.venv\Scripts\python.exe -m pytest tests -q
```

生成样例数据：

```powershell
$env:PYTHONPATH = "src"
.venv\Scripts\python.exe scripts\generate_data.py --count 1000 --seed 42
```

运行生命周期分析：

```powershell
$env:PYTHONPATH = "src"
.venv\Scripts\python.exe scripts\run_analysis.py
.venv\Scripts\python.exe scripts\compute_summary.py
```

训练流失预测模型：

```powershell
$env:PYTHONPATH = "src"
.venv\Scripts\python.exe scripts\train_churn_model.py --count 5000 --seed 42
```

执行 Notebook：

```powershell
$env:PYTHONPATH = "src"
.venv\Scripts\python.exe -m nbconvert --execute --to notebook --inplace notebooks\lifecycle_analysis.ipynb
.venv\Scripts\python.exe -m nbconvert --execute --to notebook --inplace notebooks\churn_modeling.ipynb
```

## 当前模型结果

在 5,000 用户模拟数据上，最终选择 Logistic Regression：

| 指标 | Test Set |
| --- | ---: |
| PR-AUC | 0.5371 |
| ROC-AUC | 0.6942 |
| Brier Score | 0.2227 |
| F1 | 0.5884 |
| Threshold | 0.41 |

## 设计原则

- 本地可复现优先，所有数据由脚本生成。
- 标签构造与模型特征严格按时间切分，避免使用未来信息。
- MVP 聚焦算法、数据流和 Notebook 展示，不引入不必要的服务组件。
- 代码结构保持模块化，便于 Enterprise 版本复用。
