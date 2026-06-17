# Phase 2 建模报告

## 目标

Phase 2 的目标是在 MVP 数据流程基础上增加流失预测能力，并保证模型训练过程可复现、可解释、无标签泄漏。

## 完成内容

- 构建 `model_features.py`，仅使用注册后前 7 天行为生成模型特征。
- 增加 `check_label_leakage` 校验，防止训练特征使用未来窗口信息。
- 实现按 `signup_timestamp` 排序的 60/20/20 train、validation、test 时间切分。
- 训练 Logistic Regression baseline 与 HistGradientBoostingClassifier。
- 使用 validation PR-AUC 选择模型，test set 只评估一次。
- 输出 PR-AUC、ROC-AUC、log loss、Brier score、F1、precision、recall 等指标。
- 输出 logistic 系数、permutation importance、分群指标和校准结果。
- 将模型风险接入 Next Best Action，推荐逻辑不使用真实 label。

## 数据隔离

训练脚本默认使用 `data/training/`，不会覆盖仓库中的 `data/sample/` 和 `data/processed/labels.csv`。这样可以同时保留正式样例数据和较大规模训练数据。

## 结果

| 指标 | Test Set |
| --- | ---: |
| PR-AUC | 0.5371 |
| ROC-AUC | 0.6942 |
| Brier Score | 0.2227 |
| F1 | 0.5884 |
| Threshold | 0.41 |

最终选择 Logistic Regression。该模型虽然简单，但在 validation PR-AUC 上更稳定，也更容易解释，适合作为 MVP 阶段的可复现 baseline。

## 运行方式

```powershell
$env:PYTHONPATH = "src"
.venv\Scripts\python.exe scripts\train_churn_model.py --count 5000 --seed 42
```

## 风险与后续优化

- 当前数据为模拟数据，指标用于证明流程完整性，不代表真实线上效果。
- 如果接入真实数据，需要重新审查标签定义、隐私合规和偏差风险。
- 后续可以加入更细粒度的特征监控、漂移检测和模型版本管理。
