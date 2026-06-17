# 流失预测模型卡

## 模型用途

模型用于预测 AI 职业平台用户在注册后早期阶段的流失风险，帮助增长团队识别需要干预的用户，并为 Next Best Action 提供风险输入。

## 训练数据

训练数据由本地脚本合成，模拟用户注册、onboarding、简历上传、岗位推荐、岗位收藏、成长任务和职业报告生成等行为。训练特征只使用注册后前 7 天内可观测信息。

## 标签定义

流失标签基于注册后第 8 天到第 21 天的活跃情况构造。如果用户在该观察窗口内没有有效活跃行为，则标记为 churn。标签窗口不进入模型特征。

## 特征范围

- 用户静态属性：渠道、国家、设备、职业阶段、意向强度等。
- 实验信息：onboarding variant。
- 早期漏斗行为：onboarding、profile、resume、recommendation、save、task、report。
- 行为聚合：会话数、活跃天数、事件数、事件类型数、首日行为数、最近两日行为数、首次行为耗时等。

## 模型选择

候选模型：

- Logistic Regression baseline
- HistGradientBoostingClassifier

最终按 validation PR-AUC 选择 Logistic Regression，并在 test set 上做一次最终评估。

## 最终指标

| 指标 | Test Set |
| --- | ---: |
| PR-AUC | 0.5371 |
| ROC-AUC | 0.6942 |
| Brier Score | 0.2227 |
| F1 | 0.5884 |
| Threshold | 0.41 |

## 模型阈值与 Next Best Action 阈值

- **模型分类阈值（0.41）**：由 validation F1 选择，用于平衡精确率和召回率，生成二分类的 churn 预测标签。
- **业务干预阈值（0.70）**：Next Best Action 中触发高风险挽回动作的独立阈值，用于避免对中等风险用户过度触达。该阈值不影响模型分类，也不使用真实 label。

两者分别服务于不同目标：模型阈值优化预测性能，业务阈值控制运营触达强度。

## 使用限制

- 当前模型基于模拟数据，只能用于本地项目展示和流程验证。
- 不能直接用于真实用户的高风险决策。
- Next Best Action 只能使用预测风险和可解释因素，不能使用真实 label。
- 如果迁移到真实业务，需要重新训练、校准、监控漂移，并评估公平性。
