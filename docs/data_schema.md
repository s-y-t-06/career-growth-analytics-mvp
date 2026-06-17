# 数据 Schema

## users.csv

用户基础表，每行代表一个注册用户。

| 字段 | 含义 |
| --- | --- |
| user_id | 用户唯一标识 |
| signup_timestamp | 注册时间 |
| acquisition_channel | 获客渠道 |
| country | 国家或地区 |
| device_type | 设备类型 |
| user_intent_level | 求职/成长意向强度 |
| career_stage | 职业阶段 |
| language | 语言 |
| timezone | 时区 |
| marketing_consent | 是否允许营销触达 |

## events.csv

用户行为事件表，用于漏斗、留存和模型特征构造。

| 字段 | 含义 |
| --- | --- |
| event_id | 事件唯一标识 |
| user_id | 用户 ID |
| event_timestamp | 事件发生时间 |
| event_name | 事件名称 |
| event_source | 事件来源 |
| session_id | 会话 ID |

核心事件包括 `signup`、`onboarding_complete`、`profile_complete`、`resume_upload`、`job_recommendation_view`、`job_save`、`growth_task_complete` 和 `career_report_generate`。

## experiment_assignments.csv

实验分组表。

| 字段 | 含义 |
| --- | --- |
| user_id | 用户 ID |
| experiment_id | 实验 ID |
| variant_id | 实验版本 |
| assigned_at | 分组时间 |

## interventions.csv

干预记录表，用于模拟用户触达和增长动作。

| 字段 | 含义 |
| --- | --- |
| intervention_id | 干预记录 ID |
| user_id | 用户 ID |
| channel | 干预渠道 |
| template_id | 干预模板 |
| sent_at | 发送时间 |

## labels.csv

模型标签表。

| 字段 | 含义 |
| --- | --- |
| user_id | 用户 ID |
| churned | 是否流失 |
| label_window_start | 标签观察窗口开始 |
| label_window_end | 标签观察窗口结束 |
