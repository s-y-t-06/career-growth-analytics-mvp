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
| marketing_consent | 是否允许营销触达 |
| language | 语言 |
| timezone | 时区 |
| initial_plan_type | 初始计划类型 |

## events.csv

用户行为事件表，用于漏斗、留存和模型特征构造。

| 字段 | 含义 |
| --- | --- |
| event_id | 事件唯一标识 |
| user_id | 用户 ID |
| session_id | 会话 ID |
| event_name | 事件名称 |
| event_timestamp | 事件发生时间 |
| event_properties | 事件属性（JSON） |
| page_name | 页面名称 |
| platform | 平台 |
| event_source | 事件来源 |
| experiment_id | 关联实验 ID |
| variant_id | 关联实验版本 |

核心事件包括 `signup`、`onboarding_complete`、`profile_complete`、`resume_upload`、`job_recommendation_view`、`job_save`、`growth_task_complete` 和 `career_report_generate`。

## experiment_assignments.csv

实验分组表。

| 字段 | 含义 |
| --- | --- |
| experiment_id | 实验 ID |
| user_id | 用户 ID |
| variant_id | 实验版本 |
| assignment_time | 分组时间 |
| experiment_name | 实验名称 |
| experiment_type | 实验类型 |
| traffic_allocation | 流量分配比例 |
| is_exposed | 是否已曝光 |
| is_converted | 是否已转化 |

## interventions.csv

干预记录表，用于模拟用户触达和增长动作。

| 字段 | 含义 |
| --- | --- |
| message_id | 消息唯一标识 |
| user_id | 用户 ID |
| action_name | 干预动作名称 |
| channel | 干预渠道 |
| send_time | 发送时间 |
| open_time | 打开时间 |
| click_time | 点击时间 |
| conversion_time | 转化时间 |
| experiment_id | 关联实验 ID |

## labels.csv

模型标签表。

| 字段 | 含义 |
| --- | --- |
| user_id | 用户 ID |
| signup_timestamp | 注册时间 |
| prediction_cutoff | 预测截止时间（注册后第 7 天） |
| label_start | 标签观察窗口开始（第 8 天） |
| label_end | 标签观察窗口结束（第 21 天） |
| is_churned | 是否流失 |
