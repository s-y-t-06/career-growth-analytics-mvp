# Data Schema

This document defines the data schema for the AI Career Platform Lifecycle Growth and Experimentation System MVP.

All timestamps are stored in UTC. The observation period is anchored to each user's `signup_timestamp`.

## 1. users

User-level static attributes.

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | `string` | UUID primary key. |
| `signup_timestamp` | `datetime` | Exact registration time. |
| `acquisition_channel` | `string` | `organic_search`, `social_ads`, `campus_event`, `referral`, `content_marketing`. |
| `country` | `string` | User country, e.g. `US`, `CN`, `IN`, `GB`. |
| `device_type` | `string` | `desktop`, `mobile`, `tablet`. |
| `language` | `string` | Primary language code. |
| `timezone` | `string` | IANA timezone name. |
| `initial_plan_type` | `string` | Always `free` in the MVP. |
| `user_intent_level` | `string` | Latent intent bucket used only for data generation: `low`, `medium`, `high`. |
| `career_stage` | `string` | `student`, `new_graduate`, `early_career`. |
| `marketing_consent` | `boolean` | Whether the user consents to marketing communications. |

## 2. events

Event-level behavior log.

| Column | Type | Description |
|--------|------|-------------|
| `event_id` | `string` | UUID primary key. |
| `user_id` | `string` | Foreign key to `users`. |
| `session_id` | `string` | Session identifier. |
| `event_name` | `string` | Event type (see below). |
| `event_timestamp` | `datetime` | Exact event time. |
| `event_properties` | `JSON string` | Optional extra payload. |
| `page_name` | `string` | Page or screen where the event occurred. |
| `platform` | `string` | `web`, `ios`, `android`. |
| `event_source` | `string` | `user_action`, `system`, `campaign`. |
| `experiment_id` | `string / null` | Experiment the event belongs to, if any. |
| `variant_id` | `string / null` | Experiment variant, if any. |

### Core event names

- `signup`
- `onboarding_start`
- `onboarding_complete`
- `profile_complete`
- `resume_upload`
- `job_recommendation_view`
- `job_detail_view`
- `job_save`
- `ai_assistant_interaction`
- `growth_task_complete`
- `career_report_generate`
- `session_start`
- `session_end`
- `return_visit`

## 3. experiment_assignments

Stable per-user experiment assignment.

| Column | Type | Description |
|--------|------|-------------|
| `experiment_id` | `string` | Experiment identifier. |
| `user_id` | `string` | Foreign key to `users`. |
| `variant_id` | `string` | Assigned variant. |
| `assignment_time` | `datetime` | When the assignment was created. |
| `experiment_name` | `string` | Human-readable experiment name. |
| `experiment_type` | `string` | `onboarding`, `messaging`, `feature_education`. |
| `traffic_allocation` | `float` | Expected traffic fraction for this variant. |
| `is_exposed` | `boolean` | Whether the user was exposed to the treatment. |
| `is_converted` | `boolean` | Whether the user hit the experiment primary metric. |

The MVP contains one experiment:

- `exp_onboarding_v1`
  - `control`: standard five-step onboarding (40%)
  - `personalized`: adaptive onboarding (30%)
  - `simplified`: two-step onboarding (30%)

## 4. interventions

Optional record of growth interventions (messages, prompts) sent to users.

| Column | Type | Description |
|--------|------|-------------|
| `message_id` | `string` | UUID primary key. |
| `user_id` | `string` | Foreign key to `users`. |
| `action_name` | `string` | Recommended action identifier. |
| `channel` | `string` | `email`, `push`, `in_app`. |
| `send_time` | `datetime` | When the intervention was sent. |
| `open_time` | `datetime / null` | When the user opened it. |
| `click_time` | `datetime / null` | When the user clicked it. |
| `conversion_time` | `datetime / null` | When the user performed the target action. |
| `experiment_id` | `string / null` | Related experiment, if any. |

## 5. lifecycle_stages (derived)

Derived table, not part of the raw synthetic data.

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | `string` | Primary key. |
| `current_stage` | `string` | `new_user`, `onboarding`, `activated`, `engaged`, `at_risk`, `churned`, `reactivated`. |
| `previous_stage` | `string` | Previous stage. |
| `stage_start_time` | `datetime` | When the current stage started. |
| `activation_score` | `float` | Activation probability score. |
| `churn_risk_score` | `float / null` | Churn risk score (empty until model stage). |
| `last_active_date` | `date` | Date of last valid `user_action`. |
| `days_since_last_active` | `int` | Days since last valid `user_action`. |
| `next_best_action` | `string` | Recommended next action. |

## 6. Churn label definition

- **Prediction cutoff**: `signup_timestamp + 7 days`.
- **Feature window**: events from signup up to and including the cutoff.
- **Label window**: days 8 through 21 after signup (`signup_timestamp + 8 days` to `signup_timestamp + 21 days`).
- **Active event**: any event with `event_source == "user_action"`.
- **`is_churned = 1`**: no active event in the label window.
- **`is_churned = 0`**: at least one active event in the label window.
- Only users with at least 21 full days of observation are included in training samples.
