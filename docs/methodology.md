# Methodology

This document describes how synthetic data is generated and how churn labels are constructed.

## 1. Generative causal order

The data is generated in a causal order that mirrors real user behavior:

1. Generate user base attributes (`users`).
2. Draw hidden propensity variables for each user.
3. Assign each user to an onboarding experiment variant using a stable hash.
4. Generate the first-week event sequence based on attributes, hidden propensities, and treatment effects.
5. Generate the second/third-week event sequence based on early behavior, hidden propensities, and decay.
6. Derive `is_churned` and all analytics purely from the generated event log.

Hidden propensity variables exist only inside the data generator and are never written to disk or used as model features.

## 2. Hidden propensity variables

For each user we draw four latent variables from Beta distributions:

- `intrinsic_engagement` ~ Beta(2, 2) - general tendency to engage.
- `career_urgency` ~ Beta(2.5, 2) - how urgently the user needs career help.
- `product_fit` ~ Beta(2, 2) - how well the product matches the user's needs.
- `notification_sensitivity` ~ Beta(2, 3) - responsiveness to prompts.

These variables influence event probabilities but are not observable by the model.

## 3. Encoded user attributes

Categorical attributes are mapped to numeric scores for probability calculations:

| Attribute | low | medium | high |
|-----------|-----|--------|------|
| `user_intent_level` | 0.30 | 0.60 | 0.90 |

| Attribute | score |
|-----------|-------|
| `career_stage == student` | 0.70 |
| `career_stage == new_graduate` | 0.85 |
| `career_stage == early_career` | 0.75 |
| `device_type == desktop` | 1.00 |
| `device_type == mobile` | 0.85 |
| `device_type == tablet` | 0.75 |

## 4. Core event probability formulas

Let

- `ie = intrinsic_engagement`
- `cu = career_urgency`
- `pf = product_fit`
- `intent = encoded user_intent_level`
- `device = encoded device_type`
- `stage = encoded career_stage`
- `direct_effect` = treatment-specific effect applied to `onboarding_start` and `onboarding_complete`; `0.0` for `control`, `0.30` for `personalized`, `0.15` for `simplified`

The probability of each core event is computed with a clipped logistic function:

```
logit(p) = intercept
           + coefficient_ie * ie
           + coefficient_cu * cu
           + coefficient_pf * pf
           + coefficient_intent * intent
           + coefficient_device * device
           + coefficient_stage * stage
           + direct_effect
           + state_bonus

p = clip(sigmoid(logit(p)), 0.01, 0.99)
```

The specific coefficients are defined in `src/career_growth/data_generation/events.py`.

Event sequence dependencies:

- `onboarding_start` is required before `onboarding_complete`.
- `onboarding_complete` increases the probability of later events.
- `profile_complete` is required before `resume_upload`.
- `resume_upload` increases the probability of `job_recommendation_view`.
- `job_recommendation_view` is required before `job_save`.
- `job_save` increases the probability of `growth_task_complete`.
- `growth_task_complete` increases the probability of `career_report_generate`.

If an upstream event does not occur, the downstream event can still occur with a reduced probability, reflecting organic exploration.

## 5. First-week daily activity

For each day `d` in `[0, 7]`:

```
engagement_score = 0.30 * ie
                   + 0.20 * pf
                   + 0.15 * cu
                   + 0.10 * intent

p_active_d = clip(0.01 + 0.50 * engagement_score + 0.05 * onboarding_complete, 0.005, 0.60)
```

If the day is active, a `session_start` is generated, followed by a random subset of contextually appropriate user actions (for example, `job_detail_view`, `ai_assistant_interaction`, `return_visit`).

## 6. Second/third-week daily activity

For each day `d` in `[8, 21]`:

```
state_boost = 0.015 * num_core_actions_completed
p_active_d = clip(0.001 + 0.08 * engagement_score + state_boost, 0.001, 0.25)
```

`num_core_actions_completed` counts `onboarding_complete`, `profile_complete`, `resume_upload`, `job_recommendation_view`, `job_save`, `growth_task_complete`, `career_report_generate`.

A user who completed more core actions in the first week has a higher chance of remaining active in the label window.

## 7. Treatment effect injection (synthetic causal mechanism)

The onboarding experiment is a synthetic demonstration. It does not reflect a real business outcome; it is calibrated only to exercise the analytics pipeline.

The treatment is injected as follows:

- `direct_effect` is applied only to `onboarding_start` and `onboarding_complete`. Current values are `0.0` for `control`, `0.30` for `personalized`, and `0.15` for `simplified`.
- No direct treatment effect is added to `profile_complete`, `resume_upload`, `job_save`, `career_report_generate`, or to late-phase retention.
- Downstream lifts in profile completion, resume upload, job interactions, and retention emerge indirectly because completing onboarding increases the state bonuses that govern later events and because completers are more likely to remain active during the first week.

This design keeps the primary treatment localized to the onboarding step while allowing the funnel and retention metrics to move through user-state mediation.

## 8. Noise and anomalies

- All probability decisions include independent random noise per event.
- A small fraction of users (around 3%) has inverted behavior (high propensity but low activity) to simulate external life events.
- A small fraction of events (around 1%) has slightly out-of-order timestamps within a session to test event ordering validation.
- A small fraction of records contains missing optional fields (around 0.5%).

## 9. Churn label derivation

After all events are generated, we compute for each user:

```
label_start = signup_timestamp + 8 days
label_end = signup_timestamp + 21 days

active_events = events where
    user_id == user.user_id
    and event_source == "user_action"
    and label_start <= event_timestamp <= label_end

is_churned = 1 if len(active_events) == 0 else 0
```

Only users whose `signup_timestamp + 21 days` is on or before the global data generation end date are included in training samples. This guarantees a complete observation window.

## 10. Leakage protection

The following rules prevent data leakage:

- Model features are computed only from events before the prediction cutoff (`signup_timestamp + 7 days`).
- The label is computed only from events after the cutoff.
- Hidden propensity variables are never persisted or used as features.
- `last_active_date` and `days_since_last_active` derived from the full event log are not used as features.
- `lifecycle_stages` fields that contain future information are excluded from feature engineering.
- `event_source == "system"` and `event_source == "campaign"` events are excluded from activity-based labels.

## 11. Retention definitions

All retention metrics are based on valid `user_action` events.

- **D1 retention**: at least one active event on calendar day `signup_date + 1`.
- **D7 retention**: at least one active event on calendar day `signup_date + 7`.
- **D14 retention**: at least one active event on calendar day `signup_date + 14`.
- **Rolling retention**: at least one active event on or after the target date.

## 12. Funnel rules

The core funnel uses these steps in order:

1. `signup`
2. `onboarding_complete`
3. `profile_complete`
4. `resume_upload`
5. `job_recommendation_view`
6. `job_save`
7. `career_report_generate`

A user is counted at step `n` only if they have completed step `n-1` **before** the timestamp of step `n`. Users cannot skip steps and be counted at a later step.

## 13. A/B analysis

For binary metrics we use a two-proportion z-test:

```
z = (p_treatment - p_control) / sqrt(p_pool * (1 - p_pool) * (1/n_t + 1/n_c))
p_value = 2 * (1 - CDF(|z|))
```

Where `p_pool = (x_t + x_c) / (n_t + n_c)`.

Lift metrics:

- `absolute_lift = p_treatment - p_control`
- `relative_lift = (p_treatment - p_control) / p_control` when `p_control > 0`

Sample Ratio Mismatch (SRM) is checked with a chi-squared test against expected allocation.
