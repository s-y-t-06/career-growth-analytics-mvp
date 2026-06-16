# Phase 2 Modeling Report: Churn Prediction

## 1. Objective

Build a reproducible churn-prediction MVP that scores users at the end of their first week, identifies the most impactful pre-cutoff behavioral signals, and feeds the risk score into the rule-based Next Best Action engine.

## 2. Label Definition

- Prediction cutoff: `signup_timestamp + 7 days`.
- Label window: days 8-21 after signup (inclusive).
- `is_churned = 1` if the user has zero `event_source == "user_action"` events in the label window.

## 3. Feature Engineering

Features are computed in `src/career_growth/features/model_features.py` and are strictly limited to pre-cutoff events:

- Static user attributes: `acquisition_channel`, `country`, `device_type`, `user_intent_level`, `career_stage`, `marketing_consent`, `language`, `timezone`.
- Temporal signals: `signup_hour`, `signup_day_of_week`.
- Experiment feature: `onboarding_variant`.
- Core funnel indicators: `onboarding_started`, `onboarding_complete`, `profile_complete`, `resume_upload`, `job_recommendation_view`, `job_save`, `growth_task_complete`, `career_report_generate`.
- Behavioral aggregates: `num_core_actions_completed`, `num_sessions`, `num_user_actions`, `num_days_active`, `num_email_sent`, `num_push_sent`, `num_in_app_sent`, `avg_events_per_session`, `max_events_in_session`, `total_user_actions_in_sessions`, `unique_event_type_count`, `first_day_event_count`, `last_2_days_event_count`, `ai_assistant_interaction_count`, `job_detail_view_count`, `return_visit_count`.
- Time-based signals: `hours_to_first_action`, `hours_since_last_action_at_cutoff`. For users with no pre-cutoff actions these are encoded as missing and imputed with the median in the pipeline.

Leakage guardrails:

- `check_label_leakage` rejects forbidden prefixes (`future_`, `post_`, `label_`) and columns such as `last_active_date`, `days_since_last_active`, `is_churned`.
- Hidden propensity variables are never persisted or used as features.

Boundary test: adding post-cutoff events to the event log does not change any model feature values.

## 4. Data Split

Chronological 60% / 20% / 20% split by `signup_timestamp`:

- No user overlap across splits.
- Training set precedes validation set; validation set precedes test set.

Sizes on the 5,000-user dataset:

- Train: 3,000 users
- Validation: 1,000 users
- Test: 1,000 users

## 5. Models

Two scikit-learn pipelines are trained:

1. **Logistic Regression baseline**:
   - OneHotEncoder (`handle_unknown="infrequent_if_exist"`, `min_frequency=0.01`).
   - SimpleImputer (median) + StandardScaler for numeric features.
   - `class_weight="balanced"`.

2. **HistGradientBoostingClassifier**:
   - OneHotEncoder for categorical features.
   - SimpleImputer (median) for numeric features.
   - Early stopping with `n_iter_no_change=10`.

## 6. Selection and Evaluation Protocol

- Selection metric: PR-AUC on the validation set.
- Operating threshold: chosen on the validation set using F1 score (configurable via `--threshold-criterion`).
- Final evaluation: computed exactly once on the held-out test set.
- Metrics reported: PR-AUC, ROC-AUC, log loss, Brier score, precision, recall, F1, accuracy, confusion matrix.
- Calibration assessed with a reliability diagram.

### 6.1 Candidate validation metrics

| Model | PR-AUC | ROC-AUC | Log loss | Brier score |
|---|---|---|---|---|
| logistic_regression | 0.5515 | 0.7079 | 0.6221 | 0.2167 |
| hist_gradient_boosting | 0.5203 | 0.6860 | 0.6135 | 0.2119 |

### 6.2 Final test metrics (selected: logistic_regression, threshold = 0.41)

| Metric | Value |
|---|---|
| PR-AUC | 0.5371 |
| ROC-AUC | 0.6942 |
| Log loss | 0.6362 |
| Brier score | 0.2227 |
| Precision | 0.4636 |
| Recall | 0.8049 |
| F1 score | 0.5884 |
| Accuracy | 0.5900 |

### 6.3 Confusion matrix (test set)

| | Predicted 0 | Predicted 1 |
|---|---|---|
| Actual 0 | 297 | 339 |
| Actual 1 | 71 | 293 |

## 7. Artifacts

The training script `scripts/train_churn_model.py` produces:

- `artifacts/churn_model.joblib`
- `artifacts/model_metadata.json`
- `artifacts/metrics.json`
- `artifacts/feature_schema.json`
- `artifacts/explainability.json`
- `artifacts/user_explanations.json`
- `artifacts/subgroup_metrics.csv` / `subgroup_metrics.json`
- `artifacts/nba_examples.csv` / `nba_examples.json`
- `artifacts/plots/pr_curve.png`
- `artifacts/plots/roc_curve.png`
- `artifacts/plots/calibration.png`
- `artifacts/plots/confusion_matrix.png`
- `artifacts/plots/risk_distribution.png`
- `artifacts/plots/feature_importance.png`

Training data is written to a dedicated directory (default `data/training/`) so that the committed 1,000-user sample data under `data/sample/` is not overwritten. The engineered feature matrix is saved to `data/training/processed/model_features.csv`.

## 8. Explainability

- Logistic-regression coefficients are reported when the linear model is selected.
- Permutation importance (drop in PR-AUC) is computed on the validation set for both candidate models.
- Global top 15 features by permutation importance are saved and plotted.
- User-level explanations list the strongest positive and negative factors for at least 3 test users.

### 8.1 Top 15 features by permutation importance

1. `num_core_actions_completed` (0.0215)
2. `job_save` (0.0200)
3. `profile_complete` (0.0144)
4. `growth_task_complete` (0.0114)
5. `max_events_in_session` (0.0109)
6. `onboarding_complete` (0.0083)
7. `onboarding_started` (0.0078)
8. `job_detail_view_count` (0.0075)
9. `user_intent_level` (0.0029)
10. `resume_upload` (0.0028)
11. `num_days_active` (0.0023)
12. `hours_since_last_action_at_cutoff` (0.0023)
13. `return_visit_count` (0.0016)
14. `job_recommendation_view` (0.0011)
15. `first_day_event_count` (0.0008)

### 8.2 User explanation summary

Three test users were explained. Their predicted risks, predicted classes, actual labels, and top positive/negative factors are stored in `artifacts/user_explanations.json`. These explanations are associations captured by the model, not causal effects.

## 9. Subgroup Evaluation

Test-set metrics by `acquisition_channel`, `career_stage`, and `device_type` are stored in `artifacts/subgroup_metrics.json`. All groups in this run exceeded the small-sample threshold (n >= 50).

Summary:

- `device_type == desktop`: F1 = 0.617, churn rate = 38.0%
- `device_type == mobile`: F1 = 0.576, churn rate = 34.8%
- `device_type == tablet`: F1 = 0.525, churn rate = 35.7%
- `career_stage == early_career`: F1 = 0.621, churn rate = 39.8%
- `career_stage == new_graduate`: F1 = 0.516, churn rate = 33.1%
- `career_stage == student`: F1 = 0.620, churn rate = 36.9%

## 10. Next Best Action Integration

The trained model probabilities are passed to `recommend_next_action` using the validation-selected threshold. The true churn label is never used to decide the action. Example recommendations for 10 test users are stored in `artifacts/nba_examples.json`.

## 11. Limitations and Next Steps

- Data is synthetic; causal effects are calibrated for pipeline demonstration only.
- Feature importances and user explanations are model associations, not causal claims.
- No API, database, or frontend is introduced in this phase.
- Future work may include hyper-parameter tuning, real-data retraining, subgroup fairness analysis, and a lightweight scoring service.
