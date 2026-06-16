"""Integrate churn scoring with the Next Best Action engine."""

import pandas as pd
from sklearn.pipeline import Pipeline

from career_growth.decisions.next_best_action import recommend_next_action
from career_growth.features.model_features import build_model_features


def score_users(
    model: Pipeline,
    users: pd.DataFrame,
    events: pd.DataFrame,
    feature_columns: list[str],
    threshold: float,
    experiment_assignments: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Score users with the trained model and recommend the next best action.

    Features are rebuilt from ``users`` and ``events`` so that callers only need
    the raw tables. The recommendation uses the predicted probability and the
    validation-selected threshold. The true churn label is never passed to the
    Next Best Action engine.
    """
    users = users.copy()
    users["signup_timestamp"] = pd.to_datetime(users["signup_timestamp"])

    events = events.copy()
    events["event_timestamp"] = pd.to_datetime(events["event_timestamp"])

    features = build_model_features(users, events, experiment_assignments)
    feature_matrix = features[feature_columns].reset_index(drop=True)
    probabilities = model.predict_proba(feature_matrix)[:, 1]

    records = []
    for idx, user in users.reset_index(drop=True).iterrows():
        prob = float(probabilities[idx])
        rec = recommend_next_action(
            user,
            events,
            cutoff=None,
            churn_risk_score=prob,
        )
        records.append(
            {
                "user_id": user["user_id"],
                "predicted_risk": prob,
                "predicted_class": int(prob >= threshold),
                "action_name": rec["action_name"],
                "channel": rec["channel"],
                "reason": rec["reason"],
            }
        )

    return pd.DataFrame(records)


def generate_nba_examples(
    model: Pipeline,
    test_df: pd.DataFrame,
    users: pd.DataFrame,
    events: pd.DataFrame,
    feature_columns: list[str],
    threshold: float,
    n_examples: int = 10,
    experiment_assignments: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generate example scoring and Next Best Action recommendations.

    The examples are sampled across the risk spectrum to show both high-risk and
    low-risk recommendations.
    """
    test_users = users[users["user_id"].isin(test_df["user_id"])].copy()
    scored = score_users(
        model, test_users, events, feature_columns, threshold, experiment_assignments
    )

    high_risk = scored[scored["predicted_class"] == 1]
    low_risk = scored[scored["predicted_class"] == 0]

    examples: list[pd.DataFrame] = []
    if len(high_risk) > 0:
        examples.append(high_risk.head(n_examples // 2))
    if len(low_risk) > 0:
        examples.append(low_risk.head(n_examples // 2))

    if examples:
        return pd.concat(examples, ignore_index=True).head(n_examples)
    return scored.head(n_examples)
