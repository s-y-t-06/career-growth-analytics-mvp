"""Churn label construction with strict temporal boundaries."""

import pandas as pd

from career_growth import config


def build_labels(users: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Build churn labels from events.

    The label is derived only from `event_source == "user_action"` events that
    fall between day 8 and day 21 after signup (inclusive).

    Only users whose full 21-day observation window is available are included.
    This is enforced by the data generator, which does not generate signups
    later than `END_DATE - 21 days`.
    """
    user_actions = events[events["event_source"] == "user_action"].copy()

    records = []
    for _, user in users.iterrows():
        signup = user["signup_timestamp"]
        cutoff = signup + pd.Timedelta(days=config.PREDICTION_CUTOFF_DAY)
        label_start = signup + pd.Timedelta(days=config.LABEL_WINDOW_START_DAY)
        label_end = signup + pd.Timedelta(days=config.LABEL_WINDOW_END_DAY)

        user_events = user_actions[user_actions["user_id"] == user["user_id"]]
        active_in_label = user_events[
            (user_events["event_timestamp"] >= label_start)
            & (user_events["event_timestamp"] <= label_end)
        ]
        is_churned = int(len(active_in_label) == 0)

        records.append(
            {
                "user_id": user["user_id"],
                "signup_timestamp": signup,
                "prediction_cutoff": cutoff,
                "label_start": label_start,
                "label_end": label_end,
                "is_churned": is_churned,
            }
        )

    return pd.DataFrame(records)


def check_label_leakage(features: pd.DataFrame, labels: pd.DataFrame) -> list[str]:
    """Return a list of leakage issues if feature columns contain future info."""
    forbidden_prefixes = ("future_", "post_", "label_")
    forbidden_columns = {
        "last_active_date",
        "days_since_last_active",
        "current_stage",
        "churn_risk_score",
        "conversion_propensity_score",
        "is_churned",
    }
    issues = []
    for col in features.columns:
        if col.startswith(forbidden_prefixes) or col in forbidden_columns:
            issues.append(f"Feature column contains potential leakage: {col}")
    return issues
