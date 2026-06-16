"""Model feature engineering using only pre-cutoff data."""

from datetime import timedelta

import numpy as np
import pandas as pd

from career_growth import config
from career_growth.features.labels import check_label_leakage


CATEGORICAL_FEATURES: list[str] = [
    "acquisition_channel",
    "country",
    "device_type",
    "user_intent_level",
    "career_stage",
    "onboarding_variant",
    "language",
    "timezone",
]

NUMERIC_FEATURES: list[str] = [
    "signup_hour",
    "signup_day_of_week",
    "marketing_consent",
    "onboarding_started",
    "onboarding_complete",
    "profile_complete",
    "resume_upload",
    "job_recommendation_view",
    "job_save",
    "growth_task_complete",
    "career_report_generate",
    "num_core_actions_completed",
    "num_sessions",
    "num_user_actions",
    "num_days_active",
    "num_email_sent",
    "num_push_sent",
    "num_in_app_sent",
    "avg_events_per_session",
    "max_events_in_session",
    "total_user_actions_in_sessions",
    "unique_event_type_count",
    "first_day_event_count",
    "last_2_days_event_count",
    "hours_to_first_action",
    "hours_since_last_action_at_cutoff",
    "ai_assistant_interaction_count",
    "job_detail_view_count",
    "return_visit_count",
]

ALL_FEATURE_COLUMNS: list[str] = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def _user_cutoff(user: pd.Series) -> pd.Timestamp:
    """Return the prediction cutoff for a user."""
    return user["signup_timestamp"] + timedelta(days=config.PREDICTION_CUTOFF_DAY)


def _build_static_features(users: pd.DataFrame) -> pd.DataFrame:
    """Build static user features."""
    records = []
    for _, user in users.iterrows():
        signup = pd.to_datetime(user["signup_timestamp"])
        records.append(
            {
                "user_id": user["user_id"],
                "signup_timestamp": signup,
                "acquisition_channel": user["acquisition_channel"],
                "country": user["country"],
                "device_type": user["device_type"],
                "user_intent_level": user["user_intent_level"],
                "career_stage": user["career_stage"],
                "language": user.get("language", "unknown"),
                "timezone": user.get("timezone", "unknown"),
                "signup_hour": signup.hour,
                "signup_day_of_week": signup.dayofweek,
                "marketing_consent": int(bool(user["marketing_consent"])),
            }
        )
    return pd.DataFrame(records)


def _build_variant_features(
    users: pd.DataFrame, experiment_assignments: pd.DataFrame
) -> pd.DataFrame:
    """Build experiment-variant features."""
    if experiment_assignments.empty:
        variants = pd.DataFrame(
            {"user_id": users["user_id"].values, "onboarding_variant": "unknown"}
        )
    else:
        variants = experiment_assignments[
            experiment_assignments["experiment_id"] == config.ONBOARDING_EXPERIMENT_ID
        ][["user_id", "variant_id"]].rename(
            columns={"variant_id": "onboarding_variant"}
        )
        missing = set(users["user_id"]) - set(variants["user_id"])
        if missing:
            variants = pd.concat(
                [
                    variants,
                    pd.DataFrame(
                        {
                            "user_id": list(missing),
                            "onboarding_variant": ["unknown"] * len(missing),
                        }
                    ),
                ],
                ignore_index=True,
            )
    return variants


def _hours_between(start: pd.Timestamp, end: pd.Timestamp) -> float:
    """Return the number of hours between two timestamps."""
    return (end - start).total_seconds() / 3600.0


def _build_behavior_features(
    users: pd.DataFrame, events: pd.DataFrame
) -> pd.DataFrame:
    """Build behavior features using only events before the prediction cutoff."""
    records = []
    for _, user in users.iterrows():
        user_id = user["user_id"]
        cutoff = _user_cutoff(user)
        signup = pd.to_datetime(user["signup_timestamp"])

        user_events = events[events["user_id"] == user_id].copy()
        pre_cutoff = user_events[user_events["event_timestamp"] < cutoff]
        user_actions = pre_cutoff[pre_cutoff["event_source"] == "user_action"]
        sessions = pre_cutoff[pre_cutoff["event_name"] == "session_start"]

        core_action_counts: dict[str, int] = {
            name: int((user_actions["event_name"] == name).any())
            for name in config.CORE_EVENTS
            if name != "signup"
        }

        num_sessions = int(len(sessions))
        num_user_actions = int(len(user_actions))
        num_days_active = int(user_actions["event_timestamp"].dt.date.nunique())

        sent_counts = pre_cutoff[
            pre_cutoff["event_source"] == "system"
        ]["event_name"].value_counts()
        num_email_sent = int(sent_counts.get("email_sent", 0))
        num_push_sent = int(sent_counts.get("push_sent", 0))
        num_in_app_sent = int(sent_counts.get("in_app_message_sent", 0))

        if num_sessions > 0:
            session_event_counts = pre_cutoff.groupby("session_id").size()
            avg_events_per_session = float(session_event_counts.mean())
            max_events_in_session = int(session_event_counts.max())
        else:
            avg_events_per_session = 0.0
            max_events_in_session = 0

        total_user_actions_in_sessions = int(
            pre_cutoff[pre_cutoff["event_source"] == "user_action"].groupby(
                "session_id"
            ).size().sum()
        )

        unique_event_type_count = int(pre_cutoff["event_name"].nunique())

        first_day_end = signup + timedelta(days=1)
        first_day_event_count = int(
            len(pre_cutoff[pre_cutoff["event_timestamp"] < first_day_end])
        )

        last_2_days_start = cutoff - timedelta(days=2)
        last_2_days_event_count = int(
            len(pre_cutoff[pre_cutoff["event_timestamp"] >= last_2_days_start])
        )

        if user_actions.empty:
            hours_to_first_action = np.nan
            hours_since_last_action_at_cutoff = np.nan
        else:
            first_action = user_actions["event_timestamp"].min()
            last_action = user_actions["event_timestamp"].max()
            hours_to_first_action = _hours_between(signup, first_action)
            hours_since_last_action_at_cutoff = _hours_between(last_action, cutoff)

        ai_assistant_interaction_count = int(
            (pre_cutoff["event_name"] == "ai_assistant_interaction").sum()
        )
        job_detail_view_count = int(
            (pre_cutoff["event_name"] == "job_detail_view").sum()
        )
        return_visit_count = int(
            (pre_cutoff["event_name"] == "return_visit").sum()
        )

        records.append(
            {
                "user_id": user_id,
                "onboarding_started": int(
                    (user_actions["event_name"] == "onboarding_start").any()
                ),
                "num_core_actions_completed": sum(core_action_counts.values()),
                "num_sessions": num_sessions,
                "num_user_actions": num_user_actions,
                "num_days_active": num_days_active,
                "num_email_sent": num_email_sent,
                "num_push_sent": num_push_sent,
                "num_in_app_sent": num_in_app_sent,
                "avg_events_per_session": avg_events_per_session,
                "max_events_in_session": max_events_in_session,
                "total_user_actions_in_sessions": total_user_actions_in_sessions,
                "unique_event_type_count": unique_event_type_count,
                "first_day_event_count": first_day_event_count,
                "last_2_days_event_count": last_2_days_event_count,
                "hours_to_first_action": hours_to_first_action,
                "hours_since_last_action_at_cutoff": hours_since_last_action_at_cutoff,
                "ai_assistant_interaction_count": ai_assistant_interaction_count,
                "job_detail_view_count": job_detail_view_count,
                "return_visit_count": return_visit_count,
                **core_action_counts,
            }
        )

    return pd.DataFrame(records)


def build_model_features(
    users: pd.DataFrame,
    events: pd.DataFrame,
    experiment_assignments: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a feature matrix from pre-cutoff events.

    Parameters
    ----------
    users: DataFrame containing user metadata with a ``signup_timestamp`` column.
    events: DataFrame containing event stream with ``user_id``, ``event_timestamp``,
        ``event_name``, ``event_source``, and ``session_id`` columns.
    experiment_assignments: Optional DataFrame with experiment assignments. If
        provided, the onboarding variant is joined to the feature matrix.

    Returns
    -------
    DataFrame with one row per user and only pre-cutoff features.
    """
    users = users.copy()
    users["signup_timestamp"] = pd.to_datetime(users["signup_timestamp"])

    events = events.copy()
    events["event_timestamp"] = pd.to_datetime(events["event_timestamp"])

    static_features = _build_static_features(users)
    behavior_features = _build_behavior_features(users, events)

    if experiment_assignments is None:
        experiment_assignments = pd.DataFrame(
            columns=["user_id", "experiment_id", "variant_id"]
        )
    variant_features = _build_variant_features(users, experiment_assignments)

    features = static_features.merge(behavior_features, on="user_id", how="left")
    features = features.merge(variant_features, on="user_id", how="left")
    features["onboarding_variant"] = features["onboarding_variant"].fillna("unknown")

    features = features[["user_id", "signup_timestamp"] + ALL_FEATURE_COLUMNS]
    features["signup_timestamp"] = pd.to_datetime(features["signup_timestamp"])

    numeric_cols = [c for c in NUMERIC_FEATURES if c in features.columns]
    # Preserve NaN for time-based features; imputation happens in the pipeline.
    for col in numeric_cols:
        if col not in {"hours_to_first_action", "hours_since_last_action_at_cutoff"}:
            features[col] = features[col].fillna(0).astype(float)
        else:
            features[col] = features[col].astype(float)

    return features


def attach_labels(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Attach churn labels to a feature matrix and validate leakage."""
    labels = labels[["user_id", "is_churned"]].copy()
    merged = features.merge(labels, on="user_id", how="inner")

    issues = check_label_leakage(merged.drop(columns=["is_churned"]), labels)
    if issues:
        raise ValueError("Feature leakage detected: " + "; ".join(issues))

    return merged


def prepare_model_matrix(
    users: pd.DataFrame,
    events: pd.DataFrame,
    labels: pd.DataFrame,
    experiment_assignments: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build features, attach labels, and return the complete modeling matrix."""
    features = build_model_features(users, events, experiment_assignments)
    return attach_labels(features, labels)


def save_model_features(
    features: pd.DataFrame, output_path: str = "data/training/processed/model_features.csv"
) -> None:
    """Save the engineered feature matrix to disk."""
    features.to_csv(output_path, index=False)
