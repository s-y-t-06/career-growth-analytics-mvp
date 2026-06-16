"""Retention and cohort analytics."""

import pandas as pd

from career_growth import config


def _normalize_date(series: pd.Series) -> pd.Series:
    """Return a tz-naive floor-to-day Series for consistent date comparisons."""
    return series.dt.tz_localize(None).dt.floor("D")


def compute_day_retention(
    users: pd.DataFrame,
    events: pd.DataFrame,
    day: int,
    group_by: str | None = None,
) -> pd.DataFrame:
    """Compute retention for a specific day after signup.

    A user is retained on day N if they have at least one user_action event
    whose calendar day equals signup_date + N.
    """
    active_events = events[events["event_source"] == "user_action"].copy()
    active_events["event_date"] = _normalize_date(active_events["event_timestamp"])

    users_copy = users.copy()
    users_copy["signup_date"] = _normalize_date(users_copy["signup_timestamp"])
    target_date = users_copy["signup_date"] + pd.Timedelta(days=day)

    # Vectorized approach: build a set of (user_id, event_date) pairs.
    user_date_pairs = set(zip(active_events["user_id"], active_events["event_date"]))
    users_copy["target_date"] = target_date
    users_copy["retained"] = users_copy.apply(
        lambda row: (row["user_id"], row["target_date"]) in user_date_pairs, axis=1
    )

    if group_by is None:
        overall = users_copy["retained"].mean()
        return pd.DataFrame({"day": [day], "retention_rate": [overall]})

    grouped = users_copy.groupby(group_by)["retained"].mean().reset_index()
    grouped = grouped.rename(columns={"retained": "retention_rate"})
    grouped["day"] = day
    return grouped[[group_by, "day", "retention_rate"]]


def compute_cohort_retention(
    users: pd.DataFrame,
    events: pd.DataFrame,
    days: list[int] = (1, 7, 14),
    cohort_col: str = "signup_week",
) -> pd.DataFrame:
    """Compute a cohort retention matrix.

    Cohorts are defined by `cohort_col`. Supported values:
    - signup_week
    - acquisition_channel
    - device_type
    - variant_id (requires merging experiment assignments)
    """
    active_events = events[events["event_source"] == "user_action"].copy()
    active_events["event_date"] = _normalize_date(active_events["event_timestamp"])

    users_copy = users.copy()
    users_copy["signup_date"] = _normalize_date(users_copy["signup_timestamp"])

    if cohort_col == "signup_week":
        users_copy["signup_week"] = users_copy["signup_date"].dt.to_period("W").astype(str)
    elif cohort_col == "variant_id":
        raise ValueError("Merge experiment assignments before calling with variant_id")

    user_date_pairs = set(zip(active_events["user_id"], active_events["event_date"]))

    records = []
    groups = sorted(users_copy[cohort_col].dropna().unique())
    for group_value in groups:
        subset = users_copy[users_copy[cohort_col] == group_value]
        total = len(subset)
        for day in days:
            target_date = subset["signup_date"] + pd.Timedelta(days=day)
            retained = sum(
                (uid, td) in user_date_pairs
                for uid, td in zip(subset["user_id"], target_date)
            )
            records.append(
                {
                    cohort_col: group_value,
                    "day": day,
                    "users": total,
                    "retained": retained,
                    "retention_rate": retained / total if total > 0 else 0.0,
                }
            )

    return pd.DataFrame(records)


def compute_rolling_retention(
    users: pd.DataFrame,
    events: pd.DataFrame,
    day: int,
) -> float:
    """Compute overall rolling retention: share of users active on or after day N."""
    active_events = events[events["event_source"] == "user_action"].copy()
    user_last_event = active_events.groupby("user_id")["event_timestamp"].max()

    users_copy = users.copy()
    users_copy["cutoff"] = users_copy["signup_timestamp"] + pd.Timedelta(days=day)
    users_copy = users_copy.merge(user_last_event.rename("last_event"), on="user_id", how="left")
    retained = (users_copy["last_event"] >= users_copy["cutoff"]).sum()
    return retained / len(users_copy) if len(users_copy) > 0 else 0.0


def compute_retention_by_variant(
    users: pd.DataFrame,
    events: pd.DataFrame,
    experiment_assignments: pd.DataFrame,
    day: int,
    experiment_id: str = config.ONBOARDING_EXPERIMENT_ID,
) -> pd.DataFrame:
    """Compute day-N retention grouped by experiment variant.

    The function merges experiment assignments onto a copy of ``users`` and
    delegates to ``compute_day_retention``.
    """
    users_with_variant = users.merge(
        experiment_assignments[
            experiment_assignments["experiment_id"] == experiment_id
        ][["user_id", "variant_id"]],
        on="user_id",
        how="left",
    )
    return compute_day_retention(users_with_variant, events, day, group_by="variant_id")
