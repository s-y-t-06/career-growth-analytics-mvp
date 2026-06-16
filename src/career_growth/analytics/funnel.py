"""Funnel analytics for the AI career platform."""

from typing import Any

import pandas as pd

from career_growth import config


def compute_funnel(
    users: pd.DataFrame,
    events: pd.DataFrame,
    group_by: str | None = None,
) -> pd.DataFrame:
    """Compute the core user lifecycle funnel.

    A user is counted at step n only if they performed step n and all previous
    steps in the funnel in the correct temporal order.

    Args:
        users: users DataFrame.
        events: events DataFrame.
        group_by: optional column in users to group by (e.g. acquisition_channel).

    Returns:
        DataFrame with columns: step, users, conversion_rate, drop_off_rate,
        and optionally a group column.
    """
    active_events = events[events["event_source"] == "user_action"].copy()
    first_occurrence = (
        active_events.groupby(["user_id", "event_name"])["event_timestamp"]
        .min()
        .unstack()
    )

    # Ensure signup column exists.
    if "signup" not in first_occurrence.columns:
        first_occurrence["signup"] = pd.NaT

    merged = users.merge(
        first_occurrence.reset_index(),
        on="user_id",
        how="left",
    )

    groups = [None] if group_by is None else sorted(merged[group_by].dropna().unique())
    results: list[dict[str, Any]] = []

    for group_value in groups:
        subset = merged if group_value is None else merged[merged[group_by] == group_value]
        total_users = len(subset)
        previous_users = total_users
        previous_timestamp = subset["signup"].copy()

        for step in config.FUNNEL_STEPS:
            if step == "signup":
                step_users = total_users
                valid_mask = subset["user_id"].notna()
            else:
                step_ts = subset[step]
                # Must have step timestamp and it must be after previous step timestamp.
                valid_mask = step_ts.notna()
                if step != config.FUNNEL_STEPS[0]:
                    valid_mask = valid_mask & (step_ts > previous_timestamp)
                step_users = int(valid_mask.sum())
                previous_timestamp = step_ts.where(valid_mask, pd.NaT)

            conversion_rate = step_users / total_users if total_users > 0 else 0.0
            drop_off_rate = (
                (previous_users - step_users) / previous_users if previous_users > 0 else 0.0
            )

            row: dict[str, Any] = {
                "step": step,
                "users": step_users,
                "conversion_rate": conversion_rate,
                "drop_off_rate": drop_off_rate,
            }
            if group_by:
                row[group_by] = group_value
            results.append(row)
            previous_users = step_users

    return pd.DataFrame(results)
