"""Synthetic intervention log generation.

Interventions are secondary data and do not influence the churn label or the
main event stream. They are generated after labels are computed so that win-back
campaigns target only users who are truly churned under the label definition.
"""

import uuid
from datetime import timedelta

import numpy as np
import pandas as pd

from career_growth import config


def generate_interventions(
    users: pd.DataFrame,
    events: pd.DataFrame,
    labels: pd.DataFrame,
    rng: np.random.Generator,
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """Generate a small set of marketing interventions based on user behavior.

    Args:
        users: users DataFrame (may be the public version without hidden columns).
        events: events DataFrame.
        labels: labels DataFrame with ``is_churned`` for each user.
        rng: random number generator.
        seed: generation seed used for deterministic ID derivation.
    """
    if events.empty:
        return pd.DataFrame(
            columns=[
                "message_id",
                "user_id",
                "action_name",
                "channel",
                "send_time",
                "open_time",
                "click_time",
                "conversion_time",
                "experiment_id",
            ]
        )

    churned_by_user = labels.set_index("user_id")["is_churned"].to_dict()
    user_actions = events[events["event_source"] == "user_action"].copy()

    rows = []
    message_counter = 0

    def next_message_id(user_id: str) -> str:
        nonlocal message_counter
        message_counter += 1
        return str(
            uuid.uuid5(
                uuid.NAMESPACE_OID,
                f"intervention-{seed}-{user_id}-{message_counter}",
            )
        )

    for _, user in users.iterrows():
        user_id = user["user_id"]
        signup = user["signup_timestamp"]
        consent = user["marketing_consent"]

        user_events = events[events["user_id"] == user_id]
        active_days_first_week = set()
        for _, ev in user_events.iterrows():
            if ev["event_source"] == "user_action":
                day = (ev["event_timestamp"] - signup).days
                if 0 <= day <= config.PREDICTION_CUTOFF_DAY:
                    active_days_first_week.add(day)

        onboarding_done = (
            user_events[
                (user_events["event_name"] == "onboarding_complete")
                & (user_events["event_source"] == "user_action")
            ].shape[0]
            > 0
        )

        # Rule 1: incomplete onboarding by day 3 -> prompt to complete onboarding.
        if not onboarding_done:
            send_time = signup + timedelta(days=3)
            channel = "in_app" if not consent else rng.choice(["email", "push", "in_app"])
            rows.append(
                _build_intervention_row(
                    next_message_id(user_id),
                    user_id,
                    "complete_onboarding",
                    channel,
                    send_time,
                    rng,
                )
            )
            continue

        # Rule 2: low engagement in first week -> reengagement.
        if len(active_days_first_week) <= 1:
            send_time = signup + timedelta(days=7)
            channel = "in_app" if not consent else rng.choice(["email", "push"])
            rows.append(
                _build_intervention_row(
                    next_message_id(user_id),
                    user_id,
                    "send_reengagement_message",
                    channel,
                    send_time,
                    rng,
                )
            )
            continue

        # Rule 3: churned users after label window -> win-back.
        # Use the official churn label instead of a loose last-action heuristic.
        if churned_by_user.get(user_id, 0) == 1:
            label_end = signup + timedelta(days=config.LABEL_WINDOW_END_DAY)
            send_time = label_end + timedelta(days=1)
            channel = "in_app" if not consent else rng.choice(["email", "push"])
            rows.append(
                _build_intervention_row(
                    next_message_id(user_id),
                    user_id,
                    "send_win_back",
                    channel,
                    send_time,
                    rng,
                )
            )

    return pd.DataFrame(rows)


def _build_intervention_row(
    message_id: str,
    user_id: str,
    action_name: str,
    channel: str,
    send_time: pd.Timestamp,
    rng: np.random.Generator,
) -> dict:
    open_time = send_time + timedelta(hours=int(rng.integers(1, 48))) if rng.random() < 0.20 else None
    click_time = (
        open_time + timedelta(minutes=int(rng.integers(1, 60)))
        if open_time is not None and rng.random() < 0.30
        else None
    )
    conversion_time = (
        click_time + timedelta(hours=int(rng.integers(1, 24)))
        if click_time is not None and rng.random() < 0.10
        else None
    )
    return {
        "message_id": message_id,
        "user_id": user_id,
        "action_name": action_name,
        "channel": channel,
        "send_time": send_time,
        "open_time": open_time,
        "click_time": click_time,
        "conversion_time": conversion_time,
        "experiment_id": None,
    }
