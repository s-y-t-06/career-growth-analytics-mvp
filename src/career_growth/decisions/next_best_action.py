"""Rule-based Next Best Action engine.

This module implements the approved priority rules. It does not require a
churn risk model score; when a score is provided it is used, otherwise a
simple activity heuristic is applied for high-risk detection.
"""

from datetime import timedelta
from typing import Any

import pandas as pd

from career_growth import config


def compute_user_state(user_id: str, events: pd.DataFrame, cutoff: pd.Timestamp) -> dict[str, Any]:
    """Compute the lifecycle state for a user up to a cutoff timestamp."""
    user_events = events[
        (events["user_id"] == user_id)
        & (events["event_timestamp"] <= cutoff)
        & (events["event_source"] == "user_action")
    ].copy()

    event_names = set(user_events["event_name"])

    recent_start = cutoff - timedelta(days=2)
    recent_events = user_events[user_events["event_timestamp"] >= recent_start]
    recently_active = len(recent_events) > 0

    return {
        "onboarding_completed": "onboarding_complete" in event_names,
        "profile_completed": "profile_complete" in event_names,
        "resume_uploaded": "resume_upload" in event_names,
        "job_recommendation_viewed": "job_recommendation_view" in event_names,
        "job_saved": "job_save" in event_names,
        "growth_task_completed": "growth_task_complete" in event_names,
        "career_report_generated": "career_report_generate" in event_names,
        "recently_active": recently_active,
    }


def recommend_next_action(
    user: pd.Series,
    events: pd.DataFrame,
    cutoff: pd.Timestamp | None = None,
    churn_risk_score: float | None = None,
) -> dict[str, Any]:
    """Return the recommended next best action for a single user.

    Args:
        user: row from users DataFrame.
        events: events DataFrame.
        cutoff: timestamp used to compute state. Defaults to signup + 7 days.
        churn_risk_score: optional model score; if None, recent inactivity is used.

    Returns:
        Dictionary with action_name, channel, and reason.
    """
    if cutoff is None:
        cutoff = user["signup_timestamp"] + timedelta(days=config.PREDICTION_CUTOFF_DAY)

    state = compute_user_state(user["user_id"], events, cutoff)
    consent = bool(user["marketing_consent"])

    # Rule 1: high churn risk and consent -> reengagement.
    # When no model score is available, we only flag users who completed
    # onboarding but have become inactive; new users are routed to onboarding.
    high_risk = (
        (churn_risk_score is not None and churn_risk_score >= 0.70)
        or (
            churn_risk_score is None
            and state["onboarding_completed"]
            and not state["recently_active"]
        )
    )
    if high_risk:
        if consent:
            return {
                "action_name": "send_reengagement_message",
                "channel": "email",
                "reason": "high churn risk with marketing consent",
            }
        return {
            "action_name": "send_reengagement_message",
            "channel": "in_app",
            "reason": "high churn risk without marketing consent",
        }

    # Rule 2: incomplete onboarding.
    if not state["onboarding_completed"]:
        return {
            "action_name": "complete_onboarding",
            "channel": "in_app",
            "reason": "onboarding not completed",
        }

    # Rule 3: no profile.
    if not state["profile_completed"]:
        return {
            "action_name": "complete_profile",
            "channel": "in_app",
            "reason": "profile not completed",
        }

    # Rule 4: no resume.
    if not state["resume_uploaded"]:
        return {
            "action_name": "upload_resume",
            "channel": "in_app",
            "reason": "resume not uploaded",
        }

    # Rule 5: no job recommendation view.
    if not state["job_recommendation_viewed"]:
        return {
            "action_name": "view_job_recommendations",
            "channel": "in_app",
            "reason": "job recommendations not viewed",
        }

    # Rule 6: viewed but not saved.
    if not state["job_saved"]:
        return {
            "action_name": "save_relevant_job",
            "channel": "in_app",
            "reason": "job viewed but not saved",
        }

    # Rule 7: no growth task.
    if not state["growth_task_completed"]:
        return {
            "action_name": "complete_growth_task",
            "channel": "in_app",
            "reason": "growth task not completed",
        }

    # Rule 8: no career report.
    if not state["career_report_generated"]:
        return {
            "action_name": "generate_career_report",
            "channel": "in_app",
            "reason": "career report not generated",
        }

    # Rule 9: all core actions done.
    return {
        "action_name": "continue_weekly_engagement",
        "channel": "email" if consent else "in_app",
        "reason": "all core actions completed",
    }
