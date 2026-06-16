"""Synthetic event stream generation for the AI career platform."""

import json
import uuid
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from career_growth import config


# Logistic parameters for core milestone events.
# Each tuple is (intercept, ie_coef, pf_coef, cu_coef, intent_coef, device_coef, stage_coef)
CORE_EVENT_LOGIT_PARAMS: dict[str, tuple[float, float, float, float, float, float, float]] = {
    "onboarding_start": (-0.60, 0.50, 0.30, 0.20, 0.20, 0.10, 0.10),
    "onboarding_complete": (-2.20, 1.80, 1.30, 0.70, 0.45, 0.25, 0.15),
    "profile_complete": (-1.50, 1.50, 0.40, 1.00, 0.35, 0.25, 0.15),
    "resume_upload": (-2.20, 1.30, 0.25, 1.30, 0.25, 0.25, 0.08),
    "job_recommendation_view": (-1.70, 1.20, 1.00, 0.25, 0.25, 0.15, 0.08),
    "job_save": (-1.20, 1.00, 0.65, 0.15, 0.25, 0.08, 0.08),
    "growth_task_complete": (-1.50, 1.30, 0.60, 0.35, 0.25, 0.08, 0.08),
    "career_report_generate": (-1.00, 1.00, 0.40, 0.40, 0.25, 0.08, 0.08),
}

# Day ranges for assigning milestone timestamps.
MILESTONE_DAY_RANGES: dict[str, tuple[int, int]] = {
    "onboarding_start": (0, 0),
    "onboarding_complete": (0, 0),
    "profile_complete": (0, 2),
    "resume_upload": (0, 3),
    "job_recommendation_view": (1, 5),
    "job_save": (2, 6),
    "growth_task_complete": (3, 7),
    "career_report_generate": (4, 7),
}

# State bonuses for downstream milestones.
# Positive bonus if prerequisite occurred; negative penalty if it did not.
STATE_BONUSES: dict[str, dict[str, tuple[float, float]]] = {
    "onboarding_complete": {"onboarding_start": (0.0, -3.0)},
    "profile_complete": {"onboarding_start": (0.10, -0.80), "onboarding_complete": (0.25, -1.50)},
    "resume_upload": {"onboarding_start": (0.05, -0.80), "profile_complete": (0.80, -0.80)},
    "job_recommendation_view": {"onboarding_start": (0.05, -0.60), "resume_upload": (0.50, -0.60)},
    "job_save": {"onboarding_start": (0.05, -0.50), "job_recommendation_view": (0.40, -0.50)},
    "growth_task_complete": {"onboarding_start": (0.05, -0.50), "job_save": (0.50, -0.50)},
    "career_report_generate": {"onboarding_start": (0.05, -0.50), "growth_task_complete": (0.40, -0.40)},
}

PAGE_NAMES: dict[str, str] = {
    "signup": "signup_page",
    "onboarding_start": "onboarding",
    "onboarding_complete": "onboarding",
    "profile_complete": "profile_edit",
    "resume_upload": "resume_upload",
    "job_recommendation_view": "job_recommendations",
    "job_detail_view": "job_detail",
    "job_save": "job_detail",
    "ai_assistant_interaction": "ai_assistant",
    "growth_task_complete": "growth_task",
    "career_report_generate": "career_report",
    "session_start": "session",
    "session_end": "session",
    "return_visit": "dashboard",
    "email_sent": "intervention",
    "push_sent": "intervention",
    "in_app_message_sent": "intervention",
}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _event_probability(
    params: tuple[float, float, float, float, float, float, float],
    row: pd.Series,
    direct_effect: float,
    state_bonus: float = 0.0,
) -> float:
    intercept, ie_coef, pf_coef, cu_coef, intent_coef, device_coef, stage_coef = params
    logit = (
        intercept
        + ie_coef * row["intrinsic_engagement"]
        + pf_coef * row["product_fit"]
        + cu_coef * row["career_urgency"]
        + intent_coef * row["intent_score"]
        + device_coef * row["device_score"]
        + stage_coef * row["career_stage_score"]
        + direct_effect
        + state_bonus
    )
    return float(np.clip(_sigmoid(logit), 0.01, 0.99))


def _platform_for_device(device_type: str, rng: np.random.Generator) -> str:
    if device_type == "desktop" or device_type == "tablet":
        return "web"
    return "ios" if rng.random() < 0.5 else "android"


def _random_time_on_day(signup: datetime, day_offset: int, rng: np.random.Generator) -> datetime:
    seconds = rng.integers(0, 86_400)
    return signup + timedelta(days=day_offset, seconds=int(seconds))


def _generate_milestones(
    row: pd.Series,
    variant_effect: float,
    rng: np.random.Generator,
) -> dict[str, datetime]:
    """Decide which core milestones occur and assign timestamps.

    The onboarding treatment only directly affects onboarding events.
    Downstream effects emerge from the user state and funnel prerequisites.
    """
    milestones: dict[str, datetime] = {}
    occurred: set[str] = set()

    for event_name in config.CORE_EVENTS:
        if event_name == "signup":
            # Signup is always at signup_timestamp and handled separately.
            continue

        params = CORE_EVENT_LOGIT_PARAMS[event_name]
        state_bonus = 0.0
        for prerequisite, effect in STATE_BONUSES.get(event_name, {}).items():
            bonus, penalty = effect
            if prerequisite in occurred:
                state_bonus += bonus
            else:
                state_bonus += penalty

        # Direct treatment effect is applied only to onboarding events.
        direct_effect = variant_effect if event_name in {"onboarding_start", "onboarding_complete"} else 0.0
        probability = _event_probability(params, row, direct_effect, state_bonus)
        if rng.random() < probability:
            low, high = MILESTONE_DAY_RANGES[event_name]
            # Shift day forward if a prerequisite occurred later than the default low bound.
            for prerequisite in STATE_BONUSES.get(event_name, {}):
                if prerequisite in milestones:
                    prereq_day = (milestones[prerequisite] - row["signup_timestamp"]).days
                    low = max(low, prereq_day)
            day_offset = rng.integers(low, high + 1)
            milestones[event_name] = _random_time_on_day(row["signup_timestamp"], int(day_offset), rng)
            occurred.add(event_name)

    return milestones


def _daily_activity_probability(
    row: pd.Series,
    onboarding_complete: bool,
    num_core_actions: int,
    late_phase: bool,
) -> float:
    ie = row["intrinsic_engagement"]
    pf = row["product_fit"]
    cu = row["career_urgency"]
    intent = row["intent_score"]
    onboarding_flag = 1.0 if onboarding_complete else 0.0

    engagement_score = (
        0.30 * ie
        + 0.20 * pf
        + 0.15 * cu
        + 0.10 * intent
    )
    if late_phase:
        state_boost = 0.015 * num_core_actions
        late_p = 0.001 + 0.08 * engagement_score + state_boost
        return float(np.clip(late_p, 0.001, 0.25))

    first_week_p = 0.01 + 0.50 * engagement_score + 0.05 * onboarding_flag
    return float(np.clip(first_week_p, 0.005, 0.60))


def _extra_user_actions(
    rng: np.random.Generator,
    row: pd.Series,
    milestones: dict[str, datetime],
    day_offset: int,
) -> list[str]:
    """Generate contextual extra user_action events for an active day."""
    extras: list[str] = []
    if rng.random() < 0.30:
        extras.append("job_detail_view")
    if rng.random() < 0.15:
        extras.append("ai_assistant_interaction")
    if day_offset > 0 and rng.random() < 0.25:
        extras.append("return_visit")
    if "job_save" in milestones and rng.random() < 0.10:
        extras.append("job_detail_view")
    return extras


def generate_events_for_user(
    row: pd.Series,
    experiment_assignment: dict[str, Any],
    rng: np.random.Generator,
    seed: int = config.RANDOM_SEED,
) -> list[dict[str, Any]]:
    """Generate the complete event stream for a single user."""
    signup = row["signup_timestamp"]
    platform = _platform_for_device(row["device_type"], rng)
    experiment_id = experiment_assignment["experiment_id"]
    variant_id = experiment_assignment["variant_id"]
    variant_effect = next(
        v["effect"] for v in config.ONBOARDING_VARIANTS if v["variant_id"] == variant_id
    )

    milestones = _generate_milestones(row, variant_effect, rng)
    onboarding_complete = "onboarding_complete" in milestones
    num_core_actions = sum(1 for e in milestones if e in config.CORE_EVENTS)

    user_id = row["user_id"]
    event_counter = 0
    session_counter = 0
    job_counter = 0

    def next_event_id() -> str:
        nonlocal event_counter
        event_counter += 1
        return str(uuid.uuid5(uuid.NAMESPACE_OID, f"event-{seed}-{user_id}-{event_counter}"))

    def next_session_id() -> str:
        nonlocal session_counter
        session_counter += 1
        return str(uuid.uuid5(uuid.NAMESPACE_OID, f"session-{seed}-{user_id}-{session_counter}"))

    def next_job_id() -> str:
        nonlocal job_counter
        job_counter += 1
        return str(uuid.uuid5(uuid.NAMESPACE_OID, f"job-{seed}-{user_id}-{job_counter}"))

    # Event queue: (timestamp, sort_order, event_name, event_source, properties, page_name, session_id)
    event_queue: list[tuple] = []

    # Signup event.
    event_queue.append(
        (
            signup,
            0,
            "signup",
            "user_action",
            {"variant_id": variant_id},
            PAGE_NAMES["signup"],
            next_session_id(),
        )
    )

    # Add milestone events.
    for event_name, ts in milestones.items():
        props: dict[str, Any] = {"variant_id": variant_id}
        if event_name == "job_save":
            props["job_id"] = next_job_id()
        if event_name == "growth_task_complete":
            props["task_type"] = rng.choice(["skill_assessment", "resume_review", "mock_interview"])
        event_queue.append(
            (
                ts,
                1,
                event_name,
                "user_action",
                props,
                PAGE_NAMES[event_name],
                next_session_id(),
            )
        )

    # Daily sessions for days 0-7.
    active_days_first_week: set[int] = set()
    for day_offset in range(0, config.PREDICTION_CUTOFF_DAY + 1):
        p_active = _daily_activity_probability(
            row, onboarding_complete, num_core_actions, late_phase=False
        )
        if rng.random() < p_active or (
            day_offset in {int((ts - signup).days) for ts in milestones.values()}
        ):
            active_days_first_week.add(day_offset)

    for day_offset in active_days_first_week:
        session_id = next_session_id()
        session_time = _random_time_on_day(signup, day_offset, rng)
        event_queue.append(
            (
                session_time,
                2,
                "session_start",
                "user_action",
                {"variant_id": variant_id},
                PAGE_NAMES["session_start"],
                session_id,
            )
        )

        extras = _extra_user_actions(rng, row, milestones, day_offset)
        for extra_name in extras:
            extra_time = session_time + timedelta(seconds=int(rng.integers(30, 600)))
            event_queue.append(
                (
                    extra_time,
                    3,
                    extra_name,
                    "user_action",
                    {"variant_id": variant_id},
                    PAGE_NAMES[extra_name],
                    session_id,
                )
            )

        end_time = session_time + timedelta(seconds=int(rng.integers(120, 1800)))
        event_queue.append(
            (
                end_time,
                4,
                "session_end",
                "user_action",
                {"variant_id": variant_id},
                PAGE_NAMES["session_end"],
                session_id,
            )
        )

    # Daily activity for days 8-21 (label window).
    for day_offset in range(config.LABEL_WINDOW_START_DAY, config.LABEL_WINDOW_END_DAY + 1):
        p_active = _daily_activity_probability(
            row, onboarding_complete, num_core_actions, late_phase=True
        )
        if rng.random() < p_active:
            session_id = next_session_id()
            session_time = _random_time_on_day(signup, day_offset, rng)
            event_queue.append(
                (
                    session_time,
                    2,
                    "session_start",
                    "user_action",
                    {"variant_id": variant_id},
                    PAGE_NAMES["session_start"],
                    session_id,
                )
            )
            extras = _extra_user_actions(rng, row, milestones, day_offset)
            for extra_name in extras:
                extra_time = session_time + timedelta(seconds=int(rng.integers(30, 600)))
                event_queue.append(
                    (
                        extra_time,
                        3,
                        extra_name,
                        "user_action",
                        {"variant_id": variant_id},
                        PAGE_NAMES[extra_name],
                        session_id,
                    )
                )
            end_time = session_time + timedelta(seconds=int(rng.integers(120, 1800)))
            event_queue.append(
                (
                    end_time,
                    4,
                    "session_end",
                    "user_action",
                    {"variant_id": variant_id},
                    PAGE_NAMES["session_end"],
                    session_id,
                )
            )

    # Campaign events: simulate a small number of system/campaign messages.
    if row["marketing_consent"] and rng.random() < 0.10:
        campaign_day = rng.integers(1, config.PREDICTION_CUTOFF_DAY + 1)
        campaign_time = _random_time_on_day(signup, int(campaign_day), rng)
        campaign_name = rng.choice(["email_sent", "push_sent", "in_app_message_sent"])
        event_queue.append(
            (
                campaign_time,
                5,
                campaign_name,
                "campaign",
                {"variant_id": variant_id, "campaign_type": "onboarding_reminder"},
                PAGE_NAMES[campaign_name],
                next_session_id(),
            )
        )

    # Sort by timestamp, then by internal order to maintain logical sequence.
    event_queue.sort(key=lambda x: (x[0], x[1]))

    events: list[dict[str, Any]] = []
    for ts, _, event_name, source, props, page, session_id in event_queue:
        events.append(
            {
                "event_id": next_event_id(),
                "user_id": row["user_id"],
                "session_id": session_id,
                "event_name": event_name,
                "event_timestamp": ts,
                "event_properties": json.dumps(props if props else {}),
                "page_name": page,
                "platform": platform,
                "event_source": source,
                "experiment_id": experiment_id if event_name in config.CORE_EVENTS else None,
                "variant_id": variant_id if event_name in config.CORE_EVENTS else None,
            }
        )

    return events
