"""Tests for the Next Best Action rule engine."""

import pandas as pd

from career_growth.decisions.next_best_action import recommend_next_action


def test_nba_recommends_onboarding_for_new_user(synthetic_data):
    user = synthetic_data["users"].iloc[0]
    events = synthetic_data["events"][
        synthetic_data["events"]["user_id"] != user["user_id"]
    ].copy()
    events = pd.concat(
        [
            events,
            pd.DataFrame(
                [
                    {
                        "event_id": "test-signup",
                        "user_id": user["user_id"],
                        "session_id": "test-session",
                        "event_name": "signup",
                        "event_timestamp": user["signup_timestamp"],
                        "event_properties": "{}",
                        "page_name": "signup_page",
                        "platform": "web",
                        "event_source": "user_action",
                        "experiment_id": None,
                        "variant_id": None,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    rec = recommend_next_action(user, events)
    assert rec["action_name"] == "complete_onboarding"


def test_nba_uses_in_app_without_consent(synthetic_data):
    user = synthetic_data["users"].iloc[0].copy()
    user["marketing_consent"] = False
    rec = recommend_next_action(user, synthetic_data["events"])
    assert rec["channel"] == "in_app"
