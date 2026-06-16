"""Tests for feature engineering and label construction."""

import pandas as pd

from career_growth.features.labels import build_labels, check_label_leakage


def test_label_only_uses_user_action(synthetic_data):
    labels = synthetic_data["labels"]
    churned = labels[labels["is_churned"] == 1]
    assert len(churned) > 0

    for _, user_label in churned.head(20).iterrows():
        user_events = synthetic_data["events"][
            synthetic_data["events"]["user_id"] == user_label["user_id"]
        ]
        active_in_window = user_events[
            (user_events["event_source"] == "user_action")
            & (user_events["event_timestamp"] >= user_label["label_start"])
            & (user_events["event_timestamp"] <= user_label["label_end"])
        ]
        assert len(active_in_window) == 0


def test_label_window_bounds(synthetic_data):
    user = synthetic_data["users"].iloc[0]
    label = synthetic_data["labels"][
        synthetic_data["labels"]["user_id"] == user["user_id"]
    ].iloc[0]
    signup = user["signup_timestamp"]
    assert label["label_start"] == signup + pd.Timedelta(days=8)
    assert label["label_end"] == signup + pd.Timedelta(days=21)


def test_leakage_checker_flags_forbidden_columns():
    bad_features = pd.DataFrame({"future_event_count": [1, 2], "is_churned": [0, 1]})
    labels = pd.DataFrame({"user_id": ["a", "b"], "is_churned": [0, 1]})
    issues = check_label_leakage(bad_features, labels)
    assert len(issues) >= 2
