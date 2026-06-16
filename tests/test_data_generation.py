"""Tests for the synthetic data generation pipeline."""

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from career_growth import config
from career_growth.data_generation.generator import generate_all_data


CSV_FILES = [
    "sample/users.csv",
    "sample/events.csv",
    "sample/experiment_assignments.csv",
    "sample/interventions.csv",
    "processed/labels.csv",
]


def _file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def test_generation_reproducibility(tmp_path):
    """Same seed must produce byte-identical CSV outputs."""
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    generate_all_data(count=200, seed=123, output_dir=str(dir_a))
    generate_all_data(count=200, seed=123, output_dir=str(dir_b))

    for relative in CSV_FILES:
        path_a = dir_a / relative
        path_b = dir_b / relative
        assert path_a.exists(), f"Missing file: {path_a}"
        assert path_b.exists(), f"Missing file: {path_b}"
        assert _file_hash(path_a) == _file_hash(path_b), (
            f"{relative} differs between reproducibility runs"
        )

    # Also compare full DataFrames after round-trip for one key file.
    users_a = pd.read_csv(dir_a / "sample/users.csv")
    users_b = pd.read_csv(dir_b / "sample/users.csv")
    pd.testing.assert_frame_equal(users_a, users_b)


def test_users_schema_and_no_hidden_variables(synthetic_data):
    hidden_cols = [
        "intrinsic_engagement",
        "career_urgency",
        "product_fit",
        "notification_sensitivity",
        "intent_score",
        "career_stage_score",
        "device_score",
    ]
    for col in hidden_cols:
        assert col not in synthetic_data["users"].columns


def test_events_after_signup(synthetic_data):
    merged = synthetic_data["events"].merge(
        synthetic_data["users"][["user_id", "signup_timestamp"]], on="user_id"
    )
    assert (merged["event_timestamp"] >= merged["signup_timestamp"]).all()


def test_churn_rate_within_target(synthetic_data):
    churn_rate = synthetic_data["labels"]["is_churned"].mean()
    assert 0.25 <= churn_rate <= 0.45


def test_onboarding_treatment_effect_direction(synthetic_data):
    rates = (
        synthetic_data["experiment_assignments"]
        .groupby("variant_id")["is_converted"]
        .mean()
    )
    assert rates["personalized"] > rates["control"]
    assert rates["simplified"] > rates["control"]


def test_experiment_assignment_proportions(synthetic_data):
    counts = synthetic_data["experiment_assignments"]["variant_id"].value_counts(normalize=True)
    assert abs(counts["control"] - 0.40) < 0.05
    assert abs(counts["personalized"] - 0.30) < 0.05
    assert abs(counts["simplified"] - 0.30) < 0.05


def test_active_events_only_for_label(synthetic_data):
    churned_user = synthetic_data["labels"][synthetic_data["labels"]["is_churned"] == 1].iloc[0]
    label_start = churned_user["label_start"]
    label_end = churned_user["label_end"]
    active_events = synthetic_data["events"][
        (synthetic_data["events"]["user_id"] == churned_user["user_id"])
        & (synthetic_data["events"]["event_source"] == "user_action")
        & (synthetic_data["events"]["event_timestamp"] >= label_start)
        & (synthetic_data["events"]["event_timestamp"] <= label_end)
    ]
    assert len(active_events) == 0


def test_intervention_win_back_targets_churned(synthetic_data):
    """Win-back interventions must only be sent to churned users."""
    interventions = synthetic_data["interventions"]
    labels = synthetic_data["labels"]

    merged = interventions.merge(labels[["user_id", "is_churned"]], on="user_id")
    win_backs = merged[merged["action_name"] == "send_win_back"]

    assert len(win_backs) > 0, "expected at least one win-back intervention"
    assert (win_backs["is_churned"] == 1).all(), (
        "win-back sent to retained user(s)"
    )

    retained_users = set(labels[labels["is_churned"] == 0]["user_id"])
    retained_win_backs = interventions[
        (interventions["action_name"] == "send_win_back")
        & (interventions["user_id"].isin(retained_users))
    ]
    assert len(retained_win_backs) == 0


def test_intervention_reproducibility(tmp_path):
    """Intervention records must be deterministic for a fixed seed."""
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    generate_all_data(count=200, seed=7, output_dir=str(dir_a))
    generate_all_data(count=200, seed=7, output_dir=str(dir_b))

    interventions_a = pd.read_csv(dir_a / "sample/interventions.csv")
    interventions_b = pd.read_csv(dir_b / "sample/interventions.csv")
    pd.testing.assert_frame_equal(interventions_a, interventions_b)
