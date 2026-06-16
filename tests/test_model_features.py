"""Tests for churn model feature engineering."""

import numpy as np
import pandas as pd
import pytest

from career_growth.features.model_features import (
    ALL_FEATURE_COLUMNS,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    attach_labels,
    build_model_features,
    prepare_model_matrix,
)


@pytest.fixture
def sample_data(synthetic_data):
    """Return a small deterministic dataset for feature tests."""
    return {
        "users": synthetic_data["users"].head(200).copy(),
        "events": synthetic_data["events"].copy(),
        "experiment_assignments": synthetic_data["experiment_assignments"].copy(),
        "labels": synthetic_data["labels"].copy(),
    }


def test_build_model_features_columns(sample_data):
    """The feature matrix must contain the expected columns and no leakage columns."""
    features = build_model_features(
        sample_data["users"],
        sample_data["events"],
        sample_data["experiment_assignments"],
    )

    assert "user_id" in features.columns
    for col in ALL_FEATURE_COLUMNS:
        assert col in features.columns, f"Missing feature column: {col}"

    forbidden = {"last_active_date", "days_since_last_active", "is_churned", "future_"}
    for col in features.columns:
        for prefix in forbidden:
            assert not col.startswith(prefix), f"Forbidden column found: {col}"


def test_build_model_features_no_null_categoricals(sample_data):
    """Categorical features must be populated."""
    features = build_model_features(
        sample_data["users"],
        sample_data["events"],
        sample_data["experiment_assignments"],
    )

    for col in CATEGORICAL_FEATURES:
        assert features[col].notna().all(), f"Null values found in {col}"


def test_time_features_use_missing_for_inactive_users(sample_data):
    """Time-based features for users with no pre-cutoff actions must be NaN."""
    users = sample_data["users"].head(10).copy()
    # Remove all events for the first user so it has no pre-cutoff behavior.
    events = sample_data["events"][
        sample_data["events"]["user_id"] != users.iloc[0]["user_id"]
    ].copy()

    features = build_model_features(
        users,
        events,
        sample_data["experiment_assignments"],
    )

    row = features[features["user_id"] == users.iloc[0]["user_id"]].iloc[0]
    for col in ["hours_to_first_action", "hours_since_last_action_at_cutoff"]:
        assert pd.isna(row[col]), f"Expected NaN in {col} for inactive user"


def test_numeric_features_are_numeric(sample_data):
    """Numeric columns must have a numeric dtype."""
    features = build_model_features(
        sample_data["users"],
        sample_data["events"],
        sample_data["experiment_assignments"],
    )

    for col in NUMERIC_FEATURES:
        assert pd.api.types.is_numeric_dtype(features[col]), f"{col} is not numeric"


def test_build_model_features_ignores_post_cutoff_events(sample_data):
    """Adding post-cutoff events must not change any model feature values."""
    users = sample_data["users"].head(20).copy()
    events = sample_data["events"].copy()

    features_before = build_model_features(users, events)

    # Inject synthetic post-cutoff events for every user.
    post_cutoff_events = []
    for _, user in users.iterrows():
        cutoff = pd.to_datetime(user["signup_timestamp"]) + pd.Timedelta(days=7)
        for i in range(5):
            post_cutoff_events.append(
                {
                    "event_id": f"post-{user['user_id']}-{i}",
                    "user_id": user["user_id"],
                    "session_id": f"post-session-{user['user_id']}-{i}",
                    "event_name": "job_detail_view",
                    "event_timestamp": cutoff + pd.Timedelta(hours=i + 1),
                    "event_properties": "{}",
                    "page_name": "post_page",
                    "platform": "web",
                    "event_source": "user_action",
                    "experiment_id": None,
                    "variant_id": None,
                }
            )
    events_with_post = pd.concat(
        [events, pd.DataFrame(post_cutoff_events)], ignore_index=True
    )
    features_after = build_model_features(users, events_with_post)

    feature_cols = [c for c in features_before.columns if c != "signup_timestamp"]
    pd.testing.assert_frame_equal(
        features_before[feature_cols].sort_values("user_id").reset_index(drop=True),
        features_after[feature_cols].sort_values("user_id").reset_index(drop=True),
        check_dtype=False,
    )


def test_variant_feature_joined(sample_data):
    """The onboarding variant feature must match experiment assignments."""
    features = build_model_features(
        sample_data["users"],
        sample_data["events"],
        sample_data["experiment_assignments"],
    )

    merged = features.merge(
        sample_data["experiment_assignments"][
            sample_data["experiment_assignments"]["experiment_id"] == "exp_onboarding_v1"
        ][["user_id", "variant_id"]],
        on="user_id",
        how="left",
    )
    assert (merged["onboarding_variant"] == merged["variant_id"].fillna("unknown")).all()


def test_attach_labels(sample_data):
    """Labels must be attached correctly and leakage must raise an error."""
    features = build_model_features(
        sample_data["users"],
        sample_data["events"],
        sample_data["experiment_assignments"],
    )
    labeled = attach_labels(features, sample_data["labels"])

    assert "is_churned" in labeled.columns
    assert labeled["is_churned"].isin({0, 1}).all()

    leaked_features = features.copy()
    leaked_features["future_events"] = 0
    with pytest.raises(ValueError, match="Feature leakage detected"):
        attach_labels(leaked_features, sample_data["labels"])


def test_prepare_model_matrix(sample_data):
    """The complete modeling matrix must preserve all users with labels."""
    matrix = prepare_model_matrix(
        sample_data["users"],
        sample_data["events"],
        sample_data["labels"],
        sample_data["experiment_assignments"],
    )

    expected_users = set(sample_data["users"]["user_id"]) & set(
        sample_data["labels"]["user_id"]
    )
    assert set(matrix["user_id"]) == expected_users
    assert "is_churned" in matrix.columns
