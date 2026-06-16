"""Tests for churn-score integration with Next Best Action."""

import joblib
import numpy as np
import pandas as pd
import pytest

from career_growth.decisions.next_best_action import recommend_next_action
from career_growth.features.model_features import prepare_model_matrix
from career_growth.modeling.nba_integration import score_users
from career_growth.modeling.pipeline import build_logistic_regression_pipeline
from career_growth.modeling.split import chronological_split
from career_growth.modeling.train import train_and_select_model


def _make_high_risk_user() -> pd.Series:
    """Return a user row with marketing consent enabled."""
    return pd.Series(
        {
            "user_id": "high-risk-user",
            "signup_timestamp": pd.Timestamp("2026-01-01 00:00:00+0000"),
            "acquisition_channel": "organic_search",
            "country": "US",
            "device_type": "desktop",
            "user_intent_level": "medium",
            "career_stage": "student",
            "marketing_consent": True,
            "language": "en",
            "timezone": "America/New_York",
            "initial_plan_type": "free",
        }
    )


def test_high_risk_with_consent_uses_email():
    """High churn risk with marketing consent should trigger email reengagement."""
    user = _make_high_risk_user()
    events = pd.DataFrame(
        columns=[
            "event_id",
            "user_id",
            "session_id",
            "event_name",
            "event_timestamp",
            "event_properties",
            "page_name",
            "platform",
            "event_source",
            "experiment_id",
            "variant_id",
        ]
    )
    rec = recommend_next_action(user, events, churn_risk_score=0.85)
    assert rec["action_name"] == "send_reengagement_message"
    assert rec["channel"] == "email"


def test_high_risk_without_consent_uses_in_app():
    """High churn risk without marketing consent should trigger in-app reengagement."""
    user = _make_high_risk_user()
    user["marketing_consent"] = False
    events = pd.DataFrame(
        columns=[
            "event_id",
            "user_id",
            "session_id",
            "event_name",
            "event_timestamp",
            "event_properties",
            "page_name",
            "platform",
            "event_source",
            "experiment_id",
            "variant_id",
        ]
    )
    rec = recommend_next_action(user, events, churn_risk_score=0.85)
    assert rec["action_name"] == "send_reengagement_message"
    assert rec["channel"] == "in_app"


def test_nba_input_does_not_contain_label(synthetic_data):
    """The recommend_next_action signature must not accept a true label argument."""
    import inspect

    sig = inspect.signature(recommend_next_action)
    params = set(sig.parameters.keys())
    assert "is_churned" not in params
    assert "label" not in params


def test_save_and_load_model_preserves_nba_recommendations(tmp_path, synthetic_data):
    """Saving and loading the model must preserve NBA recommendations."""
    model_matrix = prepare_model_matrix(
        synthetic_data["users"],
        synthetic_data["events"],
        synthetic_data["labels"],
        synthetic_data["experiment_assignments"],
    )
    train_df, val_df, test_df = chronological_split(
        model_matrix, train_frac=0.6, val_frac=0.2
    )

    result = train_and_select_model(
        train_df,
        val_df,
        test_df,
        threshold_criterion="f1",
        random_state=42,
    )

    test_users = synthetic_data["users"][
        synthetic_data["users"]["user_id"].isin(test_df["user_id"])
    ].head(5)

    before = score_users(
        result.model,
        test_users,
        synthetic_data["events"],
        result.feature_columns,
        result.threshold,
    )

    model_path = tmp_path / "churn_model.joblib"
    joblib.dump(result.model, model_path)
    loaded_model = joblib.load(model_path)

    after = score_users(
        loaded_model,
        test_users,
        synthetic_data["events"],
        result.feature_columns,
        result.threshold,
    )

    assert np.allclose(before["predicted_risk"].to_numpy(), after["predicted_risk"].to_numpy())
    assert (before["action_name"] == after["action_name"]).all()
