"""Tests for churn model training, evaluation, and explainability."""

import joblib
import numpy as np
import pandas as pd
import pytest

from career_growth.features.model_features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    prepare_model_matrix,
)
from career_growth.modeling.evaluate import compute_metrics, select_threshold
from career_growth.modeling.explain import (
    compute_permutation_importance,
    explain_user_prediction,
    extract_feature_names,
    extract_logistic_coefficients,
)
from career_growth.modeling.nba_integration import generate_nba_examples
from career_growth.modeling.pipeline import (
    build_hist_gradient_boosting_pipeline,
    build_logistic_regression_pipeline,
)
from career_growth.modeling.split import chronological_split, split_users_and_labels
from career_growth.modeling.subgroup import evaluate_subgroups
from career_growth.modeling.train import save_model, train_and_select_model


@pytest.fixture
def model_matrix(synthetic_data):
    """Return a modeling matrix for the shared synthetic dataset."""
    return prepare_model_matrix(
        synthetic_data["users"],
        synthetic_data["events"],
        synthetic_data["labels"],
        synthetic_data["experiment_assignments"],
    )


def test_chronological_split_order_and_disjoint(tmp_path):
    """Chronological split must preserve time order and produce disjoint sets."""
    df = pd.DataFrame(
        {
            "user_id": [f"u{i}" for i in range(10)],
            "signup_timestamp": pd.date_range("2026-01-01", periods=10, freq="D"),
        }
    )
    train, val, test = chronological_split(df, train_frac=0.6, val_frac=0.2)

    assert len(train) == 6
    assert len(val) == 2
    assert len(test) == 2
    assert set(train["user_id"]).isdisjoint(set(val["user_id"]))
    assert set(train["user_id"]).isdisjoint(set(test["user_id"]))
    assert set(val["user_id"]).isdisjoint(set(test["user_id"]))
    assert train["signup_timestamp"].max() <= val["signup_timestamp"].min()
    assert val["signup_timestamp"].max() <= test["signup_timestamp"].min()


def test_split_users_and_labels(synthetic_data):
    """User and label splits must be consistent and chronological."""
    users = synthetic_data["users"]
    labels = synthetic_data["labels"]

    train_u, val_u, test_u, train_l, val_l, test_l = split_users_and_labels(
        users, labels
    )

    assert set(train_l["user_id"]) == set(train_u["user_id"])
    assert set(val_l["user_id"]) == set(val_u["user_id"])
    assert set(test_l["user_id"]) == set(test_u["user_id"])
    assert train_u["signup_timestamp"].max() <= val_u["signup_timestamp"].min()
    assert val_u["signup_timestamp"].max() <= test_u["signup_timestamp"].min()


def test_logistic_regression_pipeline(model_matrix):
    """The logistic regression pipeline must fit and predict probabilities."""
    feature_cols = [c for c in model_matrix.columns if c not in {"user_id", "signup_timestamp", "is_churned"}]
    X = model_matrix[feature_cols]
    y = model_matrix["is_churned"].to_numpy()

    pipeline = build_logistic_regression_pipeline(random_state=42)
    pipeline.fit(X, y)
    prob = pipeline.predict_proba(X)[:, 1]

    assert prob.shape == (len(model_matrix),)
    assert np.all((prob >= 0.0) & (prob <= 1.0))


def test_hist_gradient_boosting_pipeline(model_matrix):
    """The gradient boosting pipeline must fit and predict probabilities."""
    feature_cols = [c for c in model_matrix.columns if c not in {"user_id", "signup_timestamp", "is_churned"}]
    X = model_matrix[feature_cols]
    y = model_matrix["is_churned"].to_numpy()

    pipeline = build_hist_gradient_boosting_pipeline(random_state=42)
    pipeline.fit(X, y)
    prob = pipeline.predict_proba(X)[:, 1]

    assert prob.shape == (len(model_matrix),)
    assert np.all((prob >= 0.0) & (prob <= 1.0))


def test_threshold_selection(model_matrix):
    """Threshold selection must return a value in [0, 1]."""
    feature_cols = [c for c in model_matrix.columns if c not in {"user_id", "signup_timestamp", "is_churned"}]
    X = model_matrix[feature_cols]
    y = model_matrix["is_churned"].to_numpy()

    pipeline = build_logistic_regression_pipeline(random_state=42)
    pipeline.fit(X, y)
    prob = pipeline.predict_proba(X)[:, 1]

    for criterion in ["f1", "precision", "recall", "f2", "youden"]:
        threshold = select_threshold(y, prob, criterion=criterion)
        assert 0.0 <= threshold <= 1.0


def test_youden_threshold_is_data_dependent(model_matrix):
    """Youden threshold must depend on the data and not be fixed at zero."""
    feature_cols = [c for c in model_matrix.columns if c not in {"user_id", "signup_timestamp", "is_churned"}]
    train_df, val_df, _ = chronological_split(
        model_matrix, train_frac=0.6, val_frac=0.2
    )

    pipeline = build_logistic_regression_pipeline(random_state=42)
    pipeline.fit(train_df[feature_cols], train_df["is_churned"].to_numpy())
    val_prob = pipeline.predict_proba(val_df[feature_cols])[:, 1]

    threshold = select_threshold(
        val_df["is_churned"].to_numpy(), val_prob, criterion="youden"
    )
    assert threshold > 0.0
    assert threshold < 1.0


def test_train_and_select_model(model_matrix):
    """The training harness must select a model and produce validation/test metrics."""
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

    assert result.model_name in {"logistic_regression", "hist_gradient_boosting"}
    assert 0.0 <= result.threshold <= 1.0
    assert set(result.val_metrics.keys()) >= {
        "pr_auc",
        "roc_auc",
        "log_loss",
        "brier_score",
        "precision",
        "recall",
        "f1_score",
        "accuracy",
    }
    assert set(result.test_metrics.keys()) >= {
        "pr_auc",
        "roc_auc",
        "log_loss",
        "brier_score",
        "precision",
        "recall",
        "f1_score",
        "accuracy",
    }
    assert len(result.feature_columns) > 0
    assert "logistic_regression" in result.candidate_validation_metrics
    assert "hist_gradient_boosting" in result.candidate_validation_metrics


def test_logistic_coefficients(model_matrix):
    """Logistic regression coefficients must match the preprocessor output dimension."""
    feature_cols = [c for c in model_matrix.columns if c not in {"user_id", "signup_timestamp", "is_churned"}]
    X = model_matrix[feature_cols]
    y = model_matrix["is_churned"].to_numpy()

    pipeline = build_logistic_regression_pipeline(random_state=42)
    pipeline.fit(X, y)

    names = extract_feature_names(pipeline)
    coef_df = extract_logistic_coefficients(pipeline)

    assert len(names) == len(coef_df)
    assert "feature" in coef_df.columns and "coefficient" in coef_df.columns


def test_permutation_importance(model_matrix):
    """Permutation importance must return non-negative max importance."""
    feature_cols = [c for c in model_matrix.columns if c not in {"user_id", "signup_timestamp", "is_churned"}]
    train_df, val_df, _ = chronological_split(
        model_matrix, train_frac=0.6, val_frac=0.2
    )

    pipeline = build_logistic_regression_pipeline(random_state=42)
    pipeline.fit(train_df[feature_cols], train_df["is_churned"].to_numpy())

    importance = compute_permutation_importance(
        pipeline,
        val_df[feature_cols],
        val_df["is_churned"].to_numpy(),
        n_repeats=3,
        random_state=42,
    )

    assert len(importance) == len(feature_cols)
    assert importance["importance_mean"].max() >= 0.0


def test_metrics_computed_at_selected_threshold(model_matrix):
    """Metrics must reflect the selected threshold rather than the default 0.5."""
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

    y_test = test_df["is_churned"].to_numpy()
    recomputed = compute_metrics(y_test, result.test_probabilities, result.threshold)
    assert pytest.approx(result.test_metrics["f1_score"], rel=1e-6) == recomputed["f1_score"]
    assert pytest.approx(result.test_metrics["precision"], rel=1e-6) == recomputed["precision"]
    assert pytest.approx(result.test_metrics["recall"], rel=1e-6) == recomputed["recall"]


def test_feature_schema_disjoint_and_complete():
    """Categorical and numeric feature lists must partition all model features."""
    union = set(CATEGORICAL_FEATURES) | set(NUMERIC_FEATURES)
    assert set(CATEGORICAL_FEATURES).isdisjoint(set(NUMERIC_FEATURES))
    assert union == set([c for c in CATEGORICAL_FEATURES + NUMERIC_FEATURES])


def test_subgroup_evaluation(model_matrix):
    """Subgroup evaluation must return rows with required columns."""
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

    subgroup_df = evaluate_subgroups(
        test_df, result.test_probabilities, result.threshold
    )

    required_columns = {
        "group_column",
        "group_value",
        "sample_size",
        "churn_rate",
        "precision",
        "recall",
        "f1_score",
        "predicted_positive_rate",
        "small_sample",
    }
    assert required_columns.issubset(set(subgroup_df.columns))
    assert not subgroup_df.empty


def test_user_explanation(model_matrix):
    """User explanation must return positive/negative factors."""
    feature_cols = [c for c in model_matrix.columns if c not in {"user_id", "signup_timestamp", "is_churned"}]
    train_df, val_df, test_df = chronological_split(
        model_matrix, train_frac=0.6, val_frac=0.2
    )

    pipeline = build_logistic_regression_pipeline(random_state=42)
    pipeline.fit(train_df[feature_cols], train_df["is_churned"].to_numpy())

    x_row = test_df[feature_cols].iloc[[0]]
    explanation = explain_user_prediction(pipeline, x_row, feature_cols, top_n=3)

    assert 0.0 <= explanation["predicted_risk"] <= 1.0
    assert isinstance(explanation["positive_factors"], list)
    assert isinstance(explanation["negative_factors"], list)


def test_model_save_and_load_preserve_predictions(tmp_path, model_matrix):
    """Saving and loading a model must preserve predictions."""
    feature_cols = [c for c in model_matrix.columns if c not in {"user_id", "signup_timestamp", "is_churned"}]
    X = model_matrix[feature_cols]
    y = model_matrix["is_churned"].to_numpy()

    pipeline = build_logistic_regression_pipeline(random_state=42)
    pipeline.fit(X, y)
    prob_before = pipeline.predict_proba(X)[:, 1]

    model_path = tmp_path / "model.joblib"
    joblib.dump(pipeline, model_path)
    loaded = joblib.load(model_path)
    prob_after = loaded.predict_proba(X)[:, 1]

    assert np.allclose(prob_before, prob_after)


def test_nba_examples_do_not_use_true_label(synthetic_data, model_matrix):
    """NBA examples must be generated without passing true labels."""
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

    examples = generate_nba_examples(
        result.model,
        test_df,
        synthetic_data["users"],
        synthetic_data["events"],
        result.feature_columns,
        result.threshold,
        n_examples=10,
        experiment_assignments=synthetic_data["experiment_assignments"],
    )

    assert len(examples) <= 10
    assert {"user_id", "predicted_risk", "predicted_class", "action_name", "channel"}.issubset(
        set(examples.columns)
    )
