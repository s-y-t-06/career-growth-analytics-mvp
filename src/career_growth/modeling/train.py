"""Train and select churn prediction models."""

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from career_growth import config
from career_growth.modeling.evaluate import compute_metrics, select_threshold
from career_growth.modeling.pipeline import (
    build_hist_gradient_boosting_pipeline,
    build_logistic_regression_pipeline,
)


TARGET_COLUMN: str = "is_churned"
ID_COLUMNS: set[str] = {"user_id", "signup_timestamp"}


def _split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Split a modeling matrix into feature matrix X and label vector y."""
    feature_cols = [c for c in df.columns if c not in ID_COLUMNS and c != TARGET_COLUMN]
    X = df[feature_cols].reset_index(drop=True)
    y = df[TARGET_COLUMN].astype(int).to_numpy()
    return X, y


@dataclass
class TrainingResult:
    """Container for a trained model and its evaluation artifacts."""

    model: Pipeline
    model_name: str
    candidate_validation_metrics: dict[str, dict[str, float]]
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]
    threshold: float
    feature_columns: list[str]
    val_probabilities: np.ndarray
    test_probabilities: np.ndarray


def train_and_select_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    threshold_criterion: str = "f1",
    random_state: int = config.RANDOM_SEED,
) -> TrainingResult:
    """Train baseline and gradient-boosting models and select the best by PR-AUC.

    Model selection is performed using the validation set PR-AUC. The operating
    threshold is chosen on the validation set using ``threshold_criterion``.
    Final test metrics are computed exactly once for the selected model.

    Parameters
    ----------
    train_df: Training matrix with features and ``is_churned`` label.
    val_df: Validation matrix with features and ``is_churned`` label.
    test_df: Test matrix with features and ``is_churned`` label.
    threshold_criterion: Criterion passed to ``select_threshold``.
    random_state: Random seed for reproducibility.

    Returns
    -------
    A ``TrainingResult`` with the best model and all metrics.
    """
    X_train, y_train = _split_xy(train_df)
    X_val, y_val = _split_xy(val_df)
    X_test, y_test = _split_xy(test_df)

    candidates: dict[str, Pipeline] = {
        "logistic_regression": build_logistic_regression_pipeline(random_state),
        "hist_gradient_boosting": build_hist_gradient_boosting_pipeline(random_state),
    }

    candidate_validation_metrics: dict[str, dict[str, float]] = {}
    best_model_name: str | None = None
    best_pr_auc: float = -np.inf
    best_model: Pipeline | None = None
    best_val_metrics: dict[str, float] | None = None
    best_threshold: float = 0.5

    for name, pipeline in candidates.items():
        fitted = pipeline.fit(X_train, y_train)
        val_prob = fitted.predict_proba(X_val)[:, 1]
        val_metrics = compute_metrics(y_val, val_prob, threshold=0.5)
        candidate_validation_metrics[name] = val_metrics

        if val_metrics["pr_auc"] > best_pr_auc:
            best_pr_auc = val_metrics["pr_auc"]
            best_model_name = name
            best_model = fitted
            best_val_metrics = val_metrics
            best_threshold = select_threshold(y_val, val_prob, criterion=threshold_criterion)

    if best_model is None or best_model_name is None or best_val_metrics is None:
        raise RuntimeError("No model was successfully trained.")

    val_prob = best_model.predict_proba(X_val)[:, 1]
    test_prob = best_model.predict_proba(X_test)[:, 1]

    val_metrics = compute_metrics(y_val, val_prob, threshold=best_threshold)
    test_metrics = compute_metrics(y_test, test_prob, threshold=best_threshold)

    return TrainingResult(
        model=best_model,
        model_name=best_model_name,
        candidate_validation_metrics=candidate_validation_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        threshold=best_threshold,
        feature_columns=X_train.columns.tolist(),
        val_probabilities=val_prob,
        test_probabilities=test_prob,
    )


def save_model(result: TrainingResult, path: str) -> None:
    """Persist a trained model to disk with joblib."""
    joblib.dump(result.model, path)
