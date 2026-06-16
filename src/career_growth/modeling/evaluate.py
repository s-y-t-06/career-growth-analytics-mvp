"""Model evaluation metrics and threshold selection."""

import numpy as np
import pandas as pd
from sklearn import metrics


def compute_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> dict[str, float]:
    """Compute classification metrics for a fixed probability threshold."""
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "pr_auc": float(metrics.average_precision_score(y_true, y_prob)),
        "roc_auc": float(metrics.roc_auc_score(y_true, y_prob)),
        "log_loss": float(metrics.log_loss(y_true, y_prob)),
        "brier_score": float(metrics.brier_score_loss(y_true, y_prob)),
        "threshold": float(threshold),
        "precision": float(metrics.precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(metrics.recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(metrics.f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(metrics.accuracy_score(y_true, y_pred)),
    }


def select_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    criterion: str = "f1",
    n_thresholds: int = 101,
) -> float:
    """Select an operating threshold on the validation set.

    Parameters
    ----------
    y_true: Ground-truth binary labels.
    y_prob: Predicted probabilities of the positive class.
    criterion: Metric to maximize. Supported values are ``f1``, ``precision``,
        ``recall``, ``f2``, and ``youden``.
    n_thresholds: Number of threshold candidates between 0 and 1.

    Returns
    -------
    The threshold that maximizes the chosen criterion.
    """
    if criterion == "youden":
        fpr, tpr, roc_thresholds = metrics.roc_curve(y_true, y_prob)
        youden_index = tpr - fpr
        best_idx = int(np.argmax(youden_index))
        return float(roc_thresholds[best_idx])

    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    best_threshold = 0.5
    best_score = -np.inf

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        if criterion == "f1":
            score = metrics.f1_score(y_true, y_pred, zero_division=0)
        elif criterion == "precision":
            score = metrics.precision_score(y_true, y_pred, zero_division=0)
        elif criterion == "recall":
            score = metrics.recall_score(y_true, y_pred, zero_division=0)
        elif criterion == "f2":
            score = metrics.fbeta_score(y_true, y_pred, beta=2, zero_division=0)
        else:
            raise ValueError(f"Unsupported threshold criterion: {criterion}")

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return float(best_threshold)


def compute_confusion_matrix(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> dict[str, int]:
    """Return confusion matrix counts for a fixed threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    return {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def compute_calibration_data(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> dict[str, list[float]]:
    """Compute binned calibration data for a reliability diagram.

    Bins are defined by predicted probability quantiles so that each bin has a
    comparable number of observations.
    """
    df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    df["bin"] = pd.qcut(df["y_prob"], q=n_bins, duplicates="drop")

    grouped = df.groupby("bin", observed=True).agg(
        mean_predicted=("y_prob", "mean"),
        mean_observed=("y_true", "mean"),
        count=("y_true", "size"),
    )
    grouped = grouped[grouped["count"] >= 5]

    return {
        "mean_predicted": grouped["mean_predicted"].tolist(),
        "mean_observed": grouped["mean_observed"].tolist(),
        "count": grouped["count"].tolist(),
    }
