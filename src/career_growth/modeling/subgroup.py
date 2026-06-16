"""Subgroup evaluation for churn predictions."""

import numpy as np
import pandas as pd
from sklearn import metrics


SMALL_SAMPLE_THRESHOLD: int = 50


def evaluate_subgroups(
    df: pd.DataFrame,
    y_prob: np.ndarray,
    threshold: float,
    group_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Compute test-set metrics for demographic and behavioral subgroups.

    Parameters
    ----------
    df: Test DataFrame containing feature columns and ``is_churned`` label.
    y_prob: Predicted probabilities for the test set.
    threshold: Operating threshold used to derive predicted class.
    group_columns: Columns to group by. Defaults to acquisition_channel,
        career_stage, and device_type.

    Returns
    -------
    DataFrame with one row per group and columns for sample_size, churn_rate,
    precision, recall, f1_score, and predicted_positive_rate.
    """
    if group_columns is None:
        group_columns = ["acquisition_channel", "career_stage", "device_type"]

    df = df.copy().reset_index(drop=True)
    df["y_prob"] = y_prob
    df["y_pred"] = (y_prob >= threshold).astype(int)

    records = []
    for col in group_columns:
        for group_value, group_df in df.groupby(col, observed=True):
            y_true = group_df["is_churned"].to_numpy()
            y_pred = group_df["y_pred"].to_numpy()
            sample_size = len(group_df)
            records.append(
                {
                    "group_column": col,
                    "group_value": str(group_value),
                    "sample_size": sample_size,
                    "churn_rate": float(y_true.mean()),
                    "precision": float(
                        metrics.precision_score(y_true, y_pred, zero_division=0)
                    ),
                    "recall": float(
                        metrics.recall_score(y_true, y_pred, zero_division=0)
                    ),
                    "f1_score": float(
                        metrics.f1_score(y_true, y_pred, zero_division=0)
                    ),
                    "predicted_positive_rate": float(y_pred.mean()),
                    "small_sample": sample_size < SMALL_SAMPLE_THRESHOLD,
                }
            )

    return pd.DataFrame(records)
