"""Explainability helpers for churn models."""

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def extract_feature_names(pipeline, input_features: list[str] | None = None) -> list[str]:
    """Return the feature names produced by the pipeline preprocessor."""
    preprocessor = pipeline.named_steps["preprocessor"]
    try:
        feature_names = preprocessor.get_feature_names_out(input_features)
    except AttributeError:
        feature_names = np.array([f"feature_{i}" for i in range(preprocessor.transform(pd.DataFrame()).shape[1])])
    return [str(name) for name in feature_names]


def extract_logistic_coefficients(
    pipeline, input_features: list[str] | None = None
) -> pd.DataFrame:
    """Return a DataFrame of logistic-regression feature coefficients.

    The coefficients are returned sorted by absolute value in descending order.
    """
    feature_names = extract_feature_names(pipeline, input_features)
    classifier = pipeline.named_steps["classifier"]
    coefficients = classifier.coef_.flatten()

    df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "abs_coefficient": np.abs(coefficients),
        }
    )
    return df.sort_values(by="abs_coefficient", ascending=False).reset_index(drop=True)


def compute_permutation_importance(
    pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute permutation importance for any sklearn pipeline on a validation set.

    Returns a DataFrame with one row per input feature, sorted by mean importance.
    """
    result = permutation_importance(
        pipeline,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="average_precision",
        n_jobs=-1,
    )

    df = pd.DataFrame(
        {
            "feature": X.columns.tolist(),
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return df.sort_values(by="importance_mean", ascending=False).reset_index(drop=True)


def explain_user_prediction(
    pipeline,
    X_row: pd.DataFrame,
    input_features: list[str],
    top_n: int = 5,
) -> dict[str, list[dict[str, float]]]:
    """Explain a single prediction by identifying top driving features.

    For logistic regression the explanation uses signed feature contributions
    (value * coefficient). For tree-based models it uses one-at-a-time marginal
    contributions against the median-imputed baseline.

    These explanations describe associations captured by the model; they do not
    establish causal relationships.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    feature_names = extract_feature_names(pipeline, input_features)

    x_transformed = preprocessor.transform(X_row)
    base_prob = float(classifier.predict_proba(x_transformed)[0, 1])

    contributions: list[tuple[str, float]] = []

    if hasattr(classifier, "coef_"):
        coefficients = classifier.coef_.flatten()
        for name, value in zip(feature_names, x_transformed.flatten()):
            contributions.append((name, float(value * coefficients[feature_names.index(name)])))
    else:
        # Marginal contribution for tree-based models.
        for idx, name in enumerate(feature_names):
            baseline = x_transformed.copy()
            baseline[0, idx] = 0.0
            baseline_prob = float(classifier.predict_proba(baseline)[0, 1])
            contributions.append((name, base_prob - baseline_prob))

    contributions.sort(key=lambda item: item[1], reverse=True)
    positive = [
        {"feature": name, "contribution": round(value, 6)}
        for name, value in contributions[:top_n]
        if value > 0
    ]
    negative = [
        {"feature": name, "contribution": round(value, 6)}
        for name, value in contributions[-top_n:][::-1]
        if value < 0
    ]

    return {
        "predicted_risk": round(base_prob, 6),
        "positive_factors": positive,
        "negative_factors": negative,
    }


def build_global_explanation(
    pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    input_features: list[str],
    top_n: int = 15,
) -> dict[str, list[dict[str, float]]]:
    """Build a global explanation with logistic coefficients and permutation importance."""
    explanation: dict[str, list[dict[str, float]]] = {}

    if hasattr(pipeline.named_steps["classifier"], "coef_"):
        coef_df = extract_logistic_coefficients(pipeline, input_features)
        explanation["logistic_coefficients"] = coef_df.head(top_n).to_dict(
            orient="records"
        )
    else:
        explanation["logistic_coefficients"] = []

    perm_df = compute_permutation_importance(pipeline, X, y, n_repeats=10, random_state=42)
    explanation["permutation_importance"] = perm_df.head(top_n).to_dict(
        orient="records"
    )

    return explanation
