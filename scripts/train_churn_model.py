"""Train and evaluate a churn prediction model for Career Growth Analytics."""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import joblib
import numpy as np
import pandas as pd
import sklearn

from career_growth import config
from career_growth.data_generation.generator import generate_all_data
from career_growth.features.model_features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    prepare_model_matrix,
    save_model_features,
)
from career_growth.modeling.evaluate import (
    compute_calibration_data,
    compute_confusion_matrix,
    compute_metrics,
)
from career_growth.modeling.explain import (
    build_global_explanation,
    explain_user_prediction,
)
from career_growth.modeling.nba_integration import generate_nba_examples
from career_growth.modeling.split import split_users_and_labels
from career_growth.modeling.subgroup import evaluate_subgroups
from career_growth.modeling.train import save_model, train_and_select_model


DEFAULT_OUTPUT_DIR: str = "artifacts"
PLOT_DIR: str = "plots"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train churn prediction models and save evaluation artifacts."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=config.DEFAULT_USER_COUNT,
        help="Number of synthetic users to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.RANDOM_SEED,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/training",
        help="Directory containing generated data or used for output. Defaults to 'data/training' so that the formal sample data under 'data/sample' is never overwritten.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for model artifacts and metrics.",
    )
    parser.add_argument(
        "--threshold-criterion",
        type=str,
        default="f1",
        choices=["f1", "precision", "recall", "f2", "youden"],
        help="Criterion used to select the operating threshold on validation data.",
    )
    parser.add_argument(
        "--use-existing-data",
        action="store_true",
        help="Load existing CSV files from data-dir instead of regenerating them.",
    )
    return parser.parse_args(argv)


def ensure_data(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    """Generate or load synthetic data."""
    data_dir = Path(args.data_dir)
    sample_dir = data_dir / "sample"
    processed_dir = data_dir / "processed"

    if args.use_existing_data and (sample_dir / "users.csv").exists():
        print(f"Loading existing data from {data_dir}")
        return {
            "users": pd.read_csv(sample_dir / "users.csv"),
            "events": pd.read_csv(sample_dir / "events.csv"),
            "experiment_assignments": pd.read_csv(sample_dir / "experiment_assignments.csv"),
            "interventions": pd.read_csv(sample_dir / "interventions.csv"),
            "labels": pd.read_csv(processed_dir / "labels.csv"),
        }

    print(f"Generating {args.count} users with seed {args.seed}")
    return generate_all_data(count=args.count, seed=args.seed, output_dir=args.data_dir)


def _signup_range(df: pd.DataFrame) -> str:
    """Return an ISO-formatted signup timestamp range for a split."""
    min_ts = pd.to_datetime(df["signup_timestamp"]).min()
    max_ts = pd.to_datetime(df["signup_timestamp"]).max()
    return f"{min_ts.isoformat()} to {max_ts.isoformat()}"


def plot_precision_recall(
    y_true: np.ndarray, y_prob: np.ndarray, output_path: Path
) -> None:
    """Save a precision-recall curve plot."""
    from sklearn import metrics

    precision, recall, _ = metrics.precision_recall_curve(y_true, y_prob)
    pr_auc = metrics.average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"PR-AUC = {pr_auc:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve (Test Set)")
    ax.legend(loc="lower left")
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, output_path: Path) -> None:
    """Save an ROC curve plot."""
    from sklearn import metrics

    fpr, tpr, _ = metrics.roc_curve(y_true, y_prob)
    roc_auc = metrics.roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (Test Set)")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_calibration(
    y_true: np.ndarray, y_prob: np.ndarray, output_path: Path
) -> None:
    """Save a reliability diagram."""
    calibration = compute_calibration_data(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    ax.plot(
        calibration["mean_predicted"],
        calibration["mean_observed"],
        marker="o",
        label="Model",
    )
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Reliability Diagram (Test Set)")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(
    confusion: dict[str, int], output_path: Path
) -> None:
    """Save a confusion matrix heatmap."""
    cm = np.array(
        [
            [confusion["true_negative"], confusion["false_positive"]],
            [confusion["false_negative"], confusion["true_positive"]],
        ]
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted 0", "Predicted 1"])
    ax.set_yticklabels(["Actual 0", "Actual 1"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (Test Set)")

    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_risk_distribution(
    y_true: np.ndarray, y_prob: np.ndarray, output_path: Path
) -> None:
    """Save a histogram of predicted risk scores by actual label."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(
        y_prob[y_true == 0],
        bins=30,
        alpha=0.6,
        label="Retained",
        color="green",
    )
    ax.hist(
        y_prob[y_true == 1],
        bins=30,
        alpha=0.6,
        label="Churned",
        color="red",
    )
    ax.set_xlabel("Predicted Churn Probability")
    ax.set_ylabel("Count")
    ax.set_title("Test Set Risk Distribution")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_feature_importance(importance_df: pd.DataFrame, output_path: Path) -> None:
    """Save a horizontal bar plot of the top feature importances."""
    top_n = min(15, len(importance_df))
    top_features = importance_df.head(top_n).sort_values(by="importance_mean")

    fig, ax = plt.subplots(figsize=(7, 8))
    ax.barh(
        top_features["feature"],
        top_features["importance_mean"],
        xerr=top_features.get("importance_std", 0),
    )
    ax.set_xlabel("Permutation Importance (PR-AUC)")
    ax.set_title("Top 15 Feature Importances (Validation Set)")
    ax.grid(True, axis="x", linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> int:
    """Run the full churn-model training pipeline."""
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / PLOT_DIR
    plot_dir.mkdir(parents=True, exist_ok=True)

    data = ensure_data(args)
    users = data["users"]
    events = data["events"]
    labels = data["labels"]
    experiment_assignments = data["experiment_assignments"]

    print("Building pre-cutoff features and attaching labels...")
    model_matrix = prepare_model_matrix(users, events, labels, experiment_assignments)
    print(f"Model matrix shape: {model_matrix.shape}")
    print(f"Churn rate: {model_matrix['is_churned'].mean():.2%}")

    save_model_features(
        model_matrix,
        str(Path(args.data_dir) / "processed" / "model_features.csv"),
    )

    train_users, val_users, test_users, train_labels, val_labels, test_labels = (
        split_users_and_labels(users, labels)
    )

    train_df = model_matrix[
        model_matrix["user_id"].isin(train_users["user_id"])
    ].reset_index(drop=True)
    val_df = model_matrix[
        model_matrix["user_id"].isin(val_users["user_id"])
    ].reset_index(drop=True)
    test_df = model_matrix[
        model_matrix["user_id"].isin(test_users["user_id"])
    ].reset_index(drop=True)

    print(
        f"Train/Val/Test sizes: {len(train_df)} / {len(val_df)} / {len(test_df)}"
    )

    result = train_and_select_model(
        train_df,
        val_df,
        test_df,
        threshold_criterion=args.threshold_criterion,
        random_state=args.seed,
    )

    print(f"Selected model: {result.model_name}")
    print(f"Validation metrics: {result.val_metrics}")
    print(f"Test metrics: {result.test_metrics}")

    # Save artifacts
    model_path = output_dir / "churn_model.joblib"
    save_model(result, str(model_path))
    print(f"Saved model to {model_path}")

    confusion = compute_confusion_matrix(
        test_df["is_churned"].to_numpy(), result.test_probabilities, result.threshold
    )

    metadata = {
        "model_name": result.model_name,
        "model_version": "0.2.0",
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "prediction_cutoff_day": config.PREDICTION_CUTOFF_DAY,
        "label_window_start_day": config.LABEL_WINDOW_START_DAY,
        "label_window_end_day": config.LABEL_WINDOW_END_DAY,
        "feature_columns": result.feature_columns,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "selected_threshold": result.threshold,
        "threshold_criterion": args.threshold_criterion,
        "train_signup_range": _signup_range(train_df),
        "validation_signup_range": _signup_range(val_df),
        "test_signup_range": _signup_range(test_df),
        "train_size": len(train_df),
        "validation_size": len(val_df),
        "test_size": len(test_df),
        "train_churn_rate": float(train_df["is_churned"].mean()),
        "validation_churn_rate": float(val_df["is_churned"].mean()),
        "test_churn_rate": float(test_df["is_churned"].mean()),
        "overall_churn_rate": float(model_matrix["is_churned"].mean()),
        "random_seed": args.seed,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "scikit_learn_version": sklearn.__version__,
    }
    with open(output_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    metrics_payload = {
        "candidate_validation_metrics": result.candidate_validation_metrics,
        "selected_model": result.model_name,
        "selected_threshold": result.threshold,
        "validation": result.val_metrics,
        "test": result.test_metrics,
        "confusion_matrix": confusion,
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    feature_schema = {
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
    }
    with open(output_dir / "feature_schema.json", "w", encoding="utf-8") as f:
        json.dump(feature_schema, f, indent=2)

    # Explainability
    X_val = val_df[result.feature_columns]
    global_explanation = build_global_explanation(
        result.model,
        X_val,
        val_df["is_churned"].to_numpy(),
        result.feature_columns,
        top_n=15,
    )

    user_explanations = []
    test_sample = test_df.sample(n=min(3, len(test_df)), random_state=args.seed).reset_index(drop=True)
    for _, row in test_sample.iterrows():
        uid = row["user_id"]
        x_row = test_df[test_df["user_id"] == uid][result.feature_columns].iloc[[0]]
        explanation = explain_user_prediction(
            result.model, x_row, result.feature_columns, top_n=5
        )
        prob = float(result.test_probabilities[test_df["user_id"] == uid].mean())
        user_explanations.append(
            {
                "user_id": uid,
                "predicted_risk": round(prob, 6),
                "predicted_class": int(prob >= result.threshold),
                "actual_label": int(row["is_churned"]),
                "positive_factors": explanation["positive_factors"],
                "negative_factors": explanation["negative_factors"],
                "disclaimer": "Associations captured by the model, not causal effects.",
            }
        )

    explainability = {
        **global_explanation,
        "top_15_features": global_explanation.get("permutation_importance", []),
        "user_explanations": user_explanations,
    }
    with open(output_dir / "explainability.json", "w", encoding="utf-8") as f:
        json.dump(explainability, f, indent=2)

    with open(output_dir / "user_explanations.json", "w", encoding="utf-8") as f:
        json.dump(user_explanations, f, indent=2)

    # Subgroup evaluation
    subgroup_df = evaluate_subgroups(
        test_df, result.test_probabilities, result.threshold
    )
    subgroup_df.to_csv(output_dir / "subgroup_metrics.csv", index=False)
    subgroup_df.to_json(output_dir / "subgroup_metrics.json", orient="records", indent=2)

    # Next Best Action examples
    nba_examples = generate_nba_examples(
        result.model,
        test_df,
        users,
        events,
        result.feature_columns,
        result.threshold,
        n_examples=10,
        experiment_assignments=experiment_assignments,
    )
    nba_examples.to_csv(output_dir / "nba_examples.csv", index=False)
    nba_examples.to_json(output_dir / "nba_examples.json", orient="records", indent=2)

    # Plots
    y_test = test_df["is_churned"].to_numpy()
    plot_precision_recall(y_test, result.test_probabilities, plot_dir / "pr_curve.png")
    plot_roc_curve(y_test, result.test_probabilities, plot_dir / "roc_curve.png")
    plot_calibration(y_test, result.test_probabilities, plot_dir / "calibration.png")
    plot_confusion_matrix(confusion, plot_dir / "confusion_matrix.png")
    plot_risk_distribution(y_test, result.test_probabilities, plot_dir / "risk_distribution.png")
    plot_feature_importance(
        pd.DataFrame(global_explanation["permutation_importance"]),
        plot_dir / "feature_importance.png",
    )

    print(f"Saved plots to {plot_dir}")
    print("Training complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
