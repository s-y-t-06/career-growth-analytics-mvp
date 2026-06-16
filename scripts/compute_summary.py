"""Print a concise summary of the generated dataset."""

import pandas as pd

from career_growth.analytics.experiments import analyze_experiment
from career_growth.analytics.funnel import compute_funnel
from career_growth.analytics.retention import (
    compute_day_retention,
    compute_rolling_retention,
)
from career_growth.config import ONBOARDING_EXPERIMENT_ID, PREDICTION_CUTOFF_DAY


def main() -> None:
    users = pd.read_csv("data/sample/users.csv", parse_dates=["signup_timestamp"])
    events = pd.read_csv("data/sample/events.csv", parse_dates=["event_timestamp"])
    labels = pd.read_csv("data/processed/labels.csv")
    experiment_assignments = pd.read_csv("data/sample/experiment_assignments.csv")

    print("=== Dataset Summary ===")
    print(f"Users: {len(users):,}")
    print(f"Events: {len(events):,}")
    print(f"Events per user: {len(events) / len(users):.1f}")
    print(f"Churn rate: {labels['is_churned'].mean():.2%}")
    print()

    print("=== Retention ===")
    for day in [1, 7, 14]:
        rate = compute_day_retention(users, events, day)["retention_rate"].iloc[0]
        print(f"D{day} retention: {rate:.2%}")
    print(f"D{PREDICTION_CUTOFF_DAY} rolling retention: {compute_rolling_retention(users, events, PREDICTION_CUTOFF_DAY):.2%}")
    print()

    print("=== Core Funnel ===")
    funnel = compute_funnel(users, events)
    print(funnel.to_string(index=False))
    print()

    print("=== Onboarding Experiment ===")
    results = analyze_experiment(
        users, events, experiment_assignments, ONBOARDING_EXPERIMENT_ID
    )
    print(f"SRM p-value: {results['srm_p_value']:.4f}")
    for metric_name, metric_results in results["metrics"].items():
        print(f"{metric_name}:")
        for r in metric_results:
            lift = r.get("relative_lift")
            lift_str = f"{lift:+.1%}" if lift is not None else "control"
            print(
                f"  {r['variant_id']:<12s} n={r['sample_size']:<5d} "
                f"rate={r['conversion_rate']:.2%} lift={lift_str:<8s} "
                f"p={r.get('p_value')}"
            )


if __name__ == "__main__":
    main()
