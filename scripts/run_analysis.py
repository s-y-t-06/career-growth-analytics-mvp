"""CLI script to run the full MVP analytics pipeline."""

import json

import pandas as pd

from career_growth.analytics.experiments import analyze_experiment
from career_growth.analytics.funnel import compute_funnel
from career_growth.analytics.retention import (
    compute_cohort_retention,
    compute_day_retention,
    compute_rolling_retention,
)
from career_growth.config import ONBOARDING_EXPERIMENT_ID, PREDICTION_CUTOFF_DAY
from career_growth.data_generation.experiments import create_experiment_assignments
from career_growth.decisions.next_best_action import recommend_next_action
from career_growth.validation.validator import DataValidator


def main() -> None:
    users = pd.read_csv("data/sample/users.csv", parse_dates=["signup_timestamp"])
    events = pd.read_csv("data/sample/events.csv", parse_dates=["event_timestamp"])
    experiment_assignments = pd.read_csv(
        "data/sample/experiment_assignments.csv", parse_dates=["assignment_time"]
    )
    labels = pd.read_csv("data/processed/labels.csv", parse_dates=["label_start", "label_end"])

    # Validation.
    validator = DataValidator("data")
    validator.users = users
    validator.events = events
    validator.experiment_assignments = experiment_assignments
    validator.interventions = pd.read_csv("data/sample/interventions.csv", parse_dates=["send_time"])
    validator.labels = labels
    report = validator.validate()
    print("Validation passed:", report.passed)
    if report.errors:
        print("Errors:", report.errors)
    if report.warnings:
        print("Warnings:", report.warnings)

    # Funnel.
    funnel = compute_funnel(users, events)
    print("\n=== Core Funnel ===")
    print(funnel.to_string(index=False))

    # Retention.
    print("\n=== Retention ===")
    for day in [1, 7, 14]:
        rate = compute_day_retention(users, events, day)["retention_rate"].iloc[0]
        print(f"D{day} retention: {rate:.2%}")
    rolling = compute_rolling_retention(users, events, PREDICTION_CUTOFF_DAY)
    print(f"D{PREDICTION_CUTOFF_DAY} rolling retention: {rolling:.2%}")

    # Cohort retention by signup week.
    cohort_retention = compute_cohort_retention(users, events, days=[1, 7, 14])
    print("\n=== Cohort Retention (signup week) ===")
    print(cohort_retention.to_string(index=False))

    # Experiment analysis.
    experiment_results = analyze_experiment(users, events, experiment_assignments, ONBOARDING_EXPERIMENT_ID)
    print("\n=== Experiment Analysis ===")
    print(json.dumps(experiment_results, indent=2, default=str))

    # Next best action sample.
    sample_users = users.sample(5, random_state=42)
    print("\n=== Next Best Action Sample ===")
    for _, user in sample_users.iterrows():
        recommendation = recommend_next_action(user, events)
        print(f"{user['user_id']}: {recommendation['action_name']} ({recommendation['channel']})")


if __name__ == "__main__":
    main()
