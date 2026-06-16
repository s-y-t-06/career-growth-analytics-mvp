"""Main data generation orchestrator."""

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from career_growth import config
from career_growth.data_generation.events import generate_events_for_user
from career_growth.data_generation.experiments import create_experiment_assignments
from career_growth.data_generation.interventions import generate_interventions
from career_growth.data_generation.users import generate_users
from career_growth.features.labels import build_labels


def generate_all_data(
    count: int = config.DEFAULT_USER_COUNT,
    seed: int = config.RANDOM_SEED,
    output_dir: str = "data",
) -> dict[str, pd.DataFrame]:
    """Generate the complete MVP synthetic dataset.

    Returns a dictionary containing raw and processed DataFrames and writes
    CSV files to `output_dir/sample/` and `output_dir/processed/`.
    """
    rng = np.random.default_rng(seed)

    # 1. Users.
    users = generate_users(count, rng, seed=seed)

    # 2. Experiment assignments.
    experiment_assignments = create_experiment_assignments(users)
    assignment_by_user = {
        row["user_id"]: row.to_dict() for _, row in experiment_assignments.iterrows()
    }

    # 3. Event streams.
    all_events: list[dict[str, Any]] = []
    for _, user_row in users.iterrows():
        assignment = assignment_by_user[user_row["user_id"]]
        user_events = generate_events_for_user(user_row, assignment, rng, seed=seed)
        all_events.extend(user_events)
    events = pd.DataFrame(all_events)

    # Users: drop hidden propensity columns before saving or passing downstream.
    hidden_cols = [
        "intrinsic_engagement",
        "career_urgency",
        "product_fit",
        "notification_sensitivity",
        "intent_score",
        "career_stage_score",
        "device_score",
    ]
    users_public = users.drop(columns=hidden_cols)

    # 4. Labels (must be built before interventions so win-back logic is consistent).
    labels = build_labels(users, events)

    # 5. Interventions.
    interventions = generate_interventions(users_public, events, labels, rng, seed=seed)

    # 6. Update experiment conversions (onboarding completion).
    onboarding_users = set(
        events[
            (events["event_name"] == "onboarding_complete")
            & (events["event_source"] == "user_action")
        ]["user_id"]
    )
    experiment_assignments["is_converted"] = experiment_assignments["user_id"].isin(
        onboarding_users
    )

    # 7. Persist.
    sample_dir = Path(output_dir) / "sample"
    processed_dir = Path(output_dir) / "processed"
    sample_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    users_public.to_csv(sample_dir / "users.csv", index=False)
    events.to_csv(sample_dir / "events.csv", index=False)
    experiment_assignments.to_csv(sample_dir / "experiment_assignments.csv", index=False)
    interventions.to_csv(sample_dir / "interventions.csv", index=False)
    labels.to_csv(processed_dir / "labels.csv", index=False)

    # Save generation metadata.
    metadata = {
        "user_count": count,
        "seed": seed,
        "event_count": len(events),
        "churn_rate": float(labels["is_churned"].mean()),
        "experiment_id": config.ONBOARDING_EXPERIMENT_ID,
    }
    with open(sample_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    return {
        "users": users_public,
        "events": events,
        "experiment_assignments": experiment_assignments,
        "interventions": interventions,
        "labels": labels,
    }


if __name__ == "__main__":
    data = generate_all_data()
    print(f"Generated {len(data['users'])} users, {len(data['events'])} events.")
    print(f"Churn rate: {data['labels']['is_churned'].mean():.2%}")
