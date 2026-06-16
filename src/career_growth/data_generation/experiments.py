"""Stable experiment assignment for synthetic users."""

import hashlib

import pandas as pd

from career_growth import config


def assign_variant(user_id: str, experiment_id: str, variants: list[dict], salt: str = "") -> dict:
    """Deterministically assign a user to an experiment variant.

    Uses MD5 hash of user_id + experiment_id + salt to ensure reproducibility
    and user-level stability.
    """
    raw = f"{user_id}:{experiment_id}:{salt}".encode("utf-8")
    hash_value = int(hashlib.md5(raw).hexdigest(), 16)
    normalized = (hash_value % 1_000_000) / 1_000_000.0

    cumulative = 0.0
    for variant in variants:
        cumulative += variant["allocation"]
        if normalized < cumulative:
            return variant
    return variants[-1]


def create_experiment_assignments(
    users: pd.DataFrame,
    experiment_id: str = config.ONBOARDING_EXPERIMENT_ID,
    experiment_name: str = config.ONBOARDING_EXPERIMENT_NAME,
    experiment_type: str = "onboarding",
    variants: list[dict] | None = None,
) -> pd.DataFrame:
    """Create a stable experiment assignment row for each user."""
    if variants is None:
        variants = config.ONBOARDING_VARIANTS

    rows = []
    for _, user in users.iterrows():
        variant = assign_variant(user["user_id"], experiment_id, variants)
        rows.append(
            {
                "experiment_id": experiment_id,
                "user_id": user["user_id"],
                "variant_id": variant["variant_id"],
                "assignment_time": user["signup_timestamp"],
                "experiment_name": experiment_name,
                "experiment_type": experiment_type,
                "traffic_allocation": variant["allocation"],
                "is_exposed": True,
                "is_converted": False,
            }
        )

    return pd.DataFrame(rows)
