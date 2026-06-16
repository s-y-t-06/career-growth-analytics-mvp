"""A/B experiment analysis utilities."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chisquare, norm

from career_growth import config


@dataclass
class ExperimentMetricResult:
    """Result of a single metric comparison between two variants."""

    variant_id: str
    sample_size: int
    conversions: int
    conversion_rate: float
    absolute_lift: float | None = None
    relative_lift: float | None = None
    p_value: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None


def two_proportion_z_test(
    conversions_a: int,
    n_a: int,
    conversions_b: int,
    n_b: int,
) -> tuple[float, float, float, float]:
    """Return z-statistic, p-value, absolute lift, and 95% CI for the difference."""
    p_a = conversions_a / n_a if n_a > 0 else 0.0
    p_b = conversions_b / n_b if n_b > 0 else 0.0
    p_pooled = (conversions_a + conversions_b) / (n_a + n_b) if (n_a + n_b) > 0 else 0.0

    se = np.sqrt(p_pooled * (1 - p_pooled) * (1 / n_a + 1 / n_b))
    z = (p_b - p_a) / se if se > 0 else 0.0
    p_value = 2 * (1 - norm.cdf(abs(z)))

    se_diff = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    ci_lower = (p_b - p_a) - 1.96 * se_diff
    ci_upper = (p_b - p_a) + 1.96 * se_diff

    return z, p_value, ci_lower, ci_upper


def analyze_experiment(
    users: pd.DataFrame,
    events: pd.DataFrame,
    experiment_assignments: pd.DataFrame,
    experiment_id: str,
    metrics: dict[str, str] | None = None,
    control_variant: str = "control",
) -> dict[str, Any]:
    """Analyze an A/B experiment and return metric comparisons.

    Args:
        users: users DataFrame.
        events: events DataFrame.
        experiment_assignments: experiment assignments DataFrame.
        experiment_id: experiment to analyze.
        metrics: mapping from metric name to event name used as conversion.
            Defaults to onboarding_completion, profile_completion, d7_retention.
        control_variant: variant used as control.

    Returns:
        Dictionary with sample sizes, SRM test, and metric results per variant.
    """
    if metrics is None:
        metrics = {
            "onboarding_completion_rate": "onboarding_complete",
            "profile_completion_rate": "profile_complete",
            "d7_retention_rate": "d7_retention",
        }

    subset = experiment_assignments[experiment_assignments["experiment_id"] == experiment_id].copy()
    subset = subset.merge(users[["user_id", "signup_timestamp"]], on="user_id", how="left")

    # Compute D7 retention per user.
    active_events = events[events["event_source"] == "user_action"].copy()
    active_events["event_date"] = active_events["event_timestamp"].dt.floor("D")
    signup_dates = subset.set_index("user_id")["signup_timestamp"].dt.floor("D")
    target_dates = signup_dates + pd.Timedelta(days=7)

    user_date_pairs = set(zip(active_events["user_id"], active_events["event_date"]))
    d7_retained = {
        uid: (uid, target_dates.get(uid, pd.NaT)) in user_date_pairs
        for uid in subset["user_id"]
    }

    # Compute event-based conversions per user.
    user_events = active_events.groupby("user_id")["event_name"].apply(set).to_dict()

    # SRM check using a one-way chi-squared goodness-of-fit test.
    observed = subset["variant_id"].value_counts().sort_index().values
    expected_allocations = {v["variant_id"]: v["allocation"] for v in config.ONBOARDING_VARIANTS}
    expected = np.array(
        [expected_allocations[v] * len(subset) for v in sorted(expected_allocations)]
    )
    chi2, srm_p_value = chisquare(observed, f_exp=expected)

    results = {
        "experiment_id": experiment_id,
        "sample_sizes": subset["variant_id"].value_counts().to_dict(),
        "srm_chi2": chi2,
        "srm_p_value": srm_p_value,
        "metrics": {},
    }

    control_subset = subset[subset["variant_id"] == control_variant]

    for metric_name, event_name in metrics.items():
        metric_results = []
        for variant_id in sorted(subset["variant_id"].unique()):
            variant_subset = subset[subset["variant_id"] == variant_id]
            n = len(variant_subset)

            if event_name == "d7_retention":
                conversions = sum(d7_retained.get(uid, False) for uid in variant_subset["user_id"])
            else:
                conversions = sum(
                    event_name in user_events.get(uid, set()) for uid in variant_subset["user_id"]
                )

            rate = conversions / n if n > 0 else 0.0
            result = ExperimentMetricResult(
                variant_id=variant_id,
                sample_size=n,
                conversions=conversions,
                conversion_rate=rate,
            )

            if variant_id != control_variant:
                control_n = len(control_subset)
                if event_name == "d7_retention":
                    control_conversions = sum(
                        d7_retained.get(uid, False) for uid in control_subset["user_id"]
                    )
                else:
                    control_conversions = sum(
                        event_name in user_events.get(uid, set())
                        for uid in control_subset["user_id"]
                    )

                control_rate = control_conversions / control_n if control_n > 0 else 0.0
                z, p_value, ci_lower, ci_upper = two_proportion_z_test(
                    control_conversions, control_n, conversions, n
                )
                result.absolute_lift = rate - control_rate
                result.relative_lift = (
                    (rate - control_rate) / control_rate if control_rate > 0 else None
                )
                result.p_value = p_value
                result.ci_lower = ci_lower
                result.ci_upper = ci_upper

            metric_results.append(result)

        results["metrics"][metric_name] = [
            {
                "variant_id": r.variant_id,
                "sample_size": r.sample_size,
                "conversions": r.conversions,
                "conversion_rate": r.conversion_rate,
                "absolute_lift": r.absolute_lift,
                "relative_lift": r.relative_lift,
                "p_value": r.p_value,
                "ci_lower": r.ci_lower,
                "ci_upper": r.ci_upper,
            }
            for r in metric_results
        ]

    return results
