"""Tests for analytics modules."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chisquare

from career_growth.analytics.experiments import analyze_experiment
from career_growth.analytics.funnel import compute_funnel
from career_growth.analytics.retention import (
    compute_cohort_retention,
    compute_day_retention,
    compute_retention_by_variant,
    compute_rolling_retention,
)
from career_growth.config import ONBOARDING_EXPERIMENT_ID, ONBOARDING_VARIANTS


def test_funnel_monotonic(synthetic_data):
    funnel = compute_funnel(synthetic_data["users"], synthetic_data["events"])
    users = funnel["users"].values
    assert (users[:-1] >= users[1:]).all()
    assert funnel["conversion_rate"].between(0.0, 1.0).all()


def test_funnel_group_by_channel(synthetic_data):
    funnel = compute_funnel(
        synthetic_data["users"], synthetic_data["events"], group_by="acquisition_channel"
    )
    assert "acquisition_channel" in funnel.columns
    assert len(funnel) == len(synthetic_data["users"]["acquisition_channel"].unique()) * 7


def test_retention_rates_bounded(synthetic_data):
    for day in [1, 7, 14]:
        df = compute_day_retention(synthetic_data["users"], synthetic_data["events"], day)
        assert 0.0 <= df["retention_rate"].iloc[0] <= 1.0


def test_rolling_retention_bounded(synthetic_data):
    rate = compute_rolling_retention(synthetic_data["users"], synthetic_data["events"], 7)
    assert 0.0 <= rate <= 1.0


def test_cohort_retention_shape(synthetic_data):
    cohort = compute_cohort_retention(
        synthetic_data["users"], synthetic_data["events"], days=[1, 7, 14]
    )
    assert "signup_week" in cohort.columns
    assert "retention_rate" in cohort.columns
    assert cohort["retention_rate"].between(0.0, 1.0).all()


def test_cohort_retention_not_all_zero(synthetic_data):
    cohort = compute_cohort_retention(
        synthetic_data["users"], synthetic_data["events"], days=[1, 7, 14]
    )
    for day in [1, 7, 14]:
        day_rates = cohort[cohort["day"] == day]["retention_rate"]
        assert (day_rates > 0).any(), f"all cohort retention rates are zero for day {day}"


def test_cohort_retention_consistent_with_overall(synthetic_data):
    """Weighted average of cohort retention should match overall day retention."""
    days = [1, 7, 14]
    cohort = compute_cohort_retention(
        synthetic_data["users"], synthetic_data["events"], days=days
    )
    for day in days:
        overall = compute_day_retention(
            synthetic_data["users"], synthetic_data["events"], day
        )["retention_rate"].iloc[0]
        sub = cohort[cohort["day"] == day]
        weighted = (sub["retention_rate"] * sub["users"]).sum() / sub["users"].sum()
        assert np.isclose(weighted, overall, atol=0.001), (
            f"cohert weighted retention {weighted} != overall {overall} for day {day}"
        )


def test_retention_by_variant(synthetic_data):
    df = compute_retention_by_variant(
        synthetic_data["users"],
        synthetic_data["events"],
        synthetic_data["experiment_assignments"],
        day=7,
    )
    assert "variant_id" in df.columns
    assert set(df["variant_id"]) >= {"control", "personalized", "simplified"}
    assert df["retention_rate"].between(0.0, 1.0).all()


def test_experiment_srm_not_significant(synthetic_data):
    results = analyze_experiment(
        synthetic_data["users"],
        synthetic_data["events"],
        synthetic_data["experiment_assignments"],
        ONBOARDING_EXPERIMENT_ID,
    )
    assert results["srm_p_value"] >= 0.01


def test_experiment_srm_matches_scipy(synthetic_data):
    """SRM p-value must equal scipy.stats.chisquare on the same counts."""
    results = analyze_experiment(
        synthetic_data["users"],
        synthetic_data["events"],
        synthetic_data["experiment_assignments"],
        ONBOARDING_EXPERIMENT_ID,
    )

    assignments = synthetic_data["experiment_assignments"]
    observed = assignments["variant_id"].value_counts().sort_index().values
    expected_allocations = {v["variant_id"]: v["allocation"] for v in ONBOARDING_VARIANTS}
    expected = np.array(
        [expected_allocations[v] * len(assignments) for v in sorted(expected_allocations)]
    )
    _, scipy_p = chisquare(observed, f_exp=expected)
    assert np.isclose(results["srm_p_value"], scipy_p, rtol=1e-10), (
        f"SRM p-value {results['srm_p_value']} differs from scipy {scipy_p}"
    )


def test_experiment_metrics_have_variants(synthetic_data):
    results = analyze_experiment(
        synthetic_data["users"],
        synthetic_data["events"],
        synthetic_data["experiment_assignments"],
        ONBOARDING_EXPERIMENT_ID,
    )
    assert "metrics" in results
    for metric_name, metric_results in results["metrics"].items():
        assert len(metric_results) == 3
        control = next(r for r in metric_results if r["variant_id"] == "control")
        assert 0.0 <= control["conversion_rate"] <= 1.0
