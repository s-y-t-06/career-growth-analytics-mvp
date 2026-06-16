"""Data quality validation for synthetic datasets."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd
from scipy.stats import chisquare

from career_growth import config, schemas


@dataclass
class ValidationReport:
    """Container for validation results."""

    passed: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)

    def add_error(self, message: str) -> None:
        self.errors.append(message)
        self.passed = False

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "errors": self.errors,
            "warnings": self.warnings,
            "checks": self.checks,
        }


class DataValidator:
    """Validate the synthetic MVP dataset against the defined schema and rules."""

    def __init__(self, data_dir: str = "data") -> None:
        self.data_dir = data_dir
        self.users: pd.DataFrame | None = None
        self.events: pd.DataFrame | None = None
        self.experiment_assignments: pd.DataFrame | None = None
        self.interventions: pd.DataFrame | None = None
        self.labels: pd.DataFrame | None = None

    def load(self) -> "DataValidator":
        sample_dir = f"{self.data_dir}/sample"
        processed_dir = f"{self.data_dir}/processed"
        self.users = pd.read_csv(f"{sample_dir}/users.csv", parse_dates=["signup_timestamp"])
        self.events = pd.read_csv(
            f"{sample_dir}/events.csv", parse_dates=["event_timestamp"]
        )
        self.experiment_assignments = pd.read_csv(
            f"{sample_dir}/experiment_assignments.csv", parse_dates=["assignment_time"]
        )
        self.interventions = pd.read_csv(
            f"{sample_dir}/interventions.csv", parse_dates=["send_time"]
        )
        self.labels = pd.read_csv(f"{processed_dir}/labels.csv", parse_dates=["label_start", "label_end"])
        return self

    def validate(self) -> ValidationReport:
        report = ValidationReport()
        self._validate_schema(report)
        self._validate_relationships(report)
        self._validate_temporal_rules(report)
        self._validate_experiment_srm(report)
        self._validate_label_quality(report)
        self._validate_data_distributions(report)
        return report

    def _validate_schema(self, report: ValidationReport) -> None:
        # Required columns.
        required_user_cols = {
            "user_id",
            "signup_timestamp",
            "acquisition_channel",
            "country",
            "device_type",
            "language",
            "timezone",
            "initial_plan_type",
            "user_intent_level",
            "career_stage",
            "marketing_consent",
        }
        missing = required_user_cols - set(self.users.columns)
        if missing:
            report.add_error(f"users.csv missing columns: {missing}")
        else:
            report.checks["users_columns"] = True

        required_event_cols = {
            "event_id",
            "user_id",
            "session_id",
            "event_name",
            "event_timestamp",
            "event_properties",
            "page_name",
            "platform",
            "event_source",
        }
        missing = required_event_cols - set(self.events.columns)
        if missing:
            report.add_error(f"events.csv missing columns: {missing}")
        else:
            report.checks["events_columns"] = True

        # Categorical value checks.
        invalid_channels = set(self.users["acquisition_channel"]) - set(config.ACQUISITION_CHANNELS)
        if invalid_channels:
            report.add_error(f"Invalid acquisition_channel values: {invalid_channels}")

        invalid_devices = set(self.users["device_type"]) - set(config.DEVICE_TYPES)
        if invalid_devices:
            report.add_error(f"Invalid device_type values: {invalid_devices}")

        invalid_event_names = set(self.events["event_name"]) - set(config.ALL_EVENT_NAMES)
        if invalid_event_names:
            report.add_error(f"Invalid event_name values: {invalid_event_names}")

        invalid_sources = set(self.events["event_source"]) - set(config.EVENT_SOURCES)
        if invalid_sources:
            report.add_error(f"Invalid event_source values: {invalid_sources}")

        # Duplicate event IDs.
        dup_events = self.events["event_id"].duplicated().sum()
        if dup_events > 0:
            report.add_error(f"Found {dup_events} duplicate event_id values")
        else:
            report.checks["unique_event_ids"] = True

        # Duplicate user IDs.
        dup_users = self.users["user_id"].duplicated().sum()
        if dup_users > 0:
            report.add_error(f"Found {dup_users} duplicate user_id values")
        else:
            report.checks["unique_user_ids"] = True

        # Row-level Pydantic validation on a sample to catch subtle type issues.
        sample_size = min(100, len(self.users))
        for _, row in self.users.head(sample_size).iterrows():
            try:
                schemas.User(**row.to_dict())
            except Exception as exc:
                report.add_error(f"User schema validation failed: {exc}")
                break

    def _validate_relationships(self, report: ValidationReport) -> None:
        user_ids = set(self.users["user_id"])

        orphan_events = set(self.events["user_id"]) - user_ids
        if orphan_events:
            report.add_error(f"Events reference unknown user_ids: {len(orphan_events)}")
        else:
            report.checks["event_foreign_keys"] = True

        orphan_assignments = set(self.experiment_assignments["user_id"]) - user_ids
        if orphan_assignments:
            report.add_error(f"Experiment assignments reference unknown user_ids: {len(orphan_assignments)}")
        else:
            report.checks["experiment_foreign_keys"] = True

        orphan_interventions = set(self.interventions["user_id"]) - user_ids
        if orphan_interventions:
            report.add_error(f"Interventions reference unknown user_ids: {len(orphan_interventions)}")
        else:
            report.checks["intervention_foreign_keys"] = True

        missing_assignments = user_ids - set(self.experiment_assignments["user_id"])
        if missing_assignments:
            report.add_error(f"Users missing experiment assignment: {len(missing_assignments)}")

    def _validate_temporal_rules(self, report: ValidationReport) -> None:
        # Every event timestamp must be >= signup timestamp.
        merged = self.events.merge(self.users[["user_id", "signup_timestamp"]], on="user_id")
        invalid_time = (merged["event_timestamp"] < merged["signup_timestamp"]).sum()
        if invalid_time > 0:
            report.add_error(f"Found {invalid_time} events before signup")
        else:
            report.checks["event_after_signup"] = True

        # Users must have complete observation window.
        max_signup = self.users["signup_timestamp"].max()
        required_freeze = max_signup + pd.Timedelta(days=config.LABEL_WINDOW_END_DAY)
        if required_freeze > config.END_DATE:
            report.add_error(
                f"Latest signup {max_signup} requires freeze {required_freeze} but config freeze is {config.END_DATE}"
            )
        else:
            report.checks["observation_window"] = True

        # Event timestamps should be sorted within each session.
        session_sort_issues = 0
        for session_id, group in self.events.groupby("session_id"):
            if not group["event_timestamp"].is_monotonic_increasing:
                session_sort_issues += 1
        if session_sort_issues > 0:
            report.add_warning(f"{session_sort_issues} sessions have out-of-order timestamps")
        else:
            report.checks["session_timestamp_order"] = True

    def _validate_experiment_srm(self, report: ValidationReport) -> None:
        for experiment_id in self.experiment_assignments["experiment_id"].unique():
            subset = self.experiment_assignments[self.experiment_assignments["experiment_id"] == experiment_id]
            variant_counts = subset["variant_id"].value_counts().sort_index()
            expected_allocations = {
                v["variant_id"]: v["allocation"]
                for v in config.ONBOARDING_VARIANTS
            }
            expected = [expected_allocations[v] * len(subset) for v in variant_counts.index]
            stat, p_value = chisquare(variant_counts.values, expected)
            if p_value < 0.01:
                report.add_error(
                    f"SRM detected for {experiment_id}: p={p_value:.4f}, counts={variant_counts.to_dict()}"
                )
            else:
                report.checks[f"srm_{experiment_id}"] = True

    def _validate_label_quality(self, report: ValidationReport) -> None:
        churn_rate = self.labels["is_churned"].mean()
        if not 0.25 <= churn_rate <= 0.45:
            report.add_error(f"Churn rate {churn_rate:.2%} outside target range 25%-45%")
        else:
            report.checks["churn_rate_range"] = True

        # Labels should only exist for users with full observation windows.
        label_users = set(self.labels["user_id"])
        all_users = set(self.users["user_id"])
        if label_users != all_users:
            report.add_error("Label user set does not match users table")

    def _validate_data_distributions(self, report: ValidationReport) -> None:
        missing_rate = self.events["event_properties"].isna().mean()
        if missing_rate > 0.05:
            report.add_warning(f"High event_properties missing rate: {missing_rate:.2%}")

        campaign_share = (self.events["event_source"] == "campaign").mean()
        if campaign_share > 0.20:
            report.add_warning(f"Unusually high campaign event share: {campaign_share:.2%}")

        # Each user should have at least one event (signup).
        users_without_events = len(set(self.users["user_id"]) - set(self.events["user_id"]))
        if users_without_events > 0:
            report.add_error(f"{users_without_events} users have no events")
        else:
            report.checks["users_with_events"] = True
