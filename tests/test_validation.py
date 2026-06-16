"""Tests for the data validation module."""

import pytest

from career_growth.data_generation.generator import generate_all_data
from career_growth.validation.validator import DataValidator


def test_validator_passes_on_generated_data(synthetic_data):
    validator = DataValidator("data")
    validator.users = synthetic_data["users"]
    validator.events = synthetic_data["events"]
    validator.experiment_assignments = synthetic_data["experiment_assignments"]
    validator.interventions = synthetic_data["interventions"]
    validator.labels = synthetic_data["labels"]
    report = validator.validate()
    assert report.passed, f"Validation failed: {report.errors}"


def test_validator_detects_orphan_event(tmp_path):
    output_dir = tmp_path / "orphan"
    generate_all_data(count=100, seed=42, output_dir=str(output_dir))
    validator = DataValidator(str(output_dir)).load()
    validator.events.loc[0, "user_id"] = "nonexistent-user"
    report = validator.validate()
    assert not report.passed
    assert any("unknown user_id" in err for err in report.errors)


def test_validator_detects_invalid_event_source(tmp_path):
    output_dir = tmp_path / "source"
    generate_all_data(count=100, seed=42, output_dir=str(output_dir))
    validator = DataValidator(str(output_dir)).load()
    validator.events.loc[0, "event_source"] = "bot"
    report = validator.validate()
    assert not report.passed
    assert any("event_source" in err for err in report.errors)


def test_validator_detects_duplicate_event_id(tmp_path):
    output_dir = tmp_path / "dup"
    generate_all_data(count=100, seed=42, output_dir=str(output_dir))
    validator = DataValidator(str(output_dir)).load()
    validator.events.loc[1, "event_id"] = validator.events.loc[0, "event_id"]
    report = validator.validate()
    assert not report.passed
    assert any("duplicate event_id" in err for err in report.errors)
