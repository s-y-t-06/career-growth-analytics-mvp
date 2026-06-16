"""Tests for the train_churn_model.py CLI script."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_churn_model.py"


def _load_train_script():
    """Load the training script as a module without executing __main__."""
    spec = importlib.util.spec_from_file_location("train_churn_model", TRAIN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_data_dir_is_training():
    """The training script must default to a dedicated training data directory."""
    module = _load_train_script()
    args = module.parse_args([])
    assert args.data_dir == "data/training"


def test_training_script_respects_data_dir_and_does_not_touch_sample(
    tmp_path,
):
    """The script must write generated data only to the configured data directory."""
    data_dir = tmp_path / "modeling_data"
    output_dir = tmp_path / "artifacts"

    sample_users_before = pd.read_csv(PROJECT_ROOT / "data" / "sample" / "users.csv")
    sample_labels_before = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "labels.csv")

    result = subprocess.run(
        [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--count",
            "200",
            "--seed",
            "42",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    # Generated data must live under the configured data directory.
    assert (data_dir / "sample" / "users.csv").exists()
    assert (data_dir / "processed" / "labels.csv").exists()
    assert (data_dir / "processed" / "model_features.csv").exists()

    # Formal sample data must remain untouched.
    sample_users_after = pd.read_csv(PROJECT_ROOT / "data" / "sample" / "users.csv")
    sample_labels_after = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "labels.csv")
    pd.testing.assert_frame_equal(sample_users_before, sample_users_after)
    pd.testing.assert_frame_equal(sample_labels_before, sample_labels_after)

    # Artifacts must be written to the configured output directory.
    assert (output_dir / "churn_model.joblib").exists()
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "model_metadata.json").exists()
