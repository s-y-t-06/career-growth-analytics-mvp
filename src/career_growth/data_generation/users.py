"""Synthetic user generation with hidden propensity variables."""

import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from career_growth import config


def _encode_intent(level: pd.Series) -> pd.Series:
    mapping = {"low": 0.30, "medium": 0.60, "high": 0.90}
    return level.map(mapping)


def _encode_career_stage(stage: pd.Series) -> pd.Series:
    mapping = {"student": 0.70, "new_graduate": 0.85, "early_career": 0.75}
    return stage.map(mapping)


def _encode_device(device: pd.Series) -> pd.Series:
    mapping = {"desktop": 1.00, "mobile": 0.85, "tablet": 0.75}
    return device.map(mapping)


def generate_users(
    count: int,
    rng: np.random.Generator,
    seed: int = config.RANDOM_SEED,
    study_start: datetime = config.STUDY_START_DATE,
    data_freeze: datetime = config.END_DATE,
) -> pd.DataFrame:
    """Generate the users table with hidden propensity variables.

    The hidden propensity columns are kept for event generation but must be
    removed before persisting the users table.
    """
    latest_signup = data_freeze - timedelta(days=config.LABEL_WINDOW_END_DAY)
    delta_seconds = int((latest_signup - study_start).total_seconds())

    countries = list(config.COUNTRY_CONFIG.keys())
    country_weights = config.COUNTRY_WEIGHTS

    user_ids = [str(uuid.uuid5(uuid.NAMESPACE_OID, f"career-growth-user-{seed}-{i}")) for i in range(count)]
    signup_offsets = rng.integers(0, delta_seconds, size=count)
    signup_timestamps = [study_start + timedelta(seconds=int(offset)) for offset in signup_offsets]

    df = pd.DataFrame(
        {
            "user_id": user_ids,
            "signup_timestamp": signup_timestamps,
            "acquisition_channel": rng.choice(
                config.ACQUISITION_CHANNELS,
                size=count,
                p=config.ACQUISITION_CHANNEL_WEIGHTS,
            ),
            "country": rng.choice(countries, size=count, p=country_weights),
            "device_type": rng.choice(
                config.DEVICE_TYPES,
                size=count,
                p=config.DEVICE_TYPE_WEIGHTS,
            ),
            "user_intent_level": rng.choice(
                config.USER_INTENT_LEVELS,
                size=count,
                p=config.USER_INTENT_WEIGHTS,
            ),
            "career_stage": rng.choice(
                config.CAREER_STAGES,
                size=count,
                p=config.CAREER_STAGE_WEIGHTS,
            ),
            "marketing_consent": rng.random(size=count) < 0.75,
        }
    )

    # Derived language and timezone from country.
    df["language"] = df["country"].map(lambda c: config.COUNTRY_CONFIG[c]["language"])
    df["timezone"] = df["country"].map(lambda c: config.COUNTRY_CONFIG[c]["timezone"])
    df["initial_plan_type"] = "free"

    # Hidden propensity variables used only during generation.
    df["intrinsic_engagement"] = rng.beta(2.0, 2.0, size=count)
    df["career_urgency"] = rng.beta(2.5, 2.0, size=count)
    df["product_fit"] = rng.beta(2.0, 2.0, size=count)
    df["notification_sensitivity"] = rng.beta(2.0, 3.0, size=count)

    # Encoded scores used in probability formulas.
    df["intent_score"] = _encode_intent(df["user_intent_level"])
    df["career_stage_score"] = _encode_career_stage(df["career_stage"])
    df["device_score"] = _encode_device(df["device_type"])

    # Introduce a small anomaly set: ~3% of users have inverted behavior.
    anomaly_mask = rng.random(size=count) < 0.03
    df.loc[anomaly_mask, "intrinsic_engagement"] = 1.0 - df.loc[anomaly_mask, "intrinsic_engagement"]

    return df
