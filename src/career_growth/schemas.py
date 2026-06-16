"""Pydantic schemas for data validation and type safety."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from career_growth import config


class User(BaseModel):
    """Row-level schema for the users table."""

    user_id: str
    signup_timestamp: datetime
    acquisition_channel: str
    country: str
    device_type: str
    language: str
    timezone: str
    initial_plan_type: str = "free"
    user_intent_level: str
    career_stage: str
    marketing_consent: bool

    @field_validator("acquisition_channel")
    @classmethod
    def _validate_channel(cls, value: str) -> str:
        if value not in config.ACQUISITION_CHANNELS:
            raise ValueError(f"Invalid acquisition_channel: {value}")
        return value

    @field_validator("device_type")
    @classmethod
    def _validate_device(cls, value: str) -> str:
        if value not in config.DEVICE_TYPES:
            raise ValueError(f"Invalid device_type: {value}")
        return value

    @field_validator("user_intent_level")
    @classmethod
    def _validate_intent(cls, value: str) -> str:
        if value not in config.USER_INTENT_LEVELS:
            raise ValueError(f"Invalid user_intent_level: {value}")
        return value

    @field_validator("career_stage")
    @classmethod
    def _validate_career_stage(cls, value: str) -> str:
        if value not in config.CAREER_STAGES:
            raise ValueError(f"Invalid career_stage: {value}")
        return value


class Event(BaseModel):
    """Row-level schema for the events table."""

    event_id: str
    user_id: str
    session_id: str
    event_name: str
    event_timestamp: datetime
    event_properties: dict[str, Any] | None = None
    page_name: str | None = None
    platform: str
    event_source: str
    experiment_id: str | None = None
    variant_id: str | None = None

    @field_validator("event_name")
    @classmethod
    def _validate_event_name(cls, value: str) -> str:
        if value not in config.ALL_EVENT_NAMES:
            raise ValueError(f"Invalid event_name: {value}")
        return value

    @field_validator("event_source")
    @classmethod
    def _validate_event_source(cls, value: str) -> str:
        if value not in config.EVENT_SOURCES:
            raise ValueError(f"Invalid event_source: {value}")
        return value

    @field_validator("platform")
    @classmethod
    def _validate_platform(cls, value: str) -> str:
        if value not in {"web", "ios", "android"}:
            raise ValueError(f"Invalid platform: {value}")
        return value


class ExperimentAssignment(BaseModel):
    """Row-level schema for experiment assignments."""

    experiment_id: str
    user_id: str
    variant_id: str
    assignment_time: datetime
    experiment_name: str
    experiment_type: str
    traffic_allocation: float = Field(..., ge=0.0, le=1.0)
    is_exposed: bool
    is_converted: bool


class Intervention(BaseModel):
    """Row-level schema for intervention logs."""

    message_id: str
    user_id: str
    action_name: str
    channel: str
    send_time: datetime
    open_time: datetime | None = None
    click_time: datetime | None = None
    conversion_time: datetime | None = None
    experiment_id: str | None = None

    @field_validator("channel")
    @classmethod
    def _validate_channel(cls, value: str) -> str:
        if value not in {"email", "push", "in_app"}:
            raise ValueError(f"Invalid channel: {value}")
        return value
