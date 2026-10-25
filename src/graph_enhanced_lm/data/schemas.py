# src/graph_enhanced_lm/data/schemas.py
"""Canonical Pydantic data schemas for users, posts, and graph interactions."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class ContentCategory(str, Enum):
    NORMAL = "NORMAL"
    HATE = "HATE"
    OFFENSIVE = "OFFENSIVE"
    COMMUNITY_TARGETED = "COMMUNITY_TARGETED"
    SPAM = "SPAM"
    UNKNOWN = "UNKNOWN"


class InteractionType(str, Enum):
    RETWEETED = "RETWEETED"
    REPLIED_TO = "REPLIED_TO"
    MENTIONED = "MENTIONED"
    QUOTED = "QUOTED"


class CanonicalUser(BaseModel):
    user_id: str = Field(..., min_length=1, description="Stable unique user identifier")
    username: str = Field(..., min_length=1, description="Public handle or synthetic handle")
    dataset_source: str = Field(..., min_length=1)
    is_synthetic: bool = Field(default=False)
    synthetic_ground_truth_community: Optional[int] = Field(default=None)


class CanonicalPost(BaseModel):
    post_id: str = Field(..., min_length=1)
    user_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    timestamp: str = Field(..., description="ISO-8601 UTC timestamp string")
    language: str = Field(default="en")
    hashtags: List[str] = Field(default_factory=list)
    content_label: ContentCategory = Field(default=ContentCategory.UNKNOWN)
    label_source: str = Field(default="unlabeled", description="human_annotation | model_prediction | synthetic_ground_truth")
    classification_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    classifier_model: str = Field(default="none")
    classifier_version: str = Field(default="0.0.0")
    dataset_source: str = Field(..., min_length=1)
    is_synthetic: bool = Field(default=False)

    @field_validator("timestamp")
    @classmethod
    def validate_iso_timestamp(cls, value: str) -> str:
        normalized = value.replace("Z", "+00:00")
        try:
            datetime.fromisoformat(normalized)
        except ValueError as exc:
            raise ValueError(f"Invalid ISO-8601 timestamp: {value}") from exc
        return value


class CanonicalInteraction(BaseModel):
    interaction_id: str = Field(..., min_length=1)
    source_user_id: str = Field(..., min_length=1)
    target_post_id: Optional[str] = Field(default=None)
    target_user_id: Optional[str] = Field(default=None)
    interaction_type: InteractionType
    timestamp: str
    dataset_source: str
    is_synthetic: bool = Field(default=False)