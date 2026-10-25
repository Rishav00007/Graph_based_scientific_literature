# src/graph_enhanced_lm/data/preprocessing.py
"""Text normalization, hashtag extraction, timestamp standardization, and deduplication."""
from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import List
import pandas as pd

HASHTAG_REGEX = re.compile(r"#(\w+)")
WHITESPACE_REGEX = re.compile(r"\s+")


def extract_hashtags(text: str) -> List[str]:
    """Extract lowercase hashtags without the leading '#' symbol."""
    if not isinstance(text, str):
        return []
    return sorted({match.group(1).lower() for match in HASHTAG_REGEX.finditer(text)})


def clean_post_text(text: str) -> str:
    """Normalize whitespace while preserving mentions, hashtags, and semantics for Transformers."""
    if not isinstance(text, str):
        return ""
    return WHITESPACE_REGEX.sub(" ", text).strip()


def normalize_timestamp_utc(ts_value: str | int | float | datetime) -> str:
    """Convert timestamps into ISO-8601 UTC format (YYYY-MM-DDTHH:MM:SSZ)."""
    if isinstance(ts_value, datetime):
        dt = ts_value.astimezone(timezone.utc) if ts_value.tzinfo else ts_value.replace(tzinfo=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    if isinstance(ts_value, (int, float)):
        dt = datetime.fromtimestamp(float(ts_value), tz=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    if isinstance(ts_value, str):
        cleaned = ts_value.strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(cleaned)
        dt = dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    raise ValueError(f"Unsupported timestamp format: {ts_value!r}")


def deduplicate_canonical_tables(
    users_df: pd.DataFrame, posts_df: pd.DataFrame, interactions_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Deduplicate canonical tables deterministically by primary identifiers."""
    clean_users = users_df.drop_duplicates(subset=["user_id"], keep="first").copy()
    clean_posts = posts_df.drop_duplicates(subset=["post_id"], keep="first").copy()
    clean_posts["text"] = clean_posts["text"].apply(clean_post_text)
    clean_posts["timestamp"] = clean_posts["timestamp"].apply(normalize_timestamp_utc)
    clean_inter = interactions_df.drop_duplicates(subset=["interaction_id"], keep="first").copy()
    clean_inter["timestamp"] = clean_inter["timestamp"].apply(normalize_timestamp_utc)
    return clean_users, clean_posts, clean_inter