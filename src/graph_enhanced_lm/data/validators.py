# src/graph_enhanced_lm/data/validators.py
"""Dataset validation and integrity checks."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List
import pandas as pd

from graph_enhanced_lm.data.schemas import CanonicalInteraction, CanonicalPost, CanonicalUser
from graph_enhanced_lm.utils.errors import DatasetValidationError


@dataclass
class ValidationReport:
    is_valid: bool
    num_users: int
    num_posts: int
    num_interactions: int
    duplicate_post_ids: int
    missing_timestamps: int
    orphan_post_users: int
    synthetic_records_count: int
    errors: List[str] = field(default_factory=list)
    label_distribution: Dict[str, int] = field(default_factory=dict)


def validate_canonical_dataframes(
    users_df: pd.DataFrame,
    posts_df: pd.DataFrame,
    interactions_df: pd.DataFrame,
    strict: bool = True,
) -> ValidationReport:
    """Validate canonical users, posts, and interactions DataFrames."""
    errors: List[str] = []

    required_user_cols = {"user_id", "username", "dataset_source", "is_synthetic"}
    required_post_cols = {"post_id", "user_id", "text", "timestamp", "content_label", "dataset_source", "is_synthetic"}
    required_inter_cols = {"interaction_id", "source_user_id", "interaction_type", "timestamp", "dataset_source", "is_synthetic"}

    if not required_user_cols.issubset(users_df.columns):
        errors.append(f"Missing user columns: {required_user_cols - set(users_df.columns)}")
    if not required_post_cols.issubset(posts_df.columns):
        errors.append(f"Missing post columns: {required_post_cols - set(posts_df.columns)}")
    if not required_inter_cols.issubset(interactions_df.columns):
        errors.append(f"Missing interaction columns: {required_inter_cols - set(interactions_df.columns)}")

    if errors:
        if strict:
            raise DatasetValidationError("; ".join(errors))
        return ValidationReport(False, len(users_df), len(posts_df), len(interactions_df), 0, 0, 0, 0, errors)

    dup_posts = int(posts_df["post_id"].duplicated().sum())
    if dup_posts > 0:
        errors.append(f"Found {dup_posts} duplicate post_id records.")

    missing_ts = int(posts_df["timestamp"].isna().sum() + (posts_df["timestamp"] == "").sum())
    if missing_ts > 0:
        errors.append(f"Found {missing_ts} posts with missing timestamps.")

    known_users = set(users_df["user_id"].astype(str))
    post_users = set(posts_df["user_id"].astype(str))
    orphan_users = len(post_users - known_users)
    if orphan_users > 0:
        errors.append(f"Found {orphan_users} post authors not present in users table.")

    # Validate row-level schema compliance on sample/all records
    for idx, row in users_df.iterrows():
        try:
            CanonicalUser(**row.to_dict())
        except Exception as exc:
            errors.append(f"User row {idx} invalid: {exc}")
            break

    for idx, row in posts_df.iterrows():
        row_dict = row.to_dict()
        if isinstance(row_dict.get("hashtags"), str):
            row_dict["hashtags"] = [h for h in row_dict["hashtags"].split(",") if h]
        try:
            CanonicalPost(**row_dict)
        except Exception as exc:
            errors.append(f"Post row {idx} invalid: {exc}")
            break

    for idx, row in interactions_df.iterrows():
        try:
            CanonicalInteraction(**row.to_dict())
        except Exception as exc:
            errors.append(f"Interaction row {idx} invalid: {exc}")
            break

    is_valid = len(errors) == 0
    if strict and not is_valid:
        raise DatasetValidationError("; ".join(errors))

    label_dist = posts_df["content_label"].value_counts().to_dict()
    syn_count = int(posts_df["is_synthetic"].astype(bool).sum())

    return ValidationReport(
        is_valid=is_valid,
        num_users=len(users_df),
        num_posts=len(posts_df),
        num_interactions=len(interactions_df),
        duplicate_post_ids=dup_posts,
        missing_timestamps=missing_ts,
        orphan_post_users=orphan_users,
        synthetic_records_count=syn_count,
        errors=errors,
        label_distribution={str(k): int(v) for k, v in label_dist.items()},
    )