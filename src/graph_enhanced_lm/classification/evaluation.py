# src/graph_enhanced_lm/classification/evaluation.py
"""Classifier evaluation metrics and group-aware train/val/test splitting."""
from __future__ import annotations

from typing import Any, Dict, List
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def create_leakage_safe_splits(
    posts_df: pd.DataFrame,
    test_size: float = 0.2,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split posts into train and test sets, grouping by user_id when available to prevent author leakage."""
    if "user_id" in posts_df.columns and posts_df["user_id"].nunique() > 5:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
        train_idx, test_idx = next(splitter.split(posts_df, groups=posts_df["user_id"]))
        return posts_df.iloc[train_idx].copy(), posts_df.iloc[test_idx].copy()

    return train_test_split(posts_df, test_size=test_size, random_state=random_seed, stratify=posts_df.get("content_label"))


def evaluate_predictions(y_true: List[str], y_pred: List[str], labels: List[str] | None = None) -> Dict[str, Any]:
    """Compute Accuracy, Precision, Recall, Macro-F1, Per-Class metrics, and Confusion Matrix."""
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))

    acc = float(accuracy_score(y_true, y_pred))
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    per_class_report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    class_counts = pd.Series(y_true).value_counts().to_dict()

    return {
        "num_evaluated": len(y_true),
        "accuracy": round(acc, 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "labels": labels,
        "class_distribution": {str(k): int(v) for k, v in class_counts.items()},
        "per_class_metrics": per_class_report,
        "confusion_matrix": cm,
    }