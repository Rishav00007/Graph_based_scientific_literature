# src/graph_enhanced_lm/data/dataset_adapters.py
"""Configurable adapters for public datasets (HateXplain, HASOC, custom CSV/JSONL).

IMPORTANT METHODOLOGICAL NOTE:
Datasets like HateXplain and HASOC provide human-annotated text labels for hate/offensive content
classification, but DO NOT provide full social network graphs or election timestamps.
This module explicitly marks missing structural fields rather than fabricating user graphs.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List
import pandas as pd

from graph_enhanced_lm.classification.label_mapping import map_external_label
from graph_enhanced_lm.data.preprocessing import clean_post_text, extract_hashtags


def adapt_hatexplain_records(records: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    """Convert raw HateXplain JSON entries into a classification-ready DataFrame."""
    rows: List[Dict[str, Any]] = []
    for item in records:
        post_id = str(item.get("post_id", ""))
        tokens = item.get("post_tokens", [])
        text = clean_post_text(" ".join(tokens) if isinstance(tokens, list) else str(tokens))
        annotators = item.get("annotators", [])
        votes = [ann.get("label", "normal") for ann in annotators if isinstance(ann, dict)]
        majority_raw = Counter(votes).most_common(1)[0][0] if votes else "unknown"
        targets = [
            t
            for ann in annotators
            if isinstance(ann, dict)
            for t in ann.get("target", [])
            if t and t.lower() != "none"
        ]
        canonical_label = map_external_label(majority_raw, dataset_name="hatexplain")
        if canonical_label in {"HATE", "OFFENSIVE"} and len(targets) >= 2:
            # Preserve explicit target-community signal when multiple annotators mark specific target groups
            secondary_targeted = True
        else:
            secondary_targeted = False

        rows.append(
            {
                "post_id": post_id,
                "text": text,
                "raw_label": majority_raw,
                "content_label": canonical_label,
                "is_community_targeted": secondary_targeted,
                "hashtags": ",".join(extract_hashtags(text)),
                "dataset_source": "HateXplain",
                "label_source": "human_annotation",
                "is_synthetic": False,
                "has_graph_edges": False,
                "has_timestamp": False,
            }
        )
    return pd.DataFrame(rows)


def adapt_hasoc_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Convert a HASOC TSV/CSV DataFrame into our canonical classification format."""
    rows: List[Dict[str, Any]] = []
    for idx, row in raw_df.iterrows():
        post_id = str(row.get("text_id", row.get("_id", f"hasoc_{idx}")))
        text = clean_post_text(str(row.get("text", "")))
        raw_task2 = str(row.get("task_2", row.get("task_1", "NOT")))
        canonical_label = map_external_label(raw_task2, dataset_name="hasoc")
        rows.append(
            {
                "post_id": post_id,
                "text": text,
                "raw_label": raw_task2,
                "content_label": canonical_label,
                "hashtags": ",".join(extract_hashtags(text)),
                "dataset_source": "HASOC",
                "label_source": "human_annotation",
                "is_synthetic": False,
                "has_graph_edges": False,
                "has_timestamp": False,
            }
        )
    return pd.DataFrame(rows)