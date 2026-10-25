# src/graph_enhanced_lm/classification/label_mapping.py
"""Explicit, documented mapping from external dataset or model labels to project taxonomy."""
from __future__ import annotations

from typing import Dict
from graph_enhanced_lm.data.schemas import ContentCategory

DATASET_LABEL_MAPS: Dict[str, Dict[str, ContentCategory]] = {
    "hatexplain": {
        "normal": ContentCategory.NORMAL,
        "hatespeech": ContentCategory.HATE,
        "offensive": ContentCategory.OFFENSIVE,
    },
    "hasoc": {
        "not": ContentCategory.NORMAL,
        "hof": ContentCategory.OFFENSIVE,
        "hate": ContentCategory.HATE,
        "offn": ContentCategory.OFFENSIVE,
        "prfn": ContentCategory.OFFENSIVE,
    },
    "generic": {
        "normal": ContentCategory.NORMAL,
        "neutral": ContentCategory.NORMAL,
        "not_hate": ContentCategory.NORMAL,
        "hate": ContentCategory.HATE,
        "hatespeech": ContentCategory.HATE,
        "offensive": ContentCategory.OFFENSIVE,
        "abusive": ContentCategory.OFFENSIVE,
        "community_targeted": ContentCategory.COMMUNITY_TARGETED,
        "targeted": ContentCategory.COMMUNITY_TARGETED,
        "spam": ContentCategory.SPAM,
    },
}


def map_external_label(raw_label: str, dataset_name: str = "generic") -> str:
    """Map an external label string to our canonical ContentCategory value.

    Unmapped labels are preserved as 'UNKNOWN' rather than silently forced into 'NORMAL'.
    """
    if not isinstance(raw_label, str) or not raw_label.strip():
        return ContentCategory.UNKNOWN.value
    normalized = raw_label.strip().lower()
    mapping = DATASET_LABEL_MAPS.get(dataset_name.lower(), DATASET_LABEL_MAPS["generic"])
    if normalized in mapping:
        return mapping[normalized].value
    if normalized in DATASET_LABEL_MAPS["generic"]:
        return DATASET_LABEL_MAPS["generic"][normalized].value
    return ContentCategory.UNKNOWN.value