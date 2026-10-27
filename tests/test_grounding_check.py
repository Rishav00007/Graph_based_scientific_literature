# tests/test_grounding_check.py
"""Graph RAG hallucination check: numbers in an answer must come from the evidence."""
from __future__ import annotations

from graph_enhanced_lm.retrieval.evidence_formatter import find_ungrounded_numbers, format_graph_evidence_bundle

RECORDS = [
    {"community_id": 33, "total_posts": 55, "hate_posts": 23, "hate_ratio": 0.4182},
    {"community_id": 4, "total_posts": 45, "hate_posts": 1, "hate_ratio": 0.0222},
]
EVIDENCE = format_graph_evidence_bundle("Which communities have most hate?", "MATCH ... LIMIT 20", RECORDS)


def test_grounded_answer_passes() -> None:
    answer = (
        "1. [community_id=33] has 23 hate posts out of 55 (ratio 0.4182, about 41.82%).\n"
        "2. [community_id=4] has 1 hate post out of 45."
    )
    assert find_ungrounded_numbers(answer, EVIDENCE) == []


def test_hallucinated_numbers_are_flagged() -> None:
    # Real failure seen with qwen3:1.7b: counts that were never in the evidence
    answer = "1. [community_id=4] with 364 total posts and 364 hate posts (hate ratio: 1.0)"
    # 1.0 is not flagged: it equals the value 1 that does appear in the evidence (a known blind spot)
    assert find_ungrounded_numbers(answer, EVIDENCE) == ["364"]
