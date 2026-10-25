# tests/test_end_to_end.py
"""Automated unit test suite verifying all non-external acceptance criteria without paid APIs."""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest

from graph_enhanced_lm.classification.evaluation import create_leakage_safe_splits, evaluate_predictions
from graph_enhanced_lm.classification.label_mapping import map_external_label
from graph_enhanced_lm.classification.transformer_classifier import PostContentClassifier
from graph_enhanced_lm.data.preprocessing import clean_post_text, deduplicate_canonical_tables, extract_hashtags, normalize_timestamp_utc
from graph_enhanced_lm.data.validators import validate_canonical_dataframes
from graph_enhanced_lm.llm.query_validation import validate_and_sanitize_cypher
from graph_enhanced_lm.llm.qwen_client import QwenClient
from graph_enhanced_lm.llm.text_to_cypher import TextToCypherEngine
from graph_enhanced_lm.retrieval.graph_rag import GraphRAGPipeline
from graph_enhanced_lm.temporal.election_windows import analyze_temporal_dataframe, build_election_windows
from graph_enhanced_lm.utils.errors import CypherSyntaxValidationError, UnsafeCypherError
from scripts.generate_synthetic_data import generate_synthetic_dataset


def test_synthetic_data_generation_and_validation(tmp_path: Path) -> None:
    """Criteria 1 & 2: Synthetic dataset generates reproducibly and passes schema validation."""
    paths = generate_synthetic_dataset(tmp_path, seed=42, election_date="2026-11-01")
    users_df = pd.read_csv(paths["users"])
    posts_df = pd.read_csv(paths["posts"])
    inter_df = pd.read_csv(paths["interactions"]).where(pd.notnull(pd.read_csv(paths["interactions"])), None)

    assert len(users_df) == 48
    assert bool(users_df["is_synthetic"].all()) is True
    assert bool(posts_df["is_synthetic"].all()) is True

    users_df, posts_df, inter_df = deduplicate_canonical_tables(users_df, posts_df, inter_df)
    report = validate_canonical_dataframes(users_df, posts_df, inter_df, strict=True)
    assert report.is_valid is True
    assert report.duplicate_post_ids == 0
    assert report.missing_timestamps == 0


def test_preprocessing_and_label_mapping() -> None:
    """Verify hashtag extraction, UTC normalization, and explicit label mapping."""
    assert extract_hashtags("Check #Election2026 and #Neo4j #neo4j!") == ["election2026", "neo4j"]
    assert clean_post_text("  Hello   world \n ") == "Hello world"
    assert normalize_timestamp_utc("2026-10-25T10:00:00Z") == "2026-10-25T10:00:00Z"

    assert map_external_label("hatespeech", "hatexplain") == "HATE"
    assert map_external_label("HOF", "hasoc") == "OFFENSIVE"
    assert map_external_label("unseen_strange_label", "generic") == "UNKNOWN"


def test_cypher_safety_validator_rejects_mutations() -> None:
    """Criteria 12 & 13: Read-only queries pass and get LIMIT capped; destructive queries fail."""
    safe = validate_and_sanitize_cypher("MATCH (u:User) RETURN u.user_id", max_limit=25)
    assert "LIMIT 25" in safe

    capped = validate_and_sanitize_cypher("MATCH (u:User) RETURN u.user_id LIMIT 500", max_limit=50)
    assert "LIMIT 50" in capped

    for malicious in [
        "MATCH (n) DETACH DELETE n",
        "CREATE (u:User {user_id: 'hacked'}) RETURN u",
        "MATCH (u:User) SET u.admin = true RETURN u",
        "MATCH (u:User) RETURN u; MATCH (n) DELETE n",
        "CALL gds.graph.drop('my_graph') YIELD graphName RETURN graphName",
    ]:
        with pytest.raises((UnsafeCypherError, CypherSyntaxValidationError)):
            validate_and_sanitize_cypher(malicious)


def test_temporal_election_windows(tmp_path: Path) -> None:
    """Criteria 6 & 11: Election window builder and temporal aggregation."""
    windows = build_election_windows("2026-11-01")
    assert len(windows) == 4
    assert windows[1].name == "ten_days_before_election"

    paths = generate_synthetic_dataset(tmp_path, seed=42, election_date="2026-11-01")
    posts_df = pd.read_csv(paths["posts"])
    summary = analyze_temporal_dataframe(posts_df, "2026-11-01")
    assert summary["configured_election_date"] == "2026-11-01"
    assert sum(w["total_posts"] for w in summary["windows"]) > 0


def test_classifier_and_offline_graph_rag(tmp_path: Path) -> None:
    """Criteria 5, 14, 15, 16, 17, 18: Classifier evaluation and offline Graph RAG grounding."""
    paths = generate_synthetic_dataset(tmp_path, seed=42)
    posts_df = pd.read_csv(paths["posts"])
    train_df, test_df = create_leakage_safe_splits(posts_df, test_size=0.25, random_seed=42)
    assert set(train_df["user_id"]).isdisjoint(set(test_df["user_id"]))

    clf = PostContentClassifier(mode="heuristic_fallback")
    scored = clf.classify_dataframe(test_df, preserve_human_labels=False)
    metrics = evaluate_predictions(test_df["content_label"].tolist(), scored["content_label"].tolist())
    assert metrics["macro_f1"] > 0.70

    qwen = QwenClient(offline_mock=True)
    t2c = TextToCypherEngine(llm=qwen)
    rag = GraphRAGPipeline(text_to_cypher=t2c, llm=qwen)

    # Empty evidence check -> must abstain
    empty_resp = rag.answer_question("Show unknown records", fallback_records=[])
    assert empty_resp.has_sufficient_evidence is False
    assert "Insufficient graph evidence" in empty_resp.grounded_answer