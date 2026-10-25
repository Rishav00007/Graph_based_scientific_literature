# src/graph_enhanced_lm/graph/ingestion.py
"""Idempotent batch ingestion of Users, Posts, Hashtags, ContentLabels, and Interactions."""
from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Any, Dict, List
import pandas as pd

from graph_enhanced_lm.graph.neo4j_client import Neo4jClient

logger = logging.getLogger("graph_enhanced_lm")

UPSERT_USERS_CYPHER = """
UNWIND $rows AS row
MERGE (u:User {user_id: row.user_id})
ON CREATE SET
    u.username = row.username,
    u.dataset_source = row.dataset_source,
    u.is_synthetic = toBoolean(row.is_synthetic),
    u.synthetic_ground_truth_community = row.synthetic_ground_truth_community,
    u.created_at = datetime($batch_time)
ON MATCH SET
    u.username = coalesce(row.username, u.username),
    u.updated_at = datetime($batch_time)
"""

UPSERT_POSTS_CYPHER = """
UNWIND $rows AS row
MERGE (u:User {user_id: row.user_id})
MERGE (p:Post {post_id: row.post_id})
ON CREATE SET
    p.text = row.text,
    p.timestamp = datetime(row.timestamp),
    p.timestamp_str = row.timestamp,
    p.language = row.language,
    p.content_label = row.content_label,
    p.label_source = row.label_source,
    p.classification_confidence = toFloat(row.classification_confidence),
    p.classifier_model = row.classifier_model,
    p.classifier_version = row.classifier_version,
    p.dataset_source = row.dataset_source,
    p.is_synthetic = toBoolean(row.is_synthetic),
    p.ingested_at = datetime($batch_time)
ON MATCH SET
    p.text = row.text,
    p.content_label = CASE
        WHEN p.label_source = 'human_annotation' AND row.label_source <> 'human_annotation'
        THEN p.content_label
        ELSE row.content_label
    END,
    p.classification_confidence = toFloat(row.classification_confidence),
    p.classifier_model = row.classifier_model,
    p.classifier_version = row.classifier_version,
    p.updated_at = datetime($batch_time)
MERGE (u)-[:POSTED {dataset_source: row.dataset_source}]->(p)
MERGE (cl:ContentLabel {name: p.content_label})
MERGE (p)-[hl:HAS_LABEL]->(cl)
SET hl.confidence = p.classification_confidence,
    hl.classifier_model = p.classifier_model,
    hl.provenance = 'model_or_annotation'
WITH p, row
UNWIND row.hashtag_list AS tag_name
MERGE (h:Hashtag {tag: tag_name})
MERGE (p)-[:USES_HASHTAG]->(h)
"""

UPSERT_RETWEETS_CYPHER = """
UNWIND $rows AS row
MATCH (u:User {user_id: row.source_user_id})
MATCH (p:Post {post_id: row.target_post_id})
MERGE (u)-[r:RETWEETED {interaction_id: row.interaction_id}]->(p)
SET r.timestamp = datetime(row.timestamp),
    r.timestamp_str = row.timestamp,
    r.dataset_source = row.dataset_source,
    r.is_synthetic = toBoolean(row.is_synthetic),
    r.provenance = 'source_provided'
"""

UPSERT_REPLIES_CYPHER = """
UNWIND $rows AS row
MATCH (u:User {user_id: row.source_user_id})
MATCH (p:Post {post_id: row.target_post_id})
MERGE (u)-[r:REPLIED_TO {interaction_id: row.interaction_id}]->(p)
SET r.timestamp = datetime(row.timestamp),
    r.timestamp_str = row.timestamp,
    r.dataset_source = row.dataset_source,
    r.is_synthetic = toBoolean(row.is_synthetic),
    r.provenance = 'source_provided'
"""

UPSERT_MENTIONS_CYPHER = """
UNWIND $rows AS row
MATCH (u:User {user_id: row.source_user_id})
MATCH (target:User {user_id: row.target_user_id})
MERGE (u)-[r:MENTIONED {interaction_id: row.interaction_id}]->(target)
SET r.timestamp = datetime(row.timestamp),
    r.timestamp_str = row.timestamp,
    r.dataset_source = row.dataset_source,
    r.is_synthetic = toBoolean(row.is_synthetic),
    r.provenance = 'source_provided'
"""


def ingest_canonical_data(
    client: Neo4jClient,
    users_df: pd.DataFrame,
    posts_df: pd.DataFrame,
    interactions_df: pd.DataFrame,
    batch_size: int = 200,
) -> Dict[str, int]:
    """Ingest canonical users, posts, hashtags, and interactions into Neo4j using idempotent MERGE."""
    batch_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    user_rows = users_df.to_dict(orient="records")
    for i in range(0, len(user_rows), batch_size):
        client.execute_write(UPSERT_USERS_CYPHER, {"rows": user_rows[i : i + batch_size], "batch_time": batch_time})

    post_rows: List[Dict[str, Any]] = []
    for row in posts_df.to_dict(orient="records"):
        raw_tags = row.get("hashtags", "")
        if isinstance(raw_tags, str):
            tag_list = [t.strip().lower() for t in raw_tags.split(",") if t.strip()]
        elif isinstance(raw_tags, list):
            tag_list = [str(t).strip().lower() for t in raw_tags if str(t).strip()]
        else:
            tag_list = []
        row_copy = dict(row)
        row_copy["hashtag_list"] = tag_list
        post_rows.append(row_copy)

    for i in range(0, len(post_rows), batch_size):
        client.execute_write(UPSERT_POSTS_CYPHER, {"rows": post_rows[i : i + batch_size], "batch_time": batch_time})

    retweets = interactions_df[interactions_df["interaction_type"] == "RETWEETED"].to_dict(orient="records")
    replies = interactions_df[interactions_df["interaction_type"] == "REPLIED_TO"].to_dict(orient="records")
    mentions = interactions_df[interactions_df["interaction_type"] == "MENTIONED"].to_dict(orient="records")

    if retweets:
        client.execute_write(UPSERT_RETWEETS_CYPHER, {"rows": retweets})
    if replies:
        client.execute_write(UPSERT_REPLIES_CYPHER, {"rows": replies})
    if mentions:
        client.execute_write(UPSERT_MENTIONS_CYPHER, {"rows": mentions})

    logger.info(
        "Ingested %d users, %d posts, and %d interactions into Neo4j.",
        len(user_rows),
        len(post_rows),
        len(interactions_df),
    )
    return {
        "users_ingested": len(user_rows),
        "posts_ingested": len(post_rows),
        "interactions_ingested": len(interactions_df),
    }