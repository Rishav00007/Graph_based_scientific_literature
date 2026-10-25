# src/graph_enhanced_lm/graph/constraints.py
"""Schema constraints and indexes initialization for Neo4j 5.x."""
from __future__ import annotations

import logging
from typing import List
from graph_enhanced_lm.graph.neo4j_client import Neo4jClient

logger = logging.getLogger("graph_enhanced_lm")

SCHEMA_CONSTRAINTS_AND_INDEXES: List[str] = [
    "CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (u:User) REQUIRE u.user_id IS UNIQUE",
    "CREATE CONSTRAINT post_id_unique IF NOT EXISTS FOR (p:Post) REQUIRE p.post_id IS UNIQUE",
    "CREATE CONSTRAINT hashtag_tag_unique IF NOT EXISTS FOR (h:Hashtag) REQUIRE h.tag IS UNIQUE",
    "CREATE CONSTRAINT content_label_name_unique IF NOT EXISTS FOR (c:ContentLabel) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT community_id_unique IF NOT EXISTS FOR (cm:Community) REQUIRE cm.community_id IS UNIQUE",
    "CREATE INDEX post_timestamp_idx IF NOT EXISTS FOR (p:Post) ON (p.timestamp)",
    "CREATE INDEX post_content_label_idx IF NOT EXISTS FOR (p:Post) ON (p.content_label)",
    "CREATE INDEX user_community_idx IF NOT EXISTS FOR (u:User) ON (u.louvain_community_id)",
]


def initialize_schema(client: Neo4jClient) -> None:
    """Create all uniqueness constraints and indexes idempotently."""
    for statement in SCHEMA_CONSTRAINTS_AND_INDEXES:
        client.execute_write(statement)
    logger.info("Initialized %d Neo4j constraints and indexes.", len(SCHEMA_CONSTRAINTS_AND_INDEXES))