# src/graph_enhanced_lm/gds/community_detection.py
"""Louvain community detection, PageRank/Degree centrality, and candidate community profiling."""
from __future__ import annotations

import logging
from typing import Any, Dict, List
from graph_enhanced_lm.graph.neo4j_client import Neo4jClient

logger = logging.getLogger("graph_enhanced_lm")


class GDSAnalyticsEngine:
    """Runs Louvain, Degree, and PageRank and computes community-level content statistics."""

    def __init__(self, client: Neo4jClient) -> None:
        self.client = client

    def run_louvain_write(
        self,
        graph_name: str = "user_interaction_projection",
        write_property: str = "louvain_community_id",
    ) -> Dict[str, Any]:
        """Execute Louvain community detection in write mode and materialize :Community nodes."""
        cypher = """
        CALL gds.louvain.write(
            $graph_name,
            {
                relationshipWeightProperty: 'weight',
                writeProperty: $write_property
            }
        )
        YIELD communityCount, modularity, modularities, computeMillis
        RETURN communityCount, modularity, modularities, computeMillis
        """
        result = self.client.execute_write(
            cypher, {"graph_name": graph_name, "write_property": write_property}
        )
        # Materialize analytical :Community nodes for explicit graph querying
        materialize_comm = """
        MATCH (u:User)
        WHERE u.louvain_community_id IS NOT NULL
        MERGE (c:Community {community_id: u.louvain_community_id})
        MERGE (u)-[m:BELONGS_TO_COMMUNITY]->(c)
        SET m.provenance = 'derived_analytical_louvain'
        """
        self.client.execute_write(materialize_comm)
        return result[0] if result else {}

    def run_centrality_write(self, graph_name: str = "user_interaction_projection") -> Dict[str, Any]:
        """Compute Degree Centrality and PageRank on the projection and write properties to :User nodes."""
        deg_res = self.client.execute_write(
            """
            CALL gds.degree.write($graph_name, {writeProperty: 'degree_centrality', relationshipWeightProperty: 'weight'})
            YIELD centralityDistribution, computeMillis
            RETURN centralityDistribution, computeMillis
            """,
            {"graph_name": graph_name},
        )
        pr_res = self.client.execute_write(
            """
            CALL gds.pageRank.write($graph_name, {writeProperty: 'pagerank_score', maxIterations: 20, dampingFactor: 0.85})
            YIELD centralityDistribution, computeMillis
            RETURN centralityDistribution, computeMillis
            """,
            {"graph_name": graph_name},
        )
        return {"degree": deg_res[0] if deg_res else {}, "pagerank": pr_res[0] if pr_res else {}}

    def compute_community_profiles(
        self,
        min_posts: int = 3,
        flagged_ratio_threshold: float = 0.30,
    ) -> List[Dict[str, Any]]:
        """Aggregate post categories, interaction counts, and top influential users per Louvain community.

        ETHICAL SAFEGUARD:
        Communities exceeding the configured flagged content threshold are labeled
        'candidate_community_exhibiting_hate_spam_patterns' for human analytical review,
        NEVER automatically labeled as malicious coordination rings.
        """
        query = """
        MATCH (u:User)
        WHERE u.louvain_community_id IS NOT NULL
        OPTIONAL MATCH (u)-[:POSTED]->(p:Post)
        WITH u.louvain_community_id AS community_id,
             count(DISTINCT u) AS user_count,
             count(DISTINCT p) AS total_posts,
             sum(CASE WHEN p.content_label = 'HATE' THEN 1 ELSE 0 END) AS hate_posts,
             sum(CASE WHEN p.content_label = 'OFFENSIVE' THEN 1 ELSE 0 END) AS offensive_posts,
             sum(CASE WHEN p.content_label = 'COMMUNITY_TARGETED' THEN 1 ELSE 0 END) AS targeted_posts,
             sum(CASE WHEN p.content_label = 'SPAM' THEN 1 ELSE 0 END) AS spam_posts,
             sum(CASE WHEN p.content_label = 'NORMAL' THEN 1 ELSE 0 END) AS normal_posts
        WITH community_id, user_count, total_posts, hate_posts, offensive_posts, targeted_posts, spam_posts, normal_posts,
             CASE WHEN total_posts > 0
                  THEN round(toFloat(hate_posts + offensive_posts + targeted_posts + spam_posts) / total_posts, 4)
                  ELSE 0.0 END AS flagged_content_ratio
        ORDER BY flagged_content_ratio DESC, total_posts DESC
        RETURN community_id, user_count, total_posts, hate_posts, offensive_posts,
               targeted_posts, spam_posts, normal_posts, flagged_content_ratio
        """
        rows = self.client.execute_read(query)
        for row in rows:
            is_candidate = (
                int(row["total_posts"]) >= min_posts
                and float(row["flagged_content_ratio"]) >= flagged_ratio_threshold
            )
            row["analytical_status"] = (
                "candidate_community_exhibiting_hate_spam_propagation_patterns"
                if is_candidate
                else "within_baseline_thresholds"
            )
        return rows