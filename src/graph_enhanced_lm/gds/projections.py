# src/graph_enhanced_lm/gds/projections.py
"""Neo4j GDS in-memory graph projection management and memory estimation."""
from __future__ import annotations

import logging
from typing import Any, Dict, List
from graph_enhanced_lm.graph.neo4j_client import Neo4jClient

logger = logging.getLogger("graph_enhanced_lm")


class GDSProjectionManager:
    """Manages Neo4j GDS in-memory projections safely."""

    def __init__(self, client: Neo4jClient) -> None:
        self.client = client

    def verify_gds_installed(self) -> str:
        """Return installed GDS version or raise an informative error."""
        rows = self.client.execute_read("RETURN gds.version() AS version")
        return str(rows[0]["version"]) if rows else "unknown"

    def projection_exists(self, graph_name: str) -> bool:
        rows = self.client.execute_read(
            "CALL gds.graph.exists($graph_name) YIELD exists RETURN exists",
            {"graph_name": graph_name},
        )
        return bool(rows and rows[0].get("exists"))

    def drop_projection_if_exists(self, graph_name: str) -> None:
        if self.projection_exists(graph_name):
            self.client.execute_write(
                "CALL gds.graph.drop($graph_name, false) YIELD graphName RETURN graphName",
                {"graph_name": graph_name},
            )
            logger.info("Dropped existing GDS projection: %s", graph_name)

    def create_user_interaction_projection(self, graph_name: str = "user_interaction_projection") -> Dict[str, Any]:
        """Materialize explicit user-to-user interaction edges and create a native GDS projection.

        Why materialize (:User)-[:INTERACTED_WITH]->(:User)?
        In GDS, Louvain clusters homogeneous or monopartite node sets most interpretably when
        interactions (retweeting a user's post, replying to a user's post, or mentioning a user)
        are aggregated into weighted user-to-user edges.
        """
        self.drop_projection_if_exists(graph_name)

        # Clear old derived edges so re-running doesn't keep adding to the weights
        self.client.execute_write("MATCH ()-[iw:INTERACTED_WITH]->() DELETE iw")

        # Derive explicit weighted INTERACTED_WITH relationship with provenance='derived_analytical'
        materialize_cypher = """
        MATCH (u1:User)-[:RETWEETED|REPLIED_TO]->(p:Post)<-[:POSTED]-(u2:User)
        WHERE u1.user_id <> u2.user_id
        WITH u1, u2, count(*) AS post_interactions
        MERGE (u1)-[iw:INTERACTED_WITH]->(u2)
        SET iw.weight = toFloat(post_interactions),
            iw.provenance = 'derived_analytical'
        """
        self.client.execute_write(materialize_cypher)

        mention_cypher = """
        MATCH (u1:User)-[m:MENTIONED]->(u2:User)
        WHERE u1.user_id <> u2.user_id
        WITH u1, u2, count(m) AS mention_count
        MERGE (u1)-[iw:INTERACTED_WITH]->(u2)
        SET iw.weight = coalesce(iw.weight, 0.0) + toFloat(mention_count),
            iw.provenance = 'derived_analytical'
        """
        self.client.execute_write(mention_cypher)

        project_cypher = """
        CALL gds.graph.project(
            $graph_name,
            'User',
            {
                INTERACTED_WITH: {
                    type: 'INTERACTED_WITH',
                    orientation: 'UNDIRECTED',
                    properties: 'weight'
                }
            }
        )
        YIELD graphName, nodeCount, relationshipCount, projectMillis
        RETURN graphName, nodeCount, relationshipCount, projectMillis
        """
        rows = self.client.execute_write(project_cypher, {"graph_name": graph_name})
        return rows[0] if rows else {}

    def list_projections(self) -> List[Dict[str, Any]]:
        return self.client.execute_read(
            "CALL gds.graph.list() YIELD graphName, nodeCount, relationshipCount, memoryUsage RETURN graphName, nodeCount, relationshipCount, memoryUsage"
        )