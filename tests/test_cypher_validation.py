# tests/test_cypher_validation.py
"""Text-to-Cypher safety: the validator must block every mutating/admin query and pass normal reads."""
from __future__ import annotations

import pytest

from graph_enhanced_lm.llm.query_validation import validate_and_sanitize_cypher
from graph_enhanced_lm.utils.errors import CypherSyntaxValidationError, UnsafeCypherError

SAFE_QUERIES = [
    "MATCH (u:User) RETURN u.user_id LIMIT 10",
    "MATCH (u:User)-[:POSTED]->(p:Post) WHERE p.content_label = 'HATE' RETURN u.user_id, count(p) AS n ORDER BY n DESC LIMIT 5",
    "MATCH (p:Post) WHERE p.timestamp >= datetime('2026-10-22T00:00:00Z') RETURN p.post_id LIMIT 20",
    "OPTIONAL MATCH (c:Community) RETURN c.community_id",
    "WITH 1 AS x RETURN x",
    "MATCH (u:User) CALL { WITH u MATCH (u)-[:POSTED]->(p) RETURN count(p) AS n } RETURN u.user_id, n LIMIT 5",
    "MATCH (u:User) RETURN u.dataset_source, u.username LIMIT 5",  # 'dataset' must not trip the SET rule
    "// top users\nMATCH (u:User) RETURN u.user_id LIMIT 5;",
]

UNSAFE_QUERIES = [
    # Direct writes
    "CREATE (n:User {user_id: 'x'}) RETURN n",
    "MATCH (u:User) SET u.flag = true RETURN u",
    "MATCH (u:User) DETACH DELETE u RETURN count(*)",
    "MATCH (u:User) REMOVE u.username RETURN u",
    "MERGE (u:User {user_id: 'x'}) RETURN u",
    "MATCH (u:User) FOREACH (x IN [1] | SET u.a = x) RETURN u",
    "MATCH (n) WITH n CREATE (m:Copy) RETURN m",
    # Writes hidden in a subquery
    "MATCH (u:User) CALL { WITH u CREATE (:Log) } RETURN u",
    # Multi-statement injection
    "MATCH (u:User) RETURN u; MATCH (n) DETACH DELETE n",
    # Procedures: admin, schema, GDS writes, APOC
    "CALL dbms.security.listUsers() YIELD username RETURN username",
    "MATCH (n) CALL db.createLabel('Pwned') RETURN n",
    "MATCH (n) CALL gds.graph.drop('user_interaction_projection') YIELD graphName RETURN graphName",
    "MATCH (n) CALL gds.graph.nodeProperties.write('g', ['x']) YIELD propertiesWritten RETURN propertiesWritten",
    "MATCH (n) CALL gds.louvain.write('g', {writeProperty: 'c'}) YIELD communityCount RETURN communityCount",
    "MATCH (n) CALL apoc.cypher.runWrite('MATCH (n) DELETE n', {}) YIELD value RETURN value",
    "RETURN apoc.cypher.runFirstColumnSingle('CRE' + 'ATE (n) RETURN n', {}) AS x",
    "MATCH (n) CALL\napoc.periodic.iterate('a','b',{}) YIELD batches RETURN batches",
    # Data loading and access control
    "LOAD CSV FROM 'file:///etc/passwd' AS line RETURN line",
    "MATCH (n) RETURN n UNION LOAD CSV FROM 'http://evil' AS l RETURN l",
    "GRANT ROLE admin TO neo4j",
    "SHOW USERS",
    # Comment tricks: the forbidden word must be found after comments are stripped
    "MATCH (n) /* harmless */ DETACH /* x */ DELETE n RETURN 1",
    # Malformed
    "",
    "// only a comment",
    "MATCH (u:User)",
]


@pytest.mark.parametrize("query", SAFE_QUERIES)
def test_safe_queries_pass(query: str) -> None:
    out = validate_and_sanitize_cypher(query, max_limit=50)
    assert "LIMIT" in out.upper()


@pytest.mark.parametrize("query", UNSAFE_QUERIES)
def test_unsafe_queries_blocked(query: str) -> None:
    with pytest.raises((UnsafeCypherError, CypherSyntaxValidationError)):
        validate_and_sanitize_cypher(query)


def test_limit_is_capped_and_added() -> None:
    assert validate_and_sanitize_cypher("MATCH (u:User) RETURN u LIMIT 999", max_limit=50).endswith("LIMIT 50")
    assert validate_and_sanitize_cypher("MATCH (u:User) RETURN u", max_limit=25).endswith("LIMIT 25")
