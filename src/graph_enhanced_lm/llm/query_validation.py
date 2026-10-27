# src/graph_enhanced_lm/llm/query_validation.py
"""Strict Read-Only Cypher Safety Validator for Text-to-Cypher execution."""
from __future__ import annotations

import re
from graph_enhanced_lm.utils.errors import CypherSyntaxValidationError, UnsafeCypherError

# Block all mutating, administrative, file-loading, or schema-altering clauses
FORBIDDEN_CYPHER_PATTERNS = [
    r"\bCREATE\b",
    r"\bMERGE\b",
    r"\bDELETE\b",
    r"\bDETACH\b",
    r"\bSET\b",
    r"\bREMOVE\b",
    r"\bDROP\b",
    r"\bLOAD\b",
    r"\bFOREACH\b",
    # Any procedure call (db.*, dbms.*, gds.*, apoc.*, ...). CALL { subquery } stays allowed;
    # mutations inside a subquery are still caught by the patterns above.
    r"\bCALL\s*(?!\{)\S",
    # APOC can run arbitrary Cypher from strings (apoc.cypher.runWrite etc.), even without CALL
    r"\bAPOC\s*\.",
    r"\bADMIN\b",
    r"\bGRANT\b",
    r"\bREVOKE\b",
    r"\bDENY\b",
    r"\bTERMINATE\b",
]

ALLOWED_START_KEYWORDS = ("MATCH", "OPTIONAL MATCH", "WITH", "RETURN", "UNWIND")

LINE_COMMENT_RE = re.compile(r"//.*?$", re.MULTILINE)
BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
LIMIT_CLAUSE_RE = re.compile(r"\bLIMIT\s+(\d+)\b", re.IGNORECASE)


def strip_cypher_comments(query: str) -> str:
    """Remove single-line and multi-line comments before inspecting tokens."""
    no_block = BLOCK_COMMENT_RE.sub(" ", query)
    no_line = LINE_COMMENT_RE.sub(" ", no_block)
    return no_line.strip()


def validate_and_sanitize_cypher(query: str, max_limit: int = 50) -> str:
    """Validate that a generated Cypher query is strictly read-only, single-statement, and bounded."""
    if not isinstance(query, str) or not query.strip():
        raise CypherSyntaxValidationError("Empty Cypher query provided.")

    cleaned = strip_cypher_comments(query)
    if not cleaned:
        raise CypherSyntaxValidationError("Query contains only comments.")

    # Strip trailing semicolon if present, then reject any internal semicolons (multi-statement injection)
    if cleaned.endswith(";"):
        cleaned = cleaned[:-1].strip()
    if ";" in cleaned:
        raise UnsafeCypherError("Multiple Cypher statements separated by ';' are forbidden in read-only mode.")

    upper_cleaned = cleaned.upper()
    for pattern in FORBIDDEN_CYPHER_PATTERNS:
        if re.search(pattern, upper_cleaned):
            raise UnsafeCypherError(f"Forbidden mutating or administrative Cypher pattern detected: {pattern}")

    if not upper_cleaned.startswith(ALLOWED_START_KEYWORDS):
        raise CypherSyntaxValidationError(
            f"Query must start with one of {ALLOWED_START_KEYWORDS}. Got: {cleaned[:30]}..."
        )

    if "RETURN" not in upper_cleaned:
        raise CypherSyntaxValidationError("Read-only query must include a RETURN clause.")

    # Enforce or cap LIMIT clause
    limit_match = LIMIT_CLAUSE_RE.search(cleaned)
    if limit_match:
        requested_limit = int(limit_match.group(1))
        if requested_limit > max_limit:
            cleaned = LIMIT_CLAUSE_RE.sub(f"LIMIT {max_limit}", cleaned)
    else:
        cleaned = f"{cleaned} LIMIT {max_limit}"

    return cleaned