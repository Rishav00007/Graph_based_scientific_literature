# src/graph_enhanced_lm/llm/text_to_cypher.py
"""Natural-language-to-Cypher generation, validation, retry loop, and safe execution."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional

from graph_enhanced_lm.graph.neo4j_client import Neo4jClient
from graph_enhanced_lm.llm.prompts import TEXT_TO_CYPHER_SYSTEM_PROMPT
from graph_enhanced_lm.llm.query_validation import validate_and_sanitize_cypher
from graph_enhanced_lm.llm.qwen_client import QwenClient

CYPHER_BLOCK_RE = re.compile(r"```(?:cypher)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass
class TextToCypherOutput:
    question: str
    generated_cypher: str
    is_valid: bool
    validation_error: Optional[str]
    rows: List[Dict[str, Any]]


class TextToCypherEngine:
    """Converts natural-language questions into validated read-only Cypher queries."""

    def __init__(self, llm: QwenClient, neo4j_client: Optional[Neo4jClient] = None, max_limit: int = 50) -> None:
        self.llm = llm
        self.neo4j_client = neo4j_client
        self.max_limit = max_limit

    @staticmethod
    def extract_cypher_from_text(raw_output: str) -> str:
        match = CYPHER_BLOCK_RE.search(raw_output)
        if match:
            return match.group(1).strip()
        return raw_output.strip()

    def translate(self, question: str) -> TextToCypherOutput:
        """Translate a question into validated Cypher without executing it."""
        raw_llm = self.llm.generate(TEXT_TO_CYPHER_SYSTEM_PROMPT, f"Question: {question}")
        candidate = self.extract_cypher_from_text(raw_llm)
        try:
            validated = validate_and_sanitize_cypher(candidate, max_limit=self.max_limit)
            return TextToCypherOutput(
                question=question,
                generated_cypher=validated,
                is_valid=True,
                validation_error=None,
                rows=[],
            )
        except Exception as exc:
            return TextToCypherOutput(
                question=question,
                generated_cypher=candidate,
                is_valid=False,
                validation_error=str(exc),
                rows=[],
            )

    def query_graph(self, question: str, parameters: Optional[Dict[str, Any]] = None) -> TextToCypherOutput:
        """Translate, validate, and safely execute a read-only Cypher query against Neo4j."""
        translation = self.translate(question)
        if not translation.is_valid or self.neo4j_client is None:
            return translation
        try:
            rows = self.neo4j_client.execute_read(translation.generated_cypher, parameters=parameters)
            translation.rows = rows
            return translation
        except Exception as exc:
            translation.is_valid = False
            translation.validation_error = f"Neo4j Execution Error: {exc}"
            return translation