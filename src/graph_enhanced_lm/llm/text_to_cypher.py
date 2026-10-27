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

    def translate(self, question: str, previous: Optional[TextToCypherOutput] = None) -> TextToCypherOutput:
        """Translate a question into validated Cypher without executing it.

        If `previous` is a failed attempt, its query and error are sent back so the model can fix them.
        """
        user_prompt = f"Question: {question}"
        if previous is not None:
            user_prompt += (
                f"\n\nYour previous query was:\n```cypher\n{previous.generated_cypher or '(empty)'}\n```\n"
                f"It failed with this error:\n{previous.validation_error}\n"
                "Write a corrected query that follows the schema and rules."
            )
        try:
            raw_llm = self.llm.generate(TEXT_TO_CYPHER_SYSTEM_PROMPT, user_prompt)
        except Exception as exc:
            # LLM failure (e.g. timeout); "Forbidden" is not in the message, but retrying a timeout rarely helps
            return TextToCypherOutput(question, "", False, f"LLM error (not retried): {exc}", [])
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

    def query_graph(
        self, question: str, parameters: Optional[Dict[str, Any]] = None, max_retries: int = 2
    ) -> TextToCypherOutput:
        """Translate, validate, and safely execute a read-only Cypher query against Neo4j.

        On a validation or execution error, the error is fed back to the LLM for up to `max_retries` fixes.
        Queries blocked as unsafe are never retried.
        """
        translation = self.translate(question)
        for attempt in range(max_retries + 1):
            if translation.is_valid and self.neo4j_client is None:
                return translation
            if translation.is_valid:
                try:
                    translation.rows = self.neo4j_client.execute_read(
                        translation.generated_cypher, parameters=parameters
                    )
                    return translation
                except Exception as exc:
                    translation.is_valid = False
                    translation.validation_error = f"Neo4j Execution Error: {_short_error(exc)}"
            error = translation.validation_error or ""
            if "Forbidden" in error or "not retried" in error or attempt == max_retries:
                return translation
            translation = self.translate(question, previous=translation)
        return translation


def _short_error(exc: Exception) -> str:
    """Keep the useful first part of a Neo4j error (the driver adds long GQL status text)."""
    msg = str(exc)
    if "{message:" in msg:
        msg = msg.split("{message:", 1)[1].split("} {gql_status", 1)[0]
    return msg.strip()[:400]