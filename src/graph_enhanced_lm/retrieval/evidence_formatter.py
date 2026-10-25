# src/graph_enhanced_lm/retrieval/evidence_formatter.py & graph_rag.py
"""Graph RAG evidence formatting and grounded answer generation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
import json

from graph_enhanced_lm.llm.prompts import GRAPH_RAG_SYSTEM_PROMPT
from graph_enhanced_lm.llm.qwen_client import QwenClient
from graph_enhanced_lm.llm.text_to_cypher import TextToCypherEngine


def format_graph_evidence_bundle(
    question: str,
    cypher_query: str,
    records: List[Dict[str, Any]],
) -> str:
    """Format structured Neo4j records into a deterministic, auditable Evidence Bundle for Qwen3-8B."""
    if not records:
        return (
            f"Question: {question}\n"
            f"Executed Cypher: {cypher_query}\n"
            "Graph Evidence Bundle: No matching graph records were returned."
        )
    lines = [
        f"Question: {question}",
        f"Executed Cypher: {cypher_query}",
        f"Total Records Retrieved: {len(records)}",
        "Graph Evidence Records:",
    ]
    for idx, rec in enumerate(records, start=1):
        lines.append(f"  [Record {idx}] {json.dumps(rec, default=str)}")
    return "\n".join(lines)


@dataclass
class GraphRAGResponse:
    question: str
    executed_cypher: str
    evidence_records: List[Dict[str, Any]]
    formatted_evidence: str
    grounded_answer: str
    has_sufficient_evidence: bool


class GraphRAGPipeline:
    """End-to-end Graph RAG orchestrator separating graph retrieval from LLM answer synthesis."""

    def __init__(self, text_to_cypher: TextToCypherEngine, llm: QwenClient) -> None:
        self.text_to_cypher = text_to_cypher
        self.llm = llm

    def answer_question(self, question: str, fallback_records: List[Dict[str, Any]] | None = None) -> GraphRAGResponse:
        t2c_out = self.text_to_cypher.query_graph(question)
        records = t2c_out.rows if t2c_out.rows else (fallback_records or [])
        evidence_str = format_graph_evidence_bundle(question, t2c_out.generated_cypher, records)

        if not records:
            return GraphRAGResponse(
                question=question,
                executed_cypher=t2c_out.generated_cypher,
                evidence_records=[],
                formatted_evidence=evidence_str,
                grounded_answer="Insufficient graph evidence was retrieved to answer this question.",
                has_sufficient_evidence=False,
            )

        answer = self.llm.generate(GRAPH_RAG_SYSTEM_PROMPT, evidence_str)
        return GraphRAGResponse(
            question=question,
            executed_cypher=t2c_out.generated_cypher,
            evidence_records=records,
            formatted_evidence=evidence_str,
            grounded_answer=answer,
            has_sufficient_evidence=True,
        )