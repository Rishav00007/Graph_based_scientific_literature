# src/graph_enhanced_lm/retrieval/evidence_formatter.py & graph_rag.py
"""Graph RAG evidence formatting and grounded answer generation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List
import json
import re

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


NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def find_ungrounded_numbers(answer: str, evidence: str) -> List[str]:
    """Return numbers in the answer that do not appear anywhere in the evidence bundle.

    A cheap hallucination check: an LLM answer should only quote figures it was given.
    List numbering ("1.", "2.") and percentages of evidence ratios (0.84 -> 84%) are allowed.
    """
    known = {float(n) for n in NUMBER_RE.findall(evidence)}
    ungrounded = []
    for raw in NUMBER_RE.findall(answer):
        value = float(raw)
        if value in known or round(value / 100, 4) in known:
            continue
        if value.is_integer() and 1 <= value <= 20 and re.search(rf"(^|\n)\s*{int(value)}\.\s", answer):
            continue  # list numbering
        ungrounded.append(raw)
    return sorted(set(ungrounded), key=ungrounded.index)


@dataclass
class GraphRAGResponse:
    question: str
    executed_cypher: str
    evidence_records: List[Dict[str, Any]]
    formatted_evidence: str
    grounded_answer: str
    has_sufficient_evidence: bool
    ungrounded_numbers: List[str] = field(default_factory=list)

    @property
    def passes_grounding_check(self) -> bool:
        return not self.ungrounded_numbers


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

        try:
            answer = self.llm.generate(GRAPH_RAG_SYSTEM_PROMPT, evidence_str)
        except Exception as exc:
            return GraphRAGResponse(
                question=question,
                executed_cypher=t2c_out.generated_cypher,
                evidence_records=records,
                formatted_evidence=evidence_str,
                grounded_answer=f"Answer generation failed: {exc}",
                has_sufficient_evidence=False,
            )
        return GraphRAGResponse(
            question=question,
            executed_cypher=t2c_out.generated_cypher,
            evidence_records=records,
            formatted_evidence=evidence_str,
            grounded_answer=answer,
            has_sufficient_evidence=True,
            ungrounded_numbers=find_ungrounded_numbers(answer, f"{question}\n{evidence_str}"),
        )