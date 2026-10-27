# src/graph_enhanced_lm/retrieval/graph_rag.py
"""Graph RAG pipeline entry point (implementation lives in evidence_formatter.py)."""
from graph_enhanced_lm.retrieval.evidence_formatter import (
    GraphRAGPipeline,
    GraphRAGResponse,
    format_graph_evidence_bundle,
)

__all__ = ["GraphRAGPipeline", "GraphRAGResponse", "format_graph_evidence_bundle"]
