# src/graph_enhanced_lm/utils/errors.py
"""Custom exception hierarchy for the Graph-Enhanced LM project."""


class GraphEnhancedLMError(Exception):
    """Base exception for all project errors."""


class DatasetValidationError(GraphEnhancedLMError):
    """Raised when input dataset records violate canonical schema requirements."""


class UnsafeCypherError(GraphEnhancedLMError):
    """Raised when a Text-to-Cypher query contains forbidden or mutating operations."""


class CypherSyntaxValidationError(GraphEnhancedLMError):
    """Raised when a generated Cypher query fails structural validation."""


class Neo4jConnectionError(GraphEnhancedLMError):
    """Raised when Neo4j or GDS is unreachable or misconfigured."""