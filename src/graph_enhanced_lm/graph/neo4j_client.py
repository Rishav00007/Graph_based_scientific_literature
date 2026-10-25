# src/graph_enhanced_lm/graph/neo4j_client.py
"""Official Neo4j Python driver wrapper with read/write session isolation and timeouts."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from neo4j import GraphDatabase, Driver, RoutingControl

from graph_enhanced_lm.config import Neo4jSettings
from graph_enhanced_lm.utils.errors import Neo4jConnectionError

logger = logging.getLogger("graph_enhanced_lm")


class Neo4jClient:
    """Manages Neo4j connections, read-only query execution, and parameterized write transactions."""

    def __init__(self, settings: Optional[Neo4jSettings] = None) -> None:
        self.settings = settings or Neo4jSettings()
        self._driver: Optional[Driver] = None

    def connect(self) -> None:
        """Initialize the Neo4j driver and verify connectivity."""
        try:
            self._driver = GraphDatabase.driver(
                self.settings.uri,
                auth=(self.settings.user, self.settings.password),
            )
            self._driver.verify_connectivity()
            logger.info("Connected to Neo4j at %s", self.settings.uri)
        except Exception as exc:
            raise Neo4jConnectionError(f"Failed to connect to Neo4j at {self.settings.uri}: {exc}") from exc

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def __enter__(self) -> "Neo4jClient":
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def execute_read(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Execute a parameterized read-only Cypher query with timeout protection."""
        if self._driver is None:
            self.connect()
        assert self._driver is not None
        params = parameters or {}
        records, _, _ = self._driver.execute_query(
            query,
            parameters_=params,
            database_=self.settings.database,
            routing_=RoutingControl.READ,
            timeout=timeout or self.settings.query_timeout_seconds,
        )
        return [dict(record) for record in records]

    def execute_write(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute a parameterized write Cypher query (used only by trusted ingestion/GDS modules)."""
        if self._driver is None:
            self.connect()
        assert self._driver is not None
        params = parameters or {}
        records, _, _ = self._driver.execute_query(
            query,
            parameters_=params,
            database_=self.settings.database,
            routing_=RoutingControl.WRITE,
        )
        return [dict(record) for record in records]