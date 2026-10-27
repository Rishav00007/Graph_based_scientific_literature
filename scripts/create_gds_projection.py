#!/usr/bin/env python3
"""Build weighted User-User INTERACTED_WITH edges and project them into GDS memory."""
from __future__ import annotations

import sys

from graph_enhanced_lm.gds.projections import GDSProjectionManager
from graph_enhanced_lm.graph.neo4j_client import Neo4jClient


def main(graph_name: str = "user_interaction_projection") -> int:
    with Neo4jClient() as client:
        info = GDSProjectionManager(client).create_user_interaction_projection(graph_name)
        print(f"Projection created: {info}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
