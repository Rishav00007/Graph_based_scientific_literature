#!/usr/bin/env python3
"""Check the Neo4j connection and GDS plugin, then create constraints and indexes.

Usage: python scripts/initialize_neo4j.py [--wipe]
"""
from __future__ import annotations

import argparse
import sys

from graph_enhanced_lm.gds.projections import GDSProjectionManager
from graph_enhanced_lm.graph.constraints import initialize_schema
from graph_enhanced_lm.graph.neo4j_client import Neo4jClient
from graph_enhanced_lm.utils.errors import Neo4jConnectionError


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialize Neo4j schema.")
    parser.add_argument("--wipe", action="store_true", help="Delete ALL nodes and relationships first.")
    args = parser.parse_args()

    try:
        with Neo4jClient() as client:
            print(f"Connected to Neo4j at {client.settings.uri}")
            try:
                print(f"GDS version: {GDSProjectionManager(client).verify_gds_installed()}")
            except Exception as exc:
                print(f"WARNING: Graph Data Science plugin not available ({exc}). Louvain/PageRank will fail.")

            if args.wipe:
                client.execute_write("MATCH (n) DETACH DELETE n")
                print("Wiped all nodes and relationships.")

            initialize_schema(client)
            print("Constraints and indexes created.")
    except Neo4jConnectionError as exc:
        print(f"ERROR: {exc}\nIs Neo4j running? Try: docker compose up -d", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
