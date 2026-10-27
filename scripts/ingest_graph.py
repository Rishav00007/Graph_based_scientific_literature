#!/usr/bin/env python3
"""Load the canonical CSVs, validate them and ingest them into Neo4j.

Usage: python scripts/ingest_graph.py [--data-dir data/synthetic] [--generate]
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph_enhanced_lm.data.loaders import load_canonical_tables
from graph_enhanced_lm.data.preprocessing import deduplicate_canonical_tables
from graph_enhanced_lm.data.validators import validate_canonical_dataframes
from graph_enhanced_lm.graph.constraints import initialize_schema
from graph_enhanced_lm.graph.ingestion import ingest_canonical_data
from graph_enhanced_lm.graph.neo4j_client import Neo4jClient


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest canonical CSVs into Neo4j.")
    parser.add_argument("--data-dir", default="data/synthetic")
    parser.add_argument("--generate", action="store_true", help="(Re)generate synthetic data before ingesting.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.generate or not (Path(args.data_dir) / "users.csv").exists():
        from scripts.generate_synthetic_data import generate_synthetic_dataset

        generate_synthetic_dataset(args.data_dir, seed=args.seed)
        print(f"Generated synthetic data in {args.data_dir}")

    users, posts, inters = load_canonical_tables(args.data_dir)
    users, posts, inters = deduplicate_canonical_tables(users, posts, inters)
    report = validate_canonical_dataframes(users, posts, inters, strict=True)
    print(f"Validated: users={report.num_users} posts={report.num_posts} interactions={report.num_interactions}")

    with Neo4jClient() as client:
        initialize_schema(client)
        counts = ingest_canonical_data(client, users, posts, inters)
        print(f"Ingested: {counts}")
        summary = client.execute_read("MATCH (n) RETURN labels(n)[0] AS label, count(*) AS count ORDER BY label")
        for row in summary:
            print(f"  {row['label']}: {row['count']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
