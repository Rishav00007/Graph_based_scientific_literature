#!/usr/bin/env python3
"""Run Louvain, Degree and PageRank, profile communities and score them against the planted ground truth.

Writes outputs/metrics/community_profiles.json and outputs/metrics/louvain_vs_ground_truth.json.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from graph_enhanced_lm.config import load_yaml_config
from graph_enhanced_lm.gds.community_detection import GDSAnalyticsEngine
from graph_enhanced_lm.gds.projections import GDSProjectionManager
from graph_enhanced_lm.graph.neo4j_client import Neo4jClient


def main() -> int:
    cfg = load_yaml_config("configs/gds.yaml")["gds"]
    graph_name = cfg["default_projection_name"]
    scoring = cfg["candidate_community_scoring"]
    out_dir = Path("outputs/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)

    with Neo4jClient() as client:
        proj = GDSProjectionManager(client)
        print(f"Projection: {proj.create_user_interaction_projection(graph_name)}")

        engine = GDSAnalyticsEngine(client)
        louvain = engine.run_louvain_write(graph_name, cfg["louvain"]["write_property"])
        print(f"Louvain: communities={louvain.get('communityCount')} modularity={louvain.get('modularity', 0):.4f}")
        engine.run_centrality_write(graph_name)
        print("Degree centrality and PageRank written to :User nodes.")

        profiles = engine.compute_community_profiles(
            min_posts=scoring["min_posts"], flagged_ratio_threshold=scoring["flagged_content_ratio_threshold"]
        )
        print("\nCommunity profiles:")
        for p in profiles:
            print(
                f"  community {p['community_id']}: users={p['user_count']} posts={p['total_posts']} "
                f"flagged_ratio={p['flagged_content_ratio']} -> {p['analytical_status']}"
            )

        top_users = client.execute_read(
            "MATCH (u:User) WHERE u.pagerank_score IS NOT NULL "
            "RETURN u.user_id AS user_id, u.louvain_community_id AS community_id, "
            "round(u.pagerank_score, 4) AS pagerank ORDER BY pagerank DESC LIMIT 5"
        )
        print("\nTop 5 users by PageRank:")
        for u in top_users:
            print(f"  {u['user_id']} (community {u['community_id']}): {u['pagerank']}")

        # Compare detected communities with the communities planted by the synthetic generator.
        pairs = client.execute_read(
            "MATCH (u:User) WHERE u.synthetic_ground_truth_community IS NOT NULL "
            "AND u.louvain_community_id IS NOT NULL "
            "RETURN u.synthetic_ground_truth_community AS truth, u.louvain_community_id AS found"
        )
        recovery = {}
        if pairs:
            truth = [int(r["truth"]) for r in pairs]
            found = [int(r["found"]) for r in pairs]
            recovery = {
                "users_compared": len(pairs),
                "adjusted_rand_index": round(adjusted_rand_score(truth, found), 4),
                "normalized_mutual_info": round(normalized_mutual_info_score(truth, found), 4),
            }
            print(f"\nLouvain vs planted communities: {recovery}")

    (out_dir / "community_profiles.json").write_text(
        json.dumps({"louvain": louvain, "profiles": profiles, "top_pagerank_users": top_users}, indent=2, default=str),
        encoding="utf-8",
    )
    (out_dir / "louvain_vs_ground_truth.json").write_text(json.dumps(recovery, indent=2), encoding="utf-8")
    print(f"\nSaved results to {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
