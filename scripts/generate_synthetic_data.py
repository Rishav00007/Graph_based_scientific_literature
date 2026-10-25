#!/usr/bin/env python3
"""Generate a reproducible synthetic social-network dataset with planted communities and election windows.

All generated records are explicitly marked with `is_synthetic=True` and `dataset_source='synthetic_v1'`.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import random
from typing import Any, Dict, List
import pandas as pd

from graph_enhanced_lm.utils.reproducibility import set_global_seed

TEMPLATES = {
    "NORMAL": [
        "Reading the new peer-reviewed paper on graph neural architectures and citation networks #GraphML #Research",
        "Local polling stations open at 7 AM tomorrow, remember to bring valid ID #CivicDuty #Election2026",
        "Great seminar today on Neo4j property graphs and Cypher query optimization #DataScience #Neo4j",
    ],
    "HATE": [
        "[SYNTHETIC_HATE_EXAMPLE] Hate speech pattern targeting out-group faction traits must be eradicated #PolarizeNow",
        "[SYNTHETIC_HATE_EXAMPLE] Subhuman traitors must pay for ruining the country #ExtremistTag",
    ],
    "OFFENSIVE": [
        "[SYNTHETIC_OFFENSIVE_EXAMPLE] These corrupt clowns and stupid fools have garbage takes on policy #Rant",
        "[SYNTHETIC_OFFENSIVE_EXAMPLE] Shut up you clueless idiot nobody asked for your opinion #Debate",
    ],
    "COMMUNITY_TARGETED": [
        "[SYNTHETIC_TARGETED_EXAMPLE] Boycott community X and expel group members from every civic forum #TargetFaction",
        "[SYNTHETIC_TARGETED_EXAMPLE] Outsiders must leave our district before election day #GroupTarget",
    ],
    "SPAM": [
        "[SYNTHETIC_SPAM_EXAMPLE] Free giveaway crypto airdrop click link now for guaranteed profit promo code #FreeCrypto",
        "[SYNTHETIC_SPAM_EXAMPLE] Guaranteed profit 100x token sale click link immediately #SpamPromo",
    ],
}


def generate_synthetic_dataset(
    output_dir: str | Path = "data/synthetic",
    seed: int = 42,
    election_date: str = "2026-11-01",
) -> Dict[str, Path]:
    """Generate canonical synthetic users.csv, posts.csv, and interactions.csv."""
    set_global_seed(seed)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    community_specs = [
        {"cid": 0, "count": 14, "weights": {"NORMAL": 0.90, "HATE": 0.02, "OFFENSIVE": 0.04, "COMMUNITY_TARGETED": 0.02, "SPAM": 0.02}},
        {"cid": 1, "count": 12, "weights": {"NORMAL": 0.92, "HATE": 0.01, "OFFENSIVE": 0.03, "COMMUNITY_TARGETED": 0.01, "SPAM": 0.03}},
        {"cid": 2, "count": 11, "weights": {"NORMAL": 0.15, "HATE": 0.45, "OFFENSIVE": 0.20, "COMMUNITY_TARGETED": 0.15, "SPAM": 0.05}},
        {"cid": 3, "count": 11, "weights": {"NORMAL": 0.20, "HATE": 0.04, "OFFENSIVE": 0.06, "COMMUNITY_TARGETED": 0.05, "SPAM": 0.65}},
    ]

    users: List[Dict[str, Any]] = []
    comm_to_users: Dict[int, List[str]] = {0: [], 1: [], 2: [], 3: []}
    user_counter = 1
    for spec in community_specs:
        cid = spec["cid"]
        for _ in range(spec["count"]):
            uid = f"syn_user_{user_counter:03d}"
            users.append(
                {
                    "user_id": uid,
                    "username": f"synthetic_handle_{user_counter:03d}",
                    "dataset_source": "synthetic_v1",
                    "is_synthetic": True,
                    "synthetic_ground_truth_community": cid,
                }
            )
            comm_to_users[cid].append(uid)
            user_counter += 1

    election_dt = datetime.fromisoformat(election_date).replace(tzinfo=timezone.utc)
    posts: List[Dict[str, Any]] = []
    comm_to_posts: Dict[int, List[str]] = {0: [], 1: [], 2: [], 3: []}

    post_counter = 1
    for spec in community_specs:
        cid = spec["cid"]
        categories = list(spec["weights"].keys())
        probs = list(spec["weights"].values())
        num_comm_posts = 55 if cid in (2, 3) else 45

        for _ in range(num_comm_posts):
            author = random.choice(comm_to_users[cid])
            label = random.choices(categories, weights=probs, k=1)[0]
            text = random.choice(TEMPLATES[label])

            # Concentrate problematic community 2 & 3 activity in the 10 days before the election
            if cid in (2, 3) and label != "NORMAL" and random.random() < 0.72:
                days_offset = -random.randint(1, 10)
            else:
                days_offset = random.randint(-28, 10)

            ts = election_dt + timedelta(days=days_offset, hours=random.randint(0, 22), minutes=random.randint(0, 59))
            pid = f"syn_post_{post_counter:04d}"
            hashtags = [w[1:].lower() for w in text.split() if w.startswith("#")]

            posts.append(
                {
                    "post_id": pid,
                    "user_id": author,
                    "text": text,
                    "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "language": "en",
                    "hashtags": ",".join(hashtags),
                    "content_label": label,
                    "label_source": "synthetic_ground_truth",
                    "classification_confidence": 1.0,
                    "classifier_model": "synthetic_generator",
                    "classifier_version": "1.0.0",
                    "dataset_source": "synthetic_v1",
                    "is_synthetic": True,
                }
            )
            comm_to_posts[cid].append(pid)
            post_counter += 1

    interactions: List[Dict[str, Any]] = []
    inter_counter = 1
    post_ts_map = {p["post_id"]: datetime.fromisoformat(p["timestamp"].replace("Z", "+00:00")) for p in posts}

    # Dense intra-community retweets/replies/mentions (90% intra-community, 10% cross-community)
    for spec in community_specs:
        cid = spec["cid"]
        members = comm_to_users[cid]
        c_posts = comm_to_posts[cid]
        for _ in range(95):
            src_user = random.choice(members)
            if random.random() < 0.90:
                target_post = random.choice(c_posts)
                target_user = random.choice([u for u in members if u != src_user])
            else:
                other_cid = random.choice([c for c in (0, 1, 2, 3) if c != cid])
                target_post = random.choice(comm_to_posts[other_cid])
                target_user = random.choice(comm_to_users[other_cid])

            itype = random.choices(["RETWEETED", "REPLIED_TO", "MENTIONED"], weights=[0.55, 0.25, 0.20], k=1)[0]
            base_ts = post_ts_map[target_post] + timedelta(minutes=random.randint(5, 360))
            interactions.append(
                {
                    "interaction_id": f"syn_inter_{inter_counter:04d}",
                    "source_user_id": src_user,
                    "target_post_id": target_post if itype in {"RETWEETED", "REPLIED_TO"} else None,
                    "target_user_id": target_user if itype == "MENTIONED" else None,
                    "interaction_type": itype,
                    "timestamp": base_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "dataset_source": "synthetic_v1",
                    "is_synthetic": True,
                }
            )
            inter_counter += 1

    users_file = out_path / "users.csv"
    posts_file = out_path / "posts.csv"
    inter_file = out_path / "interactions.csv"

    pd.DataFrame(users).to_csv(users_file, index=False)
    pd.DataFrame(posts).to_csv(posts_file, index=False)
    pd.DataFrame(interactions).to_csv(inter_file, index=False)
    return {"users": users_file, "posts": posts_file, "interactions": inter_file}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic social-network dataset.")
    parser.add_argument("--output-dir", default="data/synthetic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--election-date", default="2026-11-01")
    args = parser.parse_args()
    paths = generate_synthetic_dataset(args.output_dir, args.seed, args.election_date)
    print(f"Generated synthetic dataset files: {paths}")