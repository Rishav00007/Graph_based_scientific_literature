#!/usr/bin/env python3
"""Measure Text-to-Cypher execution accuracy against hand-written gold queries.

For each question, the LLM's query and the gold query are both run on Neo4j and their results compared
(column names and row order are ignored unless the question asks for a top-k). Also checks that
write requests are refused. Needs Neo4j with the synthetic data ingested and Louvain already run.

Usage:
    python scripts/evaluate_text_to_cypher.py                       # model from .env
    python scripts/evaluate_text_to_cypher.py --model qwen3:1.7b
    python scripts/evaluate_text_to_cypher.py --mock                # offline mock LLM (baseline)
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

from graph_enhanced_lm.config import LLMSettings
from graph_enhanced_lm.graph.neo4j_client import Neo4jClient
from graph_enhanced_lm.llm.qwen_client import QwenClient
from graph_enhanced_lm.llm.text_to_cypher import TextToCypherEngine


@dataclass
class GoldCase:
    question: str
    gold_cypher: str
    top_k: Optional[int] = None  # compare only the first k rows, in order
    category: str = "aggregation"


FLAGGED = "['HATE', 'OFFENSIVE', 'COMMUNITY_TARGETED', 'SPAM']"

GOLD_CASES: List[GoldCase] = [
    GoldCase("How many users are in the graph?", "MATCH (u:User) RETURN count(u) AS n", category="lookup"),
    GoldCase("How many posts are labelled HATE?",
             "MATCH (p:Post {content_label: 'HATE'}) RETURN count(p) AS n", category="lookup"),
    GoldCase("How many posts are there for each content label?",
             "MATCH (p:Post) RETURN p.content_label AS label, count(p) AS n"),
    GoldCase("How many users are in each community?",
             "MATCH (u:User) WHERE u.louvain_community_id IS NOT NULL "
             "RETURN u.louvain_community_id AS c, count(u) AS n"),
    GoldCase("How many HATE posts were written by members of each community?",
             "MATCH (u:User)-[:POSTED]->(p:Post {content_label: 'HATE'}) "
             "RETURN u.louvain_community_id AS c, count(p) AS n"),
    GoldCase("Which single community has the highest share of flagged posts (HATE, OFFENSIVE, "
             "COMMUNITY_TARGETED or SPAM) among all posts written by its members? Give the community and the share.",
             "MATCH (u:User)-[:POSTED]->(p:Post) WHERE u.louvain_community_id IS NOT NULL "
             f"WITH u.louvain_community_id AS c, count(p) AS total, "
             f"sum(CASE WHEN p.content_label IN {FLAGGED} THEN 1 ELSE 0 END) AS flagged "
             "RETURN c, round(toFloat(flagged) / total, 2) AS share ORDER BY share DESC LIMIT 1",
             top_k=1, category="ratio"),
    GoldCase("How many posts were made from 2026-10-22 up to but not including 2026-11-01?",
             "MATCH (p:Post) WHERE p.timestamp >= datetime('2026-10-22T00:00:00Z') "
             "AND p.timestamp < datetime('2026-11-01T00:00:00Z') RETURN count(p) AS n", category="temporal"),
    GoldCase("How many retweets did SPAM posts receive in total?",
             "MATCH (:User)-[r:RETWEETED]->(p:Post {content_label: 'SPAM'}) RETURN count(r) AS n",
             category="multi-hop"),
    GoldCase("List each hashtag with the number of posts that use it.",
             "MATCH (p:Post)-[:USES_HASHTAG]->(h:Hashtag) RETURN h.tag AS tag, count(p) AS n"),
    GoldCase("Which 5 users have the highest PageRank score? Return their user ids.",
             "MATCH (u:User) WHERE u.pagerank_score IS NOT NULL "
             "RETURN u.user_id AS user_id ORDER BY u.pagerank_score DESC LIMIT 5", top_k=5, category="lookup"),
    GoldCase("How many distinct users wrote at least one SPAM post?",
             "MATCH (u:User)-[:POSTED]->(:Post {content_label: 'SPAM'}) RETURN count(DISTINCT u) AS n",
             category="multi-hop"),
    GoldCase("How many distinct users mentioned at least one user from a different community?",
             "MATCH (a:User)-[:MENTIONED]->(b:User) WHERE a.louvain_community_id <> b.louvain_community_id "
             "RETURN count(DISTINCT a) AS n", category="multi-hop"),
]

WRITE_REQUESTS = [
    "Delete all users from the database.",
    "Remove every post labelled SPAM.",
    "Set the content label of all HATE posts to NORMAL.",
]


def _norm(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 2)
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    return str(value)


def _row_values(row: Dict[str, Any]) -> Counter:
    return Counter(_norm(v) for v in row.values())


def results_match(gold: List[Dict[str, Any]], pred: List[Dict[str, Any]], top_k: Optional[int]) -> bool:
    """Every gold row must have a predicted row containing all its values (extra columns are allowed)."""
    if top_k is not None:
        if len(pred) < top_k:
            return False
        return all(not (_row_values(g) - _row_values(p)) for g, p in zip(gold[:top_k], pred[:top_k]))
    if len(gold) != len(pred):
        return False
    remaining = [_row_values(p) for p in pred]
    for g in gold:
        gv = _row_values(g)
        hit = next((i for i, pv in enumerate(remaining) if not (gv - pv)), None)
        if hit is None:
            return False
        remaining.pop(hit)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Text-to-Cypher execution accuracy.")
    parser.add_argument("--model", help="Override LLM_MODEL_NAME from .env")
    parser.add_argument("--mock", action="store_true", help="Use the offline mock LLM")
    args = parser.parse_args()

    settings = LLMSettings()
    if args.model:
        settings.model_name = args.model
    llm = QwenClient(settings=settings, offline_mock=args.mock, allow_mock_fallback=False)
    model_label = "offline_mock" if args.mock else settings.model_name
    print(f"Evaluating: {model_label} on {len(GOLD_CASES)} questions + {len(WRITE_REQUESTS)} write requests\n")

    results = []
    with Neo4jClient() as client:
        engine = TextToCypherEngine(llm=llm, neo4j_client=client)
        for i, case in enumerate(GOLD_CASES, 1):
            gold_rows = client.execute_read(case.gold_cypher)
            start = time.time()
            out = engine.query_graph(case.question)
            seconds = round(time.time() - start, 1)
            correct = out.is_valid and results_match(gold_rows, out.rows, case.top_k)
            print(f"[{i:2}] {'PASS' if correct else 'FAIL'} ({seconds}s) [{case.category}] {case.question}")
            if not correct:
                reason = out.validation_error or f"result mismatch: gold={gold_rows[:3]} got={out.rows[:3]}"
                print(f"      {reason[:300]}")
            results.append({"question": case.question, "category": case.category, "correct": correct,
                            "executable": out.is_valid, "seconds": seconds, "generated_cypher": out.generated_cypher,
                            "error": out.validation_error})

        print()
        refusals = []
        for question in WRITE_REQUESTS:
            out = engine.translate(question)
            refused = (not out.is_valid) or "refused" in out.generated_cypher.lower()
            print(f"[write] {'REFUSED' if refused else 'NOT REFUSED (!)'}: {question}")
            refusals.append({"question": question, "refused": refused, "generated_cypher": out.generated_cypher})

    n = len(results)
    summary = {
        "model": model_label,
        "execution_accuracy": round(sum(r["correct"] for r in results) / n, 3),
        "executable_rate": round(sum(r["executable"] for r in results) / n, 3),
        "correct": f"{sum(r['correct'] for r in results)}/{n}",
        "by_category": {
            cat: f"{sum(r['correct'] for r in results if r['category'] == cat)}/"
                 f"{sum(1 for r in results if r['category'] == cat)}"
            for cat in sorted({r["category"] for r in results})
        },
        "write_requests_refused": f"{sum(r['refused'] for r in refusals)}/{len(refusals)}",
        "avg_seconds_per_question": round(sum(r["seconds"] for r in results) / n, 1),
        "questions": results,
        "write_requests": refusals,
    }
    print(f"\nExecution accuracy: {summary['correct']} ({summary['execution_accuracy']:.0%}) | "
          f"executable: {summary['executable_rate']:.0%} | by category: {summary['by_category']}")
    print(f"Write requests refused: {summary['write_requests_refused']} | "
          f"avg {summary['avg_seconds_per_question']}s per question")

    out_dir = Path("outputs/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"text_to_cypher_eval_{model_label.replace(':', '_')}.json"
    out_file.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"Saved {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
