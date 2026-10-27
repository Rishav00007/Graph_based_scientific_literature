#!/usr/bin/env python3
"""Stage 1 end-to-end demo on the synthetic benchmark.

Offline (default, no services needed):
    python scripts/run_demo.py
Live (Neo4j + GDS running, and/or Ollama serving a Qwen3 model):
    python scripts/run_demo.py --neo4j --llm
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

import httpx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph_enhanced_lm.classification.evaluation import evaluate_predictions
from graph_enhanced_lm.classification.transformer_classifier import PostContentClassifier
from graph_enhanced_lm.config import LLMSettings
from graph_enhanced_lm.data.loaders import load_canonical_tables
from graph_enhanced_lm.data.preprocessing import deduplicate_canonical_tables
from graph_enhanced_lm.data.validators import validate_canonical_dataframes
from graph_enhanced_lm.llm.query_validation import validate_and_sanitize_cypher
from graph_enhanced_lm.llm.qwen_client import QwenClient
from graph_enhanced_lm.llm.text_to_cypher import TextToCypherEngine
from graph_enhanced_lm.retrieval.graph_rag import GraphRAGPipeline
from graph_enhanced_lm.temporal.election_windows import analyze_temporal_dataframe
from scripts.generate_synthetic_data import generate_synthetic_dataset

DATA_DIR = "data/synthetic"
METRICS_DIR = Path("outputs/metrics")
FLAGGED_LABELS = ["HATE", "OFFENSIVE", "COMMUNITY_TARGETED", "SPAM"]

DEMO_QUESTIONS = [
    "Which communities contain a high proportion of hate-labelled posts?",
    "Which posts were repeatedly amplified through retweets?",
    "Show the most recent hate posts and their authors.",
    "Delete all users from the database.",  # adversarial: must never produce an executed write
]

ATTACK_QUERIES = [
    "MATCH (n) DETACH DELETE n RETURN count(*)",
    "MATCH (n) CALL apoc.cypher.runWrite('MATCH (n) DELETE n', {}) YIELD value RETURN value",
    "MATCH (u:User) RETURN u; MATCH (n) SET n.pwned = true",
]

TEMPORAL_CYPHER = """
MATCH (p:Post)
WHERE p.is_synthetic = true
  AND p.timestamp >= datetime($start) AND p.timestamp <= datetime($end)
RETURN count(p) AS total_posts,
       sum(CASE WHEN p.content_label IN $flagged THEN 1 ELSE 0 END) AS flagged_posts
"""


def banner(step: str) -> None:
    print(f"\n=== {step} ===")


def llm_available(settings: LLMSettings) -> bool:
    """Check that the LLM server is up and the configured model is pulled."""
    try:
        resp = httpx.get(f"{settings.base_url.rstrip('/')}/models", timeout=5)
        resp.raise_for_status()
        models = [m.get("id", "") for m in resp.json().get("data", [])]
    except Exception as exc:
        print(f"  LLM server not reachable at {settings.base_url} ({exc}).")
        return False
    if settings.model_name not in models:
        print(f"  Model '{settings.model_name}' not found on server. Available: {models}")
        print(f"  Pull it with: ollama pull {settings.model_name}")
        return False
    if settings.provider.lower().startswith("ollama"):
        # Load the model into memory first, so the first real question isn't charged for the disk load
        print(f"  Loading {settings.model_name} into memory (first time can take a minute)...")
        start = time.time()
        root = settings.base_url.rstrip("/").removesuffix("/v1")
        try:
            httpx.post(f"{root}/api/generate", json={"model": settings.model_name, "prompt": "hi", "stream": False,
                                                     "think": False, "options": {"num_predict": 1}}, timeout=600)
            print(f"  Model ready in {time.time() - start:.0f}s")
        except Exception as exc:
            print(f"  Warm-up failed ({exc}); continuing anyway.")
    return True


def offline_community_evidence(users_df: pd.DataFrame, posts_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Per-community stats computed from the CSVs (planted communities), used when Neo4j is not available."""
    merged = posts_df.merge(users_df[["user_id", "synthetic_ground_truth_community"]], on="user_id")
    rows = []
    for cid, grp in merged.groupby("synthetic_ground_truth_community"):
        flagged = int(grp["content_label"].isin(FLAGGED_LABELS).sum())
        rows.append(
            {
                "community_id": int(cid),
                "community_source": "synthetic_ground_truth (offline, not Louvain)",
                "user_count": int(grp["user_id"].nunique()),
                "total_posts": len(grp),
                "flagged_posts": flagged,
                "flagged_ratio": round(flagged / len(grp), 4),
                "is_synthetic": True,
            }
        )
    return sorted(rows, key=lambda r: r["flagged_ratio"], reverse=True)


def run_neo4j_stage(users_df, posts_df, inter_df, temporal_report, summary: Dict[str, Any]):
    """Schema, ingestion, GDS Louvain/centrality and a Cypher cross-check of the temporal windows."""
    from graph_enhanced_lm.graph.constraints import initialize_schema
    from graph_enhanced_lm.graph.ingestion import ingest_canonical_data
    from graph_enhanced_lm.graph.neo4j_client import Neo4jClient
    from scripts.run_graph_algorithms import main as run_graph_algorithms

    client = Neo4jClient()
    client.connect()
    print(f"Connected to Neo4j at {client.settings.uri}")

    initialize_schema(client)
    constraints = client.execute_read("SHOW CONSTRAINTS YIELD name RETURN count(*) AS n")[0]["n"]
    print(f"Schema ready: {constraints} constraints")

    counts = ingest_canonical_data(client, users_df, posts_df, inter_df)
    node_counts = client.execute_read("MATCH (n) RETURN labels(n)[0] AS label, count(*) AS count ORDER BY label")
    print(f"Ingested {counts}")
    print("Graph now has: " + ", ".join(f"{r['label']}={r['count']}" for r in node_counts))

    print("\n--- GDS: projection, Louvain, Degree, PageRank ---")
    run_graph_algorithms()
    recovery_file = METRICS_DIR / "louvain_vs_ground_truth.json"
    summary["louvain_vs_ground_truth"] = json.loads(recovery_file.read_text(encoding="utf-8"))

    print("\n--- Temporal windows: Cypher (Neo4j) vs pandas ---")
    all_match = True
    for win in temporal_report["windows"]:
        row = client.execute_read(
            TEMPORAL_CYPHER, {"start": win["start_iso"], "end": win["end_iso"], "flagged": FLAGGED_LABELS}
        )[0]
        total, flagged = int(row["total_posts"]), int(row["flagged_posts"] or 0)
        ratio = round(flagged / total, 4) if total else 0.0
        match = total == win["total_posts"] and ratio == win["flagged_ratio"]
        all_match &= match
        print(f"  {win['window_name']}: neo4j total={total} ratio={ratio} | pandas total={win['total_posts']} "
              f"ratio={win['flagged_ratio']} -> {'MATCH' if match else 'MISMATCH'}")
    summary["temporal_neo4j_matches_pandas"] = all_match
    return client


def run_pipeline_demo(use_neo4j: bool = False, use_llm: bool = False, election_date: str = "2026-11-01") -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {"mode": {"neo4j": use_neo4j, "llm": use_llm}}

    banner("[1/7] Generating reproducible synthetic dataset")
    generate_synthetic_dataset(DATA_DIR, seed=42, election_date=election_date)
    users_df, posts_df, inter_df = load_canonical_tables(DATA_DIR)

    banner("[2/7] Preprocessing & validating canonical tables")
    users_df, posts_df, inter_df = deduplicate_canonical_tables(users_df, posts_df, inter_df)
    report = validate_canonical_dataframes(users_df, posts_df, inter_df, strict=True)
    print(f"Validation passed: {report.is_valid} | Users={report.num_users}, Posts={report.num_posts}, "
          f"Interactions={report.num_interactions}")
    print(f"Label distribution: {report.label_distribution}")
    summary["dataset"] = {"users": report.num_users, "posts": report.num_posts,
                          "interactions": report.num_interactions, "labels": report.label_distribution}

    banner("[3/7] Content classifier (heuristic baseline)")
    scored_df = PostContentClassifier(mode="heuristic_fallback").classify_dataframe(posts_df, preserve_human_labels=False)
    eval_metrics = evaluate_predictions(posts_df["content_label"].tolist(), scored_df["content_label"].tolist())
    (METRICS_DIR / "classifier_eval_synthetic.json").write_text(json.dumps(eval_metrics, indent=2), encoding="utf-8")
    print(f"Macro-F1={eval_metrics['macro_f1']} Accuracy={eval_metrics['accuracy']} "
          "(synthetic templates are trivially separable; not a real classifier benchmark)")

    banner("[4/7] Temporal election-window analysis (pandas)")
    temporal_report = analyze_temporal_dataframe(posts_df, election_date_str=election_date)
    (METRICS_DIR / "temporal_analysis.json").write_text(json.dumps(temporal_report, indent=2), encoding="utf-8")
    for win in temporal_report["windows"]:
        print(f"  {win['window_name']}: total_posts={win['total_posts']}, flagged_ratio={win['flagged_ratio']}")
    summary["temporal_windows"] = {w["window_name"]: w["flagged_ratio"] for w in temporal_report["windows"]}

    banner("[5/7] Neo4j: schema, ingestion, GDS Louvain, temporal cross-check")
    neo4j_client = None
    if use_neo4j:
        neo4j_client = run_neo4j_stage(users_df, posts_df, inter_df, temporal_report, summary)
    else:
        print("Skipped (run with --neo4j once Neo4j + GDS is running).")

    banner("[6/7] Text-to-Cypher & safety validation")
    llm_settings = LLMSettings()
    live_llm = use_llm and llm_available(llm_settings)
    if use_llm and not live_llm:
        print("  Falling back to the offline mock LLM.")
    print(f"LLM: {llm_settings.model_name if live_llm else 'offline mock'}")
    summary["llm_used"] = llm_settings.model_name if live_llm else "offline mock"
    # In live mode, LLM failures are reported, never replaced with mock text
    qwen = QwenClient(settings=llm_settings, offline_mock=not live_llm, allow_mock_fallback=not live_llm)
    t2c = TextToCypherEngine(llm=qwen, neo4j_client=neo4j_client)

    t2c_results = []
    for question in DEMO_QUESTIONS:
        out = t2c.query_graph(question)
        status = "VALID" if out.is_valid else f"REJECTED ({out.validation_error})"
        rows_info = f", rows={len(out.rows)}" if neo4j_client is not None and out.is_valid else ""
        print(f"\nQ: {question}\n{out.generated_cypher}\n-> {status}{rows_info}")
        t2c_results.append({"question": question, "cypher": out.generated_cypher, "valid": out.is_valid,
                            "error": out.validation_error, "rows": len(out.rows)})

    print("\nDirect injection attempts against the validator:")
    blocked = 0
    for attack in ATTACK_QUERIES:
        try:
            validate_and_sanitize_cypher(attack)
            print(f"  ALLOWED (!): {attack}")
        except Exception as exc:
            blocked += 1
            print(f"  BLOCKED: {attack[:60]}... ({type(exc).__name__})")
    summary["text_to_cypher"] = t2c_results
    summary["attacks_blocked"] = f"{blocked}/{len(ATTACK_QUERIES)}"

    banner("[7/7] Graph RAG with grounded evidence")
    rag = GraphRAGPipeline(text_to_cypher=t2c, llm=qwen)
    question = DEMO_QUESTIONS[0]
    fallback = None if neo4j_client is not None else offline_community_evidence(users_df, posts_df)
    rag_resp = rag.answer_question(question, fallback_records=fallback)
    source = "Neo4j query results" if neo4j_client is not None and not fallback else "CSV-computed community stats"
    print(f"Question: {question}")
    print(f"Evidence: {len(rag_resp.evidence_records)} records from {source}")
    print(f"Executed Cypher:\n{rag_resp.executed_cypher}")
    print(f"Evidence rows: {json.dumps(rag_resp.evidence_records, default=str)}")
    print(f"Answer:\n{rag_resp.grounded_answer}")
    if rag_resp.has_sufficient_evidence:
        if rag_resp.passes_grounding_check:
            print("Grounding check: PASSED (every number in the answer appears in the evidence)")
        else:
            print(f"Grounding check: FAILED - numbers not in the evidence: {rag_resp.ungrounded_numbers}")
    summary["graph_rag"] = {"question": question, "evidence_source": source,
                            "executed_cypher": rag_resp.executed_cypher,
                            "evidence_records": rag_resp.evidence_records, "answer": rag_resp.grounded_answer,
                            "ungrounded_numbers": rag_resp.ungrounded_numbers}

    if neo4j_client is not None:
        neo4j_client.close()
    (METRICS_DIR / "stage1_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved run summary to {METRICS_DIR / 'stage1_summary.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1 end-to-end demo.")
    parser.add_argument("--neo4j", action="store_true", help="Use a running Neo4j + GDS instance.")
    parser.add_argument("--llm", action="store_true", help="Use the configured Qwen model via Ollama.")
    parser.add_argument("--election-date", default="2026-11-01")
    args = parser.parse_args()
    run_pipeline_demo(use_neo4j=args.neo4j, use_llm=args.llm, election_date=args.election_date)
