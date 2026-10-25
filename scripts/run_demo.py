#!/usr/bin/env python3
"""End-to-end demonstration runner supporting both Live Neo4j/Qwen3-8B and Offline Mock mode."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

from graph_enhanced_lm.classification.evaluation import evaluate_predictions
from graph_enhanced_lm.classification.transformer_classifier import PostContentClassifier
from graph_enhanced_lm.data.preprocessing import deduplicate_canonical_tables
from graph_enhanced_lm.data.validators import validate_canonical_dataframes
from graph_enhanced_lm.llm.qwen_client import QwenClient
from graph_enhanced_lm.llm.text_to_cypher import TextToCypherEngine
from graph_enhanced_lm.retrieval.graph_rag import GraphRAGPipeline
from graph_enhanced_lm.temporal.election_windows import analyze_temporal_dataframe
from scripts.generate_synthetic_data import generate_synthetic_dataset


def run_pipeline_demo(offline_mock: bool = True, election_date: str = "2026-11-01") -> None:
    print("=== [1/6] Generating Reproducible Synthetic Dataset ===")
    paths = generate_synthetic_dataset("data/synthetic", seed=42, election_date=election_date)
    users_df = pd.read_csv(paths["users"])
    posts_df = pd.read_csv(paths["posts"])
    inter_df = pd.read_csv(paths["interactions"]).where(pd.notnull(pd.read_csv(paths["interactions"])), None)

    print("=== [2/6] Preprocessing & Validating Canonical Tables ===")
    users_df, posts_df, inter_df = deduplicate_canonical_tables(users_df, posts_df, inter_df)
    report = validate_canonical_dataframes(users_df, posts_df, inter_df, strict=True)
    print(f"Validation Passed: {report.is_valid} | Users={report.num_users}, Posts={report.num_posts}, Interactions={report.num_interactions}")
    print(f"Label Distribution: {report.label_distribution}")

    print("=== [3/6] Running Content Classifier & Computing Evaluation Metrics ===")
    classifier = PostContentClassifier(mode="heuristic_fallback")
    scored_df = classifier.classify_dataframe(posts_df, preserve_human_labels=False)
    eval_metrics = evaluate_predictions(
        y_true=posts_df["content_label"].tolist(),
        y_pred=scored_df["content_label"].tolist(),
    )
    Path("outputs/metrics").mkdir(parents=True, exist_ok=True)
    with open("outputs/metrics/classifier_eval_synthetic.json", "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=2)
    print(f"Classifier Macro-F1 on synthetic benchmark: {eval_metrics['macro_f1']} | Accuracy: {eval_metrics['accuracy']}")

    print("=== [4/6] Running Temporal & Election-Window Analysis ===")
    temporal_report = analyze_temporal_dataframe(posts_df, election_date_str=election_date)
    with open("outputs/metrics/temporal_analysis.json", "w", encoding="utf-8") as f:
        json.dump(temporal_report, f, indent=2)
    for win in temporal_report["windows"]:
        print(f"  Window [{win['window_name']}]: total_posts={win['total_posts']}, flagged_ratio={win['flagged_ratio']}")

    print("=== [5/6] Testing Text-to-Cypher & Safety Validation ===")
    qwen = QwenClient(offline_mock=offline_mock)
    t2c = TextToCypherEngine(llm=qwen, neo4j_client=None)
    sample_question = "Which communities contain a high proportion of hate-labelled posts?"
    t2c_out = t2c.translate(sample_question)
    print(f"Question: {sample_question}\nValidated Cypher:\n{t2c_out.generated_cypher}\nValid: {t2c_out.is_valid}")

    print("=== [6/6] Running Graph RAG with Grounded Evidence Bundle ===")
    rag = GraphRAGPipeline(text_to_cypher=t2c, llm=qwen)
    sample_evidence = [
        {
            "community_id": 2,
            "user_count": 11,
            "total_posts": 55,
            "flagged_posts": 46,
            "flagged_ratio": 0.8364,
            "analytical_status": "candidate_community_exhibiting_hate_spam_propagation_patterns",
            "is_synthetic": True,
        }
    ]
    rag_resp = rag.answer_question(sample_question, fallback_records=sample_evidence)
    print(f"Graph RAG Grounded Answer:\n{rag_resp.grounded_answer}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline-mock", action="store_true", default=True)
    parser.add_argument("--election-date", default="2026-11-01")
    args = parser.parse_args()
    run_pipeline_demo(offline_mock=args.offline_mock, election_date=args.election_date)