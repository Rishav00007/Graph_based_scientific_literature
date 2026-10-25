# src/graph_enhanced_lm/classification/transformer_classifier.py
"""Modular post content classifier supporting HuggingFace Transformers and offline fallback."""
from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import List
import pandas as pd

from graph_enhanced_lm.classification.label_mapping import map_external_label
from graph_enhanced_lm.data.schemas import ContentCategory

logger = logging.getLogger("graph_enhanced_lm")


@dataclass
class ClassificationResult:
    post_id: str
    predicted_label: str
    confidence: float
    classifier_model: str
    classifier_version: str
    label_source: str = "model_prediction"


class PostContentClassifier:
    """Reusable interface for post classification, decoupled from Qwen3-8B and Neo4j."""

    def __init__(
        self,
        mode: str = "heuristic_fallback",
        model_name: str = "cardiffnlp/twitter-roberta-base-hate-multiclass-latest",
        version: str = "1.0.0",
        device: str = "cpu",
        max_length: int = 256,
        batch_size: int = 16,
    ) -> None:
        self.mode = mode
        self.model_name = model_name
        self.version = version
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self._hf_pipeline = None

        if self.mode == "pretrained_transformer":
            self._load_hf_pipeline()

    def _load_hf_pipeline(self) -> None:
        try:
            from transformers import pipeline
            device_idx = 0 if self.device == "cuda" else -1
            self._hf_pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                tokenizer=self.model_name,
                device=device_idx,
                truncation=True,
                max_length=self.max_length,
            )
            logger.info("Loaded HuggingFace classifier: %s", self.model_name)
        except Exception as exc:
            logger.warning(
                "Could not load HuggingFace model '%s' (%s). Falling back to heuristic_fallback.",
                self.model_name,
                exc,
            )
            self.mode = "heuristic_fallback"

    def predict_batch(self, post_ids: List[str], texts: List[str]) -> List[ClassificationResult]:
        """Classify a batch of posts and return structured provenance metadata."""
        if len(post_ids) != len(texts):
            raise ValueError("post_ids and texts must have identical length.")

        if self.mode == "pretrained_transformer" and self._hf_pipeline is not None:
            raw_preds = self._hf_pipeline(texts, batch_size=self.batch_size)
            results: List[ClassificationResult] = []
            for pid, pred in zip(post_ids, raw_preds):
                mapped = map_external_label(str(pred.get("label", "unknown")), "generic")
                results.append(
                    ClassificationResult(
                        post_id=pid,
                        predicted_label=mapped,
                        confidence=round(float(pred.get("score", 0.0)), 4),
                        classifier_model=self.model_name,
                        classifier_version=self.version,
                    )
                )
            return results

        # Deterministic lexicon/pattern classifier for reproducible offline baseline & testing
        results = []
        for pid, text in zip(post_ids, texts):
            lower = text.lower()
            if any(k in lower for k in ["free giveaway", "click link", "crypto airdrop", "guaranteed profit", "promo code"]):
                label, conf = ContentCategory.SPAM.value, 0.91
            elif any(k in lower for k in ["expel group", "boycott community", "target faction", "outsiders must leave"]):
                label, conf = ContentCategory.COMMUNITY_TARGETED.value, 0.86
            elif any(k in lower for k in ["hate speech pattern", "vermin", "eradicate them", "subhuman", "traitors must pay"]):
                label, conf = ContentCategory.HATE.value, 0.89
            elif any(k in lower for k in ["idiot", "shut up", "corrupt clowns", "stupid fools", "garbage take"]):
                label, conf = ContentCategory.OFFENSIVE.value, 0.82
            else:
                label, conf = ContentCategory.NORMAL.value, 0.94

            results.append(
                ClassificationResult(
                    post_id=pid,
                    predicted_label=label,
                    confidence=conf,
                    classifier_model="deterministic_lexicon_baseline",
                    classifier_version=self.version,
                )
            )
        return results

    def classify_dataframe(self, posts_df: pd.DataFrame, preserve_human_labels: bool = True) -> pd.DataFrame:
        """Run batch inference on a posts DataFrame without overwriting human annotations if requested."""
        df = posts_df.copy()
        preds = self.predict_batch(df["post_id"].astype(str).tolist(), df["text"].astype(str).tolist())
        pred_map = {p.post_id: p for p in preds}

        for idx, row in df.iterrows():
            if preserve_human_labels and str(row.get("label_source", "")) == "human_annotation":
                continue
            res = pred_map[str(row["post_id"])]
            df.at[idx, "content_label"] = res.predicted_label
            df.at[idx, "classification_confidence"] = res.confidence
            df.at[idx, "classifier_model"] = res.classifier_model
            df.at[idx, "classifier_version"] = res.classifier_version
            if str(row.get("label_source", "")) not in {"synthetic_ground_truth", "human_annotation"}:
                df.at[idx, "label_source"] = res.label_source
        return df