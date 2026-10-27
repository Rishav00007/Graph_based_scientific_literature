# src/graph_enhanced_lm/llm/qwen_client.py
"""Qwen3-8B client supporting local Ollama/vLLM OpenAI-compatible endpoints and offline fallback."""
from __future__ import annotations

import logging
import re
from typing import Optional
import httpx

from graph_enhanced_lm.config import LLMSettings

logger = logging.getLogger("graph_enhanced_lm")


class QwenClient:
    """Clean interface for Qwen3-8B inference (local Ollama/vLLM or configurable endpoint)."""

    def __init__(
        self,
        settings: Optional[LLMSettings] = None,
        offline_mock: bool = False,
        allow_mock_fallback: bool = True,
    ) -> None:
        """allow_mock_fallback=False makes LLM failures raise instead of silently returning mock text."""
        self.settings = settings or LLMSettings()
        self.offline_mock = offline_mock
        self.allow_mock_fallback = allow_mock_fallback

    def _fallback(self, exc: Exception, system_prompt: str, user_prompt: str) -> str:
        if not self.allow_mock_fallback:
            raise RuntimeError(f"LLM call to {self.settings.model_name} failed: {exc}") from exc
        logger.warning("LLM call failed (%s). Using deterministic template fallback.", exc)
        return self._mock_response(system_prompt, user_prompt)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a completion from Qwen3-8B, stripping <think>...</think> blocks if present."""
        if self.offline_mock:
            return self._mock_response(system_prompt, user_prompt)

        if self.settings.provider.lower().startswith("ollama"):
            return self._generate_ollama_native(system_prompt, user_prompt)

        url = f"{self.settings.base_url.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {self.settings.api_key}", "Content-Type": "application/json"}
        # Qwen3 "thinks" before answering by default, which is very slow on CPU. /no_think turns it off.
        if self.settings.model_name.lower().startswith("qwen3"):
            user_prompt = f"{user_prompt} /no_think"
        payload = {
            "model": self.settings.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
        }
        try:
            with httpx.Client(timeout=self.settings.timeout_seconds) as client:
                resp = client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                # Remove Qwen3 thinking tags when returning structured Cypher or grounded answers
                cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                return cleaned
        except Exception as exc:
            return self._fallback(exc, system_prompt, user_prompt)

    def _generate_ollama_native(self, system_prompt: str, user_prompt: str) -> str:
        """Call Ollama's native /api/chat with thinking disabled.

        Ollama's OpenAI-compatible endpoint can spend the whole token budget on Qwen3's hidden
        "thinking" and return an empty answer. The native API lets us turn thinking off with think=false.
        """
        root = re.sub(r"/v1/?$", "", self.settings.base_url.rstrip("/"))
        payload = {
            "model": self.settings.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "think": False,
            "options": {"temperature": self.settings.temperature, "num_predict": self.settings.max_tokens},
        }
        try:
            with httpx.Client(timeout=self.settings.timeout_seconds) as client:
                resp = client.post(f"{root}/api/chat", json=payload)
                resp.raise_for_status()
                message = resp.json().get("message", {})
        except Exception as exc:
            return self._fallback(exc, system_prompt, user_prompt)

        content = re.sub(r"<think>.*?</think>", "", message.get("content") or "", flags=re.DOTALL).strip()
        if not content:
            # Some model/server combinations still put everything in the thinking field
            content = (message.get("thinking") or "").strip()
            logger.warning("LLM returned an empty answer; using its thinking text instead (%d chars).", len(content))
        return content

    def _mock_response(self, system_prompt: str, user_prompt: str) -> str:
        lower = user_prompt.lower()
        if "cypher" in system_prompt.lower():
            if "communit" in lower and ("proportion" in lower or "hate" in lower):
                return (
                    "```cypher\n"
                    "MATCH (u:User)-[:POSTED]->(p:Post)\n"
                    "WHERE u.louvain_community_id IS NOT NULL\n"
                    "WITH u.louvain_community_id AS community_id, count(p) AS total_posts,\n"
                    "     sum(CASE WHEN p.content_label IN ['HATE', 'OFFENSIVE', 'SPAM', 'COMMUNITY_TARGETED'] THEN 1 ELSE 0 END) AS flagged_posts\n"
                    "RETURN community_id, total_posts, flagged_posts,\n"
                    "       round(toFloat(flagged_posts) / total_posts, 4) AS flagged_ratio\n"
                    "ORDER BY flagged_ratio DESC LIMIT 20\n"
                    "```"
                )
            if "repeatedly amplified" in lower or "retweeted" in lower:
                return (
                    "```cypher\n"
                    "MATCH (u:User)-[r:RETWEETED]->(p:Post)\n"
                    "RETURN u.user_id AS user_id, p.post_id AS post_id, p.content_label AS label, count(r) AS retweet_count\n"
                    "ORDER BY retweet_count DESC LIMIT 25\n"
                    "```"
                )
            return (
                "```cypher\n"
                "MATCH (u:User)-[:POSTED]->(p:Post)\n"
                "WHERE p.content_label = 'HATE'\n"
                "RETURN u.user_id AS user_id, p.post_id AS post_id, p.timestamp_str AS timestamp, p.content_label AS label\n"
                "ORDER BY p.timestamp DESC LIMIT 25\n"
                "```"
            )
        # Mock Graph RAG grounded response
        if "No matching graph records" in user_prompt:
            return "Insufficient graph evidence was retrieved to answer this question."
        return (
            "[Offline mock answer - no LLM was called] Based on the retrieved Graph Evidence Bundle, "
            "each cited record includes its provenance and content classification. "
            "Communities showing elevated proportions of HATE or SPAM posts are designated as candidate communities "
            "exhibiting hate/spam propagation patterns and require human review before any attribution."
        )