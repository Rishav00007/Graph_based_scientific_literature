# src/graph_enhanced_lm/config.py
"""Centralized configuration loader using YAML files and environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Neo4jSettings:
    uri: str = field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    user: str = field(default_factory=lambda: os.getenv("NEO4J_USER", "neo4j"))
    password: str = field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", "password123"))
    database: str = field(default_factory=lambda: os.getenv("NEO4J_DATABASE", "neo4j"))
    readonly_user: str = field(default_factory=lambda: os.getenv("NEO4J_READONLY_USER", os.getenv("NEO4J_USER", "neo4j")))
    readonly_password: str = field(
        default_factory=lambda: os.getenv("NEO4J_READONLY_PASSWORD", os.getenv("NEO4J_PASSWORD", "password123"))
    )
    query_timeout_seconds: int = 15
    max_read_limit: int = 100


@dataclass
class LLMSettings:
    provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "ollama_openai_compat"))
    base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"))
    api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", "local-no-key-required"))
    model_name: str = field(default_factory=lambda: os.getenv("LLM_MODEL_NAME", "qwen3:8b"))
    fallback_model_name: str = field(default_factory=lambda: os.getenv("LLM_Fallback_MODEL_NAME", "qwen3:4b"))
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))
    timeout_seconds: int = int(os.getenv("LLM_TIMEOUT_SECONDS", "60"))


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file safely."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    with file_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}