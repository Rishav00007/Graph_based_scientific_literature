# src/graph_enhanced_lm/data/loaders.py
"""Load canonical users/posts/interactions CSVs into clean DataFrames."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


def _nan_to_none(df: pd.DataFrame) -> pd.DataFrame:
    """Replace NaN with None so Pydantic validation and the Neo4j driver see real nulls."""
    return df.astype(object).where(df.notna(), None)


def load_canonical_tables(data_dir: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read users.csv, posts.csv and interactions.csv from a directory."""
    base = Path(data_dir)
    missing = [f for f in ("users.csv", "posts.csv", "interactions.csv") if not (base / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in {base}. Generate data first: python scripts/generate_synthetic_data.py"
        )
    users = _nan_to_none(pd.read_csv(base / "users.csv"))
    posts = _nan_to_none(pd.read_csv(base / "posts.csv"))
    interactions = _nan_to_none(pd.read_csv(base / "interactions.csv"))
    return users, posts, interactions
