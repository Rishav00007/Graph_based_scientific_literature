# src/graph_enhanced_lm/temporal/election_windows.py
"""Configurable election-period time windows and temporal activity comparison."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
import pandas as pd


@dataclass(frozen=True)
class ElectionWindowSpec:
    name: str
    start_iso: str
    end_iso: str


def build_election_windows(
    election_date_str: str = "2026-11-01",
    baseline_days_before: int = 30,
    critical_days_before: int = 10,
    days_after: int = 10,
) -> List[ElectionWindowSpec]:
    """Construct standard pre-election baseline, 10-day surge, election-day, and post-election windows."""
    election_dt = datetime.fromisoformat(election_date_str).replace(tzinfo=timezone.utc)

    baseline_start = election_dt - timedelta(days=baseline_days_before)
    critical_start = election_dt - timedelta(days=critical_days_before)
    election_end = election_dt + timedelta(days=1) - timedelta(seconds=1)
    post_end = election_dt + timedelta(days=days_after + 1) - timedelta(seconds=1)

    return [
        ElectionWindowSpec(
            name="pre_election_baseline",
            start_iso=baseline_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            end_iso=(critical_start - timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        ),
        ElectionWindowSpec(
            name="ten_days_before_election",
            start_iso=critical_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            end_iso=(election_dt - timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        ),
        ElectionWindowSpec(
            name="election_day",
            start_iso=election_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            end_iso=election_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        ),
        ElectionWindowSpec(
            name="post_election_window",
            start_iso=(election_end + timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            end_iso=post_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        ),
    ]


def analyze_temporal_dataframe(
    posts_df: pd.DataFrame,
    election_date_str: str = "2026-11-01",
) -> Dict[str, Any]:
    """Compute window-by-window post volumes and category proportions directly on canonical posts."""
    windows = build_election_windows(election_date_str)
    df = posts_df.copy()
    df["dt"] = pd.to_datetime(df["timestamp"], utc=True)

    window_summaries: List[Dict[str, Any]] = []
    for win in windows:
        w_start = pd.to_datetime(win.start_iso, utc=True)
        w_end = pd.to_datetime(win.end_iso, utc=True)
        sub = df[(df["dt"] >= w_start) & (df["dt"] <= w_end)]
        total = len(sub)
        counts = sub["content_label"].value_counts().to_dict()
        flagged = sum(counts.get(k, 0) for k in ["HATE", "OFFENSIVE", "COMMUNITY_TARGETED", "SPAM"])
        window_summaries.append(
            {
                "window_name": win.name,
                "start_iso": win.start_iso,
                "end_iso": win.end_iso,
                "total_posts": total,
                "label_counts": {str(k): int(v) for k, v in counts.items()},
                "flagged_ratio": round(flagged / total, 4) if total > 0 else 0.0,
            }
        )

    return {
        "configured_election_date": election_date_str,
        "causal_disclaimer": (
            "Observed changes in hate/spam volume across election windows represent temporal correlation, "
            "not proof that the election causally induced the activity."
        ),
        "windows": window_summaries,
    }