# src/graph_enhanced_lm/utils/logging_config.py
"""Privacy-conscious structured logging configuration."""
from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(log_level: int = logging.INFO, log_file: str | Path | None = None) -> logging.Logger:
    """Configure root and package logger without logging sensitive credentials."""
    logger = logging.getLogger("graph_enhanced_lm")
    logger.setLevel(log_level)
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger