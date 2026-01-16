"""Logging utilities."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from .io import append_jsonl, ensure_dir, write_csv


def setup_logging(log_dir: Path, name: str = "train") -> logging.Logger:
    ensure_dir(log_dir)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(log_dir / f"{name}.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


class MetricsLogger:
    """Simple metrics logger for per-epoch JSONL and final CSV summary."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        ensure_dir(self.output_dir)
        self.jsonl_path = self.output_dir / "metrics.jsonl"
        self.summary_path = self.output_dir / "summary.csv"
        self.last_summary: Optional[Dict[str, Any]] = None

    def log(self, metrics: Dict[str, Any]) -> None:
        append_jsonl(self.jsonl_path, metrics)

    def summarize(self, summary: Dict[str, Any]) -> None:
        self.last_summary = summary
        write_csv(self.summary_path, [summary])
