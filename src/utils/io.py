"""IO helpers for configs, metrics, and environment snapshots."""

import csv
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, data: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    ensure_dir(path.parent)
    rows = list(rows)
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_jsonl(path: Path, data: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data) + "\n")


def get_git_commit_hash(repo_root: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def get_env_snapshot() -> Dict[str, Any]:
    try:
        import torch
    except Exception:
        torch = None

    env = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
    }
    if torch is not None:
        env.update(
            {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
            }
        )
        if torch.cuda.is_available():
            env["gpu_name"] = torch.cuda.get_device_name(0)
    return env


def write_env_snapshot(output_dir: Path, repo_root: Path) -> None:
    ensure_dir(output_dir)
    env = get_env_snapshot()
    env["git_commit"] = get_git_commit_hash(repo_root)
    write_json(output_dir / "env.json", env)

    pip_freeze_path = output_dir / "pip_freeze.txt"
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "freeze"], check=True, capture_output=True, text=True)
        pip_freeze_path.write_text(result.stdout, encoding="utf-8")
    except Exception:
        pip_freeze_path.write_text("pip freeze failed", encoding="utf-8")
