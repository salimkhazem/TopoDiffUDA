"""Path utilities."""

import os
from pathlib import Path


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_data_root() -> Path:
    env = os.environ.get("TOPODIFFUDA_DATA")
    if env:
        return Path(env)
    repo_root = get_repo_root()
    candidate = repo_root.parent / "data"
    if candidate.exists():
        return candidate
    return repo_root / "data"


def get_outputs_root() -> Path:
    repo_root = get_repo_root()
    return repo_root / "outputs"
