"""Checkpointing helpers."""

from pathlib import Path
from typing import Dict

import torch


def save_checkpoint(state: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: Path, device: torch.device) -> Dict:
    return torch.load(path, map_location=device)
