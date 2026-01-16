"""Seeding utilities for reproducibility."""

import os
import random
from typing import Optional

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True, warn_only: bool = False) -> None:
    """Seed python, numpy, torch, and (optionally) make algorithms deterministic."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=warn_only)
        except Exception:
            # Some ops may not support deterministic mode on all platforms.
            pass


def seed_worker(worker_id: int) -> None:
    """Seed dataloader workers deterministically."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_generator(seed: Optional[int]) -> Optional[torch.Generator]:
    if seed is None:
        return None
    g = torch.Generator()
    g.manual_seed(seed)
    return g
