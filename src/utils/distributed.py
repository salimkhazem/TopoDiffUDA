"""Distributed helpers."""

import os
from typing import Optional

import torch
import torch.distributed as dist


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed(backend: str = "nccl", init_method: Optional[str] = None) -> None:
    if dist.is_initialized():
        return
    if init_method is None:
        init_method = os.environ.get("INIT_METHOD", "env://")
    dist.init_process_group(backend=backend, init_method=init_method)


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
