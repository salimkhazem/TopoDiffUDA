"""Differentiable topology losses."""

from typing import Optional

import torch
import torch.nn.functional as F


def soft_erode(x: torch.Tensor) -> torch.Tensor:
    p1 = -F.max_pool2d(-x, (3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-x, (1, 3), stride=1, padding=(0, 1))
    return torch.min(p1, p2)


def soft_dilate(x: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(x, 3, stride=1, padding=1)


def soft_open(x: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(x))


def soft_skeletonize(x: torch.Tensor, iters: int = 10) -> torch.Tensor:
    """Approximate skeletonization for soft masks."""
    skel = torch.zeros_like(x)
    for _ in range(iters):
        opened = soft_open(x)
        delta = F.relu(x - opened)
        skel = skel + F.relu(delta - skel * delta)
        x = soft_erode(x)
    return skel


def cldice_loss(pred_prob: torch.Tensor, target_prob: torch.Tensor, iters: int = 10) -> torch.Tensor:
    eps = 1e-6
    pred_skel = soft_skeletonize(pred_prob, iters)
    target_skel = soft_skeletonize(target_prob, iters)

    tprec = (pred_skel * target_prob).sum(dim=(1, 2, 3)) / (pred_skel.sum(dim=(1, 2, 3)) + eps)
    tsens = (target_skel * pred_prob).sum(dim=(1, 2, 3)) / (target_skel.sum(dim=(1, 2, 3)) + eps)
    cldice = (2 * tprec * tsens) / (tprec + tsens + eps)
    return 1.0 - cldice.mean()


def topology_consistency_loss(pred_prob: torch.Tensor, pseudo_prob: torch.Tensor, iters: int = 10) -> torch.Tensor:
    pred_skel = soft_skeletonize(pred_prob, iters)
    pseudo_skel = soft_skeletonize(pseudo_prob, iters)
    return F.l1_loss(pred_skel, pseudo_skel)
