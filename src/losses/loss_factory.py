"""Loss factory helpers."""

from typing import Dict, Optional

import torch
import torch.nn as nn

from .ce_dice import CEDiceLoss, DiceLoss
from .topology import cldice_loss, topology_consistency_loss


def build_supervised_loss(ignore_index: Optional[int] = None) -> nn.Module:
    return CEDiceLoss(ignore_index=ignore_index)


def compute_topology_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.shape[1] == 1:
        pred_prob = torch.sigmoid(logits)
        target_prob = targets.float().unsqueeze(1)
    else:
        pred_prob = torch.softmax(logits, dim=1)[:, 1:2]
        target_prob = (targets == 1).float().unsqueeze(1)
    return cldice_loss(pred_prob, target_prob)


def compute_topology_consistency(logits: torch.Tensor, pseudo: torch.Tensor) -> torch.Tensor:
    if logits.shape[1] == 1:
        pred_prob = torch.sigmoid(logits)
        pseudo_prob = pseudo.float().unsqueeze(1)
    else:
        pred_prob = torch.softmax(logits, dim=1)[:, 1:2]
        pseudo_prob = pseudo.float().unsqueeze(1)
    return topology_consistency_loss(pred_prob, pseudo_prob)
