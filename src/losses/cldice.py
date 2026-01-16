"""clDice loss wrapper."""

import torch
from .topology import cldice_loss


def cldice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.shape[1] == 1:
        probs = torch.sigmoid(logits)
        targets = targets.float().unsqueeze(1)
    else:
        probs = torch.softmax(logits, dim=1)[:, 1:2]
        targets = (targets == 1).float().unsqueeze(1)
    return cldice_loss(probs, targets)
