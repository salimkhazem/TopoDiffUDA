"""Consistency loss for weak/strong augmentations."""

import torch
import torch.nn.functional as F


def consistency_loss(logits_weak: torch.Tensor, logits_strong: torch.Tensor) -> torch.Tensor:
    if logits_weak.shape[1] == 1:
        probs_weak = torch.sigmoid(logits_weak)
        probs_strong = torch.sigmoid(logits_strong)
    else:
        probs_weak = torch.softmax(logits_weak, dim=1)
        probs_strong = torch.softmax(logits_strong, dim=1)
    return F.mse_loss(probs_weak, probs_strong)
