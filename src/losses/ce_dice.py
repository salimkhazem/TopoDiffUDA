"""Cross-entropy and Dice losses."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(probs: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    num_classes = probs.shape[1]
    targets_onehot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = torch.sum(probs * targets_onehot, dims)
    cardinality = torch.sum(probs + targets_onehot, dims)
    dice = (2.0 * intersection + eps) / (cardinality + eps)
    return 1.0 - dice.mean()


class DiceLoss(nn.Module):
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape[1] == 1:
            probs = torch.sigmoid(logits)
            targets = targets.float().unsqueeze(1)
            intersection = torch.sum(probs * targets)
            cardinality = torch.sum(probs + targets)
            dice = (2.0 * intersection + 1e-6) / (cardinality + 1e-6)
            return 1.0 - dice
        probs = F.softmax(logits, dim=1)
        return dice_loss(probs, targets)


class CEDiceLoss(nn.Module):
    def __init__(self, weight: Optional[torch.Tensor] = None, ignore_index: Optional[int] = None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape[-2:] != targets.shape[-2:]:
            logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)
        return self.ce(logits, targets) + self.dice(logits, targets)
