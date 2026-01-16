"""DeepLabV3+ wrapper (approximation using torchvision)."""

from typing import Dict

import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50


class DeepLabV3Plus(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, pretrained: bool = False):
        super().__init__()
        self.model = deeplabv3_resnet50(pretrained=pretrained, aux_loss=False, num_classes=num_classes)
        if in_channels != 3:
            self.model.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)["out"]

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        features: Dict[str, torch.Tensor] = self.model.backbone(x)
        return features["out"]
