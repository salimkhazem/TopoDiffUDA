"""SegFormer wrapper using transformers."""

from typing import Optional

import torch
import torch.nn as nn


class SegFormer(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        try:
            from transformers import SegformerConfig, SegformerForSemanticSegmentation
        except Exception as exc:
            raise ImportError("transformers is required for SegFormer") from exc

        config = SegformerConfig(
            num_labels=num_classes,
            num_channels=in_channels,
        )
        self.model = SegformerForSemanticSegmentation(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=x)
        return outputs.logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model.segformer(pixel_values=x, output_hidden_states=True, return_dict=True)
        return outputs.hidden_states[-1]
