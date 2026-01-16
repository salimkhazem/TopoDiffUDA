"""Simplified Swin-UNet style model using timm backbone."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwinUNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, image_size: tuple[int, int] | None = None):
        super().__init__()
        try:
            import timm
        except Exception as exc:
            raise ImportError("timm is required for SwinUNet") from exc

        img_size = image_size or (224, 224)
        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=False,
            in_chans=in_channels,
            features_only=True,
            img_size=img_size,
        )
        self.feat_channels = self.backbone.feature_info.channels()[-1]
        self.head = nn.Sequential(
            nn.Conv2d(self.feat_channels, self.feat_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat_channels // 2, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        feat = feats[-1]
        if feat.dim() == 4 and feat.shape[1] != self.feat_channels and feat.shape[-1] == self.feat_channels:
            feat = feat.permute(0, 3, 1, 2).contiguous()
        logits = self.head(feat)
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        feat = feats[-1]
        if feat.dim() == 4 and feat.shape[1] != self.feat_channels and feat.shape[-1] == self.feat_channels:
            feat = feat.permute(0, 3, 1, 2).contiguous()
        return feat
