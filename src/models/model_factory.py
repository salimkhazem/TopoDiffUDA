"""Model factory."""

from typing import Dict

from .deeplabv3plus import DeepLabV3Plus
from .segformer import SegFormer
from .swin_unet import SwinUNet
from .unet import UNet


def build_model(config: Dict) -> object:
    name = config["model"]["name"]
    in_channels = config["model"].get("in_channels", 3)
    num_classes = config["model"]["num_classes"]

    if name == "unet":
        return UNet(in_channels, num_classes, base_channels=config["model"].get("base_channels", 32))
    if name == "deeplabv3p":
        return DeepLabV3Plus(in_channels, num_classes, pretrained=config["model"].get("pretrained", False))
    if name == "segformer":
        return SegFormer(in_channels, num_classes)
    if name == "swin_unet":
        image_size = tuple(config.get("dataset", {}).get("image_size", [224, 224]))
        return SwinUNet(in_channels, num_classes, image_size=image_size)

    raise ValueError(f"Unknown model name: {name}")
