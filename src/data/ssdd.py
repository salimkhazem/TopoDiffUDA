"""SSDD SAR ship segmentation dataset."""

from pathlib import Path
from typing import Dict, List, Tuple

from src.data.base_dataset import BaseSegDataset, cache_split_manifest, load_split_manifest
from src.utils.paths import get_data_root


def _collect_pairs(images_dir: Path, masks_dir: Path) -> List[Tuple[Path, Path]]:
    masks = {p.stem: p for p in masks_dir.glob("*.*")}
    items = []
    for img in sorted(images_dir.glob("*.*")):
        mask = masks.get(img.stem)
        if mask is not None:
            items.append((img, mask))
    return items


def get_splits() -> Dict[str, List[Tuple[Path, Path]]]:
    cached = load_split_manifest("ssdd")
    if cached:
        return {k: [(Path(a), Path(b)) for a, b in v] for k, v in cached.items()}

    root = get_data_root() / "ssdd"
    if not root.exists():
        raise FileNotFoundError("SSDD dataset not found. Place data in data/ssdd")

    splits = {}
    for split in ["train", "val", "test"]:
        images_dir = root / "images" / split
        masks_dir = root / "masks" / split
        items = _collect_pairs(images_dir, masks_dir)
        if not items:
            raise FileNotFoundError(f"SSDD split missing images/masks under {images_dir}")
        splits[split] = items

    cache_split_manifest("ssdd", {k: [(str(a), str(b)) for a, b in v] for k, v in splits.items()})
    return splits


class SSDDDataset(BaseSegDataset):
    def __init__(self, split: str, transforms=None, config=None):
        splits = get_splits()
        image_size = tuple(config["dataset"]["image_size"]) if config else None
        super().__init__("ssdd", splits[split], split, transforms=transforms, image_size=image_size)
