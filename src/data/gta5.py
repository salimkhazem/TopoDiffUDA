"""GTA5 synthetic dataset."""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.data.base_dataset import BaseSegDataset, cache_split_manifest, load_split_manifest
from src.utils.image import load_image, load_mask, resize_pair
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
    cached = load_split_manifest("gta5")
    if cached:
        return {k: [(Path(a), Path(b)) for a, b in v] for k, v in cached.items()}

    root = None
    for cand in [get_data_root() / "gta5", get_data_root() / "GTA5"]:
        if cand.exists():
            root = cand
            break
    if root is None:
        raise FileNotFoundError("GTA5 dataset not found. Place data in data/gta5 with images/labels")

    images_dir = root / "images"
    masks_dir = root / "labels"

    # Support split subfolders: images/train, labels/train, etc.
    if (images_dir / "train").exists() and (masks_dir / "train").exists():
        splits: Dict[str, List[Tuple[Path, Path]]] = {}
        for split in ["train", "val", "test"]:
            img_split = images_dir / split
            mask_split = masks_dir / split
            if not img_split.exists() or not mask_split.exists():
                continue
            split_items = _collect_pairs(img_split, mask_split)
            if split_items:
                splits[split] = split_items
        if "train" not in splits:
            raise FileNotFoundError("GTA5 train split not found under data/gta5/images/train and data/gta5/labels/train")
        if "val" not in splits:
            val_size = max(1, int(0.05 * len(splits["train"])))
            splits["val"] = splits["train"][-val_size:]
        if "test" not in splits:
            splits["test"] = splits["val"]
    else:
        items = _collect_pairs(images_dir, masks_dir)
        if not items:
            # Try alternate GTA5 root if available
            alt_root = get_data_root() / "GTA5"
            if alt_root.exists() and alt_root != root:
                images_dir = alt_root / "images"
                masks_dir = alt_root / "labels"
                items = _collect_pairs(images_dir, masks_dir)
        if not items:
            raise FileNotFoundError("GTA5 images/masks not found under data/gta5/images and data/gta5/labels")

        val_size = max(1, int(0.05 * len(items)))
        val_items = items[-val_size:]
        train_items = items[:-val_size]
        splits = {"train": train_items, "val": val_items, "test": val_items}
    cache_split_manifest("gta5", {k: [(str(a), str(b)) for a, b in v] for k, v in splits.items()})
    return splits


class GTA5Dataset(BaseSegDataset):
    def __init__(self, split: str, transforms=None, config=None):
        splits = get_splits()
        image_size = tuple(config["dataset"]["image_size"]) if config else None
        super().__init__("gta5", splits[split], split, transforms=transforms, image_size=image_size)

    def _load_item(self, index: int):
        image_path, mask_path = self.items[index]
        image = load_image(image_path)
        mask = load_mask(mask_path)
        if self.image_size is not None:
            image, mask = resize_pair(image, mask, self.image_size)
        mask = map_gta5_to_cityscapes(mask)
        return image, mask


_GTA5_TO_CITYSCAPES = {
    0: 255,
    1: 255,
    2: 255,
    3: 255,
    4: 255,
    5: 255,
    6: 255,
    7: 0,
    8: 1,
    9: 255,
    10: 255,
    11: 2,
    12: 3,
    13: 4,
    14: 255,
    15: 255,
    16: 255,
    17: 5,
    18: 255,
    19: 6,
    20: 7,
    21: 8,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    29: 255,
    30: 255,
    31: 16,
    32: 17,
    33: 18,
}


def map_gta5_to_cityscapes(mask: np.ndarray) -> np.ndarray:
    mapped = np.full_like(mask, 255)
    for k, v in _GTA5_TO_CITYSCAPES.items():
        mapped[mask == k] = v
    return mapped
